"""Artifact and resume checks for the A800 benchmark coordinator (CPU only)."""
from __future__ import annotations

import importlib.util
import copy
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_a800_nwm_eval.py"
SPEC = importlib.util.spec_from_file_location("run_a800_nwm_eval", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (12, 34, 56)).save(path)


class A800EvalTest(unittest.TestCase):
    def test_navigation_commands_compose_with_hydra(self):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(runner.ROOT / "conf"), version_base=None):
            for model in runner.MODELS_REQUESTED:
                if model == "rae-nwm":
                    continue  # This backend uses argparse, not Hydra.
                for dataset in ("recon", "scand", "go_stanford", "unitree_go2"):
                    with self.subTest(model=model, dataset=dataset):
                        command = runner.command_for({"model": model, "dataset": dataset,
                            "kind": "navigation", "evaluation": "cem80", "ids": [2, 7]}, Path("/tmp/a800_config_test"), 32)
                        overrides = command[command.index("planning_eval.py") + 1:]
                        config = compose(config_name="plan_config", overrides=overrides)
                        self.assertEqual(config.planning_sample_seed, 42)
                        self.assertEqual(list(config.planning_sample_indices), [2, 7])
                        self.assertEqual(config.num_samples, 80)
                        self.assertEqual(config.planning_microbatch_size, 80)
                        self.assertFalse(config.planning_write_aggregate)
                        distribution = config.plan_datasets_hyperparams[dataset]
                        self.assertEqual(len(distribution.mu), 3)
                        self.assertEqual(len(distribution.var_scale), 3)
                        if dataset == "go_stanford":
                            self.assertEqual(list(distribution.mu), [-0.1, 0.0, 0.0])
                            self.assertEqual(list(distribution.var_scale), [0.1, 0.15, 0.1])

    def test_tail_excludes_old_errors_but_keeps_new_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            log = run / "logs/coordinator.log"
            log.parent.mkdir()
            old = "[2026-09-23T08:00:00+00:00] old launch\nRuntimeError: old failure\n"
            log.write_text(old)
            runner.atomic_json(run / "preflight.json", {"at": "2026-09-23T11:00:00+00:00"})
            self.assertEqual(runner.coordinator_log_offset(run, log), len(old.encode()))
            with log.open("a") as stream:
                stream.write("[2026-09-23T11:01:00+00:00] new launch\nRuntimeError: new failure\n")
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                runner.tail_log(run)
                self.assertNotIn("old failure", output.getvalue())
                self.assertIn("new failure", output.getvalue())
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                runner.tail_log(run, history=True)
                self.assertIn("old failure", output.getvalue())

    def test_legacy_bare_traceback_is_not_hidden_after_current_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            log = run / "coordinator.log"
            log.write_text("Traceback: current failure\n")
            runner.atomic_json(run / "preflight.json", {"at": "2026-09-23T11:00:00+00:00"})
            runner.atomic_json(run / "finish.json", {
                "exit_code": 1, "finished_at": "2026-09-23T11:01:00+00:00"})
            self.assertEqual(runner.coordinator_log_offset(run, log), 0)

    def test_png_cache_invalidates_when_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            image(path)
            self.assertTrue(runner.good_png(path))
            replacement = path.with_suffix(".tmp")
            replacement.write_bytes(b"corrupt")
            replacement.replace(path)
            self.assertFalse(runner.good_png(path))
            image(path)
            self.assertTrue(runner.good_png(path))

    def test_gt_scan_is_parallel_and_preserves_dataset_order(self):
        barrier = threading.Barrier(2)
        def scan(_run, dataset):
            barrier.wait(timeout=5)
            return [{"dataset": dataset}]
        with tempfile.TemporaryDirectory() as directory:
            with (mock.patch.object(runner, "DATASETS", ("recon", "scand")),
                  mock.patch.object(runner, "gt_jobs_for_dataset", scan)):
                jobs = runner.gt_jobs(Path(directory))
            self.assertEqual([job["dataset"] for job in jobs], ["recon", "scand"])

    def test_huron_effective_population_is_pinned_and_used_by_commands(self):
        direct = runner.huron_population("time")
        rollout = runner.huron_population("rollout")
        self.assertEqual((direct["raw"], direct["valid"], direct["missing_trajectory_rows"],
                          direct["out_of_range_rows"]), (500, 329, 37, 134))
        self.assertEqual((rollout["raw"], rollout["valid"], rollout["missing_trajectory_rows"],
                          rollout["out_of_range_rows"]), (150, 103, 21, 26))
        self.assertEqual(len(direct["effective_raw_row_indices"]), 329)
        self.assertEqual(len(rollout["effective_raw_row_indices"]), 103)
        self.assertEqual(len(runner.direct_ids("huron")[16]), 329)
        self.assertIn("eval_expected_full_count=329", runner.gt_command(
            {"kind": "gt_direct", "dataset": "huron", "ids": [0, 328], "endpoints": [16]}, Path("/tmp/run")))
        self.assertIn("eval_expected_full_count=103", runner.command_for(
            {"kind": "rollout", "model": "nwm-release", "dataset": "huron",
             "evaluation": "rollout_1fps", "ids": [0, 102]}, Path("/tmp/run"), 8))
        self.assertEqual(runner.prediction_sources("nwm-release", "huron", "time",
                         runner.contract("nwm-release", "huron", "time")), [])

    def test_only_failed_huron_run_can_migrate_population(self):
        current = runner.run_contract()
        legacy = copy.deepcopy(current)
        legacy["schema_version"] = 1
        legacy.pop("effective_split_population")
        legacy["direct_sample_ids"]["huron"] = {
            str(h): list(range(500)) for h in (1, 2, 4, 8, 16)}
        for key, row in legacy["contracts"].items():
            if key.split("/")[1:] in (["huron", "time"], ["huron", "rollout"]):
                row.pop("effective_sample_count")
                row.pop("effective_mapping_sha256")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "full_8xa800_seed0_v1"
            log = run / "logs/gt_gt_direct_huron_ab69f171.log"
            runner.atomic_json(run / "finish.json", {"exit_code": 1, "error": f"GT failed: {log}"})
            runner.atomic_json(run / "jobs/gt_gt_direct_huron_ab69f171.json", {
                "status": "failed", "job": {"kind": "gt_direct", "dataset": "huron", "ids": list(range(500))}})
            changed_checkpoint = copy.deepcopy(legacy)
            changed_checkpoint["models"]["nwm-release"] = "untrusted"
            self.assertEqual(runner.migrate_huron_population_contract(run, changed_checkpoint, current), changed_checkpoint)
            image(run / "gt/huron/time/id_0/1.png")
            self.assertEqual(runner.migrate_huron_population_contract(run, legacy, current), legacy)
            (run / "gt/huron/time/id_0/1.png").unlink()
            self.assertEqual(runner.migrate_huron_population_contract(run, legacy, current), current)
            self.assertEqual(runner.read_json(run / "contract.json"), current)
            self.assertEqual(runner.read_json(run / "contract.pre_huron_population_fix.json"), legacy)
            self.assertEqual(runner.read_json(run / "contract_huron_population_migration.json")
                             ["effective_split_population"]["time"]["valid"], 329)

    def test_huron_report_explicitly_marks_filtered_population(self):
        with tempfile.TemporaryDirectory() as directory:
            items = {
                "direct/nwm-release/huron/time/0_1": {"status": "pending", "provenance": None},
                "rollout/nwm-release/huron/rollout_1fps/0": {"status": "pending", "provenance": None},
            }
            report = runner.write_report(Path(directory), items)
            self.assertIn("329 valid direct", report["huron"]["note"])
            self.assertIn("103 valid rollout", report["huron"]["note"])
            self.assertEqual(report["huron"]["time"]["sample_id_semantics"],
                             "position_after_loader_filtering")
            self.assertEqual(report["groups"]["direct/nwm-release/huron/time"]
                             ["sample_counts_by_endpoint"]["16s"], 329)
            self.assertEqual(report["groups"]["rollout/nwm-release/huron/rollout_1fps"]
                             ["sample_count"], 103)

    def test_reconstruction_and_metrics_finish_before_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_fd = os.open(root / "coordinator.lock", os.O_CREAT | os.O_RDWR, 0o644)
            nav = {"kind": "navigation", "model": "nwm-release", "dataset": "recon",
                   "evaluation": "cem80", "ids": [0], "priority": 0}
            direct = {"kind": "direct", "model": "nwm-release", "dataset": "recon",
                      "evaluation": "time", "ids": [0], "endpoints": [1], "priority": 20}
            gt = {"kind": "gt_direct", "dataset": "recon", "ids": [0], "endpoints": [1]}
            rollout = {**direct, "kind": "rollout", "evaluation": "rollout_1fps"}
            metric = {**direct, "kind": "metric"}
            events, done, guard = [], set(), threading.Lock()

            def build_plan(_run_dir):
                with guard:
                    return {}, [job for job in (nav, direct, rollout) if job["kind"] not in done]

            def metric_jobs(*_):
                with guard:
                    return [metric] if {"direct", "rollout"} <= done and "metric" not in done else []

            def run_job(job, _run_dir, _gpu, _state, *, gt=False):
                with guard:
                    events.append(("start", job["kind"], time.monotonic()))
                time.sleep(0.05 if gt else 0.08)
                with guard:
                    events.append(("end", job["kind"], time.monotonic()))
                    if not gt:
                        done.add(job["kind"])
                return {"name": runner.gt_job_name(job) if gt else runner.job_name(job),
                        "exit_code": 0, "oom": False, "peak_memory_gib": None, "log": "mock.log"}

            with (mock.patch.object(runner, "build_plan", build_plan),
                  mock.patch.object(runner, "gt_jobs", lambda _: [gt]),
                  mock.patch.object(runner, "run_job", run_job),
                  mock.patch.object(runner, "ensure_links", lambda *_: None),
                  mock.patch.object(runner, "publish_state", lambda *_, **__: None),
                  mock.patch.object(runner, "refresh_active_items", lambda *_: None),
                  mock.patch.object(runner, "metric_jobs", metric_jobs),
                  mock.patch.object(runner, "aggregate_navigation", lambda *_: None),
                  mock.patch.object(runner, "write_report", lambda *_: None)):
                runner.coordinator(root, ["0", "1"], lock_fd)
            starts = {kind: stamp for event, kind, stamp in events if event == "start"}
            ends = {kind: stamp for event, kind, stamp in events if event == "end"}
            self.assertLess(ends["gt_direct"], starts["direct"])
            for kind in ("direct", "rollout", "metric"):
                self.assertLess(ends[kind], starts["navigation"])

    def test_eight_gpus_keep_dispatching_during_slow_audit_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fd = os.open(root / "coordinator.lock", os.O_CREAT | os.O_RDWR, 0o644)
            jobs = [{"kind": "direct", "model": "nwm-release", "dataset": "recon",
                     "evaluation": "time", "ids": [i], "endpoints": [1]} for i in range(16)]
            done, devices = set(), set()
            guard = threading.Lock()
            audit_started, next_wave = threading.Event(), threading.Event()

            def work(job, _root, gpu, state):
                with guard:
                    devices.add(gpu)
                if job["ids"][0] >= 8:
                    self.assertTrue(audit_started.wait(3))
                    next_wave.set()
                time.sleep(0.05)
                with guard:
                    done.add(job["ids"][0])
                return {"name": runner.job_name(job), "exit_code": 0, "oom": False}

            def slow_audit(*_):
                audit_started.set()
                self.assertTrue(next_wave.wait(3), "audit blocked GPU dispatch")
                return []

            def slow_report(*_, **__):
                self.assertTrue(next_wave.wait(3), "report blocked GPU dispatch")

            with (mock.patch.object(runner, "build_plan", side_effect=lambda _: ({}, [j for j in jobs if j["ids"][0] not in done])) as plan,
                  mock.patch.object(runner, "gt_jobs", return_value=[]),
                  mock.patch.object(runner, "run_job", side_effect=work),
                  mock.patch.object(runner, "audit_job_outputs", side_effect=slow_audit),
                  mock.patch.object(runner, "publish_progress", side_effect=slow_report),
                  mock.patch.object(runner, "ensure_links"),
                  mock.patch.object(runner, "publish_state"),
                  mock.patch.object(runner, "metric_jobs", return_value=[]),
                  mock.patch.object(runner, "aggregate_navigation"),
                  mock.patch.object(runner, "write_report")):
                runner.coordinator(root, list(map(str, range(8))), fd)
                self.assertEqual(plan.call_count, 2)
            self.assertEqual(len(done), 16)
            self.assertEqual(devices, set(map(str, range(8))))

    def test_rae_huron_filter_preserves_effective_sample_order(self):
        import pickle
        from scripts.raenwm_infer import filter_huron_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "traj").mkdir()
            (root / "traj" / "traj_data.pkl").write_bytes(pickle.dumps({"position": list(range(20))}))
            rows = [("traj", 0, 1, 4), ("missing", 0, 1, 4),
                    ("traj", 16, 1, 4), ("traj", 5, -6, 4), ("traj", 10, 1, 4)]
            self.assertEqual(filter_huron_rows(rows, root), [rows[0], rows[4]])

    def test_rae_contract_migration_is_exact_and_rejects_huron_images(self):
        name = "scripts/raenwm_infer.py"
        old = {"entrypoint_sha256": {name: "d46634574c18a78579d0f281f16233b4e5d1e3853c8f59afbee2d509c5c1783e"}, "seed": 0}
        new = {"entrypoint_sha256": {name: "aecd6992f1b6a479f791c790396e3ab261a72f27b55f858f3d5a344867ebbc78"}, "seed": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(runner.migrate_rae_loader_contract(root, old, {**new, "seed": 1}), old)
            self.assertFalse((root / "contract.json").exists())
            self.assertEqual(runner.migrate_rae_loader_contract(root, old, new), new)
            self.assertEqual(runner.read_json(root / "contract.pre_rae_loader_fix.json"), old)
            image(root / "predictions/rae-nwm/huron/time/0.png")
            with self.assertRaisesRegex(RuntimeError, "sample numbering"):
                runner.migrate_rae_loader_contract(root, old, new)

    def test_stop_refuses_unrelated_coordinator_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "coordinator.lock").write_text("12345")
            with (mock.patch.object(runner, "process_args", return_value=["python", "other.py"]),
                  mock.patch.object(runner.os, "kill") as kill):
                with self.assertRaisesRegex(RuntimeError, "refusing to stop"):
                    runner.stop_run(run)
                kill.assert_not_called()

    def test_stop_signals_only_this_runs_worker_group(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "coordinator.lock").write_text("12345")
            coordinator_args = [str(SCRIPT), "_coordinate", "--run-id", run.name]
            calls = iter([coordinator_args, []])
            def process_args(pid):
                if pid == 12345:
                    return next(calls)
                return ["torchrun", str(run / "planning") if pid == 456 else "/other/run/planning"]
            with (mock.patch.object(runner, "process_args", side_effect=process_args),
                  mock.patch.object(runner.Path, "iterdir", return_value=iter([Path("/proc/456"), Path("/proc/789")])),
                  mock.patch.object(runner.os, "getpgid", side_effect=lambda pid: pid),
                  mock.patch.object(runner.os, "kill") as kill,
                  mock.patch.object(runner.os, "killpg", side_effect=[None, ProcessLookupError]) as killpg):
                runner.stop_run(run)
            kill.assert_called_once_with(12345, runner.signal.SIGTERM)
            self.assertEqual(killpg.call_args_list, [mock.call(456, runner.signal.SIGTERM), mock.call(456, 0)])

    def test_effective_dataset_counts(self):
        self.assertEqual({k: len(v) for k, v in runner.direct_ids("tum_rgbd").items()},
                         {1: 500, 2: 500, 4: 500, 8: 460, 16: 362})
        self.assertEqual({k: len(v) for k, v in runner.direct_ids("unitree_go2").items()},
                         {1: 500, 2: 500, 4: 500, 8: 461, 16: 339})
        self.assertEqual({k: len(v) for k, v in runner.direct_ids("planetary_rover").items()},
                         {1: 10, 2: 10, 4: 10})

    def test_unproven_audit_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "predictions/nwm-release/recon_time_audit.json"
            audit.parent.mkdir(parents=True)
            source = root / "images/recon/time"
            source.mkdir(parents=True)
            audit.write_text(json.dumps({"dataset": "recon", "eval_name": "time",
                "pred_eval_dir": str(source), "frame_indices": {"4s": 4},
                "inference": {"seed": 0, "sampler": "ddpm", "sampling_steps": 250}}))
            with mock.patch.object(runner, "NAS", root):
                self.assertEqual(runner.prediction_sources("nwm-release", "recon", "time",
                              runner.contract("nwm-release", "recon", "time")), [])

    def test_corrupt_direct_image_resumes_only_that_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "predictions/nwm-release/recon/time/id_0/1.png"
            second = root / "predictions/nwm-release/recon/time/id_1/1.png"
            image(first)
            second.parent.mkdir(parents=True)
            second.write_bytes(b"not a png")
            with (mock.patch.object(runner, "MODELS_REQUESTED", ("nwm-release",)),
                  mock.patch.object(runner, "DATASETS", ("recon",)),
                  mock.patch.object(runner, "ROLLOUT", ()),
                  mock.patch.object(runner, "NAV", ()),
                  mock.patch.object(runner, "run_contract", lambda: {"mock": 1}),
                  mock.patch.object(runner, "direct_ids", lambda _: {1: [0, 1]}),
                  mock.patch.object(runner, "prediction_sources", lambda *_: [])):
                _, untrusted_jobs = runner.build_plan(root)
                self.assertEqual(untrusted_jobs[0]["ids"], [0, 1])
                runner.atomic_json(root / "contract.json", {"mock": 1})
                items, jobs = runner.build_plan(root)
                self.assertEqual(jobs[0]["ids"], [1])
                self.assertEqual(items["direct/nwm-release/recon/time/0_1"]["status"], "complete")
                image(second)
                _, jobs = runner.build_plan(root)
                self.assertEqual(jobs, [])

    def test_partial_rollout_recomputes_entire_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fps in (1, 4):
                base = root / f"predictions/nwm-release/recon/rollout_{fps}fps"
                for frame in range(16 * fps):
                    image(base / "id_0" / f"{frame}.png")
                for frame in range(16 * fps - 1):
                    image(base / "id_1" / f"{frame}.png")
            original_count = runner.count
            with (mock.patch.object(runner, "MODELS_REQUESTED", ("nwm-release",)),
                  mock.patch.object(runner, "DATASETS", ("recon",)),
                  mock.patch.object(runner, "ROLLOUT", ("recon",)),
                  mock.patch.object(runner, "NAV", ()),
                  mock.patch.object(runner, "run_contract", lambda: {"mock": 1}),
                  mock.patch.object(runner, "direct_ids", lambda _: {}),
                  mock.patch.object(runner, "count", lambda ds, kind: 2 if kind == "rollout" else original_count(ds, kind)),
                  mock.patch.object(runner, "prediction_sources", lambda *_: [])):
                runner.atomic_json(root / "contract.json", {"mock": 1})
                _, jobs = runner.build_plan(root)
                self.assertEqual([(j["evaluation"], j["ids"]) for j in jobs],
                                 [("rollout_1fps", [1]), ("rollout_4fps", [1])])

    def test_old_rollout_is_linked_per_frame_without_writing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old/id_0"
            image(source / "0.png")
            run = root / "new"
            items = {"rollout/nwm-release/recon/rollout_1fps/0": {
                "status": "complete", "source": str(source), "provenance": "verified_old_audit", "frames": 1}}
            runner.ensure_links(run, items)
            target = run / "predictions/nwm-release/recon/rollout_1fps/id_0"
            self.assertTrue(target.is_dir())
            self.assertFalse(target.is_symlink())
            self.assertTrue((target / "0.png").is_symlink())
            original = (source / "0.png").read_bytes()
            image(target / "replacement.png")
            (target / "replacement.png").replace(target / "0.png")
            self.assertEqual((source / "0.png").read_bytes(), original)
            self.assertFalse((target / "0.png").is_symlink())
            runner.ensure_links(run, items)
            self.assertTrue((target / "0.png").is_symlink())

    def test_verified_complete_old_frame_reuses_aggregate_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            run = root / "run"
            audit = root / "audit.json"
            for sample in (0, 1):
                old_image = source / f"id_{sample}" / "4.png"
                image(old_image)
                linked = run / "predictions/nwm-release/recon/time" / f"id_{sample}" / "4.png"
                linked.parent.mkdir(parents=True, exist_ok=True)
                linked.symlink_to(old_image)
            row = {"sample_count": 2, "lpips_alex": 0.1, "dreamsim": 0.2, "psnr": 20.0}
            audit.write_text(json.dumps({"frame_indices": {"4s": 4}, "metrics": {"4s": row}}))
            with (mock.patch.object(runner, "direct_ids", lambda _: {4: [0, 1]}),
                  mock.patch.object(runner, "prediction_sources",
                                    lambda *_: [(source, str(audit), {4})])):
                self.assertEqual(runner.reusable_frame_metrics("nwm-release", "recon", "time", run)["4s"]["psnr"], 20.0)
                image(run / "predictions/nwm-release/recon/time/id_1/replacement.png")
                (run / "predictions/nwm-release/recon/time/id_1/replacement.png").replace(
                    run / "predictions/nwm-release/recon/time/id_1/4.png")
                self.assertEqual(runner.reusable_frame_metrics("nwm-release", "recon", "time", run), {})

    def test_active_job_refreshes_sample_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            job = {"kind": "direct", "model": "nwm-release", "dataset": "recon",
                   "evaluation": "time", "ids": [1], "endpoints": [2]}
            key = "direct/nwm-release/recon/time/1_2"
            items = {key: {"status": "pending", "source": None}}
            image(run / "predictions/nwm-release/recon/time/id_1/2.png")
            runner.refresh_active_items(run, items, {object(): (job, "0", {})})
            self.assertEqual(items[key]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
