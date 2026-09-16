import importlib
import json
from pathlib import Path

from PIL import Image

from scripts.nwm_benchmark_registry import (
    MODELS,
    PROTOCOLS,
    import_prediction,
    load_registry,
    new_registry,
    render_markdown,
)
from scripts.visualize_nwm_rollouts import (
    frame_index,
    frame_paths,
    render_contact_sheet,
    render_video_frame,
)


def test_unseen_rollout_protocol_is_fully_pinned() -> None:
    protocol = PROTOCOLS["go_stanford_unseen_rollout_10_v1"]

    assert protocol["sample_count"] == 10
    assert protocol["source_sample_count"] == 150
    assert protocol["sample_indices"] == list(range(10))
    assert protocol["split_sha256"] == (
        "e48af806e991d465f8b04f4dd106ff9b8db55cd65303831ee31e71dbef95cc23"
    )
    assert protocol["selected_entries_sha256"] == (
        "9cecd790344ab5ffe9bddb7a429be61a91cc26d04f51971114fd83541a85f9cc"
    )
    assert protocol["rollout_fps"] == [1, 4]
    assert protocol["horizons_seconds"] == [1, 2, 4, 8, 16]
    assert protocol["diffusion_steps"] == 250
    assert protocol["seed"] == 0
    assert protocol["execution"]["distributed_world_size"] == 2
    assert protocol["visualization_sample_ids"] == list(range(10))


def test_no_pretrain_model_is_pinned_to_completed_checkpoint() -> None:
    model = MODELS["nwm-no-pretrain"]

    assert model["checkpoint_id"] == "joint_0110000"
    assert model["checkpoint_step"] == 110000
    assert model["sha256"] == (
        "b5c914d39cf1d9ce059fb39131bf514716f2136cc7a3d9b6db1a177ceeba8787"
    )
    assert model["provenance"]["initialization"].endswith(
        "no NWM checkpoint loaded"
    )


def test_geopt_finetune_model_is_pinned_to_completed_checkpoint() -> None:
    model = MODELS["nwm-geopt-ft"]

    assert model["checkpoint_id"] == "joint_0100000"
    assert model["checkpoint_step"] == 110000
    assert model["sha256"] == (
        "4915124400ae8042c6616c645d0ab296ee508090fb18258feace417bd15b041a"
    )
    assert model["provenance"]["stage1"] == "NavAnywhere-v1 GeoPT"
    assert model["provenance"]["fine_tune_scheme"] == (
        "adapter reset; 10k warmup + 100k joint steps"
    )


def test_loading_stale_registry_adds_new_static_models(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"schema_version": 1, "models": {}}), encoding="utf-8"
    )

    registry = load_registry(registry_path)

    assert registry["models"]["nwm-no-pretrain"]["checkpoint_id"] == "joint_0110000"
    assert registry["models"]["nwm-geopt-ft"]["checkpoint_id"] == "joint_0100000"


def test_rollout_audit_uses_ten_trajectory_protocol(tmp_path: Path) -> None:
    registry = new_registry()
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "dataset": "go_stanford",
                "eval_name": "rollout_10_4fps",
                "sample_count": 10,
                "frame_indices": {
                    "1s": 3,
                    "2s": 7,
                    "4s": 15,
                    "8s": 31,
                    "16s": 63,
                },
                "metrics": {
                    horizon: {
                        "sample_count": 10,
                        "lpips_alex": 0.5,
                        "dreamsim": 0.4,
                        "psnr": 12.0,
                    }
                    for horizon in ("1s", "2s", "4s", "8s", "16s")
                },
            }
        ),
        encoding="utf-8",
    )

    import_prediction(
        registry,
        "nwm-real",
        "go_stanford",
        "rollout_10_4fps",
        audit,
        "go_stanford_unseen_rollout_10_v1",
    )

    result = registry["models"]["nwm-real"]["results"]["unseen_generalization"][
        "go_stanford"
    ]["rollout_10_4fps"]
    assert result["protocol"] == "go_stanford_unseen_rollout_10_v1"


def test_visualization_frame_mapping_and_rendering(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    for index in range(16):
        Image.new("RGB", (32, 32), (index, 0, 0)).save(sequence / f"{index}.png")

    paths = frame_paths(sequence, 16)
    assert frame_index(16, 1) == 15
    assert frame_index(4, 4) == 15
    frame = render_video_frame(
        [paths[0], paths[1]], ["Ground truth", "model"], "t = 1s", (32, 32)
    )
    assert frame.size == (64, 80)

    sheet = tmp_path / "sheet.png"
    horizons = [paths[frame_index(horizon, 1)] for horizon in (1, 2, 4, 8, 16)]
    render_contact_sheet([("Ground truth", horizons), ("model", horizons)], sheet, 1)
    assert sheet.is_file()
    assert Image.open(sheet).size == (1070, 402)


def test_markdown_renders_unseen_rollout_mode() -> None:
    registry = new_registry()
    metrics = {
        horizon: {
            "sample_count": 10,
            "lpips_alex": 0.5,
            "dreamsim": 0.4,
            "psnr": 12.0,
        }
        for horizon in ("1s", "2s", "4s", "8s", "16s")
    }
    registry["models"]["nwm-real"]["results"] = {
        "unseen_generalization": {
            "go_stanford": {"rollout_10_1fps": {"metrics": metrics}}
        }
    }

    markdown = render_markdown(registry)

    assert "| nwm-real | rollout_10_1fps | 16s |" in markdown


def test_ground_truth_inference_does_not_create_double_gt_directory(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    runner.ensure_ground_truth(
        "go_stanford",
        "rollout",
        {"exp_dir": "/models/reference"},
        ["0", "1"],
        {},
        True,
    )

    assert f"output_dir={tmp_path}" in commands[0]
    assert f"output_dir={tmp_path / 'gt'}" not in commands[0]


def test_unseen_rollout_split_hash_is_checked(monkeypatch) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")

    runner.validate_unseen_rollout_split({"protocols": PROTOCOLS})

    changed = json.loads(json.dumps(PROTOCOLS))
    changed["go_stanford_unseen_rollout_10_v1"]["split_sha256"] = "0" * 64
    try:
        runner.validate_unseen_rollout_split({"protocols": changed})
    except RuntimeError as error:
        assert "split SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("a changed split hash must be rejected")


def test_ten_trajectory_rollout_dry_run_is_isolated(tmp_path: Path, monkeypatch) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()
    runner.run_unseen_rollout_prediction(
        "nwm-real",
        registry["models"]["nwm-real"],
        ["0", "1"],
        {},
        tmp_path / "registry.json",
        registry,
        False,
        True,
    )

    flattened = [argument for command in commands for argument in command]
    assert "eval_sample_indices=[0,1,2,3,4,5,6,7,8,9]" in flattened
    assert "eval_expected_full_count=150" in flattened
    assert any("go_stanford_unseen_rollout_10_v1" in argument for argument in flattened)


def test_rollout_visualization_reads_shared_ground_truth(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    seed_root = tmp_path / "evaluation_seeds" / "seed1"
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", seed_root)
    monkeypatch.setattr(runner, "SHARED_BENCHMARK_ROOT", tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    runner.run_rollout_visualization(
        ["nwm-real"], seed_root / "benchmark_results.json", {}, True
    )

    command = commands[0]
    gt_index = command.index("--gt-root") + 1
    benchmark_index = command.index("--benchmark-root") + 1
    assert command[gt_index] == str(
        tmp_path / "protocol_runs" / runner.UNSEEN_ROLLOUT_PROTOCOL / "gt"
    )
    assert command[benchmark_index] == str(
        seed_root / "protocol_runs" / runner.UNSEEN_ROLLOUT_PROTOCOL
    )
