"""Replay training DreamSim evaluation on every saved EMA checkpoint, locally.

Logical DDP ranks are replayed independently, so fewer physical GPUs can retain
the reference sampler, batch size, worker RNG streams, and diffusion noise.
No W&B run is resumed or modified. Successful runs remove worker logs and viz.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def weight_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def test_dataset(config):
    # Same test-only constructor arguments as data_utils.prepare_datasets;
    # avoid loading/validating the large, unused training posterior cache.
    from datasets import TrainingDataset
    from misc import get_transform
    from torch.utils.data import ConcatDataset

    datasets = []
    for name, spec in config.dataset.datasets.items():
        if not spec.get("enabled", True) or "test" not in spec:
            continue
        distance = spec.get("distance", config.dataset.distance)
        datasets.append(TrainingDataset(
            data_folder=spec.data_folder, data_split_folder=spec.test,
            dataset_name=name, image_size=config.dataset.image_size,
            min_dist_cat=distance.min_dist_cat, max_dist_cat=distance.max_dist_cat,
            len_traj_pred=spec.get("len_traj_pred", config.dataset.len_traj_pred),
            context_size=config.dataset.context_size,
            normalize=config.dataset.normalize, goals_per_obs=4,
            transform=get_transform(config.dataset.image_size, config.dataset.mean, config.dataset.std),
            action_stats=config.dataset.action_stats,
            waypoint_spacing=spec.metric_waypoint_spacing,
            predefined_index=None, traj_stride=1,
            motion_condition=config.motion_condition,
            motion_types=(str(config.motion_condition.eval_type),),
        ))
    return ConcatDataset(datasets)


def worker(args):
    import logging
    import torch
    import torch.distributed as dist
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, DistributedSampler
    import hydra_utils  # noqa: F401
    from misc import get_unnormalize
    from motion_condition import motion_condition_collate
    from train_utils import evaluate, setup_diffusion, setup_model, setup_tokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.55)
    manifest = json.loads((args.output / "manifest.json").read_text())
    result_path = args.output / f"worker_{args.worker}.json"
    records = json.loads(result_path.read_text()) if args.resume and result_path.exists() else []
    completed = {(r["checkpoint"], r["rank"]) for r in records}
    ranks = list(range(args.worker, args.logical_ranks, len(args.gpus.split(","))))
    if all((c["name"], rank) in completed for c in manifest["checkpoints"]
           if not c.get("duplicate_of") for rank in ranks):
        print("All assigned shards already complete", flush=True)
        return
    config = OmegaConf.create(manifest["config"])
    HydraConfig.instance().cfg = OmegaConf.create({"hydra": {"runtime": {"cwd": str(ROOT)}}})
    device = torch.device("cuda:0")
    with tempfile.TemporaryDirectory(prefix=f"worker{args.worker}-", dir=args.output / ".temporary") as tmp:
        dist.init_process_group("nccl", init_method=f"file://{tmp}/rendezvous", rank=0, world_size=1)
        try:
            dataset = test_dataset(config)
            model = setup_model(config, device).eval().requires_grad_(False)
            tokenizer = setup_tokenizer(config, device).eval().requires_grad_(False)
            # Training uses its full diffusion object, not standalone eval's 250 steps.
            diffusion = setup_diffusion(config, for_eval=False, device=device)
            assert diffusion.num_timesteps == manifest["diffusion_steps"]
            for checkpoint in manifest["checkpoints"]:
                if checkpoint.get("duplicate_of"):
                    continue
                path = Path(checkpoint["path"])
                if all((path.name, rank) in completed for rank in ranks):
                    continue
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (checkpoint["size"], checkpoint["mtime_ns"]):
                    raise RuntimeError(f"Checkpoint changed after snapshot: {path}")
                state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                model.load_state_dict(state["ema"], strict=True)
                del state
                for rank in ranks:
                    if (path.name, rank) in completed:
                        continue
                    sampler = DistributedSampler(dataset, num_replicas=args.logical_ranks,
                                                 rank=rank, shuffle=False, seed=args.seed)
                    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                        num_workers=args.num_workers, pin_memory=True,
                                        persistent_workers=False, drop_last=True,
                                        collate_fn=motion_condition_collate,
                                        **({"multiprocessing_context": "spawn"} if args.num_workers else {}))
                    start = time.monotonic()
                    print(f"START checkpoint={path.name} logical_rank={rank}", flush=True)
                    with torch.no_grad():
                        score = evaluate(model, tokenizer, diffusion, loader, rank,
                                         int(config.model.generator.input_size), device,
                                         f"{tmp}/viz", args.seed, bool(config.bfloat16),
                                         int(config.dataset.context_size),
                                         get_unnormalize(config.dataset.mean, config.dataset.std),
                                         num_batches=args.num_batches)
                    value = float(score)
                    if not __import__('math').isfinite(value):
                        raise RuntimeError(f"Nonfinite score: {path}")
                    record = {"checkpoint": path.name, "rank": rank,
                              "eval/perceptual_loss": value,
                              "goals": args.batch_size * 4 * args.num_batches,
                              "seconds": time.monotonic() - start}
                    records.append(record)
                    write_json(result_path, records)
                    print(f"DONE {json.dumps(record)}", flush=True)
        finally:
            dist.destroy_process_group()


def snapshot(args):
    import torch
    from omegaconf import OmegaConf

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Use a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    temporary = args.output / ".temporary"
    temporary.mkdir(exist_ok=args.resume)
    checkpoints, aliases, fingerprints = [], {}, {}
    config = None
    selected = None
    if args.checkpoint:
        selected = (args.run_dir / "checkpoints" / args.checkpoint).resolve(strict=True)
        if selected.parent != (args.run_dir / "checkpoints").resolve():
            raise ValueError("--checkpoint must select a file in the run's checkpoint directory")
    for path in sorted((args.run_dir / "checkpoints").glob("*.pth.tar")):
        if selected is not None and path.resolve() != selected:
            continue
        if path.is_symlink():
            aliases[path.name] = path.resolve().name
            continue
        before = path.stat()
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        fingerprint = weight_digest(state["ema"])
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Checkpoint is still being written: {path}")
        meta = state.get("two_stage_metadata", {})
        item = {"path": str(path), "name": path.name, "size": before.st_size,
                "mtime_ns": before.st_mtime_ns, "ema_sha256": fingerprint,
                "train_steps": int(state["train_steps"]), "metadata": meta}
        if fingerprint in fingerprints:
            item["duplicate_of"] = fingerprints[fingerprint]
        else:
            fingerprints[fingerprint] = path.name
        checkpoints.append(item)
        if config is None:
            config = OmegaConf.to_container(OmegaConf.create(state["config"]), resolve=True)
        del state
    if not checkpoints:
        raise RuntimeError("No saved checkpoints found")
    checkpoints.sort(key=lambda x: (x["train_steps"], x["name"]))
    # Duplicate representative can be later in step/name order; workers skip it
    # by digest and aggregation happens only once all representatives finish.
    manifest = {"source_run": args.source_run, "run_dir": str(args.run_dir),
                "config": config, "checkpoints": checkpoints, "aliases": aliases,
                "seed": args.seed, "logical_ranks": args.logical_ranks,
                "batch_size_per_rank": args.batch_size, "num_workers": args.num_workers,
                "num_batches": args.num_batches, "weight_key": "ema",
                "diffusion_steps": 1000, "metric": "DreamSim ensemble, training evaluate()",
                "scope": "saved checkpoint snapshot at launch",
                "selected_checkpoint": args.checkpoint,
                "comparison_note": "Same training metric protocol; historical code/RNG changes may prevent exact replay of old W&B values."}
    write_json(args.output / "manifest.json", manifest)
    print(f"Snapshot: {len(checkpoints)} files, {len(fingerprints)} distinct EMA states, aliases={aliases}", flush=True)
    return manifest


def run(args):
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text())
        expected = {"run_dir": str(args.run_dir), "seed": args.seed,
                    "logical_ranks": args.logical_ranks, "batch_size_per_rank": args.batch_size,
                    "num_workers": args.num_workers, "num_batches": args.num_batches}
        for key, value in expected.items():
            if manifest[key] != value:
                raise ValueError(f"Resume protocol mismatch: {key}")
        if manifest.get("selected_checkpoint") != args.checkpoint:
            raise ValueError("Resume checkpoint selection changed")
        # Keep the original assignment: each worker owns a durable result file.
        count = len(args.gpus.split(","))
        files = list(args.output.glob("worker_*.json"))
        if not files and (args.output / "results.json").exists():
            # Successful runs remove intermediate shards. Restore them from
            # the durable results when extending to newly saved checkpoints.
            saved = json.loads((args.output / "results.json").read_text())["per_rank"]
            for index in range(count):
                write_json(args.output / f"worker_{index}.json",
                           [r for r in saved if r["rank"] % count == index])
            files = list(args.output.glob("worker_*.json"))
        if {p.name for p in files} != {f"worker_{i}.json" for i in range(count)}:
            raise ValueError("Resume requires all original worker files and the same worker count")
        for index in range(count):
            records = json.loads((args.output / f"worker_{index}.json").read_text())
            if any(r["rank"] % count != index for r in records):
                raise ValueError("Resume worker assignment changed")
        for checkpoint in manifest["checkpoints"]:
            stat = Path(checkpoint["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (checkpoint["size"], checkpoint["mtime_ns"]):
                raise RuntimeError(f"Checkpoint changed: {checkpoint['path']}")
        if args.include_new:
            manifest = snapshot(args)
    else:
        manifest = snapshot(args)
    checkpoints = manifest["checkpoints"]
    temporary = args.output / ".temporary"
    temporary.mkdir(exist_ok=True)
    write_json(args.output / "status.json", {"state": "running", "coordinator_pid": os.getpid()})
    processes = []
    for index, gpu in enumerate(args.gpus.split(",")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        log_path = temporary / f"worker_{index}.log"
        with log_path.open("w") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker", str(index)]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        print(f"Worker GPU={gpu} PID={process.pid} log={log_path}", flush=True)
    statuses = [process.wait() for process in processes]
    if any(statuses):
        write_json(args.output / "status.json", {"state": "failed", "exit_codes": statuses})
        raise RuntimeError(f"Workers failed: {statuses}; inspect {temporary}")
    records = []
    for index in range(len(processes)):
        records.extend(json.loads((args.output / f"worker_{index}.json").read_text()))
    rows = []
    for checkpoint in checkpoints:
        representative = checkpoint.get("duplicate_of", checkpoint["name"])
        selected = [r for r in records if r["checkpoint"] == representative]
        assert sorted(r["rank"] for r in selected) == list(range(args.logical_ranks))
        goals = sum(r["goals"] for r in selected)
        rows.append({"checkpoint": checkpoint["name"], "train_steps": checkpoint["train_steps"],
                     "eval/perceptual_loss": sum(r["eval/perceptual_loss"] * r["goals"] for r in selected) / goals,
                     "goal_images": goals, "duplicate_of": checkpoint.get("duplicate_of", "")})
    write_json(args.output / "results.json", {"results": rows, "per_rank": records})
    with (args.output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    distinct = [row for row in rows if not row["duplicate_of"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["train_steps"] for r in distinct], [r["eval/perceptual_loss"] for r in distinct], marker="o")
    ax.set(xlabel="Training step (warmup + joint)", ylabel="eval/perceptual_loss (DreamSim)", title=args.run_dir.name)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "perceptual_loss.png", dpi=160)
    plt.close(fig)
    shutil.rmtree(temporary)
    for index in range(len(processes)):
        (args.output / f"worker_{index}.json").unlink()
    write_json(args.output / "status.json", {"state": "complete", "exit_codes": statuses, "temporary_files_cleaned": True})
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", default="")
    parser.add_argument("--gpus", default="5,6,7")
    parser.add_argument("--checkpoint", help="Evaluate only this checkpoint filename or alias (for example latest.pth.tar)")
    parser.add_argument("--logical-ranks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse saved shards with the original protocol and worker count")
    parser.add_argument("--include-new", action="store_true", help="With --resume, refresh the snapshot to include newly saved checkpoints")
    args = parser.parse_args()
    if args.include_new and not args.resume:
        parser.error("--include-new requires --resume")
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.run_dir = args.run_dir.resolve()
    base = "/file_system/nas/algorithm/dujun.nie/nwm"
    for key, value in {"NWM_MODEL_CACHE": f"{base}/cache/dreamsim",
                       "NWM_INDEX_ROOT": f"{base}/cache/dataset_indices",
                       "TORCH_HOME": f"{base}/cache/torch",
                       "MPLCONFIGDIR": f"{base}/cache/matplotlib",
                       "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4"}.items():
        os.environ.setdefault(key, value)
    if args.worker is None:
        run(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
