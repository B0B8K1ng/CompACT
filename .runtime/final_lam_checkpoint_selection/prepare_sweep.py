#!/usr/bin/env python3
"""Create the pinned registry for the finalLAM checkpoint sweep."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


EVAL_ROOT = Path("/file_system/vepfs/algorithm/dujun.nie/code/CompACT-eval-huron-fix")
sys.path.insert(0, str(EVAL_ROOT / "scripts"))
import nwm_benchmark_registry as registry_lib


BASE = Path("/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark")
OUT = BASE / "finalLAM_reset_checkpoint_sweep_20260922"
SOURCE = BASE / "finalLAM_reset_joint0050000_direct4s_20260921"
EXP = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/"
    "nav15_pa_step60000_latentpt_reset_ft/"
    "nwm-latentpt-pixel-action-finalLAM-ft-reset"
)
SOURCE_MODEL = "nwm-latentpt-pixel-action-finalLAM-ft-reset"
STEPS = tuple(range(10_000, 100_001, 10_000))


def model_name(step: int) -> str:
    return f"finalLAM-reset-joint{step:07d}"


def checkpoint(step: int) -> Path:
    return EXP / "checkpoints" / f"joint_{step:07d}.pth.tar"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    source_registry = json.loads((SOURCE / "benchmark_results.json").read_text())
    source_model = source_registry["models"][SOURCE_MODEL]
    cache_path = OUT / "checkpoint_hashes.json"
    hashes = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    hashes[str(checkpoint(50_000))] = source_model["sha256"]
    pending = [path for path in map(checkpoint, STEPS) if str(path) not in hashes]
    for path in map(checkpoint, STEPS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if pending:
        print(f"Hashing {len(pending)} checkpoints with four readers", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(sha256, path): path for path in pending}
            for future in as_completed(futures):
                path = futures[future]
                hashes[str(path)] = future.result()
                atomic_json(cache_path, hashes)
                print(f"HASH {path.name} {hashes[str(path)]}", flush=True)

    registry_path = OUT / "benchmark_results.json"
    existing = registry_lib.load_registry(registry_path) if registry_path.exists() else None
    registry = registry_lib.new_registry()
    if existing:
        for name, entry in existing["models"].items():
            if name.startswith("finalLAM-reset-joint"):
                registry["models"][name] = entry
    for step in STEPS:
        name = model_name(step)
        previous_results = registry["models"].get(name, {}).get("results", {})
        if step == 50_000 and not previous_results:
            previous_results = deepcopy(source_model["results"])
        path = checkpoint(step)
        registry["models"][name] = {
            "architecture": source_model["architecture"],
            "exp_dir": str(EXP),
            "checkpoint_id": f"joint_{step:07d}",
            "checkpoint": str(path),
            "checkpoint_step": step + 3_000,
            "sha256": hashes[str(path)],
            "training_datasets": source_model["training_datasets"],
            "provenance": {
                "original_experiment": str(EXP),
                "wandb_run_id": "nqkuchuo",
                "joint_steps": step,
                "warmup_steps": 3_000,
            },
            "results": previous_results,
        }
    registry_lib.save_registry(registry_path, registry)
    manifest = {
        "created_at": time.time(),
        "source_registry": str(SOURCE / "benchmark_results.json"),
        "shared_benchmark_root": str(SOURCE),
        "experiment": str(EXP),
        "protocol": "direct_4s_v1",
        "datasets": [
            "recon",
            "scand",
            "huron",
            "tartan_drive",
            "go_stanford",
            "unitree_go2",
            "tum_rgbd",
            "uzh_fpv",
        ],
        "checkpoints": {
            model_name(step): {
                "joint_steps": step,
                "total_steps": step + 3_000,
                "path": str(checkpoint(step)),
                "sha256": hashes[str(checkpoint(step))],
                "reused_existing_evaluation": step == 50_000,
            }
            for step in STEPS
        },
    }
    atomic_json(OUT / "manifest.json", manifest)
    print(f"READY registry={registry_path} models={len(STEPS)}", flush=True)


if __name__ == "__main__":
    main()
