#!/usr/bin/env python3
"""One-command runner for the extensible NWM benchmark registry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from nwm_benchmark_registry import DEFAULT_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark"
)
RECON_GT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/gt/recon"
)
DREAMSIM_CACHE = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models"
)


def csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def registry_command(
    registry: Path, arguments: list[str], env: dict[str, str], dry_run: bool
) -> None:
    run(
        [sys.executable, str(PROJECT_ROOT / "scripts/nwm_benchmark_registry.py"),
         "--registry", str(registry), *arguments],
        env,
        dry_run,
    )


def torchrun_prefix(gpus: list[str]) -> list[str]:
    return ["torchrun", "--standalone", f"--nproc-per-node={len(gpus)}"]


def prediction_audit_path(model_name: str, dataset: str, evaluation: str) -> Path:
    return BENCHMARK_ROOT / "predictions" / model_name / f"{dataset}_{evaluation}_audit.json"


def ensure_ground_truth(
    dataset: str,
    reference_model: dict,
    gpus: list[str],
    env: dict[str, str],
    dry_run: bool,
) -> Path:
    if dataset == "recon":
        return RECON_GT
    gt_root = BENCHMARK_ROOT / "gt"
    gt_time = gt_root / dataset / "time"
    if gt_time.is_dir() and len(list(gt_time.glob("id_*"))) == 500:
        return gt_root / dataset
    gt_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    run(
        [
            "torchrun", "--standalone", "--nproc-per-node=1",
            "isolated_nwm_infer.py",
            f"exp_dir={reference_model['exp_dir']}",
            f"output_dir={gt_root}",
            "gt=1",
            f"datasets_to_eval=[{dataset}]",
            "eval_type=time",
            "batch_size=64",
            "num_workers=8",
            "pin_memory=false",
            "seed=0",
        ],
        gt_env,
        dry_run,
    )
    return gt_root / dataset


def run_prediction(
    model_name: str,
    model: dict,
    dataset: str,
    gpus: list[str],
    env: dict[str, str],
    registry: Path,
    force: bool,
    dry_run: bool,
) -> None:
    evaluations = ("time", "rollout_1fps", "rollout_4fps") if dataset == "recon" else ("time",)
    output_root = BENCHMARK_ROOT / "predictions" / model_name
    gt_root = ensure_ground_truth(dataset, model, gpus, env, dry_run)
    model_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}

    for inference_type in ("time", "rollout"):
        selected = [name for name in evaluations if name == inference_type or name.startswith(inference_type)]
        if not selected:
            continue
        if not force and all(prediction_audit_path(model_name, dataset, name).exists() for name in selected):
            continue
        command = [
            *torchrun_prefix(gpus),
            "isolated_nwm_infer.py",
            f"exp_dir={model['exp_dir']}",
            f"ckp={model['checkpoint_id']}",
            f"output_dir={output_root}",
            f"prediction_dir={output_root}",
            f"datasets_to_eval=[{dataset}]",
            f"eval_type={inference_type}",
            "batch_size=64",
            "num_workers=4",
            "pin_memory=false",
            "seed=0",
        ]
        if inference_type == "rollout":
            command.extend(["rollout_fps_values=[1,4]", "use_efficient_rollout=true"])
        run(command, model_env, dry_run)

    frame_specs = {
        "time": "1s:1,2s:2,4s:4,8s:8,16s:16",
        "rollout_1fps": "1s:0,2s:1,4s:3,8s:7,16s:15",
        "rollout_4fps": "1s:3,2s:7,4s:15,8s:31,16s:63",
    }
    metric_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    for evaluation in evaluations:
        audit = prediction_audit_path(model_name, dataset, evaluation)
        if force or not audit.exists():
            prediction_dir = output_root / dataset / evaluation
            command = [
                sys.executable,
                "scripts/evaluate_nwm_predictions.py",
                "--gt-dir", str(gt_root / evaluation),
                "--pred-dir", str(prediction_dir),
                "--output", str(audit),
                "--frames", frame_specs[evaluation],
                "--dataset", dataset,
                "--eval-type", "rollout" if evaluation.startswith("rollout") else "time",
                "--eval-name", evaluation,
                "--batch-size", "32",
                "--device", "cuda",
                "--dreamsim-cache", str(DREAMSIM_CACHE),
            ]
            if evaluation.startswith("rollout"):
                command.extend(["--rollout-fps", evaluation.split("_")[1].removesuffix("fps")])
            run(command, metric_env, dry_run)
        registry_command(
            registry,
            ["import-prediction", "--model", model_name, "--dataset", dataset,
             "--evaluation", evaluation, "--audit", str(audit)],
            env,
            dry_run,
        )


def planning_result_path(output_root: Path, dataset: str) -> Path | None:
    matches = sorted(output_root.glob(f"{dataset}_CEM_N80_K5_RS1_rep3_OPT1*.json"))
    matches = [path for path in matches if not path.name.endswith("_audit.json")]
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous planning results for {dataset}: {matches}")
    return matches[0]


def run_planning(
    model_name: str,
    model: dict,
    gpus: list[str],
    env: dict[str, str],
    registry: Path,
    force: bool,
    dry_run: bool,
    microbatch_size: int | None,
    datasets: list[str],
) -> None:
    output_root = BENCHMARK_ROOT / "planning" / model_name
    existing = {dataset: planning_result_path(output_root, dataset) for dataset in datasets}
    if force or any(path is None for path in existing.values()):
        planning_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
        command = [
                *torchrun_prefix(gpus),
                "planning_eval.py",
                f"exp_dir={model['exp_dir']}",
                f"ckp={model['checkpoint_id']}",
                f"datasets_to_eval=[{','.join(datasets)}]",
                f"output_dir={output_root}",
                "batch_size=1",
                "num_workers=4",
                "num_samples=80",
                "topk=5",
                "rollout_stride=1",
                "opt_steps=1",
                "num_repeat_eval=3",
                "cost_fn=lpips",
                "compute_cost_with_recon=true",
                "save_preds=false",
                "plot=false",
                f"resume_planning_samples={'false' if force else 'true'}",
            ]
        if microbatch_size is not None:
            command.append(f"planning_microbatch_size={microbatch_size}")
        run(command, planning_env, dry_run)
    for dataset in datasets:
        result = planning_result_path(output_root, dataset)
        if result is None:
            if dry_run:
                continue
            raise FileNotFoundError(f"Planning result not produced for {model_name}/{dataset}")
        registry_command(
            registry,
            ["import-planning", "--model", model_name, "--dataset", dataset,
             "--metrics", str(result)],
            env,
            dry_run,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--models", type=csv, default=csv("nwm-base,nwm-real,nwm-release"))
    parser.add_argument(
        "--tasks",
        type=csv,
        default=csv("recon_prediction,navigation,unseen"),
        help="Comma-separated: recon_prediction,navigation,unseen",
    )
    parser.add_argument("--gpus", type=csv, default=csv("0,1,2,3"))
    parser.add_argument(
        "--planning-microbatch-size",
        type=int,
        default=None,
        help="Optionally split the 80 CEM candidates to reduce peak navigation memory.",
    )
    parser.add_argument(
        "--navigation-datasets",
        type=csv,
        default=csv("recon,scand"),
        help="Comma-separated navigation datasets to evaluate.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    unknown_models = sorted(set(args.models) - set(registry["models"]))
    if unknown_models:
        raise ValueError(f"Models are not registered: {unknown_models}")
    unknown_tasks = sorted(set(args.tasks) - {"recon_prediction", "navigation", "unseen"})
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if not args.gpus:
        raise ValueError("At least one GPU is required")
    unknown_navigation_datasets = sorted(
        set(args.navigation_datasets) - {"recon", "scand"}
    )
    if unknown_navigation_datasets:
        raise ValueError(
            f"Unknown navigation datasets: {unknown_navigation_datasets}"
        )

    env = os.environ.copy()
    env.setdefault("NWM_DATA_ROOT", "/file_system/nas/algorithm/dujun.nie/nwm/data")
    env.setdefault("NWM_INDEX_ROOT", "/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices")
    env.setdefault("TORCH_HOME", "/file_system/vepfs/algorithm/dujun.nie/models")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    for model_name in args.models:
        model = registry["models"][model_name]
        if "recon_prediction" in args.tasks:
            run_prediction(model_name, model, "recon", args.gpus, env, args.registry, args.force, args.dry_run)
        if "unseen" in args.tasks:
            run_prediction(model_name, model, "go_stanford", args.gpus, env, args.registry, args.force, args.dry_run)
        if "navigation" in args.tasks:
            run_planning(
                model_name,
                model,
                args.gpus,
                env,
                args.registry,
                args.force,
                args.dry_run,
                args.planning_microbatch_size,
                args.navigation_datasets,
            )

    registry_command(
        args.registry,
        ["render", "--output", str(args.registry.with_suffix(".md"))],
        env,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
