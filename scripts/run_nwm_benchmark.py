#!/usr/bin/env python3
"""One-command runner for the extensible NWM benchmark registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from nwm_benchmark_registry import DEFAULT_REGISTRY, load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark"
)
BENCHMARK_ROOT = DEFAULT_BENCHMARK_ROOT
SHARED_BENCHMARK_ROOT: Path | None = None
EVAL_SEED = 0
RECON_GT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/gt/recon"
)
DREAMSIM_CACHE = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models"
)
UNSEEN_ROLLOUT_WORLD_SIZE = 2
UNSEEN_ROLLOUT_PROTOCOL = "go_stanford_unseen_rollout_10_v1"


def progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[benchmark][{timestamp}] {message}", flush=True)


def csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unseen_rollout_split(registry: dict) -> None:
    protocol = registry["protocols"][UNSEEN_ROLLOUT_PROTOCOL]
    split = PROJECT_ROOT / protocol["split"]
    actual = sha256_file(split)
    expected = protocol["split_sha256"]
    if actual != expected:
        raise RuntimeError(
            f"unseen rollout split SHA-256 mismatch for {split}: "
            f"expected {expected}, got {actual}"
        )
    with split.open("rb") as handle:
        entries = pickle.load(handle)
    if len(entries) != protocol["source_sample_count"]:
        raise RuntimeError(
            f"unseen rollout source split size mismatch for {split}: expected "
            f"{protocol['source_sample_count']}, got {len(entries)}"
        )
    selected = [entries[index] for index in protocol["sample_indices"]]
    selected_payload = json.dumps(
        selected, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    selected_sha256 = hashlib.sha256(selected_payload).hexdigest()
    if selected_sha256 != protocol["selected_entries_sha256"]:
        raise RuntimeError(
            f"unseen rollout selected entries SHA-256 mismatch for {split}: "
            f"expected {protocol['selected_entries_sha256']}, got {selected_sha256}"
        )


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


def sample_directory_count(path: Path) -> int:
    return sum(1 for child in path.glob("id_*") if child.is_dir())


def rollout_sequences_complete(path: Path, sample_count: int, frame_count: int) -> bool:
    expected_samples = {f"id_{index}" for index in range(sample_count)}
    samples = {child.name for child in path.glob("id_*") if child.is_dir()}
    if samples != expected_samples:
        return False
    return all(
        all((path / sample / f"{frame}.png").is_file() for frame in range(frame_count))
        for sample in expected_samples
    )


def ground_truth_root(dataset: str) -> Path:
    if dataset == "recon":
        return RECON_GT
    shared_root = SHARED_BENCHMARK_ROOT or BENCHMARK_ROOT
    return shared_root / "gt" / dataset


def ensure_ground_truth(
    dataset: str,
    inference_type: str,
    reference_model: dict,
    gpus: list[str],
    env: dict[str, str],
    dry_run: bool,
) -> Path:
    if dataset == "recon":
        return RECON_GT
    gt_dataset_root = ground_truth_root(dataset)
    expected_count = 500 if inference_type == "time" else 150
    evaluation_names = (
        ("time",) if inference_type == "time" else ("rollout_1fps", "rollout_4fps")
    )
    if all(
        sample_directory_count(gt_dataset_root / evaluation) == expected_count
        for evaluation in evaluation_names
    ):
        return gt_dataset_root
    # isolated_nwm_infer.py appends its own ``gt`` component in ground-truth
    # mode, so pass the benchmark root rather than BENCHMARK_ROOT / "gt".
    gt_output_root = SHARED_BENCHMARK_ROOT or BENCHMARK_ROOT
    gt_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    run(
        [
            "torchrun", "--standalone", "--nproc-per-node=1",
            "isolated_nwm_infer.py",
            f"exp_dir={reference_model['exp_dir']}",
            f"output_dir={gt_output_root}",
            "gt=1",
            f"datasets_to_eval=[{dataset}]",
            f"eval_type={inference_type}",
            "batch_size=64",
            "num_workers=8",
            "pin_memory=false",
            "seed=0",
        ],
        gt_env,
        dry_run,
    )
    if not dry_run:
        incomplete = {
            evaluation: sample_directory_count(gt_dataset_root / evaluation)
            for evaluation in evaluation_names
            if sample_directory_count(gt_dataset_root / evaluation) != expected_count
        }
        if incomplete:
            raise RuntimeError(
                f"incomplete {dataset}/{inference_type} ground truth: "
                f"expected {expected_count}, got {incomplete}"
            )
    return gt_dataset_root


def run_prediction(
    model_name: str,
    model: dict,
    dataset: str,
    gpus: list[str],
    env: dict[str, str],
    registry: Path,
    force: bool,
    dry_run: bool,
    evaluations: tuple[str, ...] | None = None,
) -> None:
    if evaluations is None:
        evaluations = ("time", "rollout_1fps", "rollout_4fps")
    output_root = BENCHMARK_ROOT / "predictions" / model_name
    model_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}

    for inference_type in ("time", "rollout"):
        selected = [name for name in evaluations if name == inference_type or name.startswith(inference_type)]
        if not selected:
            continue
        if not force and all(prediction_audit_path(model_name, dataset, name).exists() for name in selected):
            continue
        ensure_ground_truth(dataset, inference_type, model, gpus, env, dry_run)
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
            f"seed={EVAL_SEED}",
        ]
        if inference_type == "rollout":
            command.extend(["rollout_fps_values=[1,4]", "use_efficient_rollout=true"])
        run(command, model_env, dry_run)

    frame_specs = {
        "time": "1s:1,2s:2,4s:4,8s:8,16s:16",
        "rollout_1fps": "1s:0,2s:1,4s:3,8s:7,16s:15",
        "rollout_4fps": "1s:3,2s:7,4s:15,8s:31,16s:63",
    }
    gt_root = ground_truth_root(dataset)
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


def run_rollout_visualization(
    models: list[str], registry: Path, env: dict[str, str], dry_run: bool
) -> None:
    comparison = "_vs_".join(models)
    comparison = re.sub(r"[^A-Za-z0-9_.-]+", "-", comparison).strip("-._")
    if not comparison:
        raise ValueError("model names do not produce a safe visualization directory")
    output_root = (
        BENCHMARK_ROOT
        / "visualizations"
        / UNSEEN_ROLLOUT_PROTOCOL
        / comparison
    )
    run(
        [
            sys.executable,
            "scripts/visualize_nwm_rollouts.py",
            "--registry",
            str(registry),
            "--benchmark-root",
            str(BENCHMARK_ROOT / "protocol_runs" / UNSEEN_ROLLOUT_PROTOCOL),
            "--gt-root",
            str(shared_rollout_protocol_root() / "gt"),
            "--output-root",
            str(output_root),
            "--protocol",
            UNSEEN_ROLLOUT_PROTOCOL,
            "--models",
            ",".join(models),
            "--dataset",
            "go_stanford",
        ],
        env,
        dry_run,
    )


def rollout_protocol_root() -> Path:
    return BENCHMARK_ROOT / "protocol_runs" / UNSEEN_ROLLOUT_PROTOCOL


def shared_rollout_protocol_root() -> Path:
    shared_root = SHARED_BENCHMARK_ROOT or BENCHMARK_ROOT
    return shared_root / "protocol_runs" / UNSEEN_ROLLOUT_PROTOCOL


def rollout_audit_path(model_name: str, dataset: str, evaluation: str) -> Path:
    return (
        rollout_protocol_root()
        / "predictions"
        / model_name
        / f"{dataset}_{evaluation}_audit.json"
    )


def ensure_unseen_rollout_ground_truth(
    protocol: dict,
    reference_model: dict,
    gpus: list[str],
    env: dict[str, str],
    dry_run: bool,
) -> Path:
    protocol_root = shared_rollout_protocol_root()
    gt_dataset_root = protocol_root / "gt" / protocol["dataset"]
    source_evaluations = tuple(protocol["source_evaluations"].values())
    if all(
        rollout_sequences_complete(
            gt_dataset_root / evaluation,
            protocol["sample_count"],
            16 * int(evaluation.removeprefix("rollout_").removesuffix("fps")),
        )
        for evaluation in source_evaluations
    ):
        return gt_dataset_root

    indices = ",".join(str(index) for index in protocol["sample_indices"])
    gt_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    run(
        [
            "torchrun",
            "--standalone",
            "--nproc-per-node=1",
            "isolated_nwm_infer.py",
            f"exp_dir={reference_model['exp_dir']}",
            f"output_dir={protocol_root}",
            "gt=1",
            f"datasets_to_eval=[{protocol['dataset']}]",
            "eval_type=rollout",
            f"eval_sample_indices=[{indices}]",
            f"eval_expected_full_count={protocol['source_sample_count']}",
            "batch_size=64",
            "num_workers=4",
            "pin_memory=false",
            f"seed={protocol['seed']}",
            f"rollout_fps_values=[{','.join(map(str, protocol['rollout_fps']))}]",
        ],
        gt_env,
        dry_run,
    )
    if not dry_run:
        incomplete = {
            evaluation: sample_directory_count(gt_dataset_root / evaluation)
            for evaluation in source_evaluations
            if not rollout_sequences_complete(
                gt_dataset_root / evaluation,
                protocol["sample_count"],
                16
                * int(evaluation.removeprefix("rollout_").removesuffix("fps")),
            )
        }
        if incomplete:
            raise RuntimeError(
                f"incomplete {UNSEEN_ROLLOUT_PROTOCOL} ground truth: expected "
                f"{protocol['sample_count']}, got {incomplete}"
            )
    return gt_dataset_root


def run_unseen_rollout_prediction(
    model_name: str,
    model: dict,
    gpus: list[str],
    env: dict[str, str],
    registry_path: Path,
    registry: dict,
    force: bool,
    dry_run: bool,
) -> None:
    protocol = registry["protocols"][UNSEEN_ROLLOUT_PROTOCOL]
    dataset = protocol["dataset"]
    output_root = rollout_protocol_root() / "predictions" / model_name
    source_evaluations = protocol["source_evaluations"]
    predictions_complete = all(
        rollout_sequences_complete(
            output_root / dataset / source_evaluation,
            protocol["sample_count"],
            16
            * int(source_evaluation.removeprefix("rollout_").removesuffix("fps")),
        )
        for source_evaluation in source_evaluations.values()
    )
    gt_root = ensure_unseen_rollout_ground_truth(
        protocol, model, gpus, env, dry_run
    )

    if force or not predictions_complete:
        indices = ",".join(str(index) for index in protocol["sample_indices"])
        model_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
        run(
            [
                *torchrun_prefix(gpus),
                "isolated_nwm_infer.py",
                f"exp_dir={model['exp_dir']}",
                f"ckp={model['checkpoint_id']}",
                f"output_dir={output_root}",
                f"prediction_dir={output_root}",
                f"datasets_to_eval=[{dataset}]",
                "eval_type=rollout",
                f"eval_sample_indices=[{indices}]",
                f"eval_expected_full_count={protocol['source_sample_count']}",
                "batch_size=64",
                "num_workers=4",
                "pin_memory=false",
                f"seed={EVAL_SEED}",
                f"rollout_fps_values=[{','.join(map(str, protocol['rollout_fps']))}]",
                "use_efficient_rollout=true",
            ],
            model_env,
            dry_run,
        )

    frame_specs = {
        "rollout_10_1fps": "1s:0,2s:1,4s:3,8s:7,16s:15",
        "rollout_10_4fps": "1s:3,2s:7,4s:15,8s:31,16s:63",
    }
    metric_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    for evaluation in protocol["evaluation"]:
        source_evaluation = source_evaluations[evaluation]
        audit = rollout_audit_path(model_name, dataset, evaluation)
        if force or not audit.exists():
            fps = source_evaluation.removeprefix("rollout_").removesuffix("fps")
            run(
                [
                    sys.executable,
                    "scripts/evaluate_nwm_predictions.py",
                    "--gt-dir",
                    str(gt_root / source_evaluation),
                    "--pred-dir",
                    str(output_root / dataset / source_evaluation),
                    "--output",
                    str(audit),
                    "--frames",
                    frame_specs[evaluation],
                    "--dataset",
                    dataset,
                    "--eval-type",
                    "rollout",
                    "--eval-name",
                    evaluation,
                    "--rollout-fps",
                    fps,
                    "--batch-size",
                    "10",
                    "--device",
                    "cuda",
                    "--dreamsim-cache",
                    str(DREAMSIM_CACHE),
                ],
                metric_env,
                dry_run,
            )
        registry_command(
            registry_path,
            [
                "import-prediction",
                "--model",
                model_name,
                "--dataset",
                dataset,
                "--evaluation",
                evaluation,
                "--audit",
                str(audit),
                "--protocol",
                UNSEEN_ROLLOUT_PROTOCOL,
            ],
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
                f"seed={42 + EVAL_SEED}",
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
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Output root; defaults to the legacy root for seed 0 and an isolated seed directory otherwise.",
    )
    parser.add_argument(
        "--shared-benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Read-only source of shared ground truth generated by the seed-0 benchmark.",
    )
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--models", type=csv, default=csv("nwm-base,nwm-real,nwm-release"))
    parser.add_argument(
        "--tasks",
        type=csv,
        default=csv("recon_prediction,navigation,unseen"),
        help=(
            "Comma-separated: recon_prediction,navigation,unseen,unseen_rollout. "
            "unseen is the one-shot protocol; unseen_rollout is the fixed 10-trajectory, "
            "two-GPU autoregressive metric and visualization protocol."
        ),
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
    global BENCHMARK_ROOT, SHARED_BENCHMARK_ROOT, EVAL_SEED
    args = parse_args()
    if args.eval_seed < 0:
        raise ValueError("--eval-seed must be non-negative")
    EVAL_SEED = args.eval_seed
    SHARED_BENCHMARK_ROOT = args.shared_benchmark_root.resolve()
    if args.benchmark_root is None:
        BENCHMARK_ROOT = (
            DEFAULT_BENCHMARK_ROOT
            if EVAL_SEED == 0
            else DEFAULT_BENCHMARK_ROOT / "evaluation_seeds" / f"seed{EVAL_SEED}"
        )
    else:
        BENCHMARK_ROOT = args.benchmark_root.resolve()
    if args.registry is None:
        args.registry = BENCHMARK_ROOT / "benchmark_results.json"
    registry = load_registry(args.registry)
    unknown_models = sorted(set(args.models) - set(registry["models"]))
    if unknown_models:
        raise ValueError(f"Models are not registered: {unknown_models}")
    unknown_tasks = sorted(
        set(args.tasks)
        - {"recon_prediction", "navigation", "unseen", "unseen_rollout"}
    )
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if not args.gpus:
        raise ValueError("At least one GPU is required")
    if "unseen_rollout" in args.tasks and len(args.gpus) < UNSEEN_ROLLOUT_WORLD_SIZE:
        raise ValueError(
            f"{UNSEEN_ROLLOUT_PROTOCOL} requires at least 2 GPUs; exactly the first "
            "2 are used so the 10-sample subset is not padded and stochastic sampling "
            "is comparable across models"
        )
    if "unseen_rollout" in args.tasks:
        validate_unseen_rollout_split(registry)
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
    env["NWM_EVAL_SEED"] = str(EVAL_SEED)

    # Finish every prediction/generalization protocol for one model before
    # advancing to the next model.  Navigation remains a final, separate phase.
    rollout_gpus = args.gpus[:UNSEEN_ROLLOUT_WORLD_SIZE]
    progress(
        f"START eval_seed={EVAL_SEED} models={','.join(args.models)} "
        f"tasks={','.join(args.tasks)} gpus={','.join(args.gpus)} "
        f"benchmark_root={BENCHMARK_ROOT}"
    )
    for model_name in args.models:
        model = registry["models"][model_name]
        if "recon_prediction" in args.tasks:
            progress(f"START model={model_name} task=recon_prediction")
            run_prediction(
                model_name,
                model,
                "recon",
                args.gpus,
                env,
                args.registry,
                args.force,
                args.dry_run,
            )
            progress(f"DONE model={model_name} task=recon_prediction")
        if "unseen" in args.tasks:
            progress(f"START model={model_name} task=unseen")
            run_prediction(
                model_name,
                model,
                "go_stanford",
                args.gpus,
                env,
                args.registry,
                args.force,
                args.dry_run,
                evaluations=("time",),
            )
            progress(f"DONE model={model_name} task=unseen")
        if "unseen_rollout" in args.tasks:
            progress(f"START model={model_name} task=unseen_rollout")
            run_unseen_rollout_prediction(
                model_name,
                model,
                rollout_gpus,
                env,
                args.registry,
                registry,
                args.force,
                args.dry_run,
            )
            progress(f"DONE model={model_name} task=unseen_rollout")

    if "unseen_rollout" in args.tasks:
        progress("START task=unseen_rollout_visualization")
        run_rollout_visualization(args.models, args.registry, env, args.dry_run)
        progress("DONE task=unseen_rollout_visualization")
    if "navigation" in args.tasks:
        for model_name in args.models:
            model = registry["models"][model_name]
            progress(
                f"START model={model_name} task=navigation "
                f"datasets={','.join(args.navigation_datasets)}"
            )
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
            progress(
                f"DONE model={model_name} task=navigation "
                f"datasets={','.join(args.navigation_datasets)}"
            )

    registry_command(
        args.registry,
        ["render", "--output", str(args.registry.with_suffix(".md"))],
        env,
        args.dry_run,
    )
    progress(f"COMPLETE registry={args.registry}")


if __name__ == "__main__":
    main()
