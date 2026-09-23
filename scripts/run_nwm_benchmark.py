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

from nwm_benchmark_registry import PROTOCOLS, dataset_sample_count, load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark"
)
DEFAULT_DATA_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm/data")
BENCHMARK_ROOT = DEFAULT_BENCHMARK_ROOT
SHARED_BENCHMARK_ROOT: Path | None = None
EVAL_SEED = 0
RECON_GT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/gt/recon"
)
DREAMSIM_CACHE = Path("/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models")
UNSEEN_ROLLOUT_WORLD_SIZE = 2
DIRECT_PREDICTION_WORLD_SIZE = 4
UNSEEN_ROLLOUT_PROTOCOL = "go_stanford_unseen_rollout_v1"
DIRECT_PROTOCOL = "direct_4s_v1"
ROLLOUT_PROTOCOL = "rollout_v1"
OOD_DIRECT_PROTOCOL = "ood_direct_4s_v1"
DEFAULT_RAENWM_PYTHON = Path(
    "/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python"
)


def progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[benchmark][{timestamp}] {message}", flush=True)


def csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def datasets_grouped_by_sample_count(
    protocol: dict, datasets: list[str], evaluation: str
) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for dataset in datasets:
        count = dataset_sample_count(protocol, dataset, evaluation)
        grouped.setdefault(count, []).append(dataset)
    return grouped


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
    if len(entries) != protocol["sample_count"]:
        raise RuntimeError(
            f"unseen rollout source split size mismatch for {split}: expected "
            f"{protocol['sample_count']}, got {len(entries)}"
        )


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def registry_command(
    registry: Path, arguments: list[str], env: dict[str, str], dry_run: bool
) -> None:
    run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/nwm_benchmark_registry.py"),
            "--registry",
            str(registry),
            *arguments,
        ],
        env,
        dry_run,
    )


def torchrun_prefix(gpus: list[str]) -> list[str]:
    return ["torchrun", "--standalone", f"--nproc-per-node={len(gpus)}"]


def raenwm_torchrun_prefix(python: Path, gpus: list[str]) -> list[str]:
    torchrun = python.parent / "torchrun"
    return [str(torchrun), "--standalone", f"--nproc-per-node={len(gpus)}"]


def prediction_audit_path(model_name: str, dataset: str, evaluation: str) -> Path:
    return (
        BENCHMARK_ROOT
        / "predictions"
        / model_name
        / f"{dataset}_{evaluation}_audit.json"
    )


def protocol_prediction_audit_path(
    model_name: str,
    dataset: str,
    evaluation: str,
    protocol: str | None,
) -> Path:
    if protocol is None:
        return prediction_audit_path(model_name, dataset, evaluation)
    return (
        BENCHMARK_ROOT
        / "predictions"
        / model_name
        / f"{dataset}_{evaluation}_{protocol}_audit.json"
    )


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


def direct_time_complete(
    path: Path, sample_count: int, horizons: tuple[int, ...] = (4,)
) -> bool:
    if not path.is_dir():
        return False
    expected_samples = {f"id_{index}" for index in range(sample_count)}
    samples = {child.name for child in path.glob("id_*") if child.is_dir()}
    if samples != expected_samples:
        return False
    return all(
        all((path / sample / f"{horizon}.png").is_file() for horizon in horizons)
        for sample in expected_samples
    )


def ood_protocol_root() -> Path:
    return BENCHMARK_ROOT / "protocol_runs" / OOD_DIRECT_PROTOCOL


def shared_ood_protocol_root() -> Path:
    shared_root = SHARED_BENCHMARK_ROOT or BENCHMARK_ROOT
    return shared_root / "protocol_runs" / OOD_DIRECT_PROTOCOL


def ood_prediction_audit_path(model_name: str, dataset: str) -> Path:
    return (
        ood_protocol_root()
        / "predictions"
        / model_name
        / f"{dataset}_time_audit.json"
    )


def resolve_protocol_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def validate_ood_protocol_inputs(
    registry: dict, datasets: list[str]
) -> dict[str, dict]:
    protocol = registry["protocols"][OOD_DIRECT_PROTOCOL]
    available = protocol["datasets"]
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"Unknown {OOD_DIRECT_PROTOCOL} datasets: {unknown}")
    verified: dict[str, dict] = {}
    for dataset in datasets:
        contract = available[dataset]
        checked = {}
        for key in ("report", "prediction_split", "navigation_split"):
            metadata = contract[key]
            path = resolve_protocol_path(metadata["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != metadata["sha256"]:
                raise RuntimeError(
                    f"{dataset} {key} SHA-256 mismatch: expected "
                    f"{metadata['sha256']}, got {actual}"
                )
            checked[key] = {"path": str(path.resolve()), "sha256": actual}
        report = json.loads(Path(checked["report"]["path"]).read_text(encoding="utf-8"))
        prediction_count = dataset_sample_count(protocol, dataset, "time")
        navigation_count = dataset_sample_count(protocol, dataset, "navigation")
        reported_prediction_count = report.get(
            "prediction_sample_count", report.get("prediction_samples")
        )
        reported_navigation_count = report.get(
            "navigation_sample_count", report.get("navigation_samples")
        )
        expected_fields = {
            "dataset": dataset,
            "prediction_sample_count": prediction_count,
            "navigation_sample_count": navigation_count,
        }
        reported_fields = {
            "dataset": report.get("dataset"),
            "prediction_sample_count": reported_prediction_count,
            "navigation_sample_count": reported_navigation_count,
        }
        if "rollout_sample_count" in contract:
            expected_fields["rollout_sample_count"] = dataset_sample_count(
                protocol, dataset, "rollout"
            )
            reported_fields["rollout_sample_count"] = report.get(
                "rollout_sample_count", report.get("rollout_samples")
            )
        optional_expected_fields = {
            "context_frames": protocol["context_frames"],
            "future_frames": protocol["future_frames"],
            "input_fps": protocol["input_fps"],
            "horizon_seconds": protocol["horizons_seconds"][0],
            "trajectory_cadence": contract["trajectory_cadence"],
            "temporal_semantics": contract["temporal_semantics"],
        }
        for key, expected in optional_expected_fields.items():
            if key in report:
                expected_fields[key] = expected
                reported_fields[key] = report[key]
        for key in (
            "position_units",
            "metric_navigation_evaluation_allowed",
        ):
            if key in contract:
                expected_fields[key] = contract[key]
                reported_fields[key] = report.get(key)
        mismatched = {
            key: (expected, reported_fields.get(key))
            for key, expected in expected_fields.items()
            if reported_fields.get(key) != expected
        }
        if mismatched:
            raise RuntimeError(f"{dataset} report contract mismatch: {mismatched}")
        contract_spacing = float(contract["metric_waypoint_spacing"])
        if "metric_waypoint_spacing" in report:
            report_spacing = float(report["metric_waypoint_spacing"])
            if report_spacing != contract_spacing:
                raise RuntimeError(
                    f"{dataset} waypoint spacing mismatch: expected "
                    f"{contract_spacing}, got {report_spacing}"
                )
        checked["metric_waypoint_spacing"] = contract[
            "metric_waypoint_spacing"
        ]
        checked["trajectory_cadence"] = contract["trajectory_cadence"]
        checked["temporal_semantics"] = contract["temporal_semantics"]
        verified[dataset] = checked
    return verified


def validate_ood_data_root(protocol: dict, env: dict[str, str]) -> None:
    expected = Path(protocol["data_root"]).resolve()
    actual = Path(env["NWM_DATA_ROOT"]).resolve()
    if actual != expected:
        raise RuntimeError(
            f"{OOD_DIRECT_PROTOCOL} data root mismatch: expected {expected}, got {actual}"
        )


def validate_registered_checkpoint(model_name: str, model: dict) -> None:
    checkpoint = Path(model["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected = model.get("sha256")
    if expected is None:
        raise RuntimeError(f"{model_name} has no pinned checkpoint SHA-256")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise RuntimeError(
            f"{model_name} checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )


def validate_navigation_protocol_inputs(
    registry: dict, datasets: list[str]
) -> None:
    protocol = registry["protocols"]["navigation_cem80_v1"]
    for dataset in datasets:
        metadata = protocol["splits"][dataset]
        path = resolve_protocol_path(metadata["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != metadata["sha256"]:
            raise RuntimeError(
                f"{dataset} navigation split SHA-256 mismatch: expected "
                f"{metadata['sha256']}, got {actual}"
            )


def validate_prediction_protocol_inputs(
    registry: dict, protocol_name: str, datasets: list[str]
) -> None:
    protocol = registry["protocols"][protocol_name]
    unknown = sorted(set(datasets) - set(protocol["datasets"]))
    if unknown:
        raise ValueError(f"Unknown {protocol_name} datasets: {unknown}")
    for dataset in datasets:
        metadata = protocol["splits"][dataset]
        path = resolve_protocol_path(metadata["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != metadata["sha256"]:
            raise RuntimeError(
                f"{dataset} {protocol_name} split SHA-256 mismatch: expected "
                f"{metadata['sha256']}, got {actual}"
            )


def write_ood_run_manifest(
    registry: dict,
    model_names: list[str],
    datasets: list[str],
    verified_inputs: dict[str, dict],
    physical_execution: dict,
    dry_run: bool,
) -> None:
    output = ood_protocol_root() / "run_manifest.json"
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_name": OOD_DIRECT_PROTOCOL,
        "protocol": registry["protocols"][OOD_DIRECT_PROTOCOL],
        "physical_execution": physical_execution,
        "verified_inputs": verified_inputs,
        "models": {
            name: {
                key: value
                for key, value in registry["models"][name].items()
                if key != "results"
            }
            for name in model_names
        },
    }
    print(f"+ write {output}", flush=True)
    if dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


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
    time_horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
    expected_count: int | None = None,
) -> Path:
    if dataset == "recon":
        return RECON_GT
    gt_dataset_root = ground_truth_root(dataset)
    if expected_count is None:
        expected_count = 500 if inference_type == "time" else 150
    evaluation_names = (
        ("time",) if inference_type == "time" else ("rollout_1fps", "rollout_4fps")
    )
    if inference_type == "time":
        complete = direct_time_complete(
            gt_dataset_root / "time", expected_count, time_horizons
        )
    else:
        complete = all(
            rollout_sequences_complete(
                gt_dataset_root / evaluation,
                expected_count,
                16 * int(evaluation.removeprefix("rollout_").removesuffix("fps")),
            )
            for evaluation in evaluation_names
        )
    if complete:
        return gt_dataset_root
    # isolated_nwm_infer.py appends its own ``gt`` component in ground-truth
    # mode, so pass the benchmark root rather than BENCHMARK_ROOT / "gt".
    gt_output_root = SHARED_BENCHMARK_ROOT or BENCHMARK_ROOT
    gt_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    command = [
            "torchrun",
            "--standalone",
            "--nproc-per-node=1",
            "isolated_nwm_infer.py",
            f"exp_dir={reference_model['exp_dir']}",
            f"output_dir={gt_output_root}",
            "gt=1",
            f"datasets_to_eval=[{dataset}]",
            f"eval_type={inference_type}",
            f"eval_expected_full_count={expected_count}",
            "batch_size=64",
            "num_workers=8",
            "pin_memory=false",
            "seed=0",
        ]
    if inference_type == "time":
        command.extend(
            [
                f"eval_len_traj_pred={max(time_horizons) * 4}",
                f"time_horizons_seconds=[{','.join(map(str, time_horizons))}]",
            ]
        )
    run(command, gt_env, dry_run)
    if not dry_run:
        if inference_type == "time":
            incomplete = {} if direct_time_complete(
                gt_dataset_root / "time", expected_count, time_horizons
            ) else {"time": sample_directory_count(gt_dataset_root / "time")}
        else:
            incomplete = {
                evaluation: sample_directory_count(gt_dataset_root / evaluation)
                for evaluation in evaluation_names
                if not rollout_sequences_complete(
                    gt_dataset_root / evaluation,
                    expected_count,
                    16 * int(evaluation.removeprefix("rollout_").removesuffix("fps")),
                )
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
    raenwm_python: Path = DEFAULT_RAENWM_PYTHON,
    serialize_logical_ranks: bool = False,
    batch_size: int | None = None,
    metric_batch_size: int = 32,
    time_horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
    protocol_by_evaluation: dict[str, str] | None = None,
    ground_truth_reference_model: dict | None = None,
    inference_already_prepared: bool = False,
) -> None:
    if evaluations is None:
        evaluations = ("time", "rollout_1fps", "rollout_4fps")
    if serialize_logical_ranks:
        if len(gpus) != 1:
            raise ValueError(
                "Serialized direct prediction requires exactly one physical GPU; "
                f"got {gpus}"
            )
        if model.get("backend") == "raenwm":
            raise ValueError(
                "Serialized logical-rank execution is implemented only for the "
                "local NWM backend"
            )
        unsupported = [evaluation for evaluation in evaluations if evaluation != "time"]
        if unsupported:
            raise ValueError(
                "Serialized logical-rank execution supports direct time prediction "
                f"only, got {unsupported}"
            )
    output_root = BENCHMARK_ROOT / "predictions" / model_name
    model_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}

    backend = "rae-nwm" if model.get("backend") == "raenwm" else "nwm"
    inference_contract = {
        "backend": backend,
        "sampler": (
            f"{model['provenance']['sampling_method']}_ode"
            if backend == "rae-nwm"
            else "ddpm"
        ),
        "sampling_steps": (
            int(model["provenance"]["sampling_steps"])
            if backend == "rae-nwm"
            else 250
        ),
        "seed": EVAL_SEED,
    }
    def audit_path(evaluation: str) -> Path:
        protocol = (
            protocol_by_evaluation.get(evaluation)
            if protocol_by_evaluation
            else None
        )
        return protocol_prediction_audit_path(
            model_name, dataset, evaluation, protocol
        )

    def evaluation_sample_count(evaluation: str) -> int:
        protocol_name = (
            protocol_by_evaluation.get(evaluation)
            if protocol_by_evaluation
            else None
        )
        if protocol_name is None:
            return 500 if evaluation == "time" else 150
        return dataset_sample_count(PROTOCOLS[protocol_name], dataset, evaluation)

    for inference_type in ("time", "rollout"):
        selected = [
            name
            for name in evaluations
            if name == inference_type or name.startswith(inference_type)
        ]
        if not selected:
            continue
        selected_counts = {evaluation_sample_count(name) for name in selected}
        if len(selected_counts) != 1:
            raise ValueError(
                f"Mixed sample counts for {dataset}/{inference_type}: "
                f"{sorted(selected_counts)}"
            )
        sample_count = selected_counts.pop()
        if inference_already_prepared:
            continue
        if not force and all(
            audit_path(name).exists()
            for name in selected
        ):
            continue
        ensure_ground_truth(
            dataset,
            inference_type,
            ground_truth_reference_model or model,
            gpus,
            env,
            dry_run,
            time_horizons=time_horizons,
            expected_count=sample_count,
        )
        if backend == "rae-nwm":
            assets = Path(model["assets_root"])
            command = [
                *raenwm_torchrun_prefix(raenwm_python, gpus),
                "scripts/raenwm_infer.py",
                "--source",
                model["source_dir"],
                "--checkpoint",
                model["checkpoint"],
                "--decoder",
                str(assets / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"),
                "--normalization-stats",
                str(assets / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"),
                "--dino-model",
                str(assets / "models/dinov2-with-registers-base"),
                "--project-root",
                str(PROJECT_ROOT),
                "--data-root",
                model_env["NWM_DATA_ROOT"],
                "--output-root",
                str(output_root),
                "--datasets",
                dataset,
                "--eval-type",
                inference_type,
                "--future-frames",
                str(64 if inference_type == "rollout" else max(time_horizons) * 4),
                "--batch-size",
                str(batch_size or 16),
                "--num-workers",
                "4",
                "--sampling-method",
                model["provenance"]["sampling_method"],
                "--num-steps",
                str(model["provenance"]["sampling_steps"]),
                "--seed",
                str(EVAL_SEED),
                "--expected-sample-count",
                str(sample_count),
            ]
            if inference_type == "time":
                command.extend(["--horizons", *map(str, time_horizons)])
            else:
                command.extend(["--rollout-fps", "1", "4"])
            if force:
                command.append("--force")
            raenwm_env = {
                **model_env,
                "HF_HOME": str(assets / "hf_cache"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            }
            run(command, raenwm_env, dry_run)
        else:
            logical_ranks = (
                range(DIRECT_PREDICTION_WORLD_SIZE)
                if serialize_logical_ranks
                else (None,)
            )
            for logical_rank in logical_ranks:
                command = [
                    *torchrun_prefix(gpus),
                    "isolated_nwm_infer.py",
                    f"exp_dir={model['exp_dir']}",
                    f"ckp={model['checkpoint_id']}",
                    f"output_dir={output_root}",
                    f"prediction_dir={output_root}",
                    f"datasets_to_eval=[{dataset}]",
                    f"eval_type={inference_type}",
                    f"eval_expected_full_count={sample_count}",
                    "eval_diffusion_steps=250",
                    f"batch_size={batch_size or 64}",
                    "num_workers=4",
                    "pin_memory=false",
                ]
                if inference_type == "time":
                    command.extend(
                        [
                            f"eval_len_traj_pred={max(time_horizons) * 4}",
                            f"time_horizons_seconds=[{','.join(map(str, time_horizons))}]",
                        ]
                    )
                if logical_rank is None:
                    command.append(f"seed={EVAL_SEED}")
                else:
                    if sample_count % DIRECT_PREDICTION_WORLD_SIZE != 0:
                        raise ValueError(
                            "Serialized direct prediction requires a sample count "
                            f"divisible by {DIRECT_PREDICTION_WORLD_SIZE}; got "
                            f"{sample_count} for {dataset}"
                        )
                    sample_indices = range(
                        logical_rank, sample_count, DIRECT_PREDICTION_WORLD_SIZE
                    )
                    indices = ",".join(map(str, sample_indices))
                    command.extend(
                        [
                            f"eval_sample_indices=[{indices}]",
                            f"seed={EVAL_SEED}",
                        ]
                    )
                    progress(
                        f"SERIAL direct dataset={dataset} "
                        f"logical_rank={logical_rank}/"
                        f"{DIRECT_PREDICTION_WORLD_SIZE - 1} "
                        f"physical_gpu={gpus[0]} "
                        f"samples={sample_count // DIRECT_PREDICTION_WORLD_SIZE}"
                    )
                if inference_type == "rollout":
                    command.extend(
                        ["rollout_fps_values=[1,4]", "use_efficient_rollout=true"]
                    )
                run(command, model_env, dry_run)
        if not dry_run:
            if inference_type == "time":
                complete = direct_time_complete(
                    output_root / dataset / "time", sample_count, time_horizons
                )
            else:
                complete = all(
                    rollout_sequences_complete(
                        output_root / dataset / evaluation,
                        sample_count,
                        16 * int(evaluation.removeprefix("rollout_").removesuffix("fps")),
                    )
                    for evaluation in selected
                )
            if not complete:
                raise RuntimeError(
                    f"Incomplete {inference_type} prediction for {model_name}/{dataset}"
                )

    frame_specs = {
        "time": ",".join(f"{horizon}s:{horizon}" for horizon in time_horizons),
        "rollout_1fps": "1s:0,2s:1,4s:3,8s:7,16s:15",
        "rollout_4fps": "1s:3,2s:7,4s:15,8s:31,16s:63",
    }
    gt_root = ground_truth_root(dataset)
    metric_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    for evaluation in evaluations:
        audit = audit_path(evaluation)
        if force or not audit.exists():
            prediction_dir = output_root / dataset / evaluation
            command = [
                sys.executable,
                "scripts/evaluate_nwm_predictions.py",
                "--gt-dir",
                str(gt_root / evaluation),
                "--pred-dir",
                str(prediction_dir),
                "--output",
                str(audit),
                "--frames",
                frame_specs[evaluation],
                "--dataset",
                dataset,
                "--eval-type",
                "rollout" if evaluation.startswith("rollout") else "time",
                "--eval-name",
                evaluation,
                "--batch-size",
                str(metric_batch_size),
                "--device",
                "cuda",
                "--dreamsim-cache",
                str(DREAMSIM_CACHE),
                "--inference-backend",
                inference_contract["backend"],
                "--sampler",
                inference_contract["sampler"],
                "--sampling-steps",
                str(inference_contract["sampling_steps"]),
                "--seed",
                str(inference_contract["seed"]),
            ]
            if evaluation.startswith("rollout"):
                command.extend(
                    ["--rollout-fps", evaluation.split("_")[1].removesuffix("fps")]
                )
            run(command, metric_env, dry_run)
        registry_arguments = [
            "import-prediction",
            "--model",
            model_name,
            "--dataset",
            dataset,
            "--evaluation",
            evaluation,
            "--audit",
            str(audit),
        ]
        if protocol_by_evaluation and evaluation in protocol_by_evaluation:
            registry_arguments.extend(
                ["--protocol", protocol_by_evaluation[evaluation]]
            )
        registry_command(
            registry,
            registry_arguments,
            env,
            dry_run,
        )


def run_grouped_prediction_inference(
    model_name: str,
    model: dict,
    datasets: list[str],
    evaluations: tuple[str, ...],
    gpus: list[str],
    env: dict[str, str],
    force: bool,
    dry_run: bool,
    *,
    reference_model: dict,
    raenwm_python: Path,
    batch_size: int | None,
    time_horizons: tuple[int, ...],
) -> None:
    """Load a model once per inference type and evaluate all selected datasets."""

    output_root = BENCHMARK_ROOT / "predictions" / model_name
    model_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    backend = "rae-nwm" if model.get("backend") == "raenwm" else "nwm"
    for inference_type in ("time", "rollout"):
        selected_evaluations = [
            name
            for name in evaluations
            if name == inference_type or name.startswith(inference_type)
        ]
        if not selected_evaluations:
            continue
        protocol = PROTOCOLS[
            DIRECT_PROTOCOL if inference_type == "time" else ROLLOUT_PROTOCOL
        ]
        count_evaluation = selected_evaluations[0]
        sample_counts = {
            dataset: dataset_sample_count(protocol, dataset, count_evaluation)
            for dataset in datasets
        }
        incomplete: list[str] = []
        for dataset in datasets:
            sample_count = sample_counts[dataset]
            if force or dry_run:
                incomplete.append(dataset)
            elif inference_type == "time":
                if not direct_time_complete(
                    output_root / dataset / "time", sample_count, time_horizons
                ):
                    incomplete.append(dataset)
            elif not all(
                rollout_sequences_complete(
                    output_root / dataset / evaluation,
                    sample_count,
                    16 * int(evaluation.removeprefix("rollout_").removesuffix("fps")),
                )
                for evaluation in selected_evaluations
            ):
                incomplete.append(dataset)
        if not incomplete:
            continue

        for dataset in incomplete:
            ensure_ground_truth(
                dataset,
                inference_type,
                reference_model,
                gpus,
                env,
                dry_run,
                time_horizons=time_horizons,
                expected_count=sample_counts[dataset],
            )

        grouped = datasets_grouped_by_sample_count(
            protocol, incomplete, count_evaluation
        )
        for sample_count, grouped_datasets in grouped.items():
            if backend == "rae-nwm":
                assets = Path(model["assets_root"])
                command = [
                    *raenwm_torchrun_prefix(raenwm_python, gpus),
                    "scripts/raenwm_infer.py",
                    "--source",
                    model["source_dir"],
                    "--checkpoint",
                    model["checkpoint"],
                    "--decoder",
                    str(assets / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"),
                    "--normalization-stats",
                    str(assets / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"),
                    "--dino-model",
                    str(assets / "models/dinov2-with-registers-base"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--data-root",
                    model_env["NWM_DATA_ROOT"],
                    "--output-root",
                    str(output_root),
                    "--datasets",
                    *grouped_datasets,
                    "--eval-type",
                    inference_type,
                    "--future-frames",
                    str(64 if inference_type == "rollout" else max(time_horizons) * 4),
                    "--batch-size",
                    str(batch_size or 16),
                    "--num-workers",
                    "4",
                    "--sampling-method",
                    model["provenance"]["sampling_method"],
                    "--num-steps",
                    str(model["provenance"]["sampling_steps"]),
                    "--seed",
                    str(EVAL_SEED),
                    "--expected-sample-count",
                    str(sample_count),
                ]
                if inference_type == "time":
                    command.extend(["--horizons", *map(str, time_horizons)])
                else:
                    command.extend(["--rollout-fps", "1", "4"])
                if force:
                    command.append("--force")
                run(
                    command,
                    {
                        **model_env,
                        "HF_HOME": str(assets / "hf_cache"),
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    },
                    dry_run,
                )
            else:
                command = [
                    *torchrun_prefix(gpus),
                    "isolated_nwm_infer.py",
                    f"exp_dir={model['exp_dir']}",
                    f"ckp={model['checkpoint_id']}",
                    f"output_dir={output_root}",
                    f"prediction_dir={output_root}",
                    f"datasets_to_eval=[{','.join(grouped_datasets)}]",
                    f"eval_type={inference_type}",
                    f"eval_expected_full_count={sample_count}",
                    "eval_diffusion_steps=250",
                    f"batch_size={batch_size or 64}",
                    "num_workers=4",
                    "pin_memory=false",
                    f"seed={EVAL_SEED}",
                ]
                if inference_type == "time":
                    command.extend(
                        [
                            f"eval_len_traj_pred={max(time_horizons) * 4}",
                            f"time_horizons_seconds=[{','.join(map(str, time_horizons))}]",
                        ]
                    )
                else:
                    command.extend(
                        ["rollout_fps_values=[1,4]", "use_efficient_rollout=true"]
                    )
                run(command, model_env, dry_run)

        if not dry_run:
            for dataset in incomplete:
                sample_count = sample_counts[dataset]
                if inference_type == "time":
                    complete = direct_time_complete(
                        output_root / dataset / "time", sample_count, time_horizons
                    )
                else:
                    complete = all(
                        rollout_sequences_complete(
                            output_root / dataset / evaluation,
                            sample_count,
                            16
                            * int(
                                evaluation.removeprefix("rollout_").removesuffix("fps")
                            ),
                        )
                        for evaluation in selected_evaluations
                    )
                if not complete:
                    raise RuntimeError(
                        f"Incomplete {inference_type} prediction for "
                        f"{model_name}/{dataset}"
                    )


def ensure_ood_ground_truth(
    protocol: dict,
    reference_model: dict,
    datasets: list[str],
    gpus: list[str],
    env: dict[str, str],
    dry_run: bool,
) -> Path:
    protocol_root = shared_ood_protocol_root()
    gt_root = protocol_root / "gt"
    horizons = tuple(int(value) for value in protocol["horizons_seconds"])
    sample_counts = {
        dataset: dataset_sample_count(protocol, dataset, "time")
        for dataset in datasets
    }
    incomplete = [
        dataset
        for dataset in datasets
        if not direct_time_complete(
            gt_root / dataset / "time", sample_counts[dataset], horizons
        )
    ]
    if not incomplete:
        return gt_root
    gt_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    for sample_count, grouped_datasets in datasets_grouped_by_sample_count(
        protocol, incomplete, "time"
    ).items():
        run(
            [
                "torchrun",
                "--standalone",
                "--nproc-per-node=1",
                "isolated_nwm_infer.py",
                f"exp_dir={reference_model['exp_dir']}",
                f"output_dir={protocol_root}",
                "gt=1",
                f"datasets_to_eval=[{','.join(grouped_datasets)}]",
                "eval_type=time",
                f"eval_len_traj_pred={protocol['future_frames']}",
                f"time_horizons_seconds=[{','.join(map(str, horizons))}]",
                f"eval_expected_full_count={sample_count}",
                "batch_size=64",
                "num_workers=8",
                "pin_memory=false",
                f"seed={protocol['seed']}",
            ],
            gt_env,
            dry_run,
        )
    if not dry_run:
        remaining = [
            dataset
            for dataset in incomplete
            if not direct_time_complete(
                gt_root / dataset / "time", sample_counts[dataset], horizons
            )
        ]
        if remaining:
            raise RuntimeError(
                f"Incomplete {OOD_DIRECT_PROTOCOL} ground truth: {remaining}"
            )
    return gt_root


def run_ood_direct_prediction(
    model_name: str,
    model: dict,
    datasets: list[str],
    gpus: list[str],
    env: dict[str, str],
    registry_path: Path,
    registry: dict,
    force: bool,
    dry_run: bool,
    raenwm_python: Path = DEFAULT_RAENWM_PYTHON,
    serialize_logical_ranks: bool = False,
) -> None:
    protocol = registry["protocols"][OOD_DIRECT_PROTOCOL]
    sample_counts = {
        dataset: dataset_sample_count(protocol, dataset, "time")
        for dataset in datasets
    }
    expected_world_size = int(protocol["execution"]["distributed_world_size"])
    if serialize_logical_ranks:
        if len(gpus) != 1:
            raise ValueError(
                f"Serialized {OOD_DIRECT_PROTOCOL} requires exactly one physical GPU; "
                f"got {gpus}"
            )
    elif len(gpus) != expected_world_size or len(set(gpus)) != expected_world_size:
        raise ValueError(
            f"{OOD_DIRECT_PROTOCOL} requires exactly {expected_world_size} unique GPUs "
            f"for reproducible stochastic sampling; got {len(gpus)}"
        )
    undersized = {
        dataset: count
        for dataset, count in sample_counts.items()
        if count < expected_world_size
    }
    if undersized:
        raise ValueError(
            f"{OOD_DIRECT_PROTOCOL} requires at least one sample per logical rank; "
            f"world size {expected_world_size}, undersized counts {undersized}"
        )
    horizons = tuple(int(value) for value in protocol["horizons_seconds"])
    if horizons != (4,):
        raise ValueError(f"{OOD_DIRECT_PROTOCOL} must contain only the 4-second horizon")
    gt_root = ensure_ood_ground_truth(
        protocol,
        registry["models"]["nwm-real"],
        datasets,
        gpus,
        env,
        dry_run,
    )
    output_root = ood_protocol_root() / "predictions" / model_name
    incomplete = [
        dataset
        for dataset in datasets
        if not direct_time_complete(
            output_root / dataset / "time", sample_counts[dataset], horizons
        )
    ]
    prediction_ran = bool(force or incomplete)
    model_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    backend = "rae-nwm" if model.get("backend") == "raenwm" else "nwm"
    inference = protocol["inference"][backend]
    if prediction_ran:
        selected = datasets if force else incomplete
        if backend == "rae-nwm":
            if serialize_logical_ranks:
                raise ValueError(
                    "Serialized logical-rank execution is implemented only for the "
                    "local NWM backend; complete RAE-NWM with native distributed "
                    "execution before using this recovery mode"
                )
            assets = Path(model["assets_root"])
            model_env.update(
                {
                    "HF_HOME": str(assets / "hf_cache"),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                }
            )
        for sample_count, grouped_datasets in datasets_grouped_by_sample_count(
            protocol, selected, "time"
        ).items():
            if backend == "rae-nwm":
                command = [
                    *raenwm_torchrun_prefix(raenwm_python, gpus),
                    "scripts/raenwm_infer.py",
                    "--source",
                    model["source_dir"],
                    "--checkpoint",
                    model["checkpoint"],
                    "--decoder",
                    str(assets / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"),
                    "--normalization-stats",
                    str(assets / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"),
                    "--dino-model",
                    str(assets / "models/dinov2-with-registers-base"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--data-root",
                    model_env["NWM_DATA_ROOT"],
                    "--output-root",
                    str(output_root),
                    "--datasets",
                    *grouped_datasets,
                    "--horizons",
                    "4",
                    "--future-frames",
                    str(protocol["future_frames"]),
                    "--batch-size",
                    str(inference["batch_size_per_rank"]),
                    "--num-workers",
                    "4",
                    "--sampling-method",
                    model["provenance"]["sampling_method"],
                    "--num-steps",
                    str(inference["sampling_steps"]),
                    "--seed",
                    str(protocol["seed"]),
                    "--expected-sample-count",
                    str(sample_count),
                    "--force",
                ]
                # RAE-NWM normally skips completed images. Recompute every sample in
                # each incomplete dataset so a resumed run consumes the identical
                # random stream as an uninterrupted fixed-topology run.
                run(command, model_env, dry_run)
            else:
                logical_ranks = (
                    range(expected_world_size)
                    if serialize_logical_ranks
                    else (None,)
                )
                for logical_rank in logical_ranks:
                    command = [
                        *torchrun_prefix(gpus),
                        "isolated_nwm_infer.py",
                        f"exp_dir={model['exp_dir']}",
                        f"ckp={model['checkpoint_id']}",
                        f"output_dir={output_root}",
                        f"prediction_dir={output_root}",
                        f"datasets_to_eval=[{','.join(grouped_datasets)}]",
                        "eval_type=time",
                        f"eval_len_traj_pred={protocol['future_frames']}",
                        "time_horizons_seconds=[4]",
                        f"eval_expected_full_count={sample_count}",
                        f"eval_diffusion_steps={inference['sampling_steps']}",
                        f"batch_size={inference['batch_size_per_rank']}",
                        "num_workers=4",
                        "pin_memory=false",
                    ]
                    if logical_rank is None:
                        command.append(f"seed={protocol['seed']}")
                    else:
                        sample_indices = range(
                            logical_rank, sample_count, expected_world_size
                        )
                        indices = ",".join(map(str, sample_indices))
                        command.extend(
                            [
                                f"eval_sample_indices=[{indices}]",
                                f"seed={protocol['seed']}",
                            ]
                        )
                        progress(
                            f"SERIAL logical_rank={logical_rank}/"
                            f"{expected_world_size - 1} physical_gpu={gpus[0]} "
                            f"samples={len(range(logical_rank, sample_count, expected_world_size))}"
                        )
                    run(command, model_env, dry_run)
        if not dry_run:
            remaining = [
                dataset
                for dataset in selected
                if not direct_time_complete(
                    output_root / dataset / "time", sample_counts[dataset], horizons
                )
            ]
            if remaining:
                raise RuntimeError(
                    f"Incomplete {OOD_DIRECT_PROTOCOL} predictions for "
                    f"{model_name}: {remaining}"
                )

    metric_env = {**env, "CUDA_VISIBLE_DEVICES": gpus[0]}
    for dataset in datasets:
        audit = ood_prediction_audit_path(model_name, dataset)
        if force or prediction_ran or not audit.exists():
            run(
                [
                    sys.executable,
                    "scripts/evaluate_nwm_predictions.py",
                    "--gt-dir",
                    str(gt_root / dataset / "time"),
                    "--pred-dir",
                    str(output_root / dataset / "time"),
                    "--output",
                    str(audit),
                    "--frames",
                    "4s:4",
                    "--dataset",
                    dataset,
                    "--eval-type",
                    "time",
                    "--eval-name",
                    "time",
                    "--batch-size",
                    "32",
                    "--device",
                    "cuda",
                    "--dreamsim-cache",
                    str(DREAMSIM_CACHE),
                    "--inference-backend",
                    backend,
                    "--sampler",
                    inference["sampler"],
                    "--sampling-steps",
                    str(inference["sampling_steps"]),
                    "--seed",
                    str(protocol["seed"]),
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
                "time",
                "--audit",
                str(audit),
                "--protocol",
                OOD_DIRECT_PROTOCOL,
            ],
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
        BENCHMARK_ROOT / "visualizations" / UNSEEN_ROLLOUT_PROTOCOL / comparison
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
                16 * int(evaluation.removeprefix("rollout_").removesuffix("fps")),
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
            16 * int(source_evaluation.removeprefix("rollout_").removesuffix("fps")),
        )
        for source_evaluation in source_evaluations.values()
    )
    gt_root = ensure_unseen_rollout_ground_truth(protocol, model, gpus, env, dry_run)

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
        "rollout_1fps": "1s:0,2s:1,4s:3,8s:7,16s:15",
        "rollout_4fps": "1s:3,2s:7,4s:15,8s:31,16s:63",
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
                    "32",
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
    raenwm_python: Path = DEFAULT_RAENWM_PYTHON,
    raenwm_num_steps: int | None = None,
) -> None:
    is_raenwm = model.get("backend") == "raenwm"
    if is_raenwm and raenwm_num_steps not in (50, 250):
        raise ValueError(
            "RAE-NWM navigation requires an explicit --raenwm-planning-steps "
            "choice of 50 or 250"
        )
    output_root = BENCHMARK_ROOT / "planning" / model_name
    if is_raenwm:
        output_root = output_root / f"euler{raenwm_num_steps}"
    existing = {
        dataset: planning_result_path(output_root, dataset) for dataset in datasets
    }
    if force or any(path is None for path in existing.values()):
        planning_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
        if is_raenwm:
            assets = Path(model["assets_root"])
            planning_env.update(
                {
                    "HF_HOME": str(assets / "hf_cache"),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                }
            )
            commands = []
            for sample_count, grouped_datasets in datasets_grouped_by_sample_count(
                PROTOCOLS["navigation_cem80_v1"], datasets, "navigation"
            ).items():
                command = [
                    *raenwm_torchrun_prefix(raenwm_python, gpus),
                    "scripts/raenwm_planning_eval.py",
                    "--source",
                    model["source_dir"],
                    "--checkpoint",
                    model["checkpoint"],
                    "--decoder",
                    str(
                        assets
                        / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
                    ),
                    "--normalization-stats",
                    str(assets / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"),
                    "--dino-model",
                    str(assets / "models/dinov2-with-registers-base"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--data-root",
                    planning_env["NWM_DATA_ROOT"],
                    "--output-root",
                    str(output_root),
                    "--datasets",
                    *grouped_datasets,
                    "--num-samples",
                    "80",
                    "--topk",
                    "5",
                    "--opt-steps",
                    "1",
                    "--num-repeat-eval",
                    "3",
                    "--microbatch-size",
                    str(microbatch_size or 80),
                    "--sampling-method",
                    model["provenance"]["sampling_method"],
                    "--num-steps",
                    str(raenwm_num_steps),
                    "--seed",
                    str(42 + EVAL_SEED),
                    "--expected-sample-count",
                    str(sample_count),
                ]
                if force:
                    command.append("--no-resume")
                commands.append(command)
        else:
            commands = [
                [
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
                    f"planning_sample_seed={42 + EVAL_SEED}",
                    "cost_fn=lpips",
                    "compute_cost_with_recon=true",
                    "save_preds=false",
                    "plot=false",
                    f"resume_planning_samples={'false' if force else 'true'}",
                ]
            ]
            if microbatch_size is not None:
                commands[0].append(f"planning_microbatch_size={microbatch_size}")
        for command in commands:
            run(command, planning_env, dry_run)
    for dataset in datasets:
        result = planning_result_path(output_root, dataset)
        if result is None:
            if dry_run:
                continue
            raise FileNotFoundError(
                f"Planning result not produced for {model_name}/{dataset}"
            )
        registry_command(
            registry,
            [
                "import-planning",
                "--model",
                model_name,
                "--dataset",
                dataset,
                "--metrics",
                str(result),
            ],
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
    parser.add_argument(
        "--raenwm-python",
        type=Path,
        default=DEFAULT_RAENWM_PYTHON,
        help="Python executable in the official RAE-NWM environment.",
    )
    parser.add_argument(
        "--models", type=csv, default=csv("nwm-base,nwm-real,nwm-release")
    )
    parser.add_argument(
        "--metrics",
        type=csv,
        default=None,
        help=(
            "Unified benchmark entry point: comma-separated direct,rollout,navigation. "
            "When set, --datasets selects any subset of the nine registered datasets."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=csv,
        default=csv(
            "recon,scand,huron,tartan_drive,go_stanford,planetary_rover,"
            "unitree_go2,tum_rgbd,uzh_fpv"
        ),
        help="Dataset subset for --metrics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU inference batch size (backend default: NWM 64, RAE-NWM 16).",
    )
    parser.add_argument(
        "--metric-batch-size",
        type=int,
        default=32,
        help="Per-GPU LPIPS/DreamSim metric batch size.",
    )
    parser.add_argument(
        "--tasks",
        type=csv,
        default=csv("recon_prediction,navigation,unseen"),
        help=(
            "Comma-separated: recon_prediction,navigation,unseen,unseen_rollout,"
            "ood_direct_4s. "
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
        "--raenwm-planning-steps",
        type=int,
        choices=(50, 250),
        default=None,
        help=(
            "Required for RAE-NWM navigation: official 50-step Euler or "
            "250-step Euler. This is deliberately explicit because it changes "
            "the model-native inference budget."
        ),
    )
    parser.add_argument(
        "--navigation-datasets",
        type=csv,
        default=csv("recon,scand"),
        help="Comma-separated navigation datasets to evaluate.",
    )
    parser.add_argument(
        "--ood-datasets",
        type=csv,
        default=csv("planetary_rover,unitree_go2,tum_rgbd,uzh_fpv"),
        help=f"Comma-separated dataset subset from {OOD_DIRECT_PROTOCOL}.",
    )
    parser.add_argument(
        "--ood-serial-ranks",
        action="store_true",
        help=(
            "Run the pinned four logical OOD ranks sequentially on one physical "
            "GPU while preserving each rank's seed, sample shard, and batch size."
        ),
    )
    parser.add_argument(
        "--direct-serial-ranks",
        action="store_true",
        help=(
            "Run the four logical ranks used by standard RECON/Go Stanford direct "
            "prediction sequentially on one physical GPU, preserving their seeds, "
            "sample shards, and per-rank batch size."
        ),
    )
    parser.add_argument(
        "--recon-direct-only",
        action="store_true",
        help="For recon_prediction, evaluate direct time prediction without rollout.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_unified_benchmark(args: argparse.Namespace, registry: dict, env: dict[str, str]) -> None:
    metrics = list(dict.fromkeys(args.metrics))
    unknown_metrics = sorted(set(metrics) - {"direct", "rollout", "navigation"})
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {unknown_metrics}")
    if not metrics:
        raise ValueError("--metrics must select at least one metric")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.metric_batch_size < 1:
        raise ValueError("--metric-batch-size must be positive")
    if args.eval_seed != 0:
        raise ValueError("The comparable unified v1 protocols are pinned to --eval-seed=0")
    if args.direct_serial_ranks or args.ood_serial_ranks:
        raise ValueError(
            "Serialized logical ranks are obsolete for --metrics: sample-ID keyed "
            "randomness supports any physical GPU count directly"
        )

    datasets = list(dict.fromkeys(args.datasets))
    if "direct" in metrics:
        validate_prediction_protocol_inputs(registry, DIRECT_PROTOCOL, datasets)
    if "rollout" in metrics:
        validate_prediction_protocol_inputs(registry, ROLLOUT_PROTOCOL, datasets)
    if "navigation" in metrics:
        navigation_datasets = set(
            registry["protocols"]["navigation_cem80_v1"]["datasets"]
        )
        unknown = sorted(set(datasets) - navigation_datasets)
        if unknown:
            raise ValueError(f"Unknown navigation datasets: {unknown}")
        validate_navigation_protocol_inputs(registry, datasets)

    raenwm_models = [
        name
        for name in args.models
        if registry["models"][name].get("backend") == "raenwm"
    ]
    if raenwm_models and not args.dry_run and not args.raenwm_python.is_file():
        raise FileNotFoundError(
            f"RAE-NWM Python executable does not exist: {args.raenwm_python}"
        )
    if (
        raenwm_models
        and "navigation" in metrics
        and args.raenwm_planning_steps is None
    ):
        raise ValueError(
            "Choose --raenwm-planning-steps 50 (official Euler setting) or 250"
        )
    if not args.dry_run:
        for model_name in args.models:
            validate_registered_checkpoint(model_name, registry["models"][model_name])

    progress(
        f"START unified eval_seed={EVAL_SEED} models={','.join(args.models)} "
        f"metrics={','.join(metrics)} datasets={','.join(datasets)} "
        f"gpus={','.join(args.gpus)} batch_size={args.batch_size or 'backend-default'}"
    )
    prediction_evaluations: list[str] = []
    protocol_by_evaluation: dict[str, str] = {}
    if "direct" in metrics:
        prediction_evaluations.append("time")
        protocol_by_evaluation["time"] = DIRECT_PROTOCOL
    if "rollout" in metrics:
        prediction_evaluations.extend(("rollout_1fps", "rollout_4fps"))
        protocol_by_evaluation.update(
            {
                "rollout_1fps": ROLLOUT_PROTOCOL,
                "rollout_4fps": ROLLOUT_PROTOCOL,
            }
        )

    for model_name in args.models:
        model = registry["models"][model_name]
        if prediction_evaluations:
            run_grouped_prediction_inference(
                model_name,
                model,
                datasets,
                tuple(prediction_evaluations),
                args.gpus,
                env,
                args.force,
                args.dry_run,
                reference_model=registry["models"]["nwm-real"],
                raenwm_python=args.raenwm_python,
                batch_size=args.batch_size,
                time_horizons=(4,),
            )
        for dataset in datasets:
            if prediction_evaluations:
                progress(
                    f"START model={model_name} dataset={dataset} "
                    f"metrics={','.join(metric for metric in metrics if metric != 'navigation')}"
                )
                run_prediction(
                    model_name,
                    model,
                    dataset,
                    args.gpus,
                    env,
                    args.registry,
                    args.force,
                    args.dry_run,
                    evaluations=tuple(prediction_evaluations),
                    raenwm_python=args.raenwm_python,
                    batch_size=args.batch_size,
                    metric_batch_size=args.metric_batch_size,
                    time_horizons=(4,),
                    protocol_by_evaluation=protocol_by_evaluation,
                    ground_truth_reference_model=registry["models"]["nwm-real"],
                    inference_already_prepared=True,
                )
                progress(f"DONE model={model_name} dataset={dataset} prediction")

        if "navigation" in metrics:
            progress(
                f"START model={model_name} metrics=navigation "
                f"datasets={','.join(datasets)}"
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
                datasets,
                raenwm_python=args.raenwm_python,
                raenwm_num_steps=args.raenwm_planning_steps,
            )
            progress(f"DONE model={model_name} metrics=navigation")

    registry_command(
        args.registry,
        ["render", "--output", str(args.registry.with_suffix(".md"))],
        env,
        args.dry_run,
    )
    progress(f"COMPLETE registry={args.registry}")


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
        - {
            "recon_prediction",
            "navigation",
            "unseen",
            "unseen_rollout",
            "ood_direct_4s",
        }
    )
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if not args.gpus:
        raise ValueError("At least one GPU is required")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError(f"GPU indices must be unique, got {args.gpus}")
    if args.direct_serial_ranks and set(args.tasks) & {"recon_prediction", "unseen"}:
        if len(args.gpus) != 1:
            raise ValueError(
                f"--direct-serial-ranks requires exactly one physical GPU; got {args.gpus}"
            )
        if "recon_prediction" in args.tasks and not args.recon_direct_only:
            raise ValueError(
                "--direct-serial-ranks with recon_prediction requires "
                "--recon-direct-only because rollout has a separate protocol"
            )
        unsupported_models = [
            model_name
            for model_name in args.models
            if registry["models"][model_name].get("backend") == "raenwm"
        ]
        if unsupported_models:
            raise ValueError(
                "--direct-serial-ranks currently supports local NWM models only; "
                f"got {unsupported_models}"
            )

    env = os.environ.copy()
    env.setdefault("NWM_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    env.setdefault(
        "NWM_INDEX_ROOT",
        "/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices",
    )
    env.setdefault("TORCH_HOME", "/file_system/vepfs/algorithm/dujun.nie/models")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["NWM_EVAL_SEED"] = str(EVAL_SEED)

    if args.metrics is not None:
        run_unified_benchmark(args, registry, env)
        return

    if "unseen_rollout" in args.tasks:
        validate_unseen_rollout_split(registry)
    navigation_protocol_datasets = set(
        registry["protocols"]["navigation_cem80_v1"]["datasets"]
    )
    unknown_navigation_datasets = sorted(
        set(args.navigation_datasets) - navigation_protocol_datasets
    )
    if unknown_navigation_datasets:
        raise ValueError(f"Unknown navigation datasets: {unknown_navigation_datasets}")
    if "navigation" in args.tasks:
        validate_navigation_protocol_inputs(registry, args.navigation_datasets)
    ood_navigation_datasets = (
        sorted(
            set(args.navigation_datasets)
            & set(registry["protocols"][OOD_DIRECT_PROTOCOL]["datasets"])
        )
        if "navigation" in args.tasks
        else []
    )
    ood_validation_datasets = sorted(
        set(ood_navigation_datasets)
        | (set(args.ood_datasets) if "ood_direct_4s" in args.tasks else set())
    )
    if ood_validation_datasets:
        validate_ood_data_root(registry["protocols"][OOD_DIRECT_PROTOCOL], env)
    if "ood_direct_4s" in args.tasks:
        expected_world_size = int(
            registry["protocols"][OOD_DIRECT_PROTOCOL]["execution"][
                "distributed_world_size"
            ]
        )
        if args.ood_serial_ranks and len(args.gpus) != 1:
            raise ValueError(
                f"--ood-serial-ranks requires exactly one physical GPU; got {args.gpus}"
            )
        if not args.ood_serial_ranks and len(args.gpus) != expected_world_size:
            raise ValueError(
                f"{OOD_DIRECT_PROTOCOL} requires exactly {expected_world_size} unique "
                f"GPUs; got {args.gpus}"
            )
        protocol_seed = int(registry["protocols"][OOD_DIRECT_PROTOCOL]["seed"])
        if args.eval_seed != protocol_seed:
            raise ValueError(
                f"{OOD_DIRECT_PROTOCOL} is pinned to seed {protocol_seed}, got "
                f"--eval-seed={args.eval_seed}"
            )
    verified_ood_inputs = (
        validate_ood_protocol_inputs(registry, ood_validation_datasets)
        if ood_validation_datasets
        else {}
    )
    if "ood_direct_4s" in args.tasks:
        if not args.dry_run:
            for model_name in args.models:
                validate_registered_checkpoint(model_name, registry["models"][model_name])
        write_ood_run_manifest(
            registry,
            args.models,
            args.ood_datasets,
            {name: verified_ood_inputs[name] for name in args.ood_datasets},
            {
                "mode": (
                    "serialized_logical_ranks"
                    if args.ood_serial_ranks
                    else "native_distributed"
                ),
                "physical_gpus": args.gpus,
                "physical_world_size": len(args.gpus),
                "logical_world_size": registry["protocols"][OOD_DIRECT_PROTOCOL][
                    "execution"
                ]["distributed_world_size"],
                "sample_partition": "indices[logical_rank::logical_world_size]",
                "logical_rank_seed": "protocol_seed * logical_world_size + logical_rank",
            },
            args.dry_run,
        )
    raenwm_models = [
        model_name
        for model_name in args.models
        if registry["models"][model_name].get("backend") == "raenwm"
    ]
    unsupported_raenwm_tasks = sorted(set(args.tasks) & {"unseen_rollout"})
    if raenwm_models and unsupported_raenwm_tasks:
        raise ValueError(
            "The legacy --tasks unseen_rollout wrapper only knows the local NWM "
            "artifact layout. RAE-NWM itself supports rollout; use "
            "--metrics rollout --datasets go_stanford instead."
        )
    if (
        raenwm_models
        and "navigation" in args.tasks
        and args.raenwm_planning_steps is None
    ):
        raise ValueError(
            "Choose --raenwm-planning-steps 50 (official Euler setting) or "
            "--raenwm-planning-steps 250 (matched step-count budget)"
        )
    if raenwm_models and not args.dry_run and not args.raenwm_python.is_file():
        raise FileNotFoundError(
            f"RAE-NWM Python executable does not exist: {args.raenwm_python}. "
            "See docs/raenwm_benchmark.md."
        )

    # Finish every prediction/generalization protocol for one model before
    # advancing to the next model.  Navigation remains a final, separate phase.
    rollout_gpus = args.gpus
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
                evaluations=("time",)
                if model.get("backend") == "raenwm" or args.recon_direct_only
                else None,
                raenwm_python=args.raenwm_python,
                serialize_logical_ranks=args.direct_serial_ranks,
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
                raenwm_python=args.raenwm_python,
                serialize_logical_ranks=args.direct_serial_ranks,
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
        if "ood_direct_4s" in args.tasks:
            progress(
                f"START model={model_name} task=ood_direct_4s "
                f"datasets={','.join(args.ood_datasets)}"
            )
            run_ood_direct_prediction(
                model_name,
                model,
                args.ood_datasets,
                args.gpus,
                env,
                args.registry,
                registry,
                args.force,
                args.dry_run,
                raenwm_python=args.raenwm_python,
                serialize_logical_ranks=args.ood_serial_ranks,
            )
            progress(
                f"DONE model={model_name} task=ood_direct_4s "
                f"datasets={','.join(args.ood_datasets)}"
            )

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
                raenwm_python=args.raenwm_python,
                raenwm_num_steps=args.raenwm_planning_steps,
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
