#!/usr/bin/env python3
"""Maintain one extensible JSON registry for NWM benchmark results."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/benchmark_results.json"
)

MODELS = {
    "nwm-base": {
        "architecture": "CDiT-B/2 + SD-VAE",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260820_164519_compact_nwm_cdit_b_sdvae_bs16",
        "checkpoint_id": "0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260820_164519_compact_nwm_cdit_b_sdvae_bs16/checkpoints/0200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "76b2de9a4e6efba008583057eea6a81ae90429bea988f45cbc8f5bcab10f120b",
        "training_datasets": ["recon", "huron_public_sacson_key", "scand"],
    },
    "nwm-real": {
        "architecture": "CDiT-B/2 + SD-VAE + real-motion adapter",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260823_160459_nwm_real_recon_scand_tartan_huron_bs16",
        "checkpoint_id": "0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260823_160459_nwm_real_recon_scand_tartan_huron_bs16/checkpoints/0200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "757a511cb5efc44bca53cbd0d5d19add019c95fd229cf723538de228cbe1b846",
        "training_datasets": [
            "recon",
            "huron_public_sacson_key",
            "scand",
            "tartan_drive",
        ],
    },
    "nwm-release": {
        "architecture": "CDiT-B/2 + SD-VAE",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/nwm-release",
        "checkpoint_id": "0100000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/weights/cdit_b_100000.pth.tar",
        "checkpoint_step": 100000,
        "sha256": "2a41c71eabd20946f61bb5d1d2490264246bff59b9ffca177eee672e855d5261",
        "training_datasets": ["recon", "huron", "scand", "tartan_drive"],
        "provenance": {
            "repository": "facebook/nwm",
            "revision": "0821a1a7b1ae938539e32f13f5ad82465e0f3fde",
            "checkpoint_train_steps_field": 100000,
        },
    },
    "nwm-latent": {
        "architecture": "CDiT-B/2 + SD-VAE + real/latent-motion adapters",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260824_163328_nwm-latent-badlam_bs16",
        "checkpoint_id": "0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260824_163328_nwm-latent-badlam_bs16/checkpoints/0200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "744ca3e025907977a42994709102234a4acad4faf63a59bdfa49e9922ee17664",
        "training_datasets": [
            "recon",
            "huron_public_sacson_key",
            "scand",
            "tartan_drive",
        ],
    },
}

PROTOCOLS = {
    "recon_prediction_v1": {
        "category": "recon_prediction",
        "dataset": "recon",
        "sample_counts": {"time": 500, "rollout_1fps": 150, "rollout_4fps": 150},
        "horizons_seconds": [1, 2, 4, 8, 16],
        "diffusion_steps": 250,
        "seed": 0,
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "go_stanford_unseen_v1": {
        "category": "unseen_generalization",
        "dataset": "go_stanford",
        "evaluation": "time",
        "sample_count": 500,
        "horizons_seconds": [1, 2, 4, 8, 16],
        "paper_comparison_horizon_seconds": 4,
        "diffusion_steps": 250,
        "seed": 0,
        "paper_comparison_caveat": "NWM paper reports five-sample mean for CDiT-XL; local models are CDiT-B single-seed checkpoints",
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "navigation_cem80_v1": {
        "category": "navigation_planning",
        "datasets": ["recon", "scand"],
        "sample_count": 100,
        "population": 80,
        "topk": 5,
        "rollout_stride": 1,
        "repetitions": 3,
        "optimization_steps": 1,
        "horizon_steps": 8,
        "seconds_per_step": 0.25,
        "cost": "lpips_alex_on_vae_reconstruction",
        "seed": 42,
        "metrics": {"ate": "lower", "rpe_trans": "lower"},
        "comparability": {
            "compact_paper": "exact CEM population/protocol",
            "nwm_paper": "NWM reports population 120; compare with protocol caveat",
        },
    },
    "navigation_cem80_fast10_v1": {
        "category": "navigation_planning_fast",
        "datasets": ["recon", "scand"],
        "sample_count": 100,
        "population": 80,
        "topk": 5,
        "rollout_stride": 1,
        "repetitions": 3,
        "optimization_steps": 1,
        "horizon_steps": 8,
        "seconds_per_step": 0.25,
        "diffusion_steps": 10,
        "cost": "lpips_alex_on_vae_reconstruction",
        "seed": 42,
        "metrics": {"ate": "lower", "rpe_trans": "lower"},
        "comparability": {
            "paper": "accelerated local protocol; diffusion step count differs from the 250-step local exact run"
        },
    },
}

PAPER_BASELINES = {
    "nwm_paper": {
        "title": "Navigation World Models",
        "source": "https://arxiv.org/html/2412.03572",
        "reported_model": "CDiT-XL (1B parameters)",
        "results": {
            "navigation_planning": {
                "protocol": "CEM population 120, H=8, M=3, I=1",
                "recon": {"ate": 1.13, "ate_std": 0.02, "rpe_trans": 0.35, "rpe_trans_std": 0.01},
                "scand": {"ate": 1.28, "ate_std": 0.02, "rpe_trans": 0.33, "rpe_trans_std": 0.01},
            },
            "unseen_generalization": {
                "dataset": "go_stanford",
                "horizon_seconds": 4,
                "in_domain_data": {
                    "lpips_alex": 0.658,
                    "lpips_std": 0.002,
                    "dreamsim": 0.478,
                    "dreamsim_std": 0.001,
                    "psnr": 11.031,
                    "psnr_std": 0.036,
                },
                "plus_ego4d": {
                    "lpips_alex": 0.652,
                    "lpips_std": 0.003,
                    "dreamsim": 0.464,
                    "dreamsim_std": 0.003,
                    "psnr": 11.083,
                    "psnr_std": 0.064,
                },
            },
        },
    },
    "compact_paper": {
        "title": "Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model",
        "source": "https://arxiv.org/html/2603.05438",
        "results": {
            "navigation_planning": {
                "protocol": "CEM population 80, H=8, M=3, I=1",
                "sd_vae": {
                    "recon": {"ate": 1.262, "rpe_trans": 0.354},
                    "scand": {"ate": 1.065, "rpe_trans": 0.291},
                },
                "compact_16_tokens": {
                    "recon": {"ate": 1.330, "rpe_trans": 0.390},
                    "scand": {"ate": 1.358, "rpe_trans": 0.336},
                },
                "compact_8_tokens": {
                    "recon": {"ate": 1.373, "rpe_trans": 0.401},
                    "scand": {"ate": 1.391, "rpe_trans": 0.346},
                },
            }
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def new_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": utc_now(),
        "protocols": PROTOCOLS,
        "paper_baselines": PAPER_BASELINES,
        "models": {name: {**metadata, "results": {}} for name, metadata in MODELS.items()},
    }


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_registry()
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError(f"Unsupported registry schema: {registry.get('schema_version')}")
    registry["protocols"] = PROTOCOLS
    registry["paper_baselines"] = PAPER_BASELINES
    for name, metadata in MODELS.items():
        previous = registry.setdefault("models", {}).get(name, {})
        results = previous.get("results", {})
        registry["models"][name] = {**metadata, "results": results}
    return registry


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    registry["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_model(registry: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return registry["models"][name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}; register it first") from exc


def register_model(
    registry: dict[str, Any],
    name: str,
    exp_dir: Path,
    checkpoint_id: str,
    checkpoint: Path,
    checkpoint_step: int | None,
    sha256: str | None,
    training_datasets: list[str],
) -> None:
    previous = registry.setdefault("models", {}).get(name, {})
    registry["models"][name] = {
        "exp_dir": str(exp_dir.resolve()),
        "checkpoint_id": checkpoint_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "sha256": sha256,
        "training_datasets": training_datasets,
        "results": previous.get("results", {}),
    }


def import_prediction(
    registry: dict[str, Any], model_name: str, dataset: str, evaluation: str, audit: Path
) -> None:
    payload = json.loads(audit.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("eval_name") != evaluation:
        raise ValueError(
            f"Audit identity mismatch: expected {dataset}/{evaluation}, got "
            f"{payload.get('dataset')}/{payload.get('eval_name')}"
        )
    model = require_model(registry, model_name)
    category = "recon_prediction" if dataset == "recon" else "unseen_generalization"
    protocol = "recon_prediction_v1" if dataset == "recon" else "go_stanford_unseen_v1"
    model["results"].setdefault(category, {}).setdefault(dataset, {})[evaluation] = {
        "protocol": protocol,
        "source_audit": str(audit.resolve()),
        "sample_count": payload["sample_count"],
        "frame_indices": payload["frame_indices"],
        "metrics": payload["metrics"],
        "imported_at": utc_now(),
    }


def import_planning(
    registry: dict[str, Any],
    model_name: str,
    dataset: str,
    metrics_path: Path,
    protocol: str = "navigation_cem80_v1",
) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    prefix = f"{dataset}_"
    required = ("ate", "rpe_trans", "pos_diff_norm", "yaw_diff_norm")
    missing = [key for key in required if prefix + key not in payload]
    if missing:
        raise ValueError(f"Planning result is missing {missing}: {metrics_path}")
    if protocol not in registry["protocols"]:
        raise ValueError(f"Unknown planning protocol: {protocol}")
    category = registry["protocols"][protocol]["category"]
    model = require_model(registry, model_name)
    model["results"].setdefault(category, {})[dataset] = {
        "protocol": protocol,
        "source_metrics": str(metrics_path.resolve()),
        "sample_count": registry["protocols"][protocol]["sample_count"],
        "metrics": {key: payload[prefix + key] for key in required},
        "total_time_seconds": payload.get("total_time"),
        "imported_at": utc_now(),
    }


def sync_existing(registry: dict[str, Any]) -> None:
    comparison = Path(
        "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_comparison_20260825"
    )
    release = Path(
        "/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/nwm_cdit_b"
    )
    roots = {
        "nwm-base": comparison / "nwm_base_200k",
        "nwm-real": comparison / "nwm_real_200k",
        "nwm-release": release,
    }
    for model_name, root in roots.items():
        for evaluation in ("time", "rollout_1fps", "rollout_4fps"):
            audit = root / f"recon_{evaluation}_audit.json"
            if audit.exists():
                import_prediction(registry, model_name, "recon", evaluation, audit)
    release_planning = Path(
        "/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/"
        "planning_compact80/nwm_cdit_b/recon_CEM_N80_K5_RS1_rep3_OPT1.json"
    )
    if release_planning.exists():
        import_planning(registry, "nwm-release", "recon", release_planning)


def metric_mean(metrics: dict[str, Any], key: str) -> float:
    return sum(float(item[key]) for item in metrics.values()) / len(metrics)


def render_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# NWM benchmark registry",
        "",
        f"Updated: `{registry['updated_at']}`",
        "",
        "Lower is better for LPIPS, DreamSim, ATE and RPE; higher is better for PSNR.",
        "",
        "## RECON prediction (mean over 1/2/4/8/16 seconds)",
        "",
        "| Model | Mode | LPIPS | DreamSim | PSNR |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name, model in registry["models"].items():
        evaluations = model.get("results", {}).get("recon_prediction", {}).get("recon", {})
        for evaluation in ("time", "rollout_1fps", "rollout_4fps"):
            if evaluation not in evaluations:
                continue
            metrics = evaluations[evaluation]["metrics"]
            lines.append(
                f"| {model_name} | {evaluation} | {metric_mean(metrics, 'lpips_alex'):.6f} "
                f"| {metric_mean(metrics, 'dreamsim'):.6f} | {metric_mean(metrics, 'psnr'):.6f} |"
            )

    lines.extend(
        [
            "",
            "### RECON prediction details",
            "",
            "| Model | Mode | Horizon | LPIPS | DreamSim | PSNR | Samples |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        evaluations = model.get("results", {}).get("recon_prediction", {}).get("recon", {})
        for evaluation in ("time", "rollout_1fps", "rollout_4fps"):
            if evaluation not in evaluations:
                continue
            result = evaluations[evaluation]
            for horizon in ("1s", "2s", "4s", "8s", "16s"):
                metrics = result["metrics"][horizon]
                lines.append(
                    f"| {model_name} | {evaluation} | {horizon} | "
                    f"{metrics['lpips_alex']:.6f} | {metrics['dreamsim']:.6f} | "
                    f"{metrics['psnr']:.6f} | {metrics['sample_count']} |"
                )

    lines.extend(
        [
            "",
            "## Navigation planning (CEM-80)",
            "",
            "| Model/baseline | Dataset | ATE | RPE | Protocol note |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model_name, model in registry["models"].items():
        results = model.get("results", {}).get("navigation_planning", {})
        for dataset in ("recon", "scand"):
            if dataset in results:
                metrics = results[dataset]["metrics"]
                lines.append(
                    f"| {model_name} | {dataset} | {metrics['ate']:.6f} | "
                    f"{metrics['rpe_trans']:.6f} | measured, N=80 |"
                )
    compact = registry["paper_baselines"]["compact_paper"]["results"]["navigation_planning"]
    for baseline in ("sd_vae", "compact_16_tokens", "compact_8_tokens"):
        for dataset, metrics in compact[baseline].items():
            lines.append(
                f"| CompACT paper: {baseline} | {dataset} | {metrics['ate']:.3f} | "
                f"{metrics['rpe_trans']:.3f} | paper, N=80 |"
            )
    nwm = registry["paper_baselines"]["nwm_paper"]["results"]["navigation_planning"]
    for dataset in ("recon", "scand"):
        metrics = nwm[dataset]
        lines.append(
            f"| NWM paper | {dataset} | {metrics['ate']:.3f} | "
            f"{metrics['rpe_trans']:.3f} | paper, N=120 |"
        )

    lines.extend(
        [
            "",
            "### Navigation planning fast-10step",
            "",
            "CEM settings remain N=80, K=5, M=3 and H=8; diffusion sampling is reduced from 250 to 10 steps, so these rows are not paper-protocol equivalents.",
            "",
            "| Model | Dataset | ATE | RPE | Protocol |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model_name, model in registry["models"].items():
        results = model.get("results", {}).get("navigation_planning_fast", {})
        for dataset in ("recon", "scand"):
            if dataset in results:
                metrics = results[dataset]["metrics"]
                lines.append(
                    f"| {model_name} | {dataset} | {metrics['ate']:.6f} | "
                    f"{metrics['rpe_trans']:.6f} | fast-10step |"
                )

    lines.extend(
        [
            "",
            "## Go Stanford unseen (all measured horizons)",
            "",
            "| Model | Horizon | LPIPS | DreamSim | PSNR | Samples |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        result = (
            model.get("results", {})
            .get("unseen_generalization", {})
            .get("go_stanford", {})
            .get("time")
        )
        if result:
            for horizon in ("1s", "2s", "4s", "8s", "16s"):
                metrics = result["metrics"][horizon]
                lines.append(
                    f"| {model_name} | {horizon} | {metrics['lpips_alex']:.6f} | "
                    f"{metrics['dreamsim']:.6f} | {metrics['psnr']:.6f} | "
                    f"{metrics['sample_count']} |"
                )

    lines.extend(
        [
            "",
            "### Go Stanford paper comparison at 4 seconds",
            "",
            "| Model/baseline | LPIPS | DreamSim | PSNR |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        result = (
            model.get("results", {})
            .get("unseen_generalization", {})
            .get("go_stanford", {})
            .get("time")
        )
        if result:
            metrics = result["metrics"]["4s"]
            lines.append(
                f"| {model_name} | {metrics['lpips_alex']:.6f} | "
                f"{metrics['dreamsim']:.6f} | {metrics['psnr']:.6f} |"
            )
    unseen = registry["paper_baselines"]["nwm_paper"]["results"]["unseen_generalization"]
    for name in ("in_domain_data", "plus_ego4d"):
        metrics = unseen[name]
        lines.append(
            f"| NWM paper: {name} | {metrics['lpips_alex']:.3f} | "
            f"{metrics['dreamsim']:.3f} | {metrics['psnr']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("initialize")
    subparsers.add_parser("sync-existing")

    register = subparsers.add_parser("register-model")
    register.add_argument("--model", required=True)
    register.add_argument("--exp-dir", type=Path, required=True)
    register.add_argument("--checkpoint-id", required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--checkpoint-step", type=int)
    register.add_argument("--sha256")
    register.add_argument("--training-datasets", type=csv_list, default=[])

    prediction = subparsers.add_parser("import-prediction")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--dataset", required=True)
    prediction.add_argument("--evaluation", required=True)
    prediction.add_argument("--audit", type=Path, required=True)

    planning = subparsers.add_parser("import-planning")
    planning.add_argument("--model", required=True)
    planning.add_argument("--dataset", required=True)
    planning.add_argument("--metrics", type=Path, required=True)
    planning.add_argument("--protocol", default="navigation_cem80_v1")

    render = subparsers.add_parser("render")
    render.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.registry.with_suffix(args.registry.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        registry = load_registry(args.registry)
        if args.command == "sync-existing":
            sync_existing(registry)
        elif args.command == "register-model":
            register_model(
                registry,
                args.model,
                args.exp_dir,
                args.checkpoint_id,
                args.checkpoint,
                args.checkpoint_step,
                args.sha256,
                args.training_datasets,
            )
        elif args.command == "import-prediction":
            import_prediction(registry, args.model, args.dataset, args.evaluation, args.audit)
        elif args.command == "import-planning":
            import_planning(
                registry,
                args.model,
                args.dataset,
                args.metrics,
                args.protocol,
            )
        elif args.command == "render":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(render_markdown(registry), encoding="utf-8")
            print(args.output)
            return
        save_registry(args.registry, registry)
        if args.command in {"import-prediction", "import-planning"}:
            rendered = render_markdown(registry)
            for filename in ("benchmark_results.md", "benchmark_comparison.md"):
                output = args.registry.with_name(filename)
                output.write_text(rendered, encoding="utf-8")
        print(args.registry)


if __name__ == "__main__":
    main()
