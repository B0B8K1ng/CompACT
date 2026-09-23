#!/usr/bin/env python3
"""Rank checkpoints with equal ID/OOD and dataset/metric weighting."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import mean


BASE = Path("/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark")
OUT = BASE / "finalLAM_reset_checkpoint_sweep_20260922"
REGISTRY = OUT / "benchmark_results.json"
PREFIX = "finalLAM-reset-joint"
ID_DATASETS = ("recon", "scand", "huron", "tartan_drive")
OOD_DATASETS = ("go_stanford", "unitree_go2", "tum_rgbd", "uzh_fpv")
METRICS = {"lpips_alex": "lower", "dreamsim": "lower", "psnr": "higher"}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def average_ranks(values: dict[int, float], direction: str) -> dict[int, float]:
    ordered = sorted(values, key=values.get, reverse=direction == "higher")
    result: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            values[ordered[index]], values[ordered[end]], rel_tol=0, abs_tol=1e-12
        ):
            end += 1
        rank = mean(range(index, end))
        for step in ordered[index:end]:
            result[step] = rank / (len(ordered) - 1)
        index = end
    return result


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    models = {
        int(name.removeprefix(PREFIX)): entry
        for name, entry in registry["models"].items()
        if name.startswith(PREFIX)
    }
    expected = set(range(10_000, 100_001, 10_000))
    if set(models) != expected:
        raise RuntimeError(f"Checkpoint set mismatch: {sorted(models)}")
    raw: dict[int, dict[str, dict[str, float]]] = {}
    for step, entry in models.items():
        raw[step] = {}
        direct = entry.get("results", {}).get("direct_prediction", {})
        for dataset in (*ID_DATASETS, *OOD_DATASETS):
            try:
                metrics = direct[dataset]["time"]["metrics"]["4s"]
            except KeyError as error:
                raise RuntimeError(f"Missing direct_4s result for {step}/{dataset}") from error
            raw[step][dataset] = {metric: float(metrics[metric]) for metric in METRICS}

    cell_ranks: dict[tuple[str, str], dict[int, float]] = {}
    cell_regrets: dict[tuple[str, str], dict[int, float]] = {}
    for dataset in (*ID_DATASETS, *OOD_DATASETS):
        for metric, direction in METRICS.items():
            values = {step: raw[step][dataset][metric] for step in models}
            cell_ranks[(dataset, metric)] = average_ranks(values, direction)
            low, high = min(values.values()), max(values.values())
            span = high - low
            if not span:
                cell_regrets[(dataset, metric)] = {step: 0.0 for step in models}
            elif direction == "lower":
                cell_regrets[(dataset, metric)] = {
                    step: (value - low) / span for step, value in values.items()
                }
            else:
                cell_regrets[(dataset, metric)] = {
                    step: (high - value) / span for step, value in values.items()
                }

    rows = []
    for step, entry in sorted(models.items()):
        group_values = {}
        for group, datasets in (("id", ID_DATASETS), ("ood", OOD_DATASETS)):
            ranks = [
                cell_ranks[(dataset, metric)][step]
                for dataset in datasets
                for metric in METRICS
            ]
            regrets = [
                cell_regrets[(dataset, metric)][step]
                for dataset in datasets
                for metric in METRICS
            ]
            group_values[f"{group}_rank"] = mean(ranks)
            group_values[f"{group}_regret"] = mean(regrets)
            for metric in METRICS:
                group_values[f"{group}_{metric}"] = mean(
                    raw[step][dataset][metric] for dataset in datasets
                )
        row = {
            "joint_steps": step,
            "total_steps": step + 3_000,
            "checkpoint": entry["checkpoint"],
            **group_values,
        }
        row["balanced_rank"] = mean((row["id_rank"], row["ood_rank"]))
        row["worst_group_rank"] = max(row["id_rank"], row["ood_rank"])
        row["balanced_regret"] = mean((row["id_regret"], row["ood_regret"]))
        row["worst_group_regret"] = max(row["id_regret"], row["ood_regret"])
        rows.append(row)

    for row in rows:
        row["pareto"] = not any(
            other["id_rank"] <= row["id_rank"]
            and other["ood_rank"] <= row["ood_rank"]
            and (
                other["id_rank"] < row["id_rank"]
                or other["ood_rank"] < row["ood_rank"]
            )
            for other in rows
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            row["balanced_rank"],
            row["worst_group_rank"],
            row["balanced_regret"],
        ),
    )
    for position, row in enumerate(ordered, start=1):
        row["selection_rank"] = position
    best = ordered[0]
    best["recommended"] = True
    for row in ordered[1:]:
        row["recommended"] = False

    fields = [
        "selection_rank",
        "recommended",
        "pareto",
        "joint_steps",
        "total_steps",
        "balanced_rank",
        "worst_group_rank",
        "id_rank",
        "ood_rank",
        "balanced_regret",
        "worst_group_regret",
        "id_regret",
        "ood_regret",
        "id_lpips_alex",
        "id_dreamsim",
        "id_psnr",
        "ood_lpips_alex",
        "ood_dreamsim",
        "ood_psnr",
        "checkpoint",
    ]
    with (OUT / "selection.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    payload = {
        "recommendation": best,
        "selection_rule": {
            "primary": "minimum equal-ID/OOD mean normalized rank",
            "tie_breakers": ["worst group rank", "balanced min-max regret"],
            "dataset_weighting": "equal within ID and OOD",
            "metric_weighting": "equal LPIPS, DreamSim, PSNR",
            "directions": METRICS,
        },
        "id_datasets": ID_DATASETS,
        "ood_datasets": OOD_DATASETS,
        "rows": ordered,
        "raw": raw,
    }
    atomic_json(OUT / "selection.json", payload)
    lines = [
        "# finalLAM reset checkpoint selection",
        "",
        f"Recommended: `joint_{best['joint_steps']:07d}.pth.tar` (EMA).",
        "",
        "All checkpoints use direct_4s_v1: 500 fixed windows per dataset, seed 0, "
        "250-step DDPM. Four ID and four OOD datasets receive equal group weight; "
        "LPIPS/DreamSim are minimized and PSNR is maximized.",
        "",
        "| Rank | Joint step | ID rank ↓ | OOD rank ↓ | Balanced ↓ | Worst group ↓ | Pareto |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['selection_rank']} | {row['joint_steps']:,} | "
            f"{row['id_rank']:.4f} | {row['ood_rank']:.4f} | "
            f"{row['balanced_rank']:.4f} | {row['worst_group_rank']:.4f} | "
            f"{'yes' if row['pareto'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Recommended checkpoint group means",
            "",
            "| Group | LPIPS ↓ | DreamSim ↓ | PSNR ↑ |",
            "|---|---:|---:|---:|",
            f"| ID | {best['id_lpips_alex']:.6f} | {best['id_dreamsim']:.6f} | {best['id_psnr']:.6f} |",
            f"| OOD | {best['ood_lpips_alex']:.6f} | {best['ood_dreamsim']:.6f} | {best['ood_psnr']:.6f} |",
            "",
            f"Checkpoint: `{best['checkpoint']}`",
            "",
        ]
    )
    (OUT / "selection.md").write_text("\n".join(lines))
    print(json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    main()
