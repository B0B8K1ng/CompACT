#!/usr/bin/env python3
"""Plot Figure-4-style RECON rollout metrics from the NWM benchmark registry.

Only registered rollout results are plotted.  In particular, direct-time
predictions are never substituted for missing autoregressive rollout results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DEFAULT_REGISTRY = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/benchmark_results.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "nwm_rollout_comparison"
HORIZONS = (1, 2, 4, 8, 16)
FPS_VALUES = (1, 4)
MODELS = OrderedDict(
    (
        (
            "rae-nwm",
            {
                "label": "RAE-NWM",
                "color": "#4C78A8",
                "marker": "s",
            },
        ),
        (
            "nwm-real",
            {
                "label": "NWM",
                "color": "#54A24B",
                "marker": "o",
            },
        ),
        (
            "nwm-latentpt-reset-nwm-real-recipe-180k",
            {
                "label": "OpenNWM (180k)",
                "color": "#F58518",
                "marker": "^",
            },
        ),
    )
)
METRICS = OrderedDict(
    (
        (
            "lpips_alex",
            {
                "label": "LPIPS",
                "direction": "lower is better",
                "filename": "lpips",
            },
        ),
        (
            "dreamsim",
            {
                "label": "DreamSim",
                "direction": "lower is better",
                "filename": "dreamsim",
            },
        ),
        (
            "psnr",
            {
                "label": "PSNR (dB)",
                "direction": "higher is better",
                "filename": "psnr",
            },
        ),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def rollout_result(
    registry: dict[str, Any], model_key: str, fps: int
) -> dict[str, Any] | None:
    return (
        registry.get("models", {})
        .get(model_key, {})
        .get("results", {})
        .get("recon_prediction", {})
        .get("recon", {})
        .get(f"rollout_{fps}fps")
    )


def metric_values(result: dict[str, Any], metric: str) -> list[float] | None:
    values: list[float] = []
    for horizon in HORIZONS:
        value = result.get("metrics", {}).get(f"{horizon}s", {}).get(metric)
        if value is None:
            return None
        values.append(float(value))
    return values


def padded_limits(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    padding = span * 0.12 if span else max(abs(low) * 0.08, 0.05)
    lower = low - padding
    if low >= 0:
        lower = max(0.0, lower)
    return lower, high + padding


def plot_metric(
    registry: dict[str, Any], output_dir: Path, fps: int, metric: str, dpi: int
) -> list[str]:
    metric_spec = METRICS[metric]
    plotted: list[str] = []
    all_values: list[float] = []

    figure, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for model_key, model_spec in MODELS.items():
        result = rollout_result(registry, model_key, fps)
        if result is None:
            continue
        values = metric_values(result, metric)
        if values is None:
            continue
        axis.plot(
            HORIZONS,
            values,
            label=model_spec["label"],
            color=model_spec["color"],
            marker=model_spec["marker"],
            linewidth=2.6,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        plotted.append(model_key)
        all_values.extend(values)

    if not plotted:
        plt.close(figure)
        return []

    axis.set_xscale("log", base=2)
    axis.set_xticks(HORIZONS, [f"t+{horizon}" for horizon in HORIZONS])
    axis.set_xlim(0.82, 19.0)
    axis.set_ylim(*padded_limits(all_values))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.set_xlabel("Seconds", fontsize=14)
    axis.set_ylabel(metric_spec["label"], fontsize=14)
    axis.set_title(f"RECON autoregressive rollout · {fps} FPS", fontsize=14, pad=10)
    axis.grid(axis="y", color="#C8C8C8", linewidth=1.0, alpha=0.85)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="both", labelsize=12)
    for spine in axis.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(1.0)
    axis.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.94,
        fontsize=11,
    )

    missing_labels = [
        spec["label"] for key, spec in MODELS.items() if key not in plotted
    ]
    if missing_labels:
        figure.text(
            0.995,
            0.006,
            "Not evaluated: " + ", ".join(missing_labels),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#777777",
        )

    stem = output_dir / f"rollout_{fps}fps_{metric_spec['filename']}"
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return plotted


def write_data_csv(registry: dict[str, Any], output_dir: Path) -> None:
    path = output_dir / "rollout_metrics.csv"
    fields = (
        "model",
        "model_key",
        "fps",
        "horizon_seconds",
        "metric",
        "value",
        "sample_count",
        "source_audit",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_key, model_spec in MODELS.items():
            for fps in FPS_VALUES:
                result = rollout_result(registry, model_key, fps)
                if result is None:
                    continue
                for horizon in HORIZONS:
                    horizon_metrics = result.get("metrics", {}).get(f"{horizon}s", {})
                    for metric in METRICS:
                        if metric not in horizon_metrics:
                            continue
                        writer.writerow(
                            {
                                "model": model_spec["label"],
                                "model_key": model_key,
                                "fps": fps,
                                "horizon_seconds": horizon,
                                "metric": metric,
                                "value": f"{float(horizon_metrics[metric]):.9g}",
                                "sample_count": horizon_metrics.get(
                                    "sample_count", result.get("sample_count", "")
                                ),
                                "source_audit": result.get("source_audit", ""),
                            }
                        )


def write_availability(
    registry: dict[str, Any], registry_path: Path, output_dir: Path
) -> None:
    availability: dict[str, Any] = {
        "registry": str(registry_path.resolve()),
        "registry_updated_at": registry.get("updated_at"),
        "dataset": "recon",
        "horizons_seconds": list(HORIZONS),
        "models": {},
        "metrics_plotted": list(METRICS),
        "metrics_not_available": ["fid"],
    }
    for model_key, model_spec in MODELS.items():
        model_record: dict[str, Any] = {
            "label": model_spec["label"],
            "checkpoint": registry.get("models", {}).get(model_key, {}).get("checkpoint"),
            "checkpoint_step": registry.get("models", {})
            .get(model_key, {})
            .get("checkpoint_step"),
            "sha256": registry.get("models", {}).get(model_key, {}).get("sha256"),
            "rollout": {},
        }
        for fps in FPS_VALUES:
            result = rollout_result(registry, model_key, fps)
            model_record["rollout"][f"{fps}fps"] = {
                "available": result is not None,
                "sample_count": result.get("sample_count") if result else None,
                "source_audit": result.get("source_audit") if result else None,
                "horizons": [
                    f"{horizon}s"
                    for horizon in HORIZONS
                    if f"{horizon}s" in result.get("metrics", {})
                ]
                if result
                else [],
            }
        availability["models"][model_key] = model_record

    (output_dir / "availability.json").write_text(
        json.dumps(availability, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    for fps in FPS_VALUES:
        for metric in METRICS:
            if plot_metric(registry, args.output_dir, fps, metric, args.dpi):
                generated.append(f"rollout_{fps}fps_{METRICS[metric]['filename']}")

    write_data_csv(registry, args.output_dir)
    write_availability(registry, args.registry, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "generated_plot_stems": generated,
                "registry_updated_at": registry.get("updated_at"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
