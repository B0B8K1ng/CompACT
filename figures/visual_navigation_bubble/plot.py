"""Plot the supplied 2026-09-26 table; bubble area encodes parameters.

Run: conda run -n nwm python figures/visual_navigation_bubble/plot.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("visual_navigation_bubble"))
    args = parser.parse_args()
    with args.data.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = ["NWM", "RAE-NWM", "OpenNWM"]
    if len(rows) != 3 or set(row["method"] for row in rows) != set(methods):
        parser.error("Expected exactly one row each for NWM, RAE-NWM, and OpenNWM.")
    rows.sort(key=lambda row: methods.index(row["method"]))
    columns = ["parameters_m", "id_mean_psnr", "recon_ate", "scand_ate", "office_go2_ate"]
    missing = [f"{row['method']}: {key}" for row in rows for key in columns if not row.get(key, "").strip()]
    if missing:
        parser.error("Missing source-table values: " + "; ".join(missing))
    values = np.array([[float(row[key]) for key in columns] for row in rows])
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any() or (values[:, 1:] < 0).any():
        parser.error("Values must be finite, with positive parameter counts and nonnegative errors.")
    parameters, psnr = values[:, 0], values[:, 1]
    ate = values[:, 2:].mean(axis=1)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "mathtext.fontset": "dejavusans", "axes.labelsize": 17,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "text.color": "#25313B", "axes.labelcolor": "#25313B",
        "xtick.color": "#49545F", "ytick.color": "#49545F",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0,
    })
    fig, ax = plt.subplots(figsize=(7.0, 4.65))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.19, top=0.96)
    colors = ["#587C9C", "#B59A63", "#B84B4D"]
    area_scale = 3.6
    offsets = [(0, -31), (0, 39), (0, 29)]
    for index, method in enumerate(methods):
        color = colors[index]
        ax.scatter(psnr[index], ate[index], s=parameters[index] * area_scale,
                   facecolors=[to_rgba(color, 0.30 if index == 2 else 0.20)], edgecolors=color,
                   linewidths=1.6, zorder=3)
        ax.scatter(psnr[index], ate[index], s=32, facecolor=color, edgecolor="white",
                   linewidth=0.7, zorder=4)
        ox, oy = offsets[index]
        ax.annotate(f"{method} ({parameters[index]:.0f}M)", (psnr[index], ate[index]),
                    xytext=(ox, oy), textcoords="offset points",
                    ha="center", va="bottom" if oy > 0 else "top", fontsize=15,
                    color=color, fontweight="bold" if method == "OpenNWM" else "normal", zorder=4)
    # Standard linear axes: the lower right means higher PSNR and lower ATE.
    ax.set_xlim(13.65, 14.48)
    ax.set_ylim(1.60, 1.925)
    ax.set_xticks([13.7, 13.9, 14.1, 14.3])
    ax.set_yticks([1.65, 1.70, 1.75, 1.80, 1.85, 1.90])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(r"Visual prediction — PSNR (dB) $\uparrow$", labelpad=11)
    ax.set_ylabel(r"Navigation — ATE $\downarrow$", labelpad=11)
    ax.tick_params(axis="both", length=0, width=0.9, pad=8)
    for spine in ax.spines.values():
        spine.set_color("#AAB1B7")
    ax.grid(color="#D8DDE2", linewidth=0.65, linestyle=(0, (3, 4)), alpha=0.8)
    ax.set_axisbelow(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        destination = args.output.with_suffix("." + suffix)
        fig.savefig(destination, dpi=450, facecolor="white")
        print(destination)
    plt.close(fig)
    for index, method in enumerate(methods):
        print(f"{method}: PSNR={psnr[index]:.3f}, mean ATE={ate[index]:.9f}, parameters={parameters[index]:.0f}M")


if __name__ == "__main__":
    main()
