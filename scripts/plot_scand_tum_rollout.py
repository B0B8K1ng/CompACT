"""Export the selected ID/OOD rollout results in a one-row, four-panel figure."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir
    models = [("nwm-release", "NWM", "#377eb8"),
              ("rae-nwm", "RAE-NWM", "#ee9135"),
              ("opennwm-finalLAM-100k", "OpenNWM", "#bc263b")]
    settings = [("scand", "SCAND (ID)", 1), ("tum_rgbd", "TUM RGB-D (OOD)", 4)]
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(left=.055, right=.99, bottom=.18, top=.73, wspace=.36)
    rows = []
    for group, (dataset, title, fps) in enumerate(settings):
        points = [json.loads((root / "points" / f"{dataset}_{fps}fps_{h}s.json").read_text())
                  for h in [1, 2, 4]]
        for p in points:
            for key, name, _ in models:
                values = p["models"][key]
                rows.append(dict(dataset=dataset, domain=p["domain"], fps=fps,
                                 horizon_s=p["horizon_s"], model=name,
                                 lpips=values["lpips_alex"], fid=values["fid"],
                                 sample_count=values["sample_count"]))
        for column, (metric, label) in enumerate([("lpips_alex", "LPIPS ↓"), ("fid", "FID ↓")]):
            ax = axes[group * 2 + column]
            for key, name, color in models:
                ax.plot([1, 2, 4], [p["models"][key][metric] for p in points],
                        "o-", label=name, color=color, linewidth=2.2, markersize=5)
            ax.set(xlabel="Rollout time (s)", ylabel=label, xticks=[1, 2, 4])
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        left = axes[group * 2].get_position().x0
        right = axes[group * 2 + 1].get_position().x1
        fig.text((left + right) / 2, .80, title, ha="center", va="center", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01),
               ncol=3, frameon=False, fontsize=13)
    stem = root / "scand_tum_rollout_1x4"
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix("." + extension), dpi=240, bbox_inches="tight")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {stem}.{{png,pdf,svg,csv}}; {len(rows)} model/timepoint rows")


if __name__ == "__main__":
    main()
