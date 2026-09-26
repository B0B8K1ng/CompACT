"""Reproduce descriptive scaling figures from the complete 30-row evaluation.

Run from the repository root:
  conda run --no-capture-output -n base python figures/data_scaling/plot.py
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent
DATASETS = ["recon", "scand", "tartan_drive", "huron", "go_stanford"]
NAMES = ["RECON", "SCAND", "TartanDrive", "HuRoN", "Go Stanford"]
METRICS = ["lpips_alex", "dreamsim", "psnr", "fid"]
METRIC_NAMES = ["LPIPS ↓", "DreamSim ↓", "PSNR (dB) ↑", "FID ↓"]
SETTINGS = {
    "U": ([25, 50, 75, 100], ["u25l100", "u50l100", "u75l100", "u100l100"]),
    "L": ([0, 50, 100], ["u25l0", "u25l50", "u25l100"]),
}
COLORS = {"U": "#2476A8", "L": "#CB7540"}


def load():
    with (ROOT / "source_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {}
    for r in rows:
        key = (r["checkpoint"], r["dataset"])
        assert key not in data
        assert int(r["sample_count"]) == (329 if key[1] == "huron" else 500)
        data[key] = {k: float(r[k]) for k in METRICS}
        assert all(np.isfinite(v) for v in data[key].values())
    labels = list(dict.fromkeys(SETTINGS["U"][1] + SETTINGS["L"][1]))
    assert set(data) == {(l, d) for l in labels for d in DATASETS}
    for l in labels:
        data[l, "macro"] = {k: np.mean([data[l, d][k] for d in DATASETS]).item() for k in METRICS}
    return data, labels


def values(data, axis, dataset, metric):
    return np.array([data[l, dataset][metric] for l in SETTINGS[axis][1]])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B7C0C9")
    ax.spines[["left", "bottom"]].set_linewidth(0.65)
    ax.tick_params(length=0, pad=6, labelcolor="#46515B")
    ax.grid(axis="y", color="#E6EAED", linewidth=0.65)
    ax.set_axisbelow(True)


def save(fig, stem):
    for suffix in ["pdf", "svg", "png"]:
        path = ROOT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=250, facecolor="white")
        print(path)
    plt.close(fig)


def main_figure(data):
    """Manuscript figure: vary U only, at fixed L, on RECON DreamSim."""
    x = SETTINGS["U"][0]
    y = values(data, "U", "recon", "dreamsim")
    color = COLORS["U"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    fig.subplots_adjust(left=.18, right=.965, bottom=.23, top=.965)
    ax.plot(x, y, color=color, lw=2, marker="o", ms=6.5,
            mec="white", mew=1.2, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.5f}", (xi, yi), xytext=(0, 16),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=color)
    span = float(np.ptp(y))
    ax.set_ylim(y.min() - .28 * span, y.max() + .36 * span)
    ax.set_xlim(15, 110)
    ax.set_xticks(x)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Unlabeled data (%)", labelpad=8)
    ax.set_ylabel("DreamSim ↓", labelpad=9)
    style(ax)
    save(fig, "data_scaling")


def two_sweep_figure(data, dataset="recon", metric="dreamsim"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95))
    fig.subplots_adjust(left=.098, right=.982, bottom=.205, top=.765, wspace=.41)
    name = "Five-dataset mean" if dataset == "macro" else "RECON"
    ylabel = METRIC_NAMES[METRICS.index(metric)]
    specs = [("U", dataset, metric, "(a) Unlabeled data", f"{name} · L = 100%", ylabel, "Unlabeled data, U (%)"),
             ("L", dataset, metric, "(b) Labeled data", f"{name} · U = 25%", ylabel, "Available action labels, L (%)")]
    # Same dataset and metric: use identical y limits for direct comparison.
    all_y = np.concatenate([values(data, axis, ds, metric)
                            for axis, ds, metric, *_ in specs])
    span = float(np.ptp(all_y))
    for ax, (axis, ds, metric, title, subtitle, ylabel, xlabel) in zip(axes, specs):
        x = SETTINGS[axis][0]
        y = values(data, axis, ds, metric)
        c = COLORS[axis]
        ax.plot(x, y, color=c, lw=1.9, marker="o", ms=5.8, mec="white", mew=1.1, zorder=3)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.4f}", (xi, yi),
                        xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color=c)
        ax.set_ylim(all_y.min() - .28 * span, all_y.max() + .42 * span)
        ax.set_xlim(min(x) - 11, max(x) + 11)
        ax.set_xticks(x)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=7)
        ax.text(0, 1.30, title, transform=ax.transAxes, fontsize=11, fontweight="bold", color="#26323C")
        ax.text(0, 1.15, subtitle, transform=ax.transAxes, fontsize=8.3, color="#66717A")
        change = 100 * (y[-1] / y[0] - 1)
        direction = "lower" if change < 0 else "higher"
        prefix = "Endpoint: " if dataset == "macro" else ""
        ax.text(.03, .10, f"{prefix}{abs(change):.2f}% {direction}", transform=ax.transAxes,
                fontsize=9, color=c, fontweight="bold")
        style(ax)
    stem = "data_scaling_mean" if dataset == "macro" else "data_scaling_recon_two_sweeps"
    if metric != "dreamsim":
        stem += "_" + metric.removesuffix("_alex")
    save(fig, stem)


def macro_figure(data):
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.0))
    fig.subplots_adjust(left=.075, right=.985, bottom=.11, top=.89, wspace=.48, hspace=.57)
    for row, axis in enumerate(SETTINGS):
        for col, (metric, name) in enumerate(zip(METRICS, METRIC_NAMES)):
            ax = axes[row, col]
            ax.plot(SETTINGS[axis][0], values(data, axis, "macro", metric),
                    color=COLORS[axis], marker="o", lw=1.6, ms=4.5)
            ax.set_xticks(SETTINGS[axis][0])
            ax.set_xlabel("U (%) · L = 100%" if axis == "U" else "L (%) · U = 25%")
            ax.set_ylabel(name)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            style(ax)
    fig.suptitle("All five datasets · equal-weight means", x=.075, ha="left", fontsize=12, fontweight="bold")
    save(fig, "all_macro_trends")


def three_metric_figure(data):
    """Compare both data sweeps using the same three macro metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.3), sharey="col")
    fig.subplots_adjust(left=.075, right=.975, bottom=.13, top=.82,
                        wspace=.35, hspace=.90)
    for col, (metric, name) in enumerate(zip(METRICS[:3], METRIC_NAMES[:3])):
        all_y = np.concatenate([values(data, axis, "macro", metric) for axis in SETTINGS])
        span = float(np.ptp(all_y))
        for row, axis in enumerate(SETTINGS):
            ax = axes[row, col]
            x = SETTINGS[axis][0]
            y = values(data, axis, "macro", metric)
            color = COLORS[axis]
            ax.plot(x, y, color=color, lw=1.8, marker="o", ms=6,
                    mec="white", mew=1.1, zorder=3)
            for i, (xi, yi) in enumerate(zip(x, y)):
                label = f"{yi:.3f}" if metric == "psnr" else f"{yi:.4f}"
                neighbors = np.delete(y[max(0, i - 1):i + 2], min(i, 1))
                below = yi < neighbors.mean()
                ax.annotate(label, (xi, yi), xytext=(0, -9 if below else 9),
                            textcoords="offset points", ha="center",
                            va="top" if below else "bottom",
                            fontsize=8, color=color)
            ax.set_ylim(all_y.min() - .40 * span, all_y.max() + .36 * span)
            ax.set_xlim(min(x) - 12, max(x) + 12)
            ax.set_xticks(x)
            ax.set_xlabel(f"{axis} (%)", labelpad=7)
            ax.set_title(name, fontsize=10.5, fontweight="bold", pad=13,
                         color="#26323C")
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if metric == "psnr" else "%.3f"))
            style(ax)
            ax.tick_params(labelleft=True)
    for row, axis in enumerate(SETTINGS):
        y = axes[row, 0].get_position().y1 + .095
        title = ("(a) Unlabeled data scaling  ·  L = 100%" if axis == "U"
                 else "(b) Action-label scaling  ·  U = 25%")
        fig.text(.075, y, title, color=COLORS[axis], fontsize=11,
                 fontweight="bold", ha="left")
    fig.text(.075, .025,
             "Direct 4 s prediction · Equal-weight means over RECON, SCAND, TartanDrive, HuRoN and Go Stanford",
             fontsize=8, color="#66717A", ha="left")
    save(fig, "data_scaling_mean_three_metrics")


def full_figure(data, axis):
    fig, axes = plt.subplots(6, 4, figsize=(10.6, 12.2))
    fig.subplots_adjust(left=.10, right=.98, bottom=.05, top=.94, wspace=.40, hspace=.48)
    for row, (ds, ds_name) in enumerate(zip(DATASETS + ["macro"], NAMES + ["Equal-weight mean"])):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.plot(SETTINGS[axis][0], values(data, axis, ds, metric), color=COLORS[axis], marker="o", ms=4, lw=1.4)
            ax.set_xticks(SETTINGS[axis][0])
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            if col == 0:
                ax.set_ylabel(ds_name, labelpad=8)
            if row == 0:
                ax.set_title(METRIC_NAMES[col], pad=14, fontweight="bold")
            if row == 5:
                ax.set_xlabel(f"{axis} (%)")
            style(ax)
    fixed = "L = 100%" if axis == "U" else "U = 25%"
    fig.suptitle(f"Complete {axis} sweep · {fixed} · direct 4 s prediction", x=.10, ha="left", fontsize=13, fontweight="bold")
    save(fig, f"all_{axis.lower()}_trends")


def audit(data, labels):
    trends = []
    for axis in SETTINGS:
        for ds in DATASETS + ["macro"]:
            for metric in METRICS:
                y = values(data, axis, ds, metric)
                sign = 1 if metric == "psnr" else -1
                trends.append(dict(axis=axis, dataset=ds, metric=metric, checkpoints=SETTINGS[axis][1],
                                   values=y.tolist(), strict_monotonic_improvement=bool(np.all(sign * np.diff(y) > 0)),
                                   endpoint_change=float(y[-1] - y[0])))
    out = dict(source_sha256=hashlib.sha256((ROOT / "source_results.csv").read_bytes()).hexdigest(),
               aggregation="equal weight over five datasets; mean FID is not pooled FID",
               selection="Manuscript analyzes only U scaling at L=100% on RECON DreamSim. All four U settings are retained. Complete U/L results remain in the local audit assets, outside the focused manuscript section.",
               uncertainty="one checkpoint and one evaluation seed per setting; no repeated-training confidence intervals",
               trends=trends, macro_means={l: data[l, "macro"] for l in labels})
    (ROOT / "trend_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# Scaling 趋势核对", "", "6 个权重 × 5 个数据集；分析全部 4 项指标。以下是所有严格逐档改善的序列。", "",
             "| 消融 | 数据集 | 指标 | 数值（按数据量递增） |", "|---|---|---|---|"]
    for r in trends:
        if r["strict_monotonic_improvement"]:
            vs = " → ".join(f"{v:.5f}" for v in r["values"])
            lines.append(f"| {r['axis']} | {r['dataset']} | {r['metric']} | {vs} |")
    lines += ["", "U sweep 的四项五数据集平均指标均不单调；L sweep 只有平均 FID 单调。",
              "当前论文仅分析固定 L=100% 的 U 消融，主图使用 RECON DreamSim，保留全部四个 U 设置。完整 U/L 结果保留在本地审计文件中，未接入当前聚焦的论文章节。",
              "没有用平滑、拟合、删除中间点或重新取样来改变趋势。没有多训练种子，因此不声称统计显著性。",
              "", "## 五数据集等权平均", "", "| 权重 | LPIPS ↓ | DreamSim ↓ | PSNR ↑ | FID ↓ |", "|---|---:|---:|---:|---:|"]
    for l in labels:
        r = data[l, "macro"]
        lines.append(f"| {l} | {r['lpips_alex']:.5f} | {r['dreamsim']:.5f} | {r['psnr']:.3f} | {r['fid']:.3f} |")
    (ROOT / "trend_audit.md").write_text("\n".join(lines) + "\n")


def recon_u_table(data):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Unlabeled-data scaling on RECON at fixed $L=100\%$. DreamSim is averaged over the same 500 direct 4\,s predictions; lower is better.}",
             r"\label{tab:data_scaling_recon_u}", r"\begin{tabular}{rr}", r"\toprule",
             r"$U$ (\%) & DreamSim $\downarrow$ \\", r"\midrule"]
    for u, score in zip(SETTINGS["U"][0], values(data, "U", "recon", "dreamsim")):
        lines.append(f"{u} & {score:.5f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (ROOT / "recon_u_results.tex").write_text("\n".join(lines) + "\n")


def full_table(data, labels):
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Complete data-scaling results for direct 4\,s prediction. Six checkpoints are evaluated on identical windows within each dataset (500 each, except HuRoN: 329). Mean rows give equal-weight means across datasets; mean FID is not pooled FID. All entries are single-checkpoint, single-evaluation-seed estimates.}",
             r"\label{tab:data_scaling_complete}", r"\scriptsize", r"\setlength{\tabcolsep}{7pt}",
             r"\begin{tabular}{llrrrr}", r"\toprule", r"Dataset & Setting & LPIPS $\downarrow$ & DreamSim $\downarrow$ & PSNR $\uparrow$ & FID $\downarrow$ \\", r"\midrule"]
    for i, (d, name) in enumerate(zip(DATASETS + ["macro"], NAMES + ["Mean"])):
        if i:
            lines.append(r"\midrule")
        for l in labels:
            r = data[l, d]
            lines.append(f"{name} & {l.upper()} & {r['lpips_alex']:.5f} & {r['dreamsim']:.5f} & {r['psnr']:.3f} & {r['fid']:.3f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ROOT / "full_results.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    data, labels = load()
    audit(data, labels)
    full_table(data, labels)
    recon_u_table(data)
    main_figure(data)
    two_sweep_figure(data)
    two_sweep_figure(data, "macro")
    two_sweep_figure(data, "macro", "lpips_alex")
    three_metric_figure(data)
    macro_figure(data)
    for axis in SETTINGS:
        full_figure(data, axis)
