"""Render auditable LPIPS/FID tables, endpoint selections, and rollout curves."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = {"recon": "RECON", "scand": "SCAND", "huron": "HuRoN",
            "tartan_drive": "TartanDrive", "go_stanford": "Go Stanford",
            "tum_rgbd": "TUM RGB-D", "unitree_go2": "Unitree Go2"}
MODELS = {"nwm-release": "NWM", "rae-nwm": "RAE-NWM", "opennwm-finalLAM-100k": "OpenNWM"}
METRICS = {"lpips_alex": "LPIPS ↓", "fid": "FID ↓"}
OPEN = "opennwm-finalLAM-100k"
BASE = ["nwm-release", "rae-nwm"]
COLORS = ["#377eb8", "#ee9135", "#bc263b"]


def improvement(point, metric):
    baseline = min(point["models"][m][metric] for m in BASE)
    return 100 * (baseline - point["models"][OPEN][metric]) / baseline


def plot(out, points, settings, stem):
    fig, axes = plt.subplots(len(settings), 2, figsize=(10, 2.8 * len(settings)), squeeze=False)
    for row, s in enumerate(settings):
        ds, fps, endpoint = s["dataset"], s["fps"], s["horizon_s"]
        curve = sorted([p for (d, f, h), p in points.items() if d == ds and f == fps and h <= endpoint],
                       key=lambda p: p["horizon_s"])
        for col, (metric, label) in enumerate(METRICS.items()):
            ax = axes[row, col]
            for (model, name), color in zip(MODELS.items(), COLORS):
                ax.plot([p["horizon_s"] for p in curve], [p["models"][model][metric] for p in curve],
                        "o-", label=name, color=color, lw=2, markersize=4)
            ax.set(xlabel="Rollout time (s)", ylabel=label,
                   title=f"{DATASETS[ds]} ({s['domain']}) · {fps} FPS · {endpoint}s")
            ax.set_xticks([p["horizon_s"] for p in curve])
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .97 if len(settings) > 1 else .91))
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    points = {}
    rows = []
    for path in sorted((out / "points").glob("*.json")):
        p = json.loads(path.read_text())
        key = p["dataset"], p["fps"], p["horizon_s"]
        assert key not in points
        points[key] = p
        for model, name in MODELS.items():
            r = p["models"][model]
            rows.append(dict(dataset=p["dataset"], domain=p["domain"], fps=p["fps"],
                             horizon_s=p["horizon_s"], model=name, model_key=model,
                             lpips=r["lpips_alex"], fid=r["fid"], sample_count=r["sample_count"],
                             source=str(path)))
    assert len(points) == 58, f"Incomplete results: {len(points)}/58"
    with (out / "rollout_lpips_fid.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    candidates = []
    for (ds, fps, h), p in points.items():
        if h not in ([4, 8, 16] if p["domain"] == "ID" else [4]):
            continue
        gains = {m: improvement(p, m) for m in METRICS}
        prefix = [q for (d, f, t), q in points.items() if d == ds and f == fps and t <= h]
        prefix_gains = [improvement(q, m) for q in prefix for m in METRICS]
        candidates.append(dict(dataset=ds, domain=p["domain"], fps=fps, horizon_s=h,
                               relative_improvement_pct=gains, minimum_improvement_pct=min(gains.values()),
                               endpoint_both_win=all(g > 0 for g in gains.values()),
                               all_evaluated_points_both_win=all(g > 0 for g in prefix_gains),
                               prefix_minimum_improvement_pct=min(prefix_gains)))
    # Endpoint and full-curve recommendations are kept separate and explicit.
    selected = [max([c for c in candidates if c["dataset"] == ds],
                    key=lambda c: c["minimum_improvement_pct"]) for ds in DATASETS]
    sustained = [max([c for c in candidates if c["dataset"] == ds and c["all_evaluated_points_both_win"]],
                     key=lambda c: (c["horizon_s"], c["prefix_minimum_improvement_pct"]))
                 for ds in DATASETS if any(c["dataset"] == ds and c["all_evaluated_points_both_win"] for c in candidates)]
    payload = dict(selection_rule="Maximize minimum relative gain across LPIPS/FID versus the stronger baseline, per dataset; same FPS/horizon for all models.",
                   sustained_selection_rule="Require both metrics below both baselines at every measured horizon from 1s; prefer longest allowed endpoint then largest minimum gain.",
                   selected_endpoints=selected, sustained_curves=sustained, all_candidates=candidates)
    (out / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out / "endpoint_screening.csv").open("w") as f:
        flat = [{**{k: v for k, v in c.items() if k != "relative_improvement_pct"},
                 **{m + "_gain_pct": g for m, g in c["relative_improvement_pct"].items()}} for c in candidates]
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    for ds in DATASETS:
        p = points[ds, 1, 4]
        plot(out, points, [dict(dataset=ds, domain=p["domain"], fps=fps,
                                horizon_s=16 if p["domain"] == "ID" else 4) for fps in [1, 4]], f"curves_{ds}")
    winners = sorted([s for s in selected if s["endpoint_both_win"]], key=lambda s: -s["minimum_improvement_pct"])
    if winners:
        plot(out, points, winners, "recommended_endpoints")
    if sustained:
        plot(out, points, sustained, "recommended_sustained")
    lines = ["# NWM / RAE-NWM / OpenNWM autoregressive rollout：LPIPS 与 FID", "",
             "基于 `full_8xa800_seed0_v1` 的同批预测和 GT；seed=0。每数据集 150 条轨迹，HuRoN 为 103 条有效轨迹。所有数值越低越好。LPIPS 是逐轨迹均值；FID 是该时间点全部轨迹图像的分布距离，不能按单张图像求均值。", "",
             "## 展示时间与筛选规则", "",
             "ID 候选终点为 4/8/16s，OOD 固定 4s；FPS 可选 1 或 4，三模型使用相同时间、FPS 和样本。终点推荐最大化两指标相对各自最强基线改善的较小值，属于查看结果后的展示筛选。持续领先另行要求从 1s 到终点的每个已测时间点两项均领先；不代表未测的连续时间点。", "",
             "ID/OOD 按 NWM/OpenNWM 导航训练集划分。RAE-NWM 训练不含 TartanDrive，因此该数据集并非三模型共同 ID。", "",
             "## 各数据集最有利的合法终点", "",
             "| 数据集 | 域 | FPS | 时间 | NWM LPIPS / FID | RAE-NWM LPIPS / FID | OpenNWM LPIPS / FID | LPIPS 改善 | FID 改善 | 终点双胜 | 从1s起已测点双胜 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for s in selected:
        p = points[s["dataset"], s["fps"], s["horizon_s"]]
        vals = [f"{p['models'][m]['lpips_alex']:.4f} / {p['models'][m]['fid']:.2f}" for m in MODELS]
        g = s["relative_improvement_pct"]
        lines.append(f"| {DATASETS[s['dataset']]} | {s['domain']} | {s['fps']} | {s['horizon_s']}s | " + " | ".join(vals) +
                     f" | {g['lpips_alex']:+.2f}% | {g['fid']:+.2f}% | {'是' if s['endpoint_both_win'] else '否'} | {'是' if s['all_evaluated_points_both_win'] else '否'} |")
    lines += ["", "## 整段曲线持续领先的选择", "",
              "| 数据集 | FPS | 截止时间 | 两指标全部已测点中的最小改善 |", "|---|---:|---:|---:|"]
    for s in sustained:
        lines.append(f"| {DATASETS[s['dataset']]} | {s['fps']} | {s['horizon_s']}s | {s['prefix_minimum_improvement_pct']:.2f}% |")
    if not sustained:
        lines += ["", "当前候选中没有从 1s 起所有已测点均双指标领先的曲线。"]
    lines += ["", "## 图与完整数据", ""]
    for stem in (["recommended_endpoints"] if winners else []) + (["recommended_sustained"] if sustained else []):
        lines += [f"[{stem} PDF]({out / (stem + '.pdf')})", "", f"![{stem}]({out / (stem + '.png')})", ""]
    lines += [f"[完整 CSV]({out / 'rollout_lpips_fid.csv'}) · [全部终点筛选]({out / 'endpoint_screening.csv'}) · [选择依据 JSON]({out / 'selection.json'})", "",
              "## 论文对应与计算口径", "",
              "[NWM Figure 4](https://arxiv.org/html/2412.03572#S4.F4) 对 RECON 展示 1/4 FPS、1/2/4/8/16s 的 LPIPS 和 FID；[RAE-NWM Figure 6](https://arxiv.org/html/2603.09241#S5.F6) 对 SACSoN 展示 4 FPS 的长时序曲线。这里采用两列 LPIPS/FID 的展示方式。HuRoN 为本次运行使用的 SACSoN 数据标识。", "",
              "FID 采用 RAE-NWM 发布评测代码的 torcheval FrechetInceptionDistance，torchvision Inception V3 ImageNet 权重、2048 维特征、RGB [0,1]、双线性缩放至 299×299。逐时间点计算，使用无偏样本协方差；采用数学等价的 float64 低秩算法，并独立对照 torcheval 稠密算法。该后端并非 pytorch-fid 的 TensorFlow 转换权重，不将这里的绝对分数直接等同论文数值。", "",
              "LPIPS 复用已校验 checkpoint/split/seed/sampler/frame/sample_ids 的 A800 汇总或同图像补算；FID 在同图像上补算。样本量为 103/150，单 seed，无置信区间；小差异仅说明本次点估计领先，不声称统计显著。", "",
              f"[来源元数据]({out / 'metadata.json'}) · [FID 数值校验]({out / 'fid_validation.json'}) · [计算日志]({out / 'compute_retry.log'})", ""]
    for ds, name in DATASETS.items():
        lines += [f"## {name}：全部已测点", "", f"[曲线 PDF]({out / ('curves_' + ds + '.pdf')})", "",
                  "| FPS | 时间 | 模型 | LPIPS ↓ | FID ↓ | 样本数 |", "|---:|---:|---|---:|---:|---:|"]
        for r in sorted([r for r in rows if r["dataset"] == ds], key=lambda r: (r["fps"], r["horizon_s"], list(MODELS).index(r["model_key"]))):
            lines.append(f"| {r['fps']} | {r['horizon_s']}s | {r['model']} | {r['lpips']:.4f} | {r['fid']:.2f} | {r['sample_count']} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(selected_endpoints=selected, sustained_curves=sustained), indent=2))
    print(f"REPORT_COMPLETE: {len(points)} points, {len(rows)} model rows; {out / 'report.md'}")


if __name__ == "__main__":
    main()
