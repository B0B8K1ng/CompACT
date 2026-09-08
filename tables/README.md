# 论文表格

本目录存放论文可直接 `\input` 的表格，按实验主题组织为 5 个文件，一个文件可以包含多张独立表格。
从项目根目录的 `iclr_experiment_tables.tex` 整理而来，保留原有实验数值、精度、待填结果与协议说明。
原文件保留作为历史参考；后续论文表格在本目录维护。
导航主表已于 2026-09-08 核对外部论文并补充引用结果，筛选依据见 `navigation_comparison_sources.md`；本项目实验结果仍保持原样。

## 在论文中使用

把整个 `tables/` 目录放在论文主文件旁边，在导言区加载一次公共依赖，在正文按需插入表格：

```tex
% 导言区，放在 \begin{document} 之前
\input{tables/preamble}

% 正文，按论文需要选择表格
\input{tables/navigation_main_comparison}
\input{tables/pretraining_ablation}

% 文末；若论文已有 bibliography，将 tables/navigation_references 加入其列表
\bibliographystyle{plain} % 实际投稿时采用论文模板要求的样式
\bibliography{tables/navigation_references}
```

主表的四条引用位于 `navigation_references.bib`。也可将条目合并到论文已有的 `.bib` 文件中，保持引用键一致；不要重复添加 `\bibliographystyle` 或 `\bibliography`。

`preamble.tex` 加载 `amsmath`、`booktabs`、`multirow`、`threeparttable`，并定义带 `CompACT` 前缀的公共命令，避免覆盖论文已有的同名命令。方法显示名在 `\CompACTMethod` 中统一修改。

每张表使用独立的 `\caption` 和 `\label`，编号由论文自动生成，不再手工共用 2a/2b 之类的计数器。拆分后的表使用新标签，迁移正文时应同步修改 `\ref`。原来的三个综合预测标签和 `tab:lam_objectives` 不再对应单张表。

宽表使用 `table*`，其余使用 `table`；可根据论文模板的栏宽调整。无需把所有表放进正文：主结果放正文，完整消融和细粒度指标可放附录。

## 文件索引

| 文件 | 表格数 | 内容 |
| --- | --- | --- |
| `navigation_main_comparison.tex` | 1 | 单张主表：ATE/RPE 与 LPIPS/DreamSim/FID，按规划、RECON 自回归预测、SACSoN 单步预测分组 |
| `pretraining_ablation.tex` | 5 | 预训练消融：单步预测、1 FPS / 4 FPS rollout、未见域泛化、导航规划 |
| `finetuning_strategies.tex` | 5 | Reset / Align / Action2Latent 微调策略：四种预测评测与导航规划 |
| `lam_reconstruction.tex` | 5 | LAM 重建：RGB、DINO、动作误差、航向误差、跨域 RGB 退化 |
| `lam_downstream_ablation.tex` | 5 | LAM 目标函数对下游 NWM 的影响：四种预测评测与导航规划 |

另有 `preamble.tex`（公共依赖和命令）与 `preview.tex`（预览入口）。
`navigation_references.bib` 提供主表引用；`navigation_comparison_sources.md` 记录来源、协议及未纳入原因。
每个主题文件内部用注释标明各张表的内容，便于定位；表格仍各自拥有独立的标题、标签和编号。
`\input` 会插入该主题的全部表格；如需分放正文与附录，可将对应的完整 `table` / `table*` 环境移到所需位置。

## 阅读与维护约定

- 消融中的预测表均在 4 秒处评估，每行对应一个模型或策略，列只保留 LPIPS、DreamSim、PSNR。
- 主结果合并为一张 `table*`，使用统一表头、一个标题和一个编号，标签为 `tab:main_sota_navigation`。表内按规划、RECON 自回归预测（4 FPS）、SACSoN 直接单步预测分组；视觉结果包含 4 秒和 16 秒，每组保留同设置 NWM 基线，只能在同组、同时域内比较。原视觉表标签 `tab:main_visual_prediction` 已移除，正文引用应改为主表标签。
- 预训练预测表把重复的 Hours 列移到表注：NA-v1 为 322 小时，NA-v2 为 1200+ 小时。模型名和数据版本保持原命名。
- 主表的外部结果已核对原文。规划分组只纳入明确沿用 NWM 的 RECON 2 秒规划协议的结果；视觉分组采用各论文内同设置对比，不声称跨论文的测试索引或协议相同。其他主题文件的外部引用值未在本次重新核验。
- 规划分组要求 CEM 120 个候选、1 次迭代。本项目消融中的 CEM-80 结果不能直接搬入该组。视觉分组的本方法结果也须按相应数据集、预测方式、时域与评测设置重新核对后填入。
- 主表中不适用或未报告的指标使用破折号，本方法对应任务的结果保持待评测；合并仅调整排版，保留原有数值与精度。
- `\CompACTMissing` 表示本项目待填结果；NWM rollout 行的 `--` 表示原论文未报告。不要将缺失值当作零。
- 已有单点结果保持原精度；原有 `mean ± std` 保持原样，不补造标准差。待实验完成后再统一更新统计形式及最优值加粗。
- LAM 表分别说明 ID、OOD、CD 与汇总方式；退化表保留各指标差值的方向。
- 原文件的微调公式与结果解读段落仍在原文件中，按需移入论文方法或实验正文。

## 预览

`preview.tex` 按主题载入全部 21 张表，每个主题结束后换页；主题内部由 LaTeX 自动排版。论文正文可直接引用所需主题文件。

在项目根目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/compact-table-preview tables/preview.tex
```

`latexmk` 会自动运行 BibTeX 并完成多轮编译，预览 PDF 输出到 `/tmp/compact-table-preview/preview.pdf`。
在 Overleaf 中保留 `tables/` 目录结构，将 `tables/preview.tex` 设为主文档即可预览。
当前整理环境未安装 LaTeX 编译器，尚未执行 PDF 编译；最终栏宽与浮动位置需在论文模板中确认。
