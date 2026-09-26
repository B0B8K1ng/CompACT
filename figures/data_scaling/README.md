# Unlabeled-data scaling on RECON

当前论文只分析一个消融：固定 L=100%，增加 U=25/50/75/100%，用 RECON DreamSim 评估直接 4 s 预测。

主图：`data_scaling.pdf`（矢量），另有 SVG 和 PNG。四个原始点估计依次为 0.20059、0.19686、0.19078、0.18840；数值越低越好，首尾相对降低 6.08%。保留所有中间设置，以直线连接，不平滑、不拟合。结论限定为该数据集、指标和评测设置下的观测趋势。

图内仅保留坐标轴、曲线和四个数值；横轴为 `Unlabeled data (%)`，不显示 U 缩写。标题、数据集、固定 L 设置、预测时长和相对改善写在 caption 中。正文说明上游数据比例设置、统一的下游真实动作微调和统一评测过程。

论文文件：

- 仓库根目录 `scaling_ablation.tex`：完整 `\section{Scaling and Ablation Studies}`，只包含 U 消融。
- 仓库根目录 `scaling_ablation_appendix.tex`：该消融的评测协议和单指标结果表。
- `figure.tex`：当前单面板主图及 caption。
- `recon_u_results.tex`：从原始 CSV 自动生成的四行 RECON DreamSim 结果。
- `preview.tex`：独立 LaTeX 预览入口，需要 graphicx、booktabs、geometry、hyperref。

接入主稿：在正文使用 `\input{scaling_ablation}`，在 `\appendix` 之后使用 `\input{scaling_ablation_appendix}`。这是一段独立的新章节，没有覆盖 `experiments_revised.tex` 中讨论 standalone LAM 的旧实验设计；两者的模型和评测协议不同，合稿时不应混同。尤其不要沿用旧设计里的 62.5k LAM 训练步数描述这批 downstream NWM 结果。

实验口径：

- 四个设置使用相同的 500 个 RECON 窗口、真实动作条件、4 个上下文帧、224×224 图像、EMA 权重、250 步 DDPM 和评测 seed 0。
- DreamSim 使用 pretrained ensemble，结果为 500 个预测的均值。
- U 为潜动作学习中无标签池的保留比例；L 为该阶段的动作标签可用比例。固定 L=100% 指固定比例，不额外声称已审计固定绝对标注数量。
- 存储配置支持相同 downstream 架构、超参数、1042 步 warmup 和 10417 步 joint。上游数据子集嵌套、所有设置的 checkpoint 步数及实际训练曝光量未重新审计；正文没有声称这些额外条件已验证。
- 每个设置只有一个训练 checkpoint 和一个评测 seed，不画未经估计的误差条，不声称统计显著性或普适 scaling law。

原始记录与历史分析仍保留，未接入当前论文章节：

- `source_results.csv`：全部 6 个权重 × 5 个数据集 = 30 行原始结果，保留完整精度。
- 源结果：`/file_system/nas/algorithm/dujun.nie/nwm/results/nav1_u100l100_direct4s_20260926/comparison.csv`。
- `evaluation_protocol.json`：完整评测协议。
- `trend_audit.json` / `trend_audit.md`：全部 48 条 U/L 序列、严格单调性核对和五数据集均值；JSON 记录源 CSV 的 SHA256。
- `full_results.tex`：全部指标和数据集的结果表。
- `data_scaling_recon_two_sweeps.*`：历史 RECON DreamSim 双消融图。
- `data_scaling_mean.*` / `data_scaling_mean_lpips.*`：五数据集平均 DreamSim / LPIPS 双消融图。
- `data_scaling_mean_three_metrics.*` / `figure_mean_three_metrics.tex`：五数据集均值的 LPIPS、DreamSim、PSNR 三指标图及图注。
- `all_macro_trends.*` / `all_u_trends.*` / `all_l_trends.*`：完整趋势核对图。五数据集均值的 U 趋势并非全部单调，不能把 RECON DreamSim 的结论泛化到这些序列。
- 平均 FID 是五个独立 FID 的算术平均，不是合并特征后的 pooled FID。

复现全部图、表及审计文件：

```bash
conda run --no-capture-output -n base python figures/data_scaling/plot.py
```

当前环境没有 pdflatex、latexmk 或 tectonic；LaTeX 文件可以静态核对，但尚未完成 ICLR 模板内的实际编译。
