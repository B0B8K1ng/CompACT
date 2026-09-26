# 实验部分修订说明

> `experiments_revised.tex` 是按真实实现校正后的 working draft。实现细节与已有数值可以合入主稿，但主表和两组尚无完整结果的 LAM 消融在补齐前不能作为 submission-ready 结果提交。

## 可直接使用的内容

- `experiments_revised.tex` 已按真实实现拆分为主表协议、NWM 内部消融协议、公开 LAM 重建协议和 Pixel+Action LAM 数据消融协议。
- `tables/lam_public_comparison.tex` 的数值来自相同 record ID、统一 $[0,1]$ RGB / $240\times320$ metric pipeline 的评测；navigation-domain aggregate 为 2,400 pairs，Go Stanford transfer split 为 800 pairs，均排除 offset 0。为避免暗示所有公开 checkpoint 的训练曝光均已知，表中不再把这两列统称为所有方法的 ID/OOD。
- `tables/nwm_pretraining_ablation_available.tex` 仅保留已有完整预测数值的行与列，避免用 `--` 支撑结论。
- `tables/main_results_revised.tex` 修正了数据集/模式说明、标签和 `4\,s` 排版，但 OpenNWM 单元格仍是待填项。

## 投稿前必须处理

1. **主表仍未完成。** OpenNWM 的 visual prediction、ATE 和 RPE 均未填，正文因此没有写 `outperforms`、`consistently improves` 或 `achieves lower`。填入真实结果后再逐项写结论。
2. **FID 需要决策。** 现有本地 benchmark 只实现 LPIPS-Alex、DreamSim 和 PSNR，没有 OpenNWM 的 FID/FVD 结果。要么另行规范计算 FID，要么重做主表指标；不能把 PSNR 当作 FID 填入。FVD 应从当前稿件删除。
3. **主表 baseline 不是统一重跑。** 表内 baseline 为 published references；特别是 published NWM navigation 使用 CEM120，而当前本地规划协议是 CEM80。未做匹配评测前只能作为 contextual references。
4. **未完成表暂不纳入。** `nwm_finetuning_ablation.tex` 全为空；原 `lam_downstream_ablation`/LAM objective 表也无完整结果。不要在结果完成前恢复这些 input 或写因果结论。
5. **LatentPT 结论尚不成立。** 当前 pretraining table 的 latent 行为空，已有证据只能说明 time-only pretraining 在部分预测设置有效，且 4-FPS rollout 并未一致改善。
6. **LAM variant 不是严格 objective ablation。** Pixel/Pixel+Action 与 Action-only/旧 DINO checkpoint 的训练数据不同，只能称 training-recipe comparison。
7. **两幅图暂时移除。** 原稿没有提供 `cluster.pdf` 和 `vis-latent.png`，无法根据图中行列、颜色和语义写自洽 caption。原来的两句占位 caption 必须删除；拿到图后再恢复。
8. **Conclusion 需另行同步。** 原结论中的 `strong performance`、`establish` 等措辞依赖尚未完成的主表；`planning directly in the shared latent action space` 也不准确，因为 finalLAM-100k 在 Stage 2 使用 reset 后的 real-action adapter。
9. **公开 LAM baseline 的 BibTeX 待接入。** 修订稿已按各官方项目给出的 key 加入引用：`gao2026dreamdojo`、`chen2025villa0x0`、`zhang2026dila`、`jiang2026olaf`、`liu2026lara` 和 `chen2024moto`。当前 CompACT 仓库没有论文的 `.bib` 文件；合入主稿时需加入对应条目，若主稿已有不同 key，则统一替换。
10. **主表 direct 协议尚未冻结。** 当前 canonical local direct benchmark 只有 RECON 4\,s、500 个固定 windows；它只适用于内部消融，不能用来填主表的 SACSoN 4/16\,s direct cells。主表相应协议需单独冻结并运行。
11. **无结果的 LAM 段落仍是实验设计。** `LAM training-recipe comparison` 和 `Action-free data scale and label availability` 已按真实 recipe 校正，但尚无完整结果表；投稿时应在补齐结果后保留，或暂移 Appendix/实验计划，不能把它们写成已验证结论。
12. **ATE/RPE 单位需与 baseline 对齐。** 当前本地实现计算 waypoint-normalized translation RMSE，不做 alignment/scale correction，RPE 使用相邻帧；若要与 published baseline 做数值排名，必须先确认其单位与定义一致。
13. **合入主稿时确认 LaTeX 依赖。** 三张表需要 `booktabs`、`multirow` 和 `graphicx`（用于 `\resizebox`）；正文的 `\citep` 需要 ICLR 模板自带的 natbib 支持或等价配置。

## 三个容易混淆但必须分开的 checkpoint

- **OpenNWM 的离线 proxy extractor：** 60k-step Pixel+Action LAM。
- **OpenNWM-finalLAM-100k：** CDiT-B/2 在 60k Stage-1 NWM 预训练、3k real-action adapter warm-up 后，再 joint fine-tune 100k 的 checkpoint。
- **公开 LAM 重建表中的 Ours：** 独立的 100k-step Pixel-LAM；其 21.071/0.505 和 21.912/0.472 不能写成 Pixel+Action LAM 或 OpenNWM 的结果。

最终 OpenNWM checkpoint 的 SHA256 为
`aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4`。

## 本轮校验

- 已检查四个 LaTeX 文件的花括号和 `equation`/`table`/`table*`/`tabular` 环境配对。
- 已检查三张表的逻辑列数与每行跨度：分别为 7、5、15 列。
- 已检查本地 `\label`/`\ref` 与 `\input` 目标，以及五个新文件的尾随空白。
- 当前环境没有 `pdflatex`、`latexmk`、`tectonic`、`lacheck` 或 `chktex`，因此尚未做真正的 ICLR 主稿编译；合入完整论文工程后仍需编译一次，并接入缺失的 BibTeX 条目。
