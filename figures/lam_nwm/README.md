# LAM–NWM 论文结构图

本图围绕三个结构选择组织：LAM 学习什么表示、NWM 使用什么预训练条件、预训练后的动作接口如何微调。图中不展示实验结果，也不暗示某种方案优于其他方案。

## 文件与使用

| 文件 | 用途 |
| --- | --- |
| `overview.svg` | 完整结构图，可用 Inkscape、Illustrator 等软件编辑 |
| `overview.pdf` | 完整结构图的矢量 PDF，建议插入论文 |
| `overview.png` | 完整结构图预览 |
| `panel_a_lam.svg/.pdf/.png` | 四种 LAM 结构，可独立展示或放大 |
| `panel_b_pretraining.svg/.pdf/.png` | 四种 NWM 训练路径 |
| `panel_c_finetuning.svg/.pdf/.png` | 三种两步微调策略 |
| `build_figure.py` | 图形生成源代码，修改文字、布局或颜色后重新生成 |
| `figure_lam_nwm.tex` | LaTeX 插图环境与英文 caption |

在项目根目录运行生成脚本：

```bash
python3 figures/lam_nwm/build_figure.py
```

直接编辑 SVG 后重新运行生成脚本会覆盖相应生成文件；需要保留的调整应同步到 `build_figure.py`。

将整个 `figures/lam_nwm/` 目录放在论文主文件旁，导言区加载 `graphicx`，在正文插入：

```tex
\input{figures/lam_nwm/figure_lam_nwm}
```

默认采用普通 `figure` 环境并按 `\textwidth` 排版，适合当前 ICLR 单栏论文版式。可通过 `\ref{fig:lam-nwm-design}` 引用。若整图在实际论文尺寸下显得拥挤，可分别使用三个 panel 的 PDF；最终可读性请以论文编译出的 PDF 为准。

## 图中的符号与结构

- `E_LAM`：LAM 的帧对编码器，将两帧（或两帧的 DINO 特征）映射为 latent action `z`。
- `E_z`：NWM 的 latent action 条件编码器，将 `z` 映射为 NWM 使用的条件 embedding；它与 `E_LAM` 是两个模块。
- `E_a`：真实动作编码器。Reset / Align 中输出 NWM 条件 embedding；Action2Latent 中输出 latent action 坐标，再送入 `E_z`。
- `F_theta`：NWM。`Δt` 表示两帧的时间间隔／相对帧偏移，区别于扩散过程的 timestep。微调面板省略共同的图像输入与扩散细节。
- `L_pixel`、`L_action`、`L_DINO`：相应重建目标的符号，不指定尚未固定的具体损失形式或权重。`L_NWM` 表示 NWM 训练目标，`L_align` 表示动作 embedding 对齐目标。

LAM 的四种结构分别为：Action only 用 `z` 重建真实导航动作；Pixel 用开始帧与 `z` 重建未来帧；DINO 先编码两帧，再以开始帧特征与 `z` 重建未来帧特征；Pixel + action 同时使用像素重建与动作重建监督。图中不额外假定 DINO encoder 的冻结策略。LAM 用于 NWM 标注时提取 `z`，不使用其像素、特征或动作 decoder。

NWM-Real 直接在 Base data 上训练。TimePT、GeoPT、LatentPT 分别在 NavAnywhere 上使用无动作标签、VGGT 几何动作、LAM latent action 预训练，之后使用 Base data 微调。TimePT 保留图像上下文与时间条件。NavAnywhere v1 为 13 个数据集、322 小时；v2 在 v1 上加入 Ego4D 与 GO，为 1200+ 小时，是数据规模扩展维度。

三种微调策略比较的是 LatentPT checkpoint 的动作接口迁移，不代表所有预训练方法都与三种微调形成完整组合。仓库中的 TimePT / GeoPT 使用 Reset；Align 和 Action2Latent 需要 LatentPT 的 `E_z`。

| 策略 | Step 1 | Step 2 | 推理路径 |
| --- | --- | --- | --- |
| Reset | 移除 `E_z`；仅训练新 `E_a`，使用 `L_NWM` | 训练 `E_a` 与 `F_theta`，使用 `L_NWM` | `a → E_a → F_theta` |
| Align | 冻结 `E_z`；仅训练 `E_a`，对齐 `E_a(a)` 与 `E_z(z)` | 训练 `E_a` 与 `F_theta`，同时使用 `L_NWM` 和 `L_align`；`E_z` 始终冻结 | `a → E_a → F_theta` |
| Action2Latent | 仅训练 `E_a`；梯度经过冻结的 `E_z` 与 `F_theta`，使用 `L_NWM` | 训练 `E_a`、`E_z` 与 `F_theta`，使用 `L_NWM` | `a → E_a → E_z → F_theta` |

Action2Latent 的“对齐”通过 NWM 目标学习 latent 输入接口，不额外回归 LAM 的真实 `z`。Align 的 `z → E_z` 支路只提供训练时 teacher target。三种策略在推理与规划时均不需要未来帧或 LAM extractor。表中“全量”指所画 NWM 及相关动作接口；共享的图像 VAE 仍冻结。

## 与实验表格保持一致

- NWM 的 Base data：RECON、SCAND、TartanDrive、HuRoN，均有真实导航动作。
- 按当前 LAM 表格协议，LAM 训练使用 RECON、SCAND、HuRoN，TartanDrive 用于 OOD 评测。因此，不应将 NWM 的全部 Base data 标成 LAM 的训练集。
- LAM 下游消融固定 NWM 预训练和 Reset 微调，只改变 LAM 目标。
- 预训练表中的 `No-Pretrain` 是额外的预算匹配对照，区别于 `NWM-Real`；本图遵循本次指定的四种主路径，不新增第五种结构。
- 数据扩展中的 GO 与表格中的 GO-Stanford 未见域评测集不同，图中未将它们合并。

## 内容依据

结构以本次用户提供的方法定义为准，并对照以下仓库材料核验：

- [LAM 重建表](../../tables/lam_reconstruction.tex)与[LAM 下游消融表](../../tables/lam_downstream_ablation.tex)：四种目标与 ID/OOD 数据协议。
- [预训练消融表](../../tables/pretraining_ablation.tex)：Base data、NavAnywhere v1/v2 与各预训练路径。
- [微调策略表](../../tables/finetuning_strategies.tex)、[原始公式](../../iclr_experiment_tables.tex)、[两阶段训练说明](../../TWO_STAGE_NWM.md)：三种微调的连接、损失与冻结规则。
- [训练参数策略](../../two_stage_nwm.py)、[训练目标实现](../../two_stage_training.py)、[NWM 条件连接实现](../../models.py)：实际训练和推理行为。
- [NavAnywhere 预训练说明](../../NAVANYWHERE_STAGE1.md)：LAM 离线提取 latent，action decoder 不参与标注。

相关方法的一般背景可参见 [DreamDojo 官方 LAM 文档](https://github.com/NVIDIA/DreamDojo/blob/main/docs/LAM.md)与 [NWM 官方仓库](https://github.com/facebookresearch/nwm)。这两个链接作为背景入口；本图中新增的 DINO、动作监督及微调组合以本项目定义为准。
