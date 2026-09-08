# 导航规划与视觉预测主表：文献与评测协议核对

核对日期：2026-09-08。对应 `navigation_main_comparison.tex`。

本次检索以完整标题 “Navigation World Models”、NWM、RECON、ATE/RPE、LPIPS、DreamSim、PSNR、FID 和 citing works 为线索，核查 arXiv 原文、补充材料及作者官方代码。Google Scholar 检索页访问失败，因此不声称覆盖其全部施引文献。下列数字均为论文报告值，没有运行新的模型评测。

主结果分为两张独立表，仍放在同一个 tex 文件中，各有标题和编号。规划表（`tab:main_sota_navigation`）沿用原始 NWM 的 RECON 协议；视觉表（`tab:main_visual_prediction`）按论文的评测设置分组，每组同时引用该工作与其配套 NWM 基线。数据集、预测方式和时域不同的组不能直接排名。指标名称相同或都写作“4 秒”，不足以证明结果可比。拆分只调整排版，原有数值及其来源保持不变。

## 规划表采用的协议

基准是 Bar 等人的 [Navigation World Models，arXiv:2412.03572v2](https://arxiv.org/html/2412.03572v2)，不是任意含有 NWM 行的导航表。

| 项目 | 本表依据 |
| --- | --- |
| 数据与任务 | RECON，给定观察和目标图像，比较规划轨迹与真实轨迹 |
| 预测范围 | 2 秒，4 FPS，8 个预测步 |
| 指标 | 原论文定义的 ATE / RPE，越低越好；不把其他距离误差或 SR/SPL 当作同一指标 |
| 规划预算 | 世界模型的独立规划采用 CEM：120 个候选、1 次迭代、直线轨迹参数化 |
| 原始实现补充 | NWM 官方命令使用 top-5、每条候选 3 次随机 rollout 评分；动作位移按数据集平均步长归一化 |
| 原有数值 | Table 2：GNM 为 1.87±0.00 / 0.73±0.00；NoMaD 为 1.93±0.04 / 0.52±0.00；NWM planning 为 1.13±0.02 / 0.35±0.01 |

协议说明位于 NWM §4.4、补充材料 §7；实现参数见 [官方 README 的 Trajectory Evaluation - Planning](https://github.com/facebookresearch/nwm#trajectory-evaluation---planning)。GNM、NoMaD 是直接策略，CEM 参数针对世界模型规划。

筛选要求：论文引用 NWM，在数值表中与其比较，且明确沿用上述评测协议；公开信息出现设置差异，或只说明相同数据集和指标而关键协议仍不清楚时，暂不合并。模型架构、训练数据及模型内部的目标距离函数属于方法差异，本表并不宣称这些相同。

## 规划表已纳入：1 篇论文、3 个变体

[Learning Latent Action World Models In The Wild](https://arxiv.org/html/2601.05230v2)，Quentin Garrido、Tushar Nagarajan、Basile Terver、Nicolas Ballas、Yann LeCun、Michael Rabbat，2026，arXiv:2601.05230v2。

- §7 引用 NWM 并进行 RECON 规划比较。
- Appendix A 的 “Planning protocol for RECON” 明确声明使用 NWM 的相同协议，并逐项写明 CEM 120 候选、1 次迭代、直线参数化、8 步、4 FPS、2 秒与 ATE/RPE。
- Appendix C 的 **Table S2 右表**提供规划结果，并列出 NWM 1.13 / 0.35、NoMaD 1.93 / 0.52；左表是预测质量，未混入主表。

| 主表名称 | Table S2 原始行 | ATE | RPE |
| --- | --- | --- | --- |
| Latent action WM (sparse) | Sparse / High | 1.43 | 0.42 |
| Latent action WM (noisy) | Noisy / High | 1.40 | 0.40 |
| Latent action WM (discrete) | Discrete / High | 1.48 | 0.42 |

三行统一选用作者标记的 High 容量版本，避免把低、中、高容量消融全部展开到主表；它们属于同一篇工作。High 是原文的潜动作容量标签，不表示三个模型的参数量相同，也不表示逐指标挑选的最优配置。原文只报告单点数值，保留两位小数，不补造标准差。

纳入依据是作者明确声明的协议一致性及相应数值表。本次未逐样本核对测试索引、随机种子或重跑模型，不能将这些文献引用值描述为本项目统一复现结果。

## 未纳入规划表的候选

“待确认”表示证据不足，不等于已经证实设置不同。相同的 NWM 基线数字本身不能证明协议相同；重新评测的 NWM 数字不同也不能单独作为排除理由。

| 工作与原文 | 位置 / 已报告结果 | 本次处理依据 |
| --- | --- | --- |
| [RAE-NWM，v2](https://arxiv.org/html/2603.09241v2) | Table 2，RECON 1.36 / 0.37；DINO-Reg 1.46 / 0.36 | 虽为 2 秒、8 步、120 候选，但[官方 run_plan.sh](https://github.com/20robo/raenwm/blob/main/run_plan.sh) 为 top-3、`num_repeat_eval=1`，与原始 NWM 命令不同；未找到该表在 top-5、3 次评分下的结果。暂不纳入。 |
| [CompACT: Planning in 8 Tokens，v1](https://arxiv.org/html/2603.05438v1) | Table 4；附录的 RECON 规划设置 | CEM population 为 80，与主表的 120 不同。暂不纳入。 |
| [V-JEPA 2.1，v3](https://arxiv.org/html/2603.14482v3) | §3.4，Table 7 | 虽为 2 秒 / 4 FPS，但候选数为 480；该表第二项误差标为 RTE，也不能未经指标核对直接改写为 RPE。暂不纳入。 |
| [DR-NWM，v1](https://arxiv.org/html/2605.24761v1) | §4.3，Table 4 | 评估 4 秒轨迹，并使用 32 个候选，与本表的 2 秒 / 120 候选不同。暂不纳入。 |
| [CoME，v1](https://arxiv.org/html/2605.18813v1) | §4.1，Table 2：0.96 / 0.28 | STM 使用 12 个上下文帧；表注明 100 条采样轨迹，但未明确给出与原始协议一致的完整候选预算和测试索引。不能把这 100 条直接解释成 CEM population，也不能仅凭相同 NWM 数字合并。暂不纳入。 |
| [LS-NWM，v1](https://arxiv.org/html/2511.11011v1) | Table I，RECON 1.51 / 0.43；§IV-C | 运行时间实验明确使用 120 候选、3 次 CEM 迭代；未确认 Table I 对应的是原始 1 次迭代设置。待确认，不把运行时间实验配置直接断言为 Table I 配置。 |
| [AR Forcing，v1](https://arxiv.org/html/2605.31314v1) | Table 2，2 秒结果；§4.3、§7.2 | 已确认 8 帧、120 候选、top-5、重复评估 3 次，但没有确认 CEM 迭代次数及测试索引与原始基准一致。待确认；不把其较长时域表或重评 NWM 基线混入规划分组。 |
| [UniWM，v3](https://arxiv.org/html/2510.08713v3) | §4.1，Table 1 | 数据重新按语义子场景分段，导航过程持续到 Stop；没有确认为原始固定 2 秒轨迹评测。暂不纳入。 |
| [NavWAM，v1](https://arxiv.org/html/2606.13494v1) | §5.2，Table 1 | 该表报告 GO Stanford，非 RECON。暂不纳入。 |
| [NavWM，v1](https://arxiv.org/html/2606.24101v1) | §4.2–4.3，Tables 1–2 | 未给出与本表一致的 RECON 2 秒独立结果；导航 rollout 使用平均 43 步的语义片段。暂不纳入。 |
| [UA-NWM，v1](https://arxiv.org/html/2608.05597v1) | Appendix F，Table 9 | 使用 NoMaD 生成的 8 / 16 / 32 条候选做重排序，与当前独立 CEM 规划预算不同；也未确认该表轨迹时域一致。暂不纳入。 |
| [One-Step World Model，v1](https://arxiv.org/html/2601.12277v1) | Tables 1–2、4 | 与 NWM 比较的是生成指标、SR/SPL 和实机成功率，没有提供本表所需的 RECON 2 秒 ATE/RPE。暂不纳入。 |
| [Latent World Models with Monotone Planning Costs，v1](https://arxiv.org/html/2608.09073v1) | §4.1–4.4，Table 1 | GNM 数据集测试集、6 步轨迹，指标为 AOE/MAOE/ADE/MADE，非本表协议。暂不纳入。 |
| [Beyond Language Modeling，v1](https://arxiv.org/html/2603.03276v1) | §5.1–5.2，Figures 12–13 | 使用 NWM 协议研究预训练，但相关结果是预训练配置曲线；未找到符合要求的、与 NWM 数值行直接比较的 ATE/RPE 表。没有从曲线估读数值。 |

## 视觉表已纳入：2 篇论文、各自的同设置 NWM 对比

视觉分组保留 LPIPS、DreamSim 和 FID，均为越低越好。选择 4 秒以衔接项目已有评测，另列 16 秒以展示长时域预测；两个方法均使用相同的时域选择。没有从曲线估读数据，也没有为未报告的 PSNR 或标准差补值。

### RECON 自回归预测：AR Forcing

[AR Forcing: Towards Long-Horizon Robot Navigation World Model，arXiv:2605.31314v1](https://arxiv.org/html/2605.31314v1)，**Table 1 的 RECON 部分**。

- §4.1 引用 NWM，并明确两种方法使用相同的数据划分、指标评测脚本和超参数；该工作重新训练并评测 NWM，不能用原始 NWM 论文的单步预测数值替代这组基线。
- §4.2 明确两者均以 4 FPS 自回归预测 16 秒，Table 1 按时域报告指标。§7.2 给出 4 帧上下文、64 个预测步及该论文的测试划分与评测索引设置。
- 当前表引用 4 秒、16 秒两列。这是该论文内部的可比结果，不声称它的测试索引与本项目现有 rollout 实验完全一致。

| 方法 | 时域 | LPIPS | DreamSim | FID |
| --- | --- | --- | --- | --- |
| NWM | 4 s | 0.423 | 0.181 | 63.0 |
| AR Forcing | 4 s | 0.341 | 0.125 | 52.9 |
| NWM | 16 s | 0.533 | 0.319 | 77.3 |
| AR Forcing | 16 s | 0.463 | 0.210 | 66.0 |

### SACSoN 直接单步预测：RAE-NWM

[RAE-NWM: Navigation World Model in Dense Visual Representation Space，arXiv:2603.09241v2](https://arxiv.org/html/2603.09241v2)，**Table 1**。

- §5.1 将 NWM 列为生成质量基线；§5.2 及 Table 1 明确比较 SACSoN 上的 4 秒和 16 秒直接预测。两者都以聚合动作预测目标帧，跳过中间帧；不是连续 rollout。
- 该工作 Table 1 的数据集是 **SACSoN**，不能标成 RECON，也不能移入上面的自回归组。其 4 FPS 连续 rollout 结果主要见 Figure 6 与附录 Figure 13–14，未从这些曲线读取数值。
- 该工作规划预算尚未通过上一节的同协议筛选，不影响其视觉 Table 1 内部对比的采用。视觉比较与规划比较分别核对。

| 方法 | 时域 | LPIPS | DreamSim | FID |
| --- | --- | --- | --- | --- |
| NWM | 4 s | 0.407 | 0.229 | 26.15 |
| RAE-NWM | 4 s | 0.303 | 0.145 | 15.09 |
| NWM | 16 s | 0.470 | 0.281 | 33.06 |
| RAE-NWM | 16 s | 0.349 | 0.171 | 15.90 |

### 未采用的其他视觉结果

下列是进一步核查过的候选，不是全部施引文献清单。“待确认”不等于已证实设置不同。

| 工作与原文 | 位置 | 本次处理依据 |
| --- | --- | --- |
| [Vid2World，v3](https://arxiv.org/html/2505.14357v3) | §5.3，Table 1，Figure 6 | Table 1 的 NWM 行标为单步预测，Vid2World 行采用自回归预测；即使都是 RECON 的 4 秒结果，预测方式仍不一致。Figure 6 才是自回归对比，但没有从曲线估读数值。 |
| [DR-NWM，v1](https://arxiv.org/html/2605.24761v1) | §4.2，Table 1，Appendix C | 已有 RECON 4/16 秒的 LPIPS、DreamSim、FID 对比，但未确认该数值表的采样帧率与当前 RECON 4 FPS 组一致。待确认，未直接合并。 |
| [Mobile World Models，v1](https://arxiv.org/html/2603.07799v1) | Tables I–II | 视觉表主要在 SCAND 上，并区分 DDIM-5 / DDIM-25。未确认与现有组一致的完整采样协议；不能仅凭相同指标和时域合并。 |
| [WorldPack，v1](https://arxiv.org/html/2512.02473v1) | §6.4，Table 3 | 表中 Baseline 与 WorldPack 的实际上下文范围分别为 4 帧与 19 帧，且未明确该 Baseline 等同于原始 NWM 的完整评测配置。未并入固定上下文的 RECON 组。 |
| [One-Step World Model，v1](https://arxiv.org/html/2601.12277v1) | §4.1，Table 1 | 使用自建 Matterport3D / Habitat 数据，且 NWM-B 基线的扩散模型被替换为 shortcut model。该表的 PSNR、SSIM 等不属于现有 RECON / SACSoN 协议。 |
| [Pondering the Way / SWAM，v1](https://arxiv.org/html/2606.29908v1) | Table 2 | 比较目标条件下的联合视频/动作生成与 NWM+NoMaD 候选选择，不能当作给定真实动作的视觉预测表直接合并。 |
| [UniWM，v3](https://arxiv.org/html/2510.08713v3)；[NavWM，v1](https://arxiv.org/html/2606.24101v1) | 视觉生成指标表与实验设置 | 数据按语义片段组织，部分数值跨多个数据集汇总；未确认与现有分组一致的独立数据集、固定时域评测。 |
| [Learning Latent Action World Models In The Wild，v2](https://arxiv.org/html/2601.05230v2) | Table S2 左表 | 是该工作内部 IDM 与 Controller 的预测质量对比，没有对应 NWM 数值行；未与右侧规划指标混用。 |

## 后续维护

- 引用键保存在 `navigation_references.bib`；主表表注指出原始数值表，预览入口已加载该文献库。
- 本方法保持待评测。项目其他表中的 CEM-80 消融结果不能直接填入当前 CEM-120 规划分组；已有 4 秒预测值也不能未经数据划分和预测设置核对就填入视觉分组。
- 视觉分组各自保留对应来源的 NWM 行，不进行跨组最优值加粗。只有同一组、同一时域的行具有直接比较关系。
- 若后续取得候选论文的同协议结果或配置证据，再补充主表并更新本记录。需要比较其他协议时应另设分组或独立表，保留该协议下重新评测的 NWM 基线。
