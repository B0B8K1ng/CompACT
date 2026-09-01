# ICLR 2027：LAM 与两阶段 NWM 实验安排

> 版本：2026-09-01  
> 目标会议：ICLR 2027  
> 适用仓库：CompACT  
> 状态：实验冻结前计划；所有结果表格在获得真实运行结果前均不得填写推测值

## 0. 投稿约束与执行原则

本计划以 ICLR 2027 官方要求为准：摘要截止时间为 **2026-09-18 23:59 AoE**，全文截止时间为 **2026-09-25 23:59 AoE**。投稿正文上限为 9 页，参考文献、AI 使用声明、可复现性声明和伦理声明不计入正文页数；附录不限页数，但不能假定审稿人一定阅读。投稿必须双盲，论文、补充材料、代码包、日志截图和 W&B 链接均不得泄露作者、机构、账号或内部服务器信息。

官方入口：

- [ICLR 2027 Author Guide](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)
- [ICLR Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)
- [ICLR Code of Ethics](https://iclr.cc/public/CodeOfEthics)
- [ICLR 2027 LaTeX template](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)

所有实验遵循以下原则：

1. 先冻结问题、数据划分、主指标和主比较，再查看测试集结果。
2. 同一比较使用相同数据 recipe、训练步数、全局 batch、模型规模、VAE latent、评测样本和随机数协议。
3. 以轨迹而不是帧作为数据划分和统计重采样单元，避免相邻帧泄漏和伪重复。
4. 所有重要结论至少使用 3 个独立训练种子；单种子结果只能作为开发结果。
5. 不以“没有超过 SOTA”作为隐藏结果的理由，完整报告负结果、失败种子和计算成本。
6. 不新增 HybridPT、ShuffledPT、motion embedding 或 action-type embedding。
7. 超过 5 分钟的训练、评测和预处理必须使用持久实验运行器：

   ```bash
   codex-exp start <short-name> -- <完整且显式的环境命令>
   ```

   每个任务必须保存工作目录、完整命令、日志、退出码和最终 checkpoint 路径。

## 1. 论文定位

### 1.1 建议核心问题

在大规模无真实动作的导航视频上，能否通过离线 LAM latent proxy 学到可迁移的动作条件空间，并用少量有真实动作的数据将该空间转换为可用于预测和规划的 real-action NWM？

### 1.2 暂定核心贡献

以下内容是需要实验验证的研究假设，不是当前既成结论：

- **H1：** 保持完整时间跨度训练时，局部 LAM proxy 能给 NWM 提供比纯时间、几何 proxy 和 IDM proxy 更有效的动作条件。
- **H2：** 在完全相同的 NavAnywhere 采样和计算预算下，LatentPT 的优势来自 proxy 类型，而不是数据顺序或样本覆盖差异。
- **H3：** Embedding Alignment 能保留阶段 1 的 latent action condition space，并在阶段 2 推理时只依赖 real action，不依赖未来图像或 LAM。
- **H4：** Real-to-Latent 可以只通过 diffusion loss 学到 real action 到 latent action 的映射，不需要阶段 2 latent target 或 alignment loss。
- **H5：** 图像预测改进能转化为真实动作可控性、闭环/规划指标和未见环境泛化的改进。

### 1.3 必须控制的措辞边界

当前仓库包含 DreamDojo latent 的**离线读取与 NWM 使用路径**，但没有 DreamDojo LAM 的训练实现。因此：

- 在获得独立的 LAM 训练代码、数据、checkpoint 和许可证证据前，不能写“我们训练了 DreamDojo LAM”。
- 可以写“使用固定的预训练 LAM 作为离线 proxy extractor”，但必须给出来源、版本、checkpoint SHA256、latent 维度、输入预处理、授权范围和提取配置。
- LAM 不能在 NWM 训练或测试时在线运行；所有 proxy 都从缓存读取。
- Embedding Alignment 的 LAM 只在训练时提供 teacher target，测试和 planning 不得使用未来帧。

## 2. 实验对象与统一符号

- `LAM`：固定的 DreamDojo latent action model，仅用于离线生成局部帧对 latent。
- `E_z`：阶段 1 LatentPT 学到的 latent action encoder。
- `E_real`：阶段 2 Reset/Alignment 使用的 real action encoder。
- `G_real_to_latent`：方案 C 的 real action 到标准化 latent 坐标映射。
- `NWM`：CDiT/NWM diffusion backbone、图像/context 模块、diffusion timestep embedder、relative time embedder、positional embedding 和 final layer。
- `frame_offset`：目标帧相对当前帧的有符号整数偏移，完整范围为 `[-64, 64]`。
- `local pair`：`abs(frame_offset) <= 8`。
- `recipe`：固定样本身份、轨迹选择规则、offset 采样规则与随机种子的不可变采样文件。

动作 encoder 只编码动作，不能额外输入 `rel_t`。相对时间由独立 time embedder 编码，以便区分“动作内容”与“时间跨度”。

## 3. 数据协议

### 3.1 阶段 1：NavAnywhere only

阶段 1 只使用 NavAnywhere，不混入 RECON、SCAND、TartanDrive 或 HuRoN/SACSoN 的真实动作。

四种方法共享同一份冻结 recipe：

| 方法 | action condition | local proxy | long-range 样本 |
|---|---|---|---|
| NWM-TimePT | 无 | 无 | time-only |
| NWM-GeoPT | geometry | `|offset| <= 8` 且 cache 有效 | time-only |
| NWM-IDMPT | IDM | `|offset| <= 8` 且 cache 有效 | time-only |
| NWM-LatentPT | DreamDojo latent | `|offset| <= 8` 且 cache 有效 | time-only |

完整训练采样始终保留 `[-64, 64]`。对局部样本，只有 cache 存在、shape 正确、数值 finite 且通过有效性检查时才加入 action embedding。长间隔样本不丢弃、不裁剪 `rel_t`、不使用零 action 的条件语义、不做最近邻或在线补提取。

冻结 recipe 至少记录：

```text
schema_version
source_id
trajectory_id
current_frame_index
target_frame_index
frame_offset
split
sampling_weight
seed
recipe_sha256
```

必须完成的阶段 1 数据审计：

- [ ] 轨迹总数、有效图像数、总时长、帧率分布、轨迹长度分位数。
- [ ] 短/中/长轨迹各自被采样的次数和占比。
- [ ] `[-64,64]` 各 offset 的样本数；正向、反向和 zero offset 的比例。
- [ ] 局部 eligible、found、missing、invalid 和 used 比例。
- [ ] recipe 中不存在 train/val 轨迹交叉。
- [ ] 固定独立 validation trajectories；不能用训练轨迹的相邻帧作为验证集。
- [ ] recipe SHA256、数据 manifest SHA256 和 VAE cache manifest SHA256 写入 checkpoint metadata。

默认使用预计算的 VAE posterior latent。四种方法必须读取相同的 VAE cache key，VAE 版本、缩放系数和 dtype 完全一致。

### 3.2 阶段 2：real-action train only

阶段 2 训练集固定为：

- RECON train
- SCAND train
- TartanDrive train
- HuRoN/SACSoN train（使用当前代码中的实际名称）

Go Stanford 只能用于未见环境评测，不能进入训练、超参数选择或 checkpoint selection。

当前公开数据审计已知信息：

| 数据集 | train trajectories | test trajectories | train windows | test windows | 状态 |
|---|---:|---:|---:|---:|---|
| RECON | 9,468 / 9,468 | 2,367 / 2,367 | 132,929 | 31,711 | 已审计 |
| SCAND | 483 | 121 | 64,646 | 18,138 | 已审计 |
| HuRoN/SACSoN | 2,278 / 2,451 | 561 / 613 | 106,716 | 27,587 | 公开副本不完整，必须披露 |
| TartanDrive | 待审计 | 待审计 | 待审计 | 待审计 | 投稿前 P0 阻塞项 |

不同数据集的 real action 必须继续使用当前 NWM 已有的坐标变换和 normalization，不重新定义动作语义。混合采样要报告每个数据集、每种轨迹长度和每个 offset 的实际见样本数，不能只报告配置权重。

### 3.3 阶段 2 三种方案

| 方案 | checkpoint 来源 | Warm-up trainable | Joint trainable | loss | 推理路径 |
|---|---|---|---|---|---|
| A Reset | 任一 Stage-1 模式 | `E_real` | `E_real + NWM` | diffusion | `real -> E_real -> NWM` |
| B Alignment | LatentPT only | `E_real` | `E_real + NWM` | warm-up: align；joint: diffusion + align | `real -> E_real -> NWM` |
| C Real-to-Latent | LatentPT only | `G_real_to_latent` | `G + E_z + NWM` | diffusion only | `real -> G -> E_z -> NWM` |

方案 B 中 `E_z` 始终冻结，alignment 只在 local 且 latent cache 有效的样本上计算。方案 C 不读取阶段 2 latent target，不计算 alignment loss；warm-up 虽然冻结 `E_z` 和 NWM，但不能用 `torch.no_grad()` 截断到 `G` 的梯度。

## 4. LAM 实验轨

LAM 实验的目的不是单独追求一个 latent reconstruction 数字，而是回答两个问题：latent 是否包含动作信息，以及这种信息是否能转移到 NWM。

### 4.1 LAM 来源与缓存审计（P0）

每个 LAM cache 必须有以下 manifest：

```yaml
extractor_name: DreamDojo
extractor_version: ...
checkpoint_sha256: ...
license_or_access_terms: ...
latent_dim: 32
input_resolution: ...
frame_preprocessing: ...
pair_direction: current_to_target
max_abs_frame_offset: 8
latent_normalization:
  kind: raw_or_standardized
  statistics_source: NavAnywhere_train_only
  mean: ...
  std: ...
cache_key:
  - source_id
  - trajectory_id
  - current_frame_index
  - target_frame_index
extractor_code_commit: ...
created_at: ...
```

需要报告：总请求数、成功数、缺失数、坏 shape 数、NaN/Inf 数、重复 key 数、每个 offset 覆盖率、latent 每维均值/标准差、有效秩、L2 norm、max abs 和异常值比例。

阶段 1 normalization 统计只能由 NavAnywhere train split 计算；阶段 2 B/C 必须复用同一份统计。禁止用 validation/test 或阶段 2 数据重估统计。

### 4.2 LAM 动作信息 probing

在阶段 2 的训练划分内另取固定 probe-train/probe-val 轨迹，冻结 LAM，不更新 extractor：

1. 线性 probe：`z -> normalized [dx, dy, dyaw, ...]`。
2. 小型 MLP probe：用于判断非线性可解码性，参数量和训练预算固定。
3. 对照输入：零向量、随机高斯、time-only、geometry proxy、IDM proxy；real action 自身作为上界而非同等输入 baseline。
4. 指标：MAE、RMSE、R²、cosine similarity；转向任务可增加方向角误差。
5. 按数据集、offset 和轨迹长度分桶报告，不能只给总体平均。

probe 的拆分单位必须是 trajectory。主结论以线性 probe 为主，小 MLP 结果放附录，避免用大 probe 掩盖 latent 本身不足。

### 4.3 LAM 稳定性与退化检查

- latent 维度的方差和 effective rank，检查 collapse。
- 不同数据集之间的 latent 分布距离，只作诊断，不直接解释为语义差异。
- 同一动作范围内的 nearest-neighbor retrieval，人工检查视觉相似性与动作相似性是否混淆。
- `z(i,j)` 与 `z(j,i)` 只检查一致性，不预设二者严格互为负数，除非 LAM 定义明确保证该性质。
- 若比较 `step_100000` 和 `step_125000` checkpoint，必须固定下游配置，且在主实验前选定 primary extractor；不能依据最终 test 结果挑 checkpoint。

### 4.4 LAM 结果交付物

- Table L1：extractor provenance、cache coverage、invalid rate、normalization。
- Table L2：线性/MLP probing，与 geometry/IDM/time/random 对照。
- Figure L1：每维 std、有效秩、norm 分布。
- Figure L2：按 offset 的动作 probe 误差。
- Appendix：每个数据集的完整 probe 数字、超参数、失败样本审计。

## 5. NWM 主实验矩阵

### 5.1 主实验运行数

主结果使用种子 `{0, 1, 2}`：

| 组别 | 配置 | 每种子运行数 | 总运行数 | 目的 |
|---|---|---:|---:|---|
| Stage 1 | TimePT / GeoPT / IDMPT / LatentPT | 4 | 12 | 比较 proxy 类型 |
| Stage 2 Reset | 从四个 Stage-1 checkpoint 分别启动 | 4 | 12 | 公平 reset baseline |
| Stage 2 Align | LatentPT -> EmbeddingAlign | 1 | 3 | 主方法 |
| Stage 2 R2L | LatentPT -> RealToLatent | 1 | 3 | diffusion-only 映射 |
| Single-stage | real action 从头/legacy 训练 | 1 | 3 | 无视频预训练 baseline |
| **合计** | Stage-1 checkpoint 被复用 | **11** | **33** | 不含只评测的公开模型 |

`nwm-base`、`nwm-real`、`nwm-release`、`nwm-latent` 等已有 checkpoint 可作为评测对照，但如果训练数据或预算不可比，表中必须注明，不能伪装成严格 controlled baseline。

### 5.2 公平性约束

四个 Stage-1 模式保持一致：

- 相同 model size、初始化规则和 VAE latent。
- 相同 recipe SHA、训练步数、global batch、梯度累积和数据顺序。
- 相同 optimizer、LR schedule、weight decay、precision 和 checkpoint 间隔。
- 相同 validation sample IDs、diffusion timesteps/noise seeds。
- action dropout/CFG 概率保持原语义。
- proxy encoder 参数量差异单独报告；必要时增加参数量匹配的附录对照。

Stage-2 比较保持一致：

- 相同 four-dataset sampler 和 sample IDs。
- 相同 warm-up/joint 步数和 `adapter_lr/backbone_lr` 搜索预算。
- 每种方案超参数只用 validation 选择。
- B/C 必须验证来源为 LatentPT，latent dim、normalization、hidden dim、model size 和 context size 一致。

### 5.3 运行 ID 与目录

统一命名：

```text
<stage>-<method>-<model>-seed<seed>-recipe<sha8>-<yyyymmdd>
```

示例：

```text
s1-latentpt-cditb-seed0-recipea13f82c1-20260903
s2-align-cditb-seed0-recipea13f82c1-20260908
```

大型产物写入：

```text
/file_system/nas/algorithm/dujun.nie/nwm/
  recipes/
  vae_latents/
  proxy_cache/
  checkpoints/
  predictions/
  benchmark_reports/
```

源码保留在当前仓库。checkpoint 和报告均附带 SHA256，不能只靠 `latest.pt` 标识。

## 6. 评测协议

### 6.1 训练与条件覆盖日志

每个 epoch 至少记录：

```text
loss/diffusion
loss/alignment_cos
loss/alignment_l1
loss/total
proxy/total_samples
proxy/eligible_samples
proxy/ineligible_long_range_samples
proxy/found_samples
proxy/missing_samples
proxy/invalid_samples
proxy/used_samples
proxy/eligible_rate
proxy/used_rate
proxy/mean_abs_offset_used
proxy/mean_abs_offset_time_only
data/<dataset>/samples
data/offset_histogram
grad/adapter_norm
grad/backbone_norm
```

TimePT 额外记录 `action_conditioning_enabled=false`。方案 C 记录 predicted latent 的 mean/std/norm/max_abs。

### 6.2 图像预测

仓库当前 benchmark registry 的固定协议：

- RECON time：500 samples。
- RECON rollout 1 FPS：150 samples。
- RECON rollout 4 FPS：150 samples。
- horizons：1、2、4、8、16 秒。
- diffusion steps：250。
- 默认评测 seed：0；模型训练种子另行区分。
- 指标：LPIPS-Alex（越低越好）、DreamSim（越低越好）、PSNR（越高越好）。
- FID 只在全局样本量足够时报告，不对小分桶 FID 作主结论。

除 registry 主协议外，对 RECON、SCAND、TartanDrive、HuRoN 各自的 held-out test trajectories 使用固定 sample manifest。主表报告 macro-average 和逐数据集结果，不能让大数据集按帧数压过小数据集。

### 6.3 未见环境泛化

Go Stanford：

- 固定 500 个 sample IDs。
- horizons：1、2、4、8、16 秒。
- 主文比较 4 秒结果，其余放附录。
- 250 diffusion steps。
- 只作最终泛化评测，不参与选模型。

### 6.4 Planning

主 planning 协议使用仓库的 CEM80：

- datasets：RECON、SCAND。
- 每个数据集固定 100 samples。
- population 80，top-k 5，3 次重复，1 次优化。
- horizon 8，0.25 秒/步。
- 基于 VAE reconstruction 的 LPIPS cost。
- 固定 seed 42。
- 指标：ATE、RPE-translation，报告每次重复和均值/标准差。

CEM10 accelerated protocol 只用于开发诊断，不能与 CEM80 主结果混为一谈。

### 6.5 Action controllability（投稿前 P0 新增评测）

仅有像素指标不能证明模型使用了 action。必须增加同 context、同 diffusion noise、替换 action 的反事实评测：

1. 固定 context、target horizon、noise seed。
2. 输入原始动作、零动作、相反转向动作、来自其他样本的动作。
3. 用统一的位姿/视觉运动 estimator 测量预测运动方向和幅值。
4. 指标至少包括 endpoint/action direction agreement、转向符号准确率、动作交换后的预测差异和相同动作重复稳定性。
5. TimePT 应作为 action-insensitive control；B/C 的推理只使用 real action。

该评测实现前，不能仅凭 LPIPS/PSNR 声称“更可控”。

### 6.6 效率与部署依赖

每个方法报告：

- 总参数、trainable 参数和 adapter 参数。
- 训练 GPU-hours、峰值显存、吞吐量。
- proxy/LAM 离线提取 GPU-hours和缓存大小。
- 单步/rollout 推理时延。
- 推理需要的模块和输入。

特别说明：B 推理不需要 LAM、`z_gt` 或 `E_z` teacher；C 推理需要 `G + E_z`，但不需要未来帧或 LAM extractor。

## 7. 统计分析计划

### 7.1 种子与配对

- 训练种子：`0, 1, 2`。
- 三个种子共享 recipe 内容与评测 sample IDs，但模型初始化、训练噪声和 dropout seed 独立。
- 对同一 sample 的方法差值做配对统计。
- 以 trajectory 为 cluster 做 paired bootstrap，报告 95% confidence interval。
- 每项主指标同时报告三个种子的逐种子值、mean ± std 和 bootstrap CI。

### 7.2 预注册的 primary comparisons

主比较只保留以下四项：

1. LatentPT vs TimePT，在相同 Stage-1 validation 和下游 Reset 下比较。
2. LatentPT vs GeoPT/IDMPT，在相同 Reset 下比较。
3. LatentPT -> Align vs LatentPT -> Reset。
4. LatentPT -> RealToLatent vs LatentPT -> Reset/Align。

多个 primary comparisons 对同一指标使用 Holm correction。其余比较标为 exploratory。

### 7.3 模型选择与失败规则

- checkpoint 由预先固定的 step 或 validation 指标选择，绝不查看 test 指标。
- 某 seed 出现 NaN、数据读取损坏或明确基础设施故障时允许重跑；必须保留失败日志，并使用相同 seed。
- 不能因为结果差而替换 seed。
- 若只完成单 seed，降级为 preliminary，不形成强因果结论。
- 主张“方法有提升”至少要求对应 paired difference 的 95% CI 不跨 0，且 planning/controllability 没有与主张矛盾的明显退化；否则诚实报告为 mixed/negative result。

## 8. Ablation 计划

### P0：正文必需

- 四种 Stage-1 action mode：Time/Geo/IDM/Latent。
- 三种 Stage-2 方案：Reset/Align/RealToLatent。
- real-action single-stage/no-pretrain baseline。
- 训练时 local proxy 与 long-range time-only 的覆盖统计。
- action controllability。
- B 在推理时移除 teacher/LAM 的实测验证。
- C 无 latent target、无 alignment loss 的实测验证。

### P1：高价值附加实验

- Alignment loss：cosine only、L1 only、cosine + L1。
- warm-up：0 vs 默认步数。
- C joint 时：`E_z` frozen vs trainable。
- proxy local threshold：4 vs 8；两者总采样范围都必须保持 `[-64,64]`。
- NavAnywhere 数据量：25%、50%、100%；使用嵌套的冻结 trajectory subset recipe。
- 每种方法的训练计算量匹配结果。

### P2：时间允许时

- 不同 horizon/offset 的误差剖面。
- 场景类别、速度、转弯强度和轨迹长度的 subgroup analysis。
- 失败案例：动态障碍、低光、急转、视野遮挡和反向 offset。

本项目不新增 HybridPT 或 ShuffledPT；已有兼容代码也不作为本文新增方法或 ablation。

## 9. 计算预算与分阶段启动

不预估未经测量的 GPU-hours。先对每种代表配置运行 100-step pilot，记录稳定后的秒/step、峰值显存、I/O wait 和 checkpoint 大小，再计算：

```text
Stage1_GPUh = 12 * measured_stage1_hours_per_run * GPUs_per_run
Stage2_GPUh = 21 * measured_stage2_hours_per_run * GPUs_per_run
Eval_GPUh   = sum(all fixed benchmark jobs)
Total_GPUh  = Stage1_GPUh + Stage2_GPUh + Eval_GPUh + proxy_extraction_GPUh
```

采用 gate 控制浪费：

1. **Gate 0：** recipe、VAE cache、proxy cache、split 和 checkpoint metadata 审计通过。
2. **Gate 1：** 单卡 tiny smoke 通过，loss finite，保存/恢复与梯度测试通过。
3. **Gate 2：** 每种方法 100-step 性能 pilot，确定全量预算和并行度。
4. **Gate 3：** 先跑 seed 0：4 个 Stage-1 + 7 个 Stage-2，共 11 个训练任务。
5. **Gate 4：** 数据与指标无 bug 后，启动 seed 1/2，而不是按 seed 0 好坏选择性扩展。

GPU 任务启动前必须检查 GPU 可用显存；大缓存和 checkpoint 写入前检查 `/file_system/nas/algorithm/dujun.nie` 的可用空间。

## 10. 时间安排

| 日期 | 任务 | 必须交付 |
|---|---|---|
| 09-01～09-02 | 数据、recipe、VAE/proxy/LAM provenance 审计 | manifests、SHA、split、覆盖表、阻塞项清零 |
| 09-02～09-03 | tiny smoke、100-step pilot、W&B schema | 可恢复 checkpoint、吞吐/显存/GPUh 估算 |
| 09-03～09-05 | LAM probe 与 cache 质量分析 | Table L1/L2、latent 退化图 |
| 09-03～09-09 | Stage-1 seed 0，随后 seed 1/2 | 12 checkpoints、Stage-1 validation 表 |
| 09-06～09-13 | Stage-2 seed 0 与固定 benchmark | 7 个 seed-0 模型、预测/规划初表 |
| 09-10～09-17 | Stage-2 seed 1/2 | 全部 33 个主训练任务、三种子统计 |
| 09-13～09-18 | 方法图、主表、标题/摘要/作者确认 | 09-18 前提交摘要；作者列表冻结 |
| 09-18～09-22 | P1 ablation、可控性、失败案例 | ablation 表、定性图、统计 CI |
| 09-22～09-24 | 匿名 artifact、clean-env 复现、论文冻结 | 匿名补充材料、复现报告、AI/伦理声明 |
| 09-25 | 最终格式/匿名/链接检查并投稿 | 09-25 23:59 AoE 前完成 |

如果 09-09 前主 Stage-1 尚未稳定，应立即缩减 P1/P2，不减少三种子主实验，也不通过减小主 benchmark 样本数制造“完整结果”。

## 11. W&B 与可追溯性

建议：

```text
project: nwm-iclr2027
group: <method>-<recipe_sha8>
job_type: pretrain | warmup | joint | eval | probe
name: <stage>-<method>-<model>-seed<seed>-<timestamp>
tags:
  - navanywhere
  - recipe:<sha8>
  - action_mode:<mode>
  - finetune:<scheme>
  - seed:<seed>
```

每个 W&B run 保存：

- git commit 和 dirty diff patch/hash。
- 完整解析后的 config。
- recipe、dataset、VAE cache、proxy cache、checkpoint SHA256。
- 数据集实际采样数和 offset histogram。
- Python/PyTorch/CUDA/NCCL/driver、GPU 型号、world size 和 precision。
- 所有随机种子与 deterministic 设置。
- trainable/frozen 参数名、参数量和 optimizer group LR。
- 父 checkpoint 与 substage 恢复信息。

公开 artifact 不直接暴露内部 W&B entity、NAS 路径、用户名或组织名。投稿时优先导出匿名的 CSV/JSON、配置和静态图；若提供在线 dashboard，必须先验证匿名访问和元数据脱敏。

## 12. 论文表格、图片与正文页数安排

### 12.1 主文交付物

- **Figure 1：** 两阶段方法图，清楚区分 offline LAM extraction、Stage-1 proxy conditioning、Stage-2 A/B/C 和 test-time dependency。
- **Table 1：** 数据、模型、预算和 LAM/proxy provenance。
- **Table 2：** 四种 Stage-1 模式的 validation、proxy 覆盖和计算量。
- **Table 3：** Stage-2 A/B/C 在四个 held-out 数据集上的预测指标。
- **Table 4：** action controllability、planning 和 Go Stanford 泛化。
- **Figure 2：** NavAnywhere 数据规模或 horizon/offset 曲线。
- **Figure 3：** 配对差值 CI 和代表性成功/失败案例。

### 12.2 9 页建议分配

| 内容 | 页数 |
|---|---:|
| 摘要 + 引言 | 1.0 |
| 相关工作 | 0.7 |
| 方法 | 2.0 |
| 实验设置 | 1.1 |
| 主结果 | 2.3 |
| 分析/ablation/限制 | 1.4 |
| 结论 | 0.5 |
| **总计** | **9.0** |

可复现性、AI 使用和伦理声明不挤占正文配额，但关键实验设置不能全部藏入附录。附录提供逐数据集/逐种子结果、完整超参数、缓存 schema、额外 qualitative 和失败任务日志。

## 13. Artifact 与可复现性清单

- [ ] 匿名仓库或单一补充 PDF/压缩包，不包含作者或内部组织信息。
- [ ] 环境 lockfile、CUDA/PyTorch 版本和一键命令。
- [ ] 所有主实验 YAML、sampling recipe 和 sample manifest。
- [ ] VAE/proxy/LAM cache schema、生成命令和 SHA；不违规重新分发原数据。
- [ ] checkpoint metadata、父 checkpoint SHA 和 substage 恢复说明。
- [ ] Stage-1、Stage-2、恢复、单卡 smoke、DDP 和 benchmark 命令。
- [ ] 输出表格的聚合脚本、trajectory-cluster bootstrap 脚本和原始匿名 CSV。
- [ ] 从 clean environment 至少复现一个 tiny end-to-end run。
- [ ] 明确数据、代码、LAM checkpoint 和预训练模型的许可证。
- [ ] 失败种子、缺失公开轨迹和未运行实验如实披露。

## 14. AI、伦理和限制声明草案

ICLR 2027 要求披露生成式 AI 在研究过程中的使用。本项目已经使用 AI 编码助手协助实验设计和软件实现，因此最终论文不能写“仅用于语法润色”。建议根据实际使用更新为：

> Generative AI tools were used to assist with experiment planning, software implementation, debugging, and language editing. All research claims, code changes, experimental outputs, citations, and the final manuscript were reviewed and verified by the authors, who take full responsibility for the work.

伦理/限制至少覆盖：

- 导航视频可能包含个人、车辆牌照或地点信息；说明数据许可、隐私处理和展示图像的脱敏。
- 模型预测不等同于安全控制策略，不能直接用于未经验证的真实机器人部署。
- LAM latent 的语义可能受训练分布偏差影响。
- HuRoN/SACSoN 公开副本与原始 split 覆盖不完整，必须报告实际样本数。
- DreamDojo checkpoint、NavAnywhere、TartanDrive 和其他数据的授权边界必须在发布 artifact 前确认。

## 15. 当前阻塞项与风险登记

| 优先级 | 风险 | 当前状态 | 关闭条件 |
|---|---|---|---|
| P0 | DreamDojo LAM 训练代码/权重来源不在本仓库 | 未关闭 | 获得 checkpoint SHA、配置、许可证和 extractor provenance |
| P0 | NavAnywhere train/val split 与冻结 recipe 尚需最终确认 | 进行中 | 生成 recipe、验证无轨迹泄漏、保存 SHA |
| P0 | NavAnywhere VAE cache 尚需完成/验证 | 未关闭 | 8-GPU 预计算完成且 cache manifest/缺失率通过 |
| P0 | TartanDrive 覆盖与许可证审计缺失 | 未关闭 | 数据 manifest、split、action 和窗口统计完成 |
| P0 | action controllability evaluator 尚未完成 | 未关闭 | 反事实动作评测可复现并加入 registry |
| P0 | HuRoN/SACSoN 公开轨迹不完整 | 已知限制 | 尽量补齐；否则在表格和限制中明确实际覆盖 |
| P1 | 33 个主训练任务的 GPU 时间未知 | 未关闭 | 完成 100-step pilot 并冻结预算 |
| P1 | W&B/日志可能泄露内部身份路径 | 未关闭 | 匿名导出和人工双盲检查通过 |
| P1 | 多数据集 sampler 可能导致样本数失衡 | 待审计 | 每 epoch 实际数据集/轨迹长度/offset 统计通过 |

在 P0 项关闭前，不启动完整三种子大规模矩阵；可以运行 tiny smoke、cache audit 和 100-step pilot。

## 16. 每日实验记录模板

```markdown
### <date> / <owner> / <run-id>

- Research question:
- Config path and SHA:
- Git commit + diff hash:
- Parent checkpoint SHA:
- Dataset/recipe/cache SHA:
- Exact command:
- codex-exp name:
- Working directory:
- Log path:
- GPU allocation and free memory before start:
- Output disk free space before start:
- Start/end time and exit code:
- Final checkpoint/report path + SHA:
- W&B run ID:
- Validation result:
- Data/proxy coverage:
- Failure/anomaly:
- Decision and next action:
```

## 17. 投稿前最终验收

只有全部满足后，才把结果写成 ICLR 主结论：

1. 四种 Stage-1 方法使用完全相同的 frozen recipe 和 VAE latent。
2. 所有 Stage-1 方法保留 `[-64,64]`，proxy 只在有效 local pair 上加入。
3. A/B/C 的梯度、冻结模块、checkpoint 恢复和 inference dependency 有测试证据。
4. 三个独立种子完成，逐种子结果、mean/std 和 trajectory-cluster CI 齐全。
5. 预测、action controllability、planning 和 unseen generalization 至少形成三类互补证据。
6. LAM 来源、cache provenance、latent normalization 和许可证可追溯。
7. 数据覆盖、缺失轨迹、失败实验和实际 GPU-hours 完整披露。
8. clean-env tiny reproduction 通过，匿名 artifact 不泄露身份。
9. AI 使用声明准确反映实验设计和代码实现中的实际使用。
10. 正文证据在 9 页内自洽，不能依赖审稿人阅读附录才能验证核心主张。

