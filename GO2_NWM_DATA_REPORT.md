# Go2 数据转换与 NWM 微调可行性

日期：2026-09-09。结论：**数据已下载、解压并转换为 NWM 可读格式；可以开展 Go2 导航运动适配，优先考虑 nwm-timept-ft。当前数据不足以学全 Go2 的动作空间。没有启动 GPU 微调。**

## 数据位置与来源

- 来源：`LittleBoss/unitree-go2-data`，本机 `fastwam` 环境已有的 ModelScope 登录可以读取；匿名请求返回不存在。下载默认使用直连。
- NAS 根目录：`/file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data`。
- 原始下载：`raw/go2-cleaned-datasets.zip`，273,891,377 字节。
- 压缩包 SHA-256：`7bb845507c33be4e87a22c7b20a77a797b22708eaedd312c7c01f75fd75dd371`；与 ModelScope 文件清单一致。
- 上传文件 revision：`4fd7712be8be8daa54091ffa2d312c1196d34db0`；完整记录见 NAS 根目录 `modelscope_manifest.json`。
- 解压目录：`extracted/go2-cleaned`。37 个归档条目，解压大小 276,493,454 字节。
- NWM 数据：`nwm_4hz`。
- 审计文件：`nwm_4hz/conversion_report.json`、`normalization.json`、`validation_report.json`、`eligible_action_coverage.json`。
- 日志：`logs/download_direct.log`、`logs/convert.log`、`logs/validate.log`，对应 `.exitcode` 文件记录最终退出状态。

## 划分结果

按采集时间排序，前 5 条训练，最后 2 条完整原始轨迹测试。先按原始 episode 划分，再依据原有 `segment_id` 拆连续片段。所有子片段继承同一原始 episode 的集合归属，没有按帧随机拆分。

| 原始 episode 后缀 | 集合 | 原视频帧 | 4 Hz 保留帧 | 满足 NWM 长度的帧 | 可用观测锚点 |
| --- | --- | ---: | ---: | ---: | ---: |
| 120452_b5b9ee97 | train | 1355 | 380 | 380 | 246 |
| 121612_8de6b25b | train | 1562 | 438 | 438 | 304 |
| 122036_55eb8409 | train | 818 | 230 | 128 | 61 |
| 122302_105f3d57 | train | 1737 | 487 | 423 | 289 |
| 122858_c1741a1d | train | 987 | 277 | 277 | 143 |
| 123112_1f87c457 | test | 1041 | 292 | 282 | 215 |
| 123252_a0e820b9 | test | 670 | 188 | 188 | 121 |
| 合计 | | 8170 | 2292 | 2116 | 1379 |

所有名称前缀均为 `episode_20260909T`。测试 episode 的完整名称为：

```text
episode_20260909T123112_1f87c457
episode_20260909T123252_a0e820b9
```

总采集时长约 571.98 秒（9.53 分钟），不是大规模动作学习数据。原始清洗数据共有 15 个连续片段，转换全部保留。9 个训练片段、2 个测试片段满足 NWM 默认长度，训练锚点 1043、测试锚点 336。另 4 个短片段仍保存在磁盘，不进入默认采样索引；包括测试第一条末尾 10 帧的片段。它们不会被拼接或补帧。

每个锚点随机取 4 个目标，因此一次遍历大约产生 4172 个训练目标对，但这些目标和上下文高度相关，不能视为同等数量的独立采集数据。

`data_splits/go2/{train,test}/episode_names.txt` 保存原始轨迹划分；`traj_names.txt` 只列可用于默认 NWM 的连续片段；`all_traj_names.txt` 还包含短片段。

## 格式与时间、坐标约定

```text
nwm_4hz/
  go2/
    episode_20260909T...__seg00/
      0.jpg
      1.jpg
      ...
      traj_data.pkl
      frame_mapping.csv
      metadata.json
  data_splits/go2/train/{traj_names,all_traj_names,episode_names}.txt
  data_splits/go2/test/{traj_names,all_traj_names,episode_names}.txt
  index_cache/go2/{train,test}/...
```

`traj_data.pkl` 只包含 NWM 原读取器需要的数值数组：

```python
{
    "position": np.ndarray(shape=(N, 2), dtype=np.float64),  # 原局部里程计 x/y，米
    "yaw": np.ndarray(shape=(N,), dtype=np.float64),        # 朝向，弧度
}
```

原始 JPEG 输出保持视频的 1280×720 尺寸，质量 95；NWM 的 `misc.get_transform` 实施 4:3 中心裁剪、224×224 缩放、RGB 及 mean/std=0.5 归一化。没有先压成正方形而在读取时再次错误裁剪。原始视频与所有原始对齐 CSV 均保留。

采样使用 **`source_elapsed_s`（主机接收时间）**，在每个连续片段内建立 0.25 秒网格，选最近的实际帧，并取该帧对应的同一 CSV 行。没有使用 `frame_index / 15` 作为物理时间，没有重新插值位姿，没有跨缺口平滑。4 Hz 与仓库既有 Go2 实验协议一致。选中帧相对目标网格最大偏差约 39.8 ms；实际帧间隔有抖动，不能宣称严格曝光同步。

`frame_mapping.csv` 保留原始帧号、接收时间、网格时间、四元数、z/roll/pitch、机器人上报速度和配对质量字段，方便逐帧追溯。`vx/vy/yaw_speed` 是源数据的上报字段，不能直接认定为下发的控制指令。

NWM 对目标帧 j、当前帧 i 构造动作：

```text
dx =  cos(yaw_i)*(x_j-x_i) + sin(yaw_i)*(y_j-y_i)
dy = -sin(yaw_i)*(x_j-x_i) + cos(yaw_i)*(y_j-y_i)
dyaw = wrap_to_pi(yaw_j-yaw_i)
a = (dx / spacing, dy / spacing, dyaw)
spacing = 0.07610250112606785 m
```

这里 dx 是当前机体朝向的前向位移，dy 是左向位移，dyaw 逆时针为正。spacing 只用满足 NWM 长度的训练片段相邻采样点位移均值估计，包含停止时刻；测试数据不参与拟合。推理与评测必须使用同一个 spacing，不能继续套用 RECON 的 0.25 或 SCAND 的 0.38。

动作进入真实动作编码器前，还会经过 checkpoint 的既有 minmax 变换；保持其原有配置，不能把 spacing 缩放与编码器归一化混淆。原实现没有把超范围值裁剪到 [-1,1]。

保持 context_size=4、len_traj_pred=64、目标偏移范围 [-64,64]；源读取器的时间条件实际为 `frame_offset / 128.0`。在 4 Hz 下，64 帧对应约 16 秒。长度 N 的连续片段有 `max(0, N-67)` 个观测锚点。

## 实际动作覆盖与局限

以下只统计进入默认 NWM 数据集的训练片段，共 1637 对相邻帧（约 0.25 秒），并非对所有长时域目标对统计：

| 观测到的运动 | 判定 | 比例 |
| --- | --- | ---: |
| 前进 | dx > 2 cm | 75.63% |
| 倒退 | dx < -2 cm | 0.24%（仅 4 对） |
| 侧移占主导 | abs(dy) > max(2 cm, abs(dx)) | 0% |
| 近静止 | 平移 < 1 cm 且 abs(dyaw) < 0.02 rad | 12.46% |
| 左转 | dyaw > 0.05 rad | 13.38% |
| 右转 | dyaw < -0.05 rad | 8.86% |

这些分类可重叠。平移 dx 的中位数约 0.0897 m/步，95 分位约 0.1250 m/步；dy 的 5–95 分位约 [-0.00591, 0.01043] m/步。倒退、纯侧移、复杂机动覆盖显著不足。长时域目标在拐弯后出现大的 dy，并不等价于采集了机体横移指令；负时间偏移产生的负位移也不是机器人实际倒退。

![Go2 trajectory and motion coverage](/file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data/nwm_4hz/motion_coverage.png)

清洗报告明确声明摄像头曝光延迟未标定，媒体时钟和接收时间约有 5% 速率差。当前转换解决了采样时间基准和已知中断问题，不能消除未知曝光/通信延迟。原 CSV 的 `error_code` 全为 100，源报告判定帧对有效；这里不臆测该固件字段的含义，也未据此额外删除数据。

测试第一条原始录制提前中断，已有数据仍可用，缺失结尾无法恢复。两个测试 episode 是独立采集记录，不等于两个全新场景；里程计轨迹存在局部接近/重叠，仍需视觉核对场景泛化范围。完整 episode 隔离和图像哈希检查不能证明物理路线完全不重叠。

## 两种 checkpoint 能否微调

已在 CPU 读取实际 checkpoint 元数据确认，二者均为 CDiT-B/2、hidden_dim=768、context_size=4。

| 初始化 | 实际状态 | Go2 适配方式 | 判断 |
| --- | --- | --- | --- |
| nwm-timept | proxy_pretrain，action_mode=none | 从主干和时间模块初始化，新建真实动作编码器，先训练编码器再小学习率联合训练 | 可以，适合作为对照 |
| nwm-timept-ft | real_finetune，action_mode=real；已完成 10k warmup + 100k joint | 保留已学到的真实动作编码器和主干，开启新的 Go2 优化器与步数 | 优先尝试，小数据下起点更合适，但需实验确认 |

实际文件：

```text
/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywhere_stage1/nwm-timept/checkpoints/latest.pth.tar
/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_timept_ft/nwm-nav1-timept-finetune/checkpoints/joint_0100000.pth.tar
```

单独继续 TimePT 的 `action_mode=none` 训练只能适应图像分布，不能学到动作条件。现有 B/C 方案依赖 LatentPT 的潜动作编码器，不适用于这里的 TimePT 初始化；TimePT 应采用真实动作 adapter reset 路径。

**当前仓库不能仅替换数据路径就正确开始这项微调。** 本次交付的数据配置 `conf/dataset/go2.yaml` 与原 `TrainingDataset` 相容，但它不是完整的 stage-2 训练配方：

1. `data_utils.prepare_datasets` 对 `real_finetune` 强制限定 RECON、SCAND、TartanDrive、SACSoN 四个来源。需要新增显式 Go2 适配协议和对应数据选择校验，保留原实验协议默认值；不能把 Go2 冒充其中一个来源。
2. 从 TimePT 开始时可以复用现有 `finetune.stage1_checkpoint` 加载逻辑，并采用 Go2 配方。
3. 从 TimePT-FT 开始时不能传给 `finetune.stage1_checkpoint`：该入口只接受 proxy_pretrain checkpoint，且 reset 会丢弃已有真实动作编码器。也不应伪造 checkpoint 元数据绕过检查。
4. `training.from_checkpoint` 是原实验断点恢复，会恢复训练计数、优化器和数据位置，并校验数据指纹。它不等价于切换到 Go2 的新微调。需要明确的“仅初始化权重”入口，加载完整模型/选定 EMA 权重，保留真实动作编码器，重新构建优化器、调度器、步数与 Go2 数据指纹。

本次没有修改上述训练核心，也没有启动微调；避免在数据审计任务中改变已有训练实验语义。

## 建议的首轮实验

优先用 TimePT-FT 做低成本适配实验。下列是待验证的起始设置，不是已得到最佳效果的参数：

- 冻结 VAE。先冻结主干、仅训练已有真实动作编码器约 100–200 步，adapter 学习率 1e-5；随后主干 1e-6、adapter 1e-5，联合训练约 300–800 步。
- 初始全局观测 batch 4–8，每观测 4 个目标；先不采用原 10k+100k 长配方，当前仅 5 条训练轨迹，过拟合风险高。
- 两条测试轨迹冻结，不用它们选学习率、停止时刻或 checkpoint。需要调参时在剩余 5 条训练 episode 内做按轨迹的验证/交叉验证，所有派生样本跟随该 fold；数据尺度统计也应按 fold 重新拟合。
- 对照至少包括未适配 TimePT-FT、Go2 适配 TimePT-FT、TimePT 加新动作编码器。输入采样率、spacing、上下文、扩散采样种子与评测预算保持一致。
- 测试短时域未来预测（例如 +1/+4/+8 帧），报告每条原始测试 episode 的 LPIPS、PSNR/SSIM；按前进、转向、静止分组，避免静止样本掩盖运动失败。
- 同一上下文对比真实动作、零动作和打乱动作，检查动作改变是否系统性改变预测。单纯画质提升不足以证明学会了动作条件。只有两条测试轨迹，不据此宣称稳定的跨场景泛化或闭环控制成功。

如果目标是覆盖更完整的 Go2 平面运动，应补采独立的倒退、左右横移、左右原地转向、不同速度的前进/转弯和启停过程，并同步保留实际下发的 vx/vy/wz、持续时间、机器人反馈位姿及相机时间信息。当前数据只支持世界模型学习“观察到的平面位移/转角对应怎样的图像变化”；不直接学习 12 个关节控制、步态切换、跳跃或电机力矩。

NWM 的 `(dx,dy,dyaw)` 是局部目标位姿变化，真机执行需要独立控制器将其转换为限幅速度指令。不能把位移直接送到速度接口，也不能把世界模型离线预测验证当作真机闭环验证。

## 复现与验证

工作目录：`/file_system/vepfs/algorithm/dujun.nie/code/CompACT`。下载使用已有登录的 `fastwam` 环境，转换使用 `nwm-preprocess`，加载核验/绘图使用 `nwm`；全部直接执行，无 GPU 操作。

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n fastwam python -u scripts/download_go2_data.py --root /file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data

/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm-preprocess python -u scripts/prepare_go2_nwm.py --source /file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data/extracted/go2-cleaned --output /file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data/nwm_4hz

/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm python -u scripts/validate_go2_nwm.py --root /file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data/nwm_4hz

/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm python scripts/plot_go2_audit.py --root /file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data/nwm_4hz

/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm-preprocess python -m unittest discover -s tests -p test_prepare_go2_nwm.py
```

转换器拒绝向已有非空输出目录再次写入，重新转换时请选择新的版本目录，避免混用旧索引和新样本。修改 FPS/划分后需要更新数据配置中的训练集 spacing。

核验覆盖：源输出文件 SHA-256、完整视频解码计数、逐帧图像与 CSV 索引一致性、所有转换 JPG 可解码、位姿有限值、四元数与 yaw 一致、无跨 segment 样本、原始 episode 集合隔离、训练/测试精确图像重复检查，以及 NWM 原读取器的全部锚点局部动作计算和训练样本形状。结果以 `validation_report.json` 为准。4 个转换单元测试检查接收时间采样、重复/无效时间戳、禁止上采样复制帧、前左坐标与角度环绕。

最终结果：下载、转换、读取核验均退出 0；2292 张转换图像全部通过解码；训练 1043 / 测试 336 个锚点全部通过动作计算检查；各集合 16 个读取器样本通过图像/动作形状检查；四元数推导 yaw 与存储 yaw 的最大差为 4.44e-16 rad；原始 episode 交叉数与精确图像交叉数均为 0；4 个单元测试全部通过。
