# Go2 微调与离线导航评测

新增入口复用 `train.py`、两阶段训练循环、数据读取器和 `planning_eval.py`。
仅 `two_stage=go2_adapt` 启用新配方；原配置的四数据集限制、EMA 0.9999、
stage1 重置和 no-pretrain 随机初始化行为保留。

## 环境与运行

在项目根目录使用 `nwm` 环境。大产物默认保存到
`/file_system/nas/algorithm/dujun.nie/nwm/compact/go2_adaptation/<run-id>`。
以下 `--gpu 0` 应替换为空闲 GPU。启动前检查磁盘、空闲显存和活跃 GPU 进程；
检查失败直接退出。只有已获 GPU 使用者同意共享时才指定 `--allow-busy-gpu`。
脚本不会停止其他实验。

```bash
# 只检查数据、权重哈希并打印将执行的命令，不生成实验产物、不使用 GPU
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py all --run-id go2_v1 --dry-run

# 固定 100 个索引、训练集 CEM 先验及环境/代码/数据快照
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py prepare --run-id go2_v1

# 独立冒烟协议：2 步 warmup + 2 步 joint，两个导航窗口，CEM/扩散参数不减小
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py all --run-id go2_smoke_v1 --models timept-ft --gpu 0 --smoke

# 全量五组：先完成所有可用模型训练，再评测微调前、后；缺少最终权重记为 pending
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py all --run-id go2_v1 --gpu 0

# 当前缩减协议：每条测试 episode 各 5 个窗口，复用旧实验已完成且通过校验的训练
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py all --run-id go2_20260910_10win_v2 --eval-windows 10 --reuse-training-from /file_system/nas/algorithm/dujun.nie/nwm/compact/go2_adaptation/go2_20260910_v1 --gpu 0 --resume

# 中断恢复：训练接续 latest 和原 WandB run ID，评测跳过已有逐样本结果
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py all --run-id go2_v1 --gpu 0 --resume

# LatentPT 最终 joint_0100000 到位后单独补齐
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py all --run-id go2_v1 --models latentpt-ft --gpu 0 --resume

# 可分别执行 train / eval；最后按全部五组汇总
conda run --no-capture-output -n nwm python -u scripts/run_go2_adaptation.py summarize --run-id go2_v1
```

长任务可由普通后台进程运行并重定向日志；入口输出工作目录、准确命令、PID、
日志路径和退出码，子进程记录保存在 `logs/*.process.json`。`all` 串行执行，
单组失败会明确退出；修复后使用相同命令加 `--resume`。

## 固定实验设置

- 源权重：TimePT/GeoPT 注册表 `joint_0100000`，No-pretrain 注册表
  `joint_0110000`，NWM-real 注册表 `0200000`；LatentPT 使用 pixel-action-l20
  原实验的最终 `joint_0100000`，缺失时不会替换成中间权重。
- `finetune.init_checkpoint` 严格加载完整 EMA，包括原有真实动作编码器、
  不活跃编码器及归一化状态；校验源配置、全部参数名称与形状和 SHA-256。
  不导入源优化器、步数、随机状态或数据位置。Go2 恢复使用
  `training.from_checkpoint`，跳过源初始化。
- 5 条原始训练 episode、2 条测试 episode；沿用 4 Hz 连续片段和
  spacing `0.07610250112606785`。训练只读训练集，关闭测试集评估。
- 种子 0，batch 8，每观测 4 个目标；200 步仅训练真实动作编码器，lr `1e-5`；
  800 步 joint，主干 lr `1e-6`。在线 VAE 编码，VAE 和其他动作编码器冻结。
  EMA `0.99`，AdamW、梯度裁剪与原无调度器设置沿用。
- 每 10 步记录，每 200 步及阶段结束保存。固定最终 `joint_0000800`，不用测试集选模型。

## 评测协议和结果

`prepare` 用 NumPy `default_rng(42)`，按原 episode 排序，每条测试 episode
从具有 4 帧上下文和 8 帧未来的有效窗口无放回取 50 个。索引包含片段名、
当前帧、固定目标偏移 8，不跨片段。评测要求实际加载全部 100 个窗口。
`--eval-windows 10` 改为两条 episode 各 5 个窗口；默认 100 保留。
所有模型共用新索引，不复用旧 100 窗口的评测缓存。
`--reuse-training-from` 校验原训练配置、数据、训练源码和完整 checkpoint 哈希，
仅允许编排脚本、测试及评测协议变化。复用训练保留原 WandB run/group，来源另存
`reused_training/`。后续恢复须继续传入相同参数。
Go2 子进程默认 `TORCH_HOME=/file_system/nas/algorithm/dujun.nie/nwm/cache/torch`，
复用 NAS 上的 LPIPS 依赖；显式环境变量可覆盖。

CEM 参数为 8 步 / 2 秒、80 候选、top-5、1 轮优化、3 次评分、250 步扩散、
重建目标图像 LPIPS，batch 1、候选 microbatch 40。每个窗口重新设种子
`42 + sample_id`，使跳过已完成窗口不改变后续随机采样。

先验仅从训练集有 4 帧上下文的有效 8 步窗口拟合。终点局部位移除以
`8 × spacing`，按原动作范围 `[-2.5,-4]` 到 `[5,4]` 映射到 `[-1,1]`；
第三维为终点相对朝向减去位移方向的环绕角差除以 π。
取均值和总体标准差，标准差下限为 `[0.02,0.02,0.05]`。

`summary.json` 保留原始归一化误差；`summary.md` 输出每条 episode 和总体的
米制 ATE、平移 RPE、终点距离误差、弧度制终点朝向误差。这些是**离线规划误差**，
不代表真机导航成功率。缺失权重或结果明确列在 pending。

## 复现与 WandB

`protocol.json` 保存完整协议、CEM 先验、索引、数据/代码/索引哈希；
`data_files.json` 保存实际输入文件哈希；`sources/` 保存来源和权重哈希；
`source/`、`workspace.patch`、`environment.json` 保存源码、工作区差异及环境版本。
完整配置、日志、checkpoint、逐样本结果分别位于 `configs/`、`logs/`、`train/`、`eval/`。

同一个实验编号的输入不可变。已有结果仅在权重、协议和代码哈希一致时复用；
改变数据、实现或配方时应换实验编号。冒烟和全量必须使用不同编号。
缺少 LatentPT 最终权重不会改变其他组协议，之后可直接补齐。

WandB 项目 `compact-nwm`，group 为实验编号，每种初始化一个 run。
记录损失、各优化器组学习率、`train/joint_phase`（0=warmup，1=joint）和完整配置。
Go2 要求在线初始化成功，否则训练退出。`train/<model>/wandb_state.json` 保存
run ID 和链接，恢复使用 `resume=must`；请保留该文件和 checkpoint。
汇总同时记录 SDK 完成状态、远端状态和 run 链接；远端无法确认时标记 `unverified`。

## 验证

新增 `tests/test_go2_adaptation.py` 覆盖两种权重格式、严格加载拒绝、warmup/joint
参数更新与 EMA、跨阶段恢复、固定窗口、先验数据来源、米制换算、逐样本种子、
WandB ID 恢复及不可变缓存。建议同时运行原有 two-stage、no-pretrain、Go2 数据处理
和导航评测测试。实际 GPU 冒烟及全量运行结果应以保存的日志和汇总为准。
