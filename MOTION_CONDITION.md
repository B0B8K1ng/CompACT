# 统一 Motion Condition

本文说明连续 NWM（CDiT）路径的 motion condition 数据接口。框架支持
`real`、`geometry`、`latent` 和 `none`，不改变 diffusion backbone、训练目标、
optimizer 或 scheduler。

## 条件与 batch 接口

每个 dataset sample 是一个字典：

```python
{
    "video": FloatTensor[T, C, H, W],  # 预计算 VAE 模式下换成 posterior_mean/logvar
    "k": FloatTensor[G],               # G 个 goal 的 temporal condition
    "motion_type": "real",            # real | geometry | latent | none
    "motion": FloatTensor[G, D],        # none 时必须省略此字段
}
```

`D` 分别由 `real_dim`、`geometry_dim` 和 `latent_dim` 指定。`none` 不是零
action：sample 中没有 `motion`，也不会产生 null motion embedding。
为保持原 NWM 语义，当前 `k` 仍是 `goal_offset / 128.0`，本次修改没有改变 temporal
condition 的取值或 embedding。

混合 batch 由 `motion_condition_collate` 按类型分组，不会把不同维度 padding 到
同一长度。例如一个 real、一个 latent、一个 none sample 会得到：

```python
{
    "video": FloatTensor[B, T, C, H, W],
    "k": FloatTensor[B, G],
    "motion_type": ["real", "latent", "none"],
    "motion": {
        "real": {
            "sample_indices": LongTensor[N_real],
            "values": FloatTensor[N_real, G, 3],
        },
        "latent": {
            "sample_indices": LongTensor[N_latent],
            "values": FloatTensor[N_latent, G, latent_dim],
        },
    },
}
```

进入 CDiT 前，`flatten_motion_groups` 将上述分组转换为模型接口：

```python
motion = {
    "real": {
        "indices": LongTensor[N_real * G],
        "values": FloatTensor[N_real * G, 3],
    },
    "latent": {
        "indices": LongTensor[N_latent * G],
        "values": FloatTensor[N_latent * G, latent_dim],
    },
}
```

`indices` 指向展平后的 `B * G` 行，规则为 `sample_index * G + goal_index`。
`none` 行不在 `motion` 中。模型先计算

```text
base_condition = diffusion_timestep_embedding + temporal_k_embedding
```

再只对分组中存在的行加上对应独立 adapter 的输出。因此 `none` 行严格保持
`base_condition`。real、geometry、latent adapter 均为
`Linear -> SiLU -> Linear -> LayerNorm`，输入维度独立、输出均为 CDiT condition
dimension；real/geometry 在 adapter 前做连续值归一化，latent 使用无仿射
LayerNorm。

## 离线 motion 文件

geometry 和 latent extractor 应在本项目外离线运行。默认每条 trajectory 一个文件：

```text
<offline.root>/<dataset_name>/<trajectory_name>.pt
```

也支持 `.npz`（将 `file_pattern` 后缀改成 `.npz`）。两种格式使用同一 canonical
schema：

```python
{
    "schema_version": 1,
    "motion_type": "geometry",       # geometry | latent
    "dataset_name": "recon",
    "trajectory_name": "trajectory_001",
    "pair_direction": "current_to_goal",
    "normalization": "raw",          # 尚未执行 adapter 的 normalization
    # 以下四项是 geometry 文件的必需字段；latent 文件省略：
    "coordinate_frame": "current_navigation_frame",
    "translation_unit": "waypoint_spacing_units", # dataset.normalize=false 时为 meters
    "yaw_unit": "radians",
    "components": ["delta_x", "delta_y", "delta_yaw"],
    "frame_pairs": LongTensor[N, 2], # 每行 [current_frame, goal_frame]
    "motion": FloatTensor[N, D],
}
```

- `frame_pairs[n]` 的方向固定为 **current -> goal**，不能反向解释；pair 必须唯一。
- `motion[n]` 与同一行 pair 对齐、值必须有限，`D` 必须等于相应 config 维度。
- 应导出训练索引可能采到的全部 pair，包括数据集允许的反向时间 goal。
- `.pt` 保存普通字典；`.npz` 必须能以 `allow_pickle=False` 读取，字符串元数据使用
  标量字符串数组。
- canonical 文件应始终写入 common 字段；geometry 还必须写入上述坐标/单位字段。
  `strict_metadata: true` 会校验 schema、方向、raw 标记、motion type，以及 geometry
  的坐标系、单位和分量顺序；存在的 dataset/trajectory 名称也会校验。

### Geometry extractor 与坐标约定

截至 2026-08-21，本项目已接入 [VGGT-Ω 官方实现](https://github.com/facebookresearch/vggt-omega)
作为可替换的离线 camera-pose extractor；固定版本、权重、TartanDrive 提取和验证流程见
`VGGT_OMEGA_TARTAN.md`。需要长序列 VO/SLAM 一致性时，可另行评估
[MASt3R-SLAM 官方实现](https://github.com/rmurai0610/MASt3R-SLAM)；当前未集成
MASt3R-SLAM，也不会在 NWM 训练时加载或训练任何 pose extractor。
VGGT-Ω 官方仓库同时提示其公开 1B checkpoint 的部分 benchmark 可能受祖先 checkpoint
污染影响；它仍可用于与这些 benchmark 无关的下游提取，但在问题调查完成前不应把相关
表格数值作为模型选择依据。

extractor 的输出必须先转换到与 real action 相同的 navigation 表示，再写盘：

1. 明确外部模型 pose/extrinsic 的方向，把相对变换统一为 current -> goal。
2. 使用相机到机器人/navigation frame 的标定，将位移和旋转表达在**当前帧的
   navigation 局部坐标系**。
3. 仅保留 `(delta_x, delta_y, delta_yaw)`；丢弃 `delta_z`、pitch、roll，并将 yaw
   wrap 到与 real action 相同的角度范围。
4. 将平移尺度对齐到 real action。当前 dataset 启用 `normalize` 时，real action 的
   `x/y` 会除以各数据集的 `metric_waypoint_spacing`，geometry cache 也必须采用同一
   单位。写盘值是 adapter 前的 action，不要提前套用 config 中的 min-max normalization。

单目 relative pose 存在尺度歧义。必须用标定尺度、已知基线、里程计或可靠的序列级
尺度估计完成对齐；不要对每个 frame pair 独立缩放，否则 geometry 与 real action 不再
可比较。

### DreamDojo LAM

DreamDojo LAM 只作为外部离线 extractor：在独立环境中为每个 current -> goal pair
导出 latent，将 `motion_type` 写为 `latent`，并使用上面的同一 schema。项目不包含
DreamDojo 代码、checkpoint 或训练时推理图；这里只读取 `[N, latent_dim]`，维度由
`motion_condition.latent.latent_dim` 指定。navigation evaluation 不得从未来帧提取
latent action。

## Hydra 配置

默认配置位于 `conf/motion_condition/default.yaml`，由 `conf/nwm.yaml` 引入：

```yaml
motion_condition:
  enabled: true
  available_types: [real, geometry, latent, none]
  train_types: [real, latent]
  # 示例：两个公共 real anchor datasets + 一个 latent proxy dataset
  dataset_motion_types:
    recon: real
    sacson: real
    scand: latent
  sampling_strategy: round_robin
  eval_type: real
  balance_parameter_count: true
  parameter_count_reference_dim: 3

  real:
    real_dim: 3
    normalization:
      mode: minmax
      min: [-2.5, -4.0, -3.141592653589793]
      max: [5.0, 4.0, 3.141592653589793]

  geometry:
    geometry_dim: 3
    normalization: {mode: minmax, min: [-2.5, -4.0, -3.141592653589793], max: [5.0, 4.0, 3.141592653589793]}
    offline:
      root: /abs/path/to/geometry_cache
      file_pattern: "{dataset_name}/{trajectory_name}.pt"
      pairs_key: frame_pairs
      values_key: motion
      cache_size: 8
      strict_metadata: true

  latent:
    latent_dim: 32
    normalization: {mode: layer_norm}
    offline:
      root: /abs/path/to/lam_cache
      file_pattern: "{dataset_name}/{trajectory_name}.pt"
      pairs_key: frame_pairs
      values_key: motion
      cache_size: 8
      strict_metadata: true
```

`available_types` 控制 checkpoint 中可用的 adapter；为保证最终统一 real-action
evaluation，应保留 `real`。`train_types` 必须是其子集，控制训练 sample 使用哪些
motion type。推荐用 `dataset_motion_types` 把每个训练 dataset 固定到一种 condition；mapping
非空时必须完整覆盖所有训练 dataset，且实际使用的类型必须与 `train_types` 一致。
`ConcatDataset + DistributedSampler` 会把这些 dataset 打乱后共同训练，自定义 collate
允许同一 batch 出现 real、proxy 和 none。`dataset_motion_types: null` 时才回退到原来的 sample
index `round_robin`，主要用于接口实验。当前采样概率仍与各 dataset 长度成比例；严格公平
实验应保持 dataset、采样比例、总 step 和 batch size 一致。

`balance_parameter_count` 会微调各 adapter 的内部宽度，使不同输入维度的参数量尽可能
接近；也可用可选的 `adapter_hidden_dim` 设置基准内部宽度。

旧实验目录的 `.hydra/config.yaml` 不含 `motion_condition` 时，模型仍实例化原始
`y_embedder`，state-dict key 与初始化保持不变，可继续 strict load。不能把旧 checkpoint
直接配上新 config 当成 adapter checkpoint；两种参数结构不同，会明确 strict-load 失败。
如需迁移，必须另写显式、可审计的 adapter 初始化/权重转换流程。

> 实验解释注意：如果每个 proxy 实验都包含相同的公共 real datasets，且它们在
> `dataset_motion_types` 中标成 `real`，`real_action_adapter` 会在所有模型中得到训练，最终统一
> real-action evaluation 不再经过随机 adapter。仍应固定公共 real dataset、采样权重和
> 训练步数。如果配置完全不含 real 训练样本，代码会继续告警，此时原来的随机 real
> adapter 问题仍然存在。

## 运行示例

先设置导航数据路径：

```bash
export NWM_DATA_ROOT=/abs/path/to/nwm/data
```

以下命令保持默认 `available_types=[real,geometry,latent,none]` 和
`motion_condition.eval_type=real`。下面以 `recon/sacson` 为公共 real anchors、`scand`
为被替换 condition 的 proxy dataset；请按实际 dataset 名称调整。按实际 GPU 数量修改
`--nproc`。

NWM-real：

```bash
bash scripts/train.sh --nproc=8 -- \
  --config-name nwm \
  'motion_condition.train_types=[real]' \
  'motion_condition.dataset_motion_types={recon:real,sacson:real,scand:real}' \
  training.notes=nwm-real
```

NWM-geometry：

```bash
bash scripts/train.sh --nproc=8 -- \
  --config-name nwm \
  'motion_condition.train_types=[real,geometry]' \
  'motion_condition.dataset_motion_types={recon:real,sacson:real,scand:geometry}' \
  motion_condition.geometry.offline.root=/abs/path/to/geometry_cache \
  training.notes=nwm-geometry
```

NWM-latent（示例维度 32，其他维度直接改 config）：

```bash
bash scripts/train.sh --nproc=8 -- \
  --config-name nwm \
  'motion_condition.train_types=[real,latent]' \
  'motion_condition.dataset_motion_types={recon:real,sacson:real,scand:latent}' \
  motion_condition.latent.latent_dim=32 \
  motion_condition.latent.offline.root=/abs/path/to/lam_cache \
  training.notes=nwm-latent
```

当前 navigation LAM pixel 最后一步权重实验使用独立入口，避免与旧的
`step_125000/offset64` cache 混用：

```bash
NWM_VARIANT=real NWM_MOTION_VARIANT=latent_tartan_scand ./nwm_train.sh
```

该变体固定读取 `step_100000_offset8` 下的 TartanDrive 与 SCAND latent cache，且设置
`motion_condition.latent.max_frame_offset=8`。每个 goal 独立判断：frame offset 在闭区间
`[-8, 8]` 内才进入 latent adapter；超出范围的 goal 仍计算图像生成损失，但没有动作标签。

NWM-blank：

```bash
bash scripts/train.sh --nproc=8 -- \
  --config-name nwm \
  'motion_condition.train_types=[real,none]' \
  'motion_condition.dataset_motion_types={recon:real,sacson:real,scand:none}' \
  training.notes=nwm-blank
```

不按 dataset 固定类型的四类混合接口压力测试（正式公平实验不建议这样分配）：

```bash
bash scripts/train.sh --nproc=8 -- \
  --config-name nwm \
  'motion_condition.train_types=[real,latent,geometry,none]' \
  motion_condition.dataset_motion_types=null \
  motion_condition.geometry.offline.root=/abs/path/to/geometry_cache \
  motion_condition.latent.offline.root=/abs/path/to/lam_cache \
  training.notes=nwm-mixed
```

最终 navigation inference 对四类 checkpoint 都传入数据集的真实 navigation action；
`isolated_nwm_infer.py` 的 navigation wrapper 默认将 `delta` 路由为 `real`，不读取
geometry/latent cache，也不使用未来帧：

```bash
torchrun --standalone --nproc-per-node=1 isolated_nwm_infer.py \
  exp_dir=/abs/path/to/nwm-run \
  ckp=0100000 \
  eval_type=time \
  'datasets_to_eval=[recon,scand,sacson]'
```

规划评测同样通过 real-action 路径：

```bash
bash scripts/plan.sh --nproc=1 -- \
  exp_dir=/abs/path/to/nwm-run \
  ckp=0100000
```
