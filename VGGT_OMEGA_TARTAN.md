# TartanDrive VGGT-Ω Geometry Action 提取

本文是 TartanDrive 离线 geometry action 的可复现 runbook。正式协议只从图像估计
pose/action；`traj_data.pkl` 中的 position/yaw **只允许在提取结束后用于验证**，不得用于
对齐、定尺度或修正训练 cache。

## 固定资产与目录

| 资产 | 固定值 |
| --- | --- |
| 官方代码 | `https://github.com/facebookresearch/vggt-omega.git` |
| 代码 commit | `282ec70363edeff59424bf43731658092fba3d37` |
| HF 模型 | `facebook/VGGT-Omega` |
| HF revision | `ba9db085d6b7349b738fa2e37d198bb4dd077954` |
| HF endpoint | `https://huggingface.co`（gated 下载禁止镜像） |
| checkpoint | `vggt_omega_1b_512.pt` |
| checkpoint SHA-256 | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` |
| checkpoint NAS | `/file_system/nas/algorithm/dujun.nie/models/VGGT-Omega-1B-512/` |
| TartanDrive | `/file_system/nas/algorithm/dujun.nie/nwm/data/tartan` |
| split | `data_splits/tartan_drive/{train,test}/traj_names.txt` |
| 输出 NAS | `/file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/vggt_omega_tartan` |
| 完整配置 | `conf/geometry_action/vggt_omega_tartan.yaml` |

正式输出布局为：

```text
vggt_omega_tartan/
├── manifests/tartan_frames.json
├── raw_pose/tartan_drive/<trajectory>.pt
├── geometry_motion/tartan_drive/<trajectory>.pt
├── logs/rank_*.log
└── validation/report.json
```

`raw_pose` 保留 Omega 的绝对 pose、窗口信息和 provenance，便于重新生成 action；
`geometry_motion` 是 NWM `OfflineMotionStore` 可直接读取的 canonical cache。每条轨迹独立
原子写入，`--resume` 只跳过通过元数据、输入 fingerprint 和完整性校验的文件。
extractor 故意不依赖 Hydra；下面的命令是对审计配置逐项展开，改实验时应同时版本化 YAML 和
实际命令/provenance，不能只改其中一处。

## 权限、许可证与代码

先在 [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) 登录并申请 gated
权重，阅读并接受模型页条款。token 只需 read 权限。不要把 token 写入 YAML、命令行、日志、
shell history 或 manifest；在交互式 shell 中输入：

```bash
read -rsp 'HF token: ' HF_TOKEN
export HF_TOKEN
printf '\n'
export HF_XET_HIGH_PERFORMANCE=1
```

VGGT-Ω 使用 FAIR Noncommercial Research License v1，并受其中 AUP 约束。许可限制包括非商业
用途，AUP 还涉及可能造成人身风险的 transportation technologies/heavy machinery。本文流程只
面向已获项目负责人确认的离线、非部署研究，不授权机器人控制、道路部署或其他用途。每位运行者
仍须自行确认具体使用符合最新官方条款。下载后的 `LICENSE.txt` 与权重一同保存在 NAS。

官方代码已放在 `third_party/vggt_omega`。任何机器开始运行前都应验证，而不是依赖分支名：

```bash
test "$(git -C third_party/vggt_omega rev-parse HEAD)" = \
  282ec70363edeff59424bf43731658092fba3d37
test -z "$(git -C third_party/vggt_omega status --porcelain)"
```

从空目录恢复时：

```bash
git clone https://github.com/facebookresearch/vggt-omega.git third_party/vggt_omega
git -C third_party/vggt_omega checkout --detach \
  282ec70363edeff59424bf43731658092fba3d37
```

主环境当前 NumPy 2.x，而固定版本 Omega 声明 `numpy<2`。为避免污染 NWM 环境，使用隔离环境；
`--system-site-packages` 复用机器已有 CUDA PyTorch，减少安装时间，最终版本必须写入 provenance：

```bash
uv venv --python 3.12 --system-site-packages .venv-vggt-omega
uv pip install --python .venv-vggt-omega/bin/python \
  -r third_party/vggt_omega/requirements.txt
uv pip install --python .venv-vggt-omega/bin/python \
  -e third_party/vggt_omega 'huggingface-hub[hf-xet]==0.36.0' PyYAML
```

## 获取并核验权重

下载器固定 HF commit，在目标 NAS 的同一文件系统内 staging，先检查大文件不是 LFS/Xet pointer，
再计算所有文件 SHA-256。payload 逐文件原子发布，`checkpoint_manifest.json` 最后发布并作为完成
标记；并发调用由文件锁串行化。脚本不会输出或保存 token。服务器可能全局配置
`HF_ENDPOINT=https://hf-mirror.com`，但该镜像的 gated redirect 会丢失 Authorization；下载器在
导入 `huggingface_hub` 前强制使用官方 endpoint，并将其写入 manifest。可用 `--endpoint` 显式
指定 HTTPS origin，但会拒绝已知的 `hf-mirror.com` gated 下载。

```bash
.venv-vggt-omega/bin/python scripts/fetch_vggt_omega.py \
  --expected-weight-sha256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
unset HF_TOKEN
```

断点后可直接重跑。已有文件只有在 manifest、固定 revision、固定代码 commit、文件大小和 SHA-256
全部一致时才会跳过。离线复核不需要 token：

```bash
.venv-vggt-omega/bin/python scripts/fetch_vggt_omega.py \
  --verify-only \
  --expected-weight-sha256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
```

固定 revision 的官方 immutable metadata 已给出上述 ETag/SHA-256，下载器把它作为默认的独立
期望值；即使命令省略该参数也会强制逐字节匹配。显式写在命令中是为了让实验日志自包含。
下载完成后，以 NAS 中的 `checkpoint_manifest.json` 为本次实验的 checkpoint receipt。不要用
已有的原版 `VGGT-1B/model.safetensors` 替代 Omega 权重。

## 坐标、尺度与连续性协议

Omega 对一组帧联合预测，因此 `window_size=0` 会把一条轨迹放在同一 gauge 中，是正式选定
路径。它提供共同坐标系，但没有显式的时序平滑约束；连续性仍须通过相邻增量、加速度、窗口
重叠和 SE(2) composition 检查验证。

正式 `tartandrive_forward_camera` 策略完全 image-only：

- 输入 extrinsic 按 Omega 的 OpenCV world-to-camera 约定转换为 camera-to-world；
- 假设 TartanDrive 图像来自前向相机，使用 `nav_x = cam_z`、`nav_y = -cam_x`；
- yaw 由投影后的相机 forward direction 计算并 wrap 到 `[-pi, pi)`；
- 只保留 `(delta_x, delta_y, delta_yaw)`，丢弃 z/pitch/roll；
- 单目平移尺度按每条轨迹预测的相邻非零 planar step 的 robust median 归一到 1
  `waypoint_spacing_unit`，不读取真实 position/yaw；
- 无 NWM pair 的空/单帧轨迹允许 unused scale=1；有有效训练 pair 却无法估计尺度时立即报错。

这种尺度表示的是图像估计得到的轨迹内相对运动单位，不声称是米。若以后获得独立测量的
camera-to-navigation 外参和 metric scale，应使用 `fixed` policy 并把标定文件及 SHA-256 纳入
provenance；禁止使用 `per_trajectory_gt_sim2` 或任何 GT 拟合作为训练 cache。配置中记录的
TartanDrive `waypoint_spacing=0.72m` 只供数据 provenance/GT 验证报告使用，正式 image-only
提取不会读取它。

正式策略已经由分层 pilot 固定为
`resolution=384 + resize_mode=max_size + bfloat16 + window_size=0`。正式 CLI 保留 `overlap=64`，
以精确匹配已启动任务及 artifact descriptor；`window_size=0` 时它只是一项记录/兼容值，不参与
分窗。最长轨迹 484 帧仍须整轨迹联合推理；不存在自动窗口回退。已测
`window_size=256, overlap=64` 虽然更快，但 strict overlap
validation 失败，**禁止**用于正式 raw pose 或 geometry cache。若整轨迹 OOM，应停止该 rank 并
调查资源/分辨率，不能静默切换窗口。不要在同一正式实验中改变分辨率、尺度策略或窗口参数。

## 构建只读输入 manifest

NAS 小文件遍历较慢。先扫描一次 split，缓存排序后的 trajectory/frame 列表、帧数和 fingerprint；
所有 rank 只读同一 manifest，避免每个进程重复扫描：

```bash
PY=.venv-vggt-omega/bin/python
DATA=/file_system/nas/algorithm/dujun.nie/nwm/data/tartan
SPLITS="$PWD/data_splits/tartan_drive"
OUT=/file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/vggt_omega_tartan

"$PY" scripts/extract_tartandrive_vggt_omega.py manifest \
  --data-root "$DATA" \
  --split-root "$SPLITS" \
  --splits train test \
  --workers 32 \
  --manifest-path "$OUT/manifests/tartan_frames.json"
```

manifest 必须报告 1,251 条轨迹和 62,884 帧（train 1,000/49,687，test 251/13,197）。若不一致，
停止提取并先检查数据或 split；不要用旧数字强行覆盖。

## 单轨迹 smoke 与已固化 pilot receipt

选择已核验的 102 帧 train 轨迹，创建独立 smoke split，不改正式 split：

```bash
SMOKE="$OUT/smoke"
mkdir -p "$SMOKE/splits/smoke"
printf '%s\n' 20210903_heightmaps_10_20210903_257_0 \
  > "$SMOKE/splits/smoke/traj_names.txt"

"$PY" scripts/extract_tartandrive_vggt_omega.py manifest \
  --data-root "$DATA" \
  --split-root "$SMOKE/splits" \
  --splits smoke \
  --workers 1 \
  --manifest-path "$SMOKE/manifest.json"
```

先跑完整轨迹，记录 wall time 和峰值显存：

```bash
/usr/bin/time -v "$PY" scripts/extract_tartandrive_vggt_omega.py extract \
  --third-party-root "$PWD/third_party/vggt_omega" \
  --checkpoint /file_system/nas/algorithm/dujun.nie/models/VGGT-Omega-1B-512/vggt_omega_1b_512.pt \
  --checkpoint-manifest /file_system/nas/algorithm/dujun.nie/models/VGGT-Omega-1B-512/checkpoint_manifest.json \
  --expected-checkpoint-sha256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934 \
  --model-revision ba9db085d6b7349b738fa2e37d198bb4dd077954 \
  --expected-code-revision 282ec70363edeff59424bf43731658092fba3d37 \
  --data-root "$DATA" \
  --manifest-path "$SMOKE/manifest.json" \
  --output-root "$SMOKE/full" \
  --inference-path fast \
  --resolution 384 --resize-mode max_size \
  --window-size 0 --overlap 64 \
  --dtype bfloat16 --device cuda --seed 0 \
  --rank 0 --world-size 1 --shard-cost-power 2.0 --resume --allow-tf32 \
  --alignment-policy tartandrive_forward_camera \
  --degenerate-scale-policy empty_only \
  --translation-unit waypoint_spacing_units \
  --min-offset -64 --max-offset 64 --context-size 4 --len-traj-pred 64 \
  --verify-input-hashes --print-traceback \
  --compare-full-fast-frames 64
```

smoke 仍需检查：

- 图像解码、模型 forward、后处理和写盘各阶段时间；
- frames/s、峰值 `torch.cuda.max_memory_allocated` 和 GPU 利用率；
- 同一 frame 的旋转 geodesic error、相对平移方向误差和尺度一致性；
- 整轨迹 pose/action 连续性；
- canonical 文件 schema、finite、identity/inverse/composition。

选型 receipt 来自同一组 16 条分层轨迹、共 2,569 帧，所有分辨率均使用 `max_size` 和整轨迹：

| 候选 | FPS | 耗时 | strict | GT action cosine | yaw MAE (deg) | yaw sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **384（正式）** | **12.4835** | **205.791s** | **16/16 pass** | 0.981224 | 3.667664 | 0.966233 |
| 512 | 5.8646 | 438.052s | 16/16 pass | 0.984768 | 3.671569 | 0.969788 |

384 在保持全部 strict pass、相近 direction/yaw 指标且 yaw MAE 略优的情况下，吞吐约为 512 的
2.13 倍，因此选为正式设置。320 因 direction quality 明显较差被排除。另测的 256/64 窗口方案
因 strict overlap failure 被排除，速度优势不能覆盖坐标 gauge 不连续。

checkpoint 名称中的 `512` 是模型发布名称，不要求提取输入必须使用 512。后续若要改变 384 或
整轨迹策略，必须建立新的输出 namespace、重跑同等级 pilot 和 strict validation，不能覆盖本 cache。
正式全量前仍要求 smoke 完整、重复运行 fingerprint/结果 hash 一致且 strict validator 通过。

## 多 GPU 全量提取

吞吐最高且最稳的方式是一张 GPU 一个独立进程。每个进程只看到一张卡，按排序后的 trajectory
frame/window 长度估计 attention cost，并使用确定性的 balanced LPT 分片，使长轨迹在 rank 间
尽量均衡；不要用 DataParallel，也不要让两个 rank 写同一条轨迹。

本次正式 artifact 的 provenance 固定如下；这是当前产物的实际运行，不是 8 卡运行：

| 字段 | 正式值 |
|---|---|
| run log | `formal_384_whole_2gpu_20260821T164241Z` |
| actual world size | 2 |
| rank 0 | `CUDA_VISIBLE_DEVICES=2` |
| rank 1 | `CUDA_VISIBLE_DEVICES=3` |

对应的正式 2-GPU 命令为：

```bash
FORMAL_RUN_LOG=formal_384_whole_2gpu_20260821T164241Z
WORLD_SIZE=2
GPU_IDS=(2 3)
mkdir -p "$OUT/logs"
for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$RANK]}" "$PY" \
    scripts/extract_tartandrive_vggt_omega.py extract \
    --third-party-root "$PWD/third_party/vggt_omega" \
    --checkpoint /file_system/nas/algorithm/dujun.nie/models/VGGT-Omega-1B-512/vggt_omega_1b_512.pt \
    --checkpoint-manifest /file_system/nas/algorithm/dujun.nie/models/VGGT-Omega-1B-512/checkpoint_manifest.json \
    --expected-checkpoint-sha256 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934 \
    --model-revision ba9db085d6b7349b738fa2e37d198bb4dd077954 \
    --expected-code-revision 282ec70363edeff59424bf43731658092fba3d37 \
    --data-root "$DATA" \
    --manifest-path "$OUT/manifests/tartan_frames.json" \
    --output-root "$OUT" \
    --inference-path fast \
    --resolution 384 --resize-mode max_size \
    --window-size 0 --overlap 64 \
    --dtype bfloat16 --device cuda --seed 0 \
    --rank "$RANK" --world-size "$WORLD_SIZE" \
    --shard-cost-power 2.0 --resume --allow-tf32 \
    --alignment-policy tartandrive_forward_camera \
    --degenerate-scale-policy empty_only \
    --translation-unit waypoint_spacing_units \
    --min-offset -64 --max-offset 64 --context-size 4 --len-traj-pred 64 \
    --verify-input-hashes --print-traceback \
    --compare-full-fast-frames 0 \
    > "$OUT/logs/${FORMAL_RUN_LOG}_rank_${RANK}.log" 2>&1 &
done
wait
```

`--compare-full-fast-frames 0` 是本次正式 artifact 的实际参数。camera-only fast path 与
full path 的逐 tensor parity 已在独立 smoke/pilot 中用 64 帧验证；正式 rank 不重复执行 dense
heads，以保持最高吞吐。上文 smoke 命令仍应使用 `64`。

可选的通用 8-GPU 启动方式如下，但它是**另一种 sharding 运行**，不描述也不能冒充本次 2-GPU
正式 artifact。只有 8 张卡均已确认可独占时，才将上面命令的启动变量替换为：

```bash
WORLD_SIZE=8
GPU_IDS=(0 1 2 3 4 5 6 7)
# 其余 extractor 参数及循环保持与上面的完整命令一致。
```

任何卡数都必须同时更新 `WORLD_SIZE` 和 `GPU_IDS`，不能只少启动某个 rank。

任何进程中断时，保持相同 `WORLD_SIZE`、manifest、窗口参数和输出目录重跑同一命令即可 resume。
如果改动 `WORLD_SIZE`，稳定 hash/trajectory ownership 会变化，但已完成且校验通过的原子文件仍会
被跳过。正式运行期间不得修改 split、图像、checkpoint 或 manifest。

首次正式运行还应保存完整 receipt：

```bash
mkdir -p "$OUT/provenance"
cp conf/geometry_action/vggt_omega_tartan.yaml "$OUT/provenance/"
.venv-vggt-omega/bin/python -m pip freeze > "$OUT/provenance/pip-freeze.txt"
sha256sum conf/geometry_action/vggt_omega_tartan.yaml \
  data_splits/tartan_drive/train/traj_names.txt \
  data_splits/tartan_drive/test/traj_names.txt \
  > "$OUT/provenance/input-sha256.txt"
git -C third_party/vggt_omega rev-parse HEAD > "$OUT/provenance/vggt-code-commit.txt"
nvidia-smi -q > "$OUT/provenance/nvidia-smi.txt"
```

## 严格验证与交付门槛

全量结束后运行 validator。GT 参数只产生诊断指标；不提供 GT pass/fail threshold 时，GT 指标不
影响退出码，更不会写回 pose/action：

```bash
CUDA_VISIBLE_DEVICES='' "$PY" scripts/validate_tartandrive_geometry.py \
  --raw-pose-root "$OUT/raw_pose" \
  --geometry-root "$OUT/geometry_motion" \
  --data-root "$DATA" \
  --dataset-name tartan_drive \
  --split-files \
    "$SPLITS/train/traj_names.txt" \
    "$SPLITS/test/traj_names.txt" \
  --output "$OUT/validation/report.json" \
  --waypoint-spacing 0.72 \
  --no-resume --strict \
  --verify-frame-hashes --use-gt \
  --no-require-algebra-coverage \
  --flush-every 10 --max-composition-checks 10000 \
  --max-so3-orthogonality 1e-3 --max-so3-det-error 1e-3 \
  --max-identity-translation 1e-4 --max-identity-yaw-deg 1e-3 \
  --max-inverse-translation 1e-3 --max-inverse-yaw-deg 1e-2 \
  --max-composition-translation 1e-3 --max-composition-yaw-deg 1e-2 \
  --max-pose-action-translation 1e-4 --max-pose-action-yaw-deg 1e-3
```

874 条轨迹因长度不足而合法地没有 NWM pair，所以正式验证必须使用
`--no-require-algebra-coverage`；这些轨迹仍会验证 raw pose、源帧 fingerprint 和空 cache schema。
随后运行独立的固定协议审计，防止“元数据自洽但不属于本次正式配置”的 artifact 混入：

```bash
CUDA_VISIBLE_DEVICES='' "$PY" scripts/audit_tartandrive_geometry_provenance.py \
  --raw-pose-root "$OUT/raw_pose" \
  --geometry-root "$OUT/geometry_motion" \
  --split-files \
    "$SPLITS/train/traj_names.txt" \
    "$SPLITS/test/traj_names.txt" \
  --output "$OUT/validation/provenance_receipt.json" \
  --workers 16
```

本次已验证产物的总 receipt 为 `$OUT/provenance/formal_run_receipt.json`；它记录实际 2-rank
命令、模型/输入/报告 hash、源码快照、环境、轨迹和 pair 数量。正式交付要求 strict report、
protocol receipt 与 total receipt 三者都为 `status=pass`。

交付必须同时满足：

1. 所有 split trajectory 都有明确状态；需要 NWM pair 的轨迹不能 missing/failed。
2. checkpoint、代码、manifest、每帧 fingerprint 与记录一致，所有 tensor finite。
3. rotation 为合法 SO(3)，frame 顺序唯一，pose/action 方向为 current-to-goal。
4. identity、inverse、SE(2) composition 和窗口 overlap 检查通过。
5. action schema 为 v1，`motion_type=geometry`、`normalization=raw`、components 顺序固定为
   `(delta_x, delta_y, delta_yaw)`、translation unit 为 `waypoint_spacing_units`。
6. GT 报告明确包含 `only_for_validation=true`，且没有 GT-derived transform/scale 写进 cache。
7. 抽样可视化/统计中没有整体前后颠倒、yaw 符号错误、NaN、接缝尖峰或大批退化轨迹。

验证通过后，NWM geometry 配置指向：

```text
motion_condition.geometry.offline.root=
  /file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/vggt_omega_tartan/geometry_motion
```

`tartan_drive` 在默认 NWM 配置中显式关闭，避免改变历史三数据集 baseline。验证通过后，使用
`nwm_train.sh` 的 `geometry_tartan` variant 启动；launcher 会检查三个 pass receipt 和全部 1,251
条 cache，然后把它作为 geometry proxy 数据加入训练，同时保留三个完全相同的 real-action
anchor 数据集：

```bash
RUN_NOTES=nwm-geometry-tartan \
NWM_VARIANT=real \
NWM_MOTION_VARIANT=geometry_tartan \
BATCH_SIZE=16 \
./nwm_train.sh
```

该 variant 默认使用在线 VAE，因为当前四数据集预计算 posterior cache 中部分 TartanDrive 文件
不可由普通用户读取。修复整个 cache（包括 completion marker、manifest 和 trajectory payload）的
权限后，可显式设置 `USE_PRECOMPUTED_LATENTS=true`。

对应 latent-action 实验应保持 `recon/sacson/scand`、采样比例和训练步数不变，仅把
`tartan_drive:geometry` 改为 `tartan_drive:latent` 并配置 latent cache。这样每个 proxy 实验的
`real_action_adapter` 都会在相同 anchor 数据上得到训练，最终统一 real-action evaluation 才是
可解释的比较。

不要在验证通过前训练。最终 navigation evaluation 仍统一使用真实 navigation action，不从未来帧
或 Omega 输出构造测试 condition。

## 已知限制

- 单目尺度由 image-only 轨迹统计归一化，不是绝对米制，也不保证跨场景物理尺度完全一致。
- “整轨迹联合预测”提供共同 gauge，但不等于显式时间平滑；验证报告决定连续性是否可接受。
- 很多 TartanDrive 轨迹极短；空/单帧状态必须显式记录，不能静默丢失。
- 官方 2026-08-18 notice 指出公开 1B 的 ancestor checkpoint 可能污染其论文部分 benchmark。
  TartanDrive 不属于所列 benchmark，但实验记录仍应保留该 notice，不能引用受影响表格作为本任务
  的准确性证据。
- Omega 输出依赖前向相机假设；如相机安装方向得到新的独立标定，必须版本化新的 policy/cache，
  不得原地覆盖本结果。
