# NavAnywhere latent-action 收尾

在能读取、写入现有 cache 的环境运行（此前部分文件由 root 创建，权限为 600）。

```bash
cd /file_system/vepfs/algorithm/dujun.nie/code/CompACT
GPU_IDS=0,1,2,3,4,5,6,7 \
./finish_navanywhere_latent_actions_8gpu.sh
```

脚本直接在前台运行，进度输出到终端。运行环境固定为 DreamDojo 的 `.venv/bin/python`，可通过 `VENV_PYTHON` 修改。使用 `./finish_navanywhere_latent_actions_8gpu.sh --help` 查看路径等可选参数。

需要后台运行并保存日志时：

```bash
FINISH_LOG=/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_nav1_latent_actions/finish_$(date -u +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname -- "$FINISH_LOG")"
nohup env GPU_IDS=0,1,2,3,4,5,6,7 ./finish_navanywhere_latent_actions_8gpu.sh > "$FINISH_LOG" 2>&1 &
echo "PID=$! LOG=$FINISH_LOG"
tail -f "$FINISH_LOG"
```

流程：

1. 检查所有 source 目录的写权限；检测当前节点仍在运行的旧补提进程，避免重复启动。
2. 使用 24 个 CPU 进程检查文件内容。默认复用 `finalize_20260908` 中的已通过记录，确认大小和 mtime 未变；未通过或改变的文件重新完整读取。加 `--fresh-audit` 可重新读取所有文件。
3. 根据相同 recipe、训练 pair plan、checkpoint 和提取设置，仅补提缺失或不合格文件。默认八个独立 GPU worker 从队列领取 4096 个配对的小块，没有 NCCL 汇总。小块边界保持原 batch=64 的组成和顺序，保留 bf16 推理及 float32 的 32 维 z_mu。
4. 校验小块、合并轨迹、原子替换需修复文件，确认覆盖全部 recipe 和计划配对后，生成各 source manifest、metadata.json 和 `_SUCCESS.json`。

允许共享 GPU，不要求利用率为零；默认每张卡至少有 6000 MiB 空闲显存。共享任务的显存变化仍可能导致 OOM，已保存的小块会保留，可重新执行相同命令续跑。有效的已完成轨迹和小块都会复用，损坏的小块会重新计算。

检查内容包括可读取性、格式与元数据、frame inventory、训练配对覆盖、float32 `[N, 32]` 和 NaN/Inf。已有缓存不重新扫描所有原始 JPEG；待补提轨迹会扫描原始图像。脚本会为当前身份拥有的缓存文件补充组读取权限，方便后续训练使用。

默认路径：

- cache：`/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/navanywhere_nav1_pixel_action_step100000`
- 检查报告：`/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_nav1_latent_actions/finalize_20260908/final_summary.json`
- 临时完整轨迹：cache 同级的 `navanywhere_nav1_pixel_action_step100000_repairs_20260908`
- 可续跑小块：cache 同级的 `navanywhere_nav1_pixel_action_step100000_chunks_20260908`

计划固定为 seed=20260901、训练 world_size=8、每卡 batch=16、200000 steps 的已有 pair plan。成功应覆盖 67875 条轨迹和 23802046 个计划配对。允许已有完整局部配对缓存包含更多配对。

脚本不会自动停止其他运行中的任务。如果提示旧补提仍在运行，先让其完成，或由你明确停止相应实验后再运行。
