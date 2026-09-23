# LingBot-World 2.0 1.3B RECON evaluation

This adapter evaluates only 4-second visual reconstruction on the fixed
500-sample RECON `time.pkl` split. It does not run rollout, navigation, or CEM.

The input is the native 640x480 current frame. RECON poses from the current
frame through `current+16` are interpolated from 4 Hz to 65 poses at 16 Hz and
embedded as level, forward-facing OpenCV camera-to-world matrices. Because the
dataset has no camera calibration, the adapter uses the confirmed virtual
832x480 intrinsics `[fx, fy, cx, cy] = [416, 320, 416, 240]`, corresponding to
a centered 640x480 camera with square pixels and a 90-degree horizontal field
of view. LingBot normalizes translation internally, so the fixed camera height
does not affect its relative camera motion.

The 1.3B causal-fast model runs with `frame_num=65` and `chunk_size=1`. This is
required for an exact 4-second prediction: the official default chunk size of
4 truncates a 65-frame request to 61 decoded frames. The final generated frame
is bilinearly resized to 224x224 and saved as `id_N/4.png`, matching the existing
RECON metric layout. The prompt is fixed and sample seed is `42 + sample_id`.

The runner loads the model once, resumes valid existing PNG files, writes each
PNG atomically, and records the exact model revisions and protocol in
`run_manifest.json`. Per-sample action arrays use a temporary directory that is
removed automatically.

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> conda run --no-capture-output -n wanav \
  python scripts/run_lingbot_recon_eval.py --limit 20
```

After the 20-sample validation, omit `--limit` to resume and finish all 500:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> conda run --no-capture-output -n wanav \
  python scripts/run_lingbot_recon_eval.py
```

Evaluate the completed output with the repository's existing metric script:

```bash
conda run --no-capture-output -n nwm python scripts/evaluate_nwm_predictions.py \
  --gt-dir /file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/gt/recon/time \
  --pred-dir /file_system/nas/algorithm/dujun.nie/nwm/results/lingbot_world_v2_1.3b_causal_fast/recon/time \
  --output /file_system/nas/algorithm/dujun.nie/nwm/results/lingbot_world_v2_1.3b_causal_fast/recon_time_metrics.json \
  --frames 4s:4 --dataset recon --eval-type time \
  --eval-name lingbot-world-v2-1.3b-causal-fast \
  --batch-size 32 --device cuda \
  --dreamsim-cache /file_system/nas/algorithm/dujun.nie/nwm/cache/dreamsim \
  --inference-backend lingbot-world-v2 --sampler causal-fast \
  --sampling-steps 4 --seed 42
```
