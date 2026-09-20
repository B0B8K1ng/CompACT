# NWM rollout comparison

Figure-4-style line plots for the RECON autoregressive rollout benchmark. Each
metric and FPS setting is a separate figure. The horizontal points are 1, 2, 4,
8, and 16 seconds (`t+1` through `t+16`).

Model mapping:

- **RAE-NWM**: registry key `rae-nwm`, public RAE-NWM weights.
- **NWM**: registry key `nwm-real`, the 200k NWM comparison checkpoint.
- **OpenNWM (180k)**: registry key
  `nwm-latentpt-reset-nwm-real-recipe-180k`, checkpoint `joint_0180000`.

The current plots contain only models with registered rollout results. Missing
models are stated at the foot of each plot; direct-time predictions are not
substituted for rollout results. The benchmark audits provide LPIPS-Alex,
DreamSim, and PSNR. FID—the other metric in the original NWM Figure 4—was not
computed in these local rollout audits and is therefore not plotted.

Regenerate without GPU inference:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run -n nwm \
  python scripts/plot_nwm_rollout_comparison.py
```

The script also writes `rollout_metrics.csv` and `availability.json` so the
plotted values and missing entries can be audited directly.
