# Reproducible NWM benchmark pipeline

The unified runner supports the following canonical datasets:

`recon,scand,huron,tartan_drive,go_stanford,planetary_rover,unitree_go2,tum_rgbd,uzh_fpv`

HuRoN deliberately maps to the historical `sacson` data/split directory and
TartanDrive maps to the historical `tartan` data directory. The public names
above are always used in commands and result records.

## Fixed protocols

- `direct_4s_v1`: 500 fixed windows, four context frames, direct prediction at
  4 seconds, LPIPS-Alex/DreamSim/PSNR. NWM uses 250-step DDPM; RAE-NWM uses its
  official 50-step Euler ODE sampler.
- `rollout_v1`: all 150 fixed windows, autoregressive 1 fps and 4 fps rollout,
  measured at 1/2/4/8/16 seconds with the same image metrics. The former
  10-window Go Stanford subset is not a registered protocol.
- `navigation_cem80_v1`: 100 fixed windows, CEM N=80/K=5, three stochastic
  repetitions, one optimization step, horizon eight, LPIPS-Alex cost, and
  ATE/RPE/final-position/final-yaw metrics.

Every split is checked against its registered SHA-256 before a run. Work is
sharded as `split_position[rank::world_size]` without padding. Initial noise,
every DDPM step noise, rollout-step noise, and navigation candidate/repetition
noise are derived from the protocol seed and semantic sample ID. Consequently,
changing GPU count or batch/microbatch size preserves the evaluated samples and
random streams; normal floating-point kernel differences are allowed.

The fixed HuRoN splits are built from the locally available public SACSoN
trajectories. Historical entries that refer to unavailable processed chunks are
replaced deterministically using trajectory coverage and evenly spaced movement
distance quantiles: 500 windows at 4 seconds, 150 rollout windows, and 100
navigation windows. Rebuild or verify them with
`scripts/prepare_nwm_benchmark_splits.py --repair-huron [--check]`.

## Commands

Run any model/dataset/metric selection on any positive number of GPUs:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run \
  --no-capture-output -n nwm \
  python scripts/run_nwm_benchmark.py \
  --models nwm-release,rae-nwm \
  --metrics direct,rollout,navigation \
  --datasets go_stanford,unitree_go2 \
  --gpus 0,1 \
  --batch-size 16 \
  --planning-microbatch-size 40 \
  --raenwm-planning-steps 50
```

Use `--dry-run` to validate selections and print every command without starting
GPU work. Omit `--batch-size` to use 64 per GPU for local NWM and 16 per GPU for
RAE-NWM. Prediction inference loads each model once per selected metric type
and processes all selected datasets before metric aggregation. Completed sample
artifacts and audits are reused unless `--force` is given.

Outputs, manifests, audits, and the file-locked registry live under
`/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark` by default.
