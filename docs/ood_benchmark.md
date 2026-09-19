# Fixed out-of-domain NWM benchmark

The registered `ood_direct_4s_v1` protocol evaluates four held-out datasets:

- `planetary_rover`: real Mars Perseverance and CE-4/Yutu-2 imagery and pose
  trajectories.
- `unitree_go2`: the seven local Unitree Go2 recordings.
- `tum_rgbd`: a pinned, selectively downloaded subset of
  [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).
- `uzh_fpv`: a pinned [UZH-FPV](https://fpv.ifi.uzh.ch/datasets/) subset with
  official continuous-time poses.

All generated data lives under
`/file_system/nas/algorithm/dujun.nie/nwm/data`. The source reports record URLs,
revisions, file hashes, pose interpolation, image selection, and metric scale.
The workspace holds only the small split manifests next to the evaluation code.

## Data contract

Every dataset has exactly 500 deterministic direct-prediction windows and 100
deterministic navigation windows. A prediction window contains four context
frames and sixteen future frames. Unitree, TUM, and UZH use nearest real camera
frames on a 4 Hz grid, so frame 16 is physically 4 seconds in the future.
Planetary Rover has no fixed-rate timestamps; its directly comparable input
shape uses a 16-waypoint spatial horizon and is explicitly marked as such in
the registry and result table.

Images are never interpolated. Actions come from real pose trajectories
(including the documented reversible Moon scale transform); quaternion poses
are reduced to planar yaw, and the generated `traj_data.pkl` is accepted by
both direct prediction and CEM navigation loaders.
The splitter deterministically chooses the closest available 4-second motion
distribution to Go Stanford, but never rescales a measured trajectory merely to
improve that match. Each report records both distributions and their residual
quantile error; this matters especially for the faster UZH drone sequences.

Validate all data and pinned splits with:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python \
  scripts/prepare_ood_benchmarks.py validate
```

## Direct prediction

The benchmark uses one-shot prediction at only the nominal 4-second/16-step
horizon, 500 samples per dataset, and seed 0. Any positive GPU count is valid. NWM
models use 250 DDPM steps; RAE-NWM uses its official 50-step Euler ODE sampler.
Checkpoint, report, prediction split, and navigation split SHA-256 values are
checked before inference starts.

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python \
  scripts/run_nwm_benchmark.py \
  --models nwm-real,rae-nwm,nwm-latentpt-reset-nwm-real-recipe-180k \
  --metrics direct \
  --datasets planetary_rover,unitree_go2,tum_rgbd,uzh_fpv \
  --gpus 0,1,2,3 \
  --raenwm-python /file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python
```

Predictions, immutable run inputs, metric audits, the JSON registry, Markdown
summary, and LaTeX table are written below
`/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark`. New runs use
the unified `direct_4s_v1` protocol; older measured rows under
`protocol_runs/ood_direct_4s_v1` remain readable for provenance.

When only one physical GPU is available, run the same protocol directly:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run \
  --no-capture-output -n nwm \
  python scripts/run_nwm_benchmark.py \
  --models nwm-real,rae-nwm,nwm-latentpt-reset-nwm-real-recipe-180k \
  --metrics direct \
  --datasets planetary_rover,unitree_go2,tum_rgbd,uzh_fpv \
  --gpus 7 \
  --raenwm-python /file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python
```

Exact strided sharding and sample-ID keyed stochastic noise make one- and
multi-GPU execution comparable up to floating-point kernel differences.

## Navigation support

All four datasets are also registered in `navigation_cem80_v1` with their fixed
100-window `navigation_eval.pkl` and measured waypoint spacing. For example,
the three requested models can be evaluated with the same CEM N=80, K=5,
H=8, three-repeat protocol using:

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python \
  scripts/run_nwm_benchmark.py \
  --models nwm-real,rae-nwm,nwm-latentpt-reset-nwm-real-recipe-180k \
  --metrics navigation \
  --datasets planetary_rover,unitree_go2,tum_rgbd,uzh_fpv \
  --gpus 0,1,2,3 \
  --planning-microbatch-size 80 \
  --raenwm-planning-steps 50
```

OOD navigation uses one shared proposal prior. Candidate and world-model noise
is keyed by sample/candidate/repetition identity, so interrupted runs can resume
with a different GPU count or microbatch size.
