# NavAnywhere Stage-1 pretraining

This is the focused runbook for NavAnywhere-only NWM-TimePT, NWM-GeoPT,
NWM-IDMPT, and NWM-LatentPT pretraining. The default path reads precomputed
SD-VAE posterior statistics, keeps the full signed `[-64, 64]` target-frame
range, and writes training curves to Weights & Biases.

## 1. One-time VAE posterior precompute

The eight-GPU launcher activates the `nwm` conda environment, checks GPU and
disk capacity, creates or validates the sampling recipe, and starts a durable
`codex-exp` job. It is resumable at trajectory granularity.

```bash
./precompute_navanywhere_vae_latents_8gpu.sh
```

The default data root is
`/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere`; cache, recipe, and
logs are written below `/file_system/nas/algorithm/dujun.nie/nwm/compact`.
Override any setting in front of the command:

```bash
EXPERIMENT_NAME=nav-vae-v1 \
GPU_IDS=0,1,2,3,4,5,6,7 \
VAE_BATCH_SIZE=128 LOADER_THREADS=8 \
VAE_LATENT_ROOT=/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/nav-vae-v1 \
./precompute_navanywhere_vae_latents_8gpu.sh
```

Both launchers reject a selected GPU above 20% utilization or below 30 GiB
free memory by default, so they do not accidentally share another experiment.
Choose another `GPU_IDS`, tune `MAX_GPU_UTILIZATION`/`MIN_FREE_GPU_MB`, or set
`REQUIRE_IDLE_GPUS=0` only when deliberate GPU sharing is safe.

Monitor the detached job with the experiment name printed by the launcher:

```bash
codex-exp status nav-vae-v1
codex-exp logs nav-vae-v1
```

Each shard is `root/{source_id}/{trajectory_id}.pt` and contains exact
`frame_indices`, `posterior_mean`, and `posterior_logvar`. The posterior tensors
are BF16 `[num_frames, 4, 28, 28]` for the default 224-pixel SD-VAE transform.
Source manifests, global metadata, tensor/file fingerprints, and `_SUCCESS.json`
are validated before training. Invalid or incomplete caches fail closed; the
trainer does not silently decode JPEGs or invoke the VAE when cache use is
enabled. The precompute `VAE_BATCH_SIZE` is part of cache provenance but does
not constrain the later training batch size: cached posterior statistics are
stored per frame. In cached mode the trainer also does not load the VAE weights
onto each training GPU; it retains only the recorded posterior scaling factor.

## 2. Coverage-oriented sampling recipe

The recipe is a compact, versioned JSON file, not a huge expanded pair list.
It freezes the sorted `(source_id, trajectory_id, frame_index)` inventory,
seed, epoch size, offset range, four signed offset strata, and the algorithms
used to map `(epoch, logical_sample_index)` to frames.

Observation sampling is balanced in three levels:

1. Sources are sampled uniformly, so a large source cannot drown out a small
   source.
2. Trajectories are sampled uniformly within each source, so a long trajectory
   cannot drown out shorter trajectories.
3. Current-frame positions use a deterministic coprime affine cycle, which
   visits every valid observation in a trajectory before repeating.

For four goals per observation, one goal is drawn from each signed stratum:
`[-64,-9]`, `[-8,-1]`, `[0,8]`, and `[9,64]`. Near trajectory boundaries, an
empty stratum falls back to another valid offset; target indices are never
clipped. This deliberately covers past/future and local/long-range motion.

The default epoch length equals the number of usable observations in the
frozen inventory. `SAMPLES_PER_EPOCH` may be increased for more balanced
exposure per epoch, but it must be chosen when the recipe is first created.
An existing recipe is immutable: launchers validate it instead of rewriting it.

TimePT, GeoPT, IDMPT, and LatentPT use exactly the same current/target frame
pairs when all of the following stay fixed:

- the same recipe file and SHA-256;
- the same `SAMPLING_SEED` and recipe epoch size;
- the same training epoch/cursor and distributed sampler contract.

Action mode is intentionally absent from the recipe. Geo/IDM/latent proxy
lookup occurs only after a pair has been selected. Proxy availability therefore
cannot change which video samples are trained. For exact resume ordering, also
keep world size, per-GPU batch size, worker count, and dataset configuration
unchanged; checkpoints validate that resume fingerprint.

The full sampling interval remains `[-64,64]`. GeoPT/IDMPT/LatentPT use a
proxy only when `abs(frame_offset) <= 8` and the exact cached record is valid.
Long-range pairs remain in the diffusion loss and use only diffusion-timestep
plus relative-time conditioning.

To create the recipe alone:

```bash
conda run -n nwm python scripts/build_navanywhere_sampling_recipe.py \
  --root /file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere \
  --output /file_system/nas/algorithm/dujun.nie/nwm/compact/recipes/nav-balanced.json \
  --seed 20260901 --context-size 4 --goals-per-obs 4
```

## 3. One-command TimePT training

After the VAE cache is complete, the default command launches TimePT on eight
GPUs as a durable experiment. It activates the conda environment, configures
CUDA/NCCL/Hugging Face/PyTorch/W&B cache variables, checks capacity and W&B
authentication, and uses the precomputed posterior cache by default.

```bash
./run_navanywhere_stage1.sh
```

A practical named launch is:

```bash
EXPERIMENT_NAME=nwm-timept-nav-v1 \
WANDB_PROJECT=compact-nwm-navanywhere \
WANDB_RUN_NAME=nwm-timept-nav-v1 \
GPU_IDS=0,1,2,3,4,5,6,7 \
MAX_TRAIN_STEPS=200000 \
./run_navanywhere_stage1.sh
```

W&B online logging is enabled by default. Authenticate once with `wandb login`
or export `WANDB_API_KEY`. Use `WANDB_MODE=offline` only for local debugging.
The run records losses, optimizer-group learning rates, proxy statistics,
recipe path/SHA/summary, and normal training metadata.

Important experiment knobs are together near the top of
`run_navanywhere_stage1.sh`; all can also be environment overrides:

```text
STAGE1_MODE, SAMPLING_SEED, SAMPLES_PER_EPOCH, SAMPLING_RECIPE
VAE_LATENT_ROOT, GPU_IDS, NPROC, BATCH_SIZE, NUM_WORKERS
MAX_TRAIN_STEPS, LEARNING_RATE, WEIGHT_DECAY, LOG_EVERY, CKPT_EVERY
WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_NAME, RESULTS_DIR
REQUIRE_IDLE_GPUS, MAX_GPU_UTILIZATION, MIN_FREE_GPU_MB
```

Resume from an existing checkpoint without changing the sampling contract:

```bash
EXPERIMENT_NAME=nwm-timept-nav-v1-resume \
WANDB_RUN_NAME=nwm-timept-nav-v1-resume \
RESUME_CHECKPOINT=/path/to/checkpoints/timept_step_50000.pt \
./run_navanywhere_stage1.sh
```

## 4. Identical GeoPT, IDMPT, and LatentPT sampling

Point every run at the recipe and VAE cache used by TimePT. Only the mode and
its proxy root change:

```bash
STAGE1_MODE=geopt EXPERIMENT_NAME=nwm-geopt-nav-v1 \
SAMPLING_RECIPE=/file_system/nas/algorithm/dujun.nie/nwm/compact/recipes/nav-balanced.json \
GEOMETRY_PROXY_ROOT=/path/to/geometry-proxy \
./run_navanywhere_stage1.sh

STAGE1_MODE=idmpt EXPERIMENT_NAME=nwm-idmpt-nav-v1 \
SAMPLING_RECIPE=/file_system/nas/algorithm/dujun.nie/nwm/compact/recipes/nav-balanced.json \
IDM_PROXY_ROOT=/path/to/idm-proxy \
./run_navanywhere_stage1.sh

STAGE1_MODE=latentpt EXPERIMENT_NAME=nwm-latentpt-nav-v1 \
SAMPLING_RECIPE=/file_system/nas/algorithm/dujun.nie/nwm/compact/recipes/nav-balanced.json \
LATENT_PROXY_ROOT=/path/to/dreamdojo-latent-proxy \
./run_navanywhere_stage1.sh
```

Proxy keys are the exact tuple `(source_id, trajectory_id,
current_frame_index, target_frame_index)`. Batch index and random sampler index
are never used as cache keys. The mode launchers refuse a missing proxy root,
and strict proxy loading provides the full sample key in any local-pair error.

## 5. Attached/debug launch

The default launch detaches because full preprocessing and training exceed five
minutes. For a command preview without data, GPUs, or W&B:

```bash
DRY_RUN=1 WANDB_MODE=offline ./run_navanywhere_stage1.sh
```

For a deliberately short attached debug run after the real cache exists:

```bash
DETACH=0 WANDB_MODE=offline GPU_IDS=0 NPROC=1 \
MAX_TRAIN_STEPS=3 NUM_WORKERS=0 ./run_navanywhere_stage1.sh
```
