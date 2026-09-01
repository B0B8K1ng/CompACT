# CompACT CDiT-B + SD-VAE baseline on NAS data

The complete Hydra configuration is `conf/nwm.yaml`; the launcher is
`nwm_train.sh`. This is the continuous SD-VAE baseline reported as the NWM
reproduction in CompACT, rather than the default discrete CompactTok model.

## Precompute SD-VAE posteriors

The accelerated path stores the unclipped posterior mean and the Diffusers-
clamped log variance for every existing training frame. It does **not** store a
single sampled latent: training still draws fresh posterior noise on every
access. The cache is BF16 and was encoded with fixed 128-image VAE batches to
match the paper launch (`16 observations/GPU x 8 images/observation`).

Run the full extractor on an otherwise idle eight-GPU machine:

```bash
./nwm_precompute_latents.sh
```

The defaults are:

- GPUs `0,1,2,3,4,5,6,7`;
- output
  `/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/vae_latents_sd_vae_ft_ema_224`;
- BF16 posterior mean/logvar;
- fixed VAE batch 128;
- resumable trajectory-level atomic files;
- full structural, finite-value, source-coverage, and direct-VAE semantic
  verification before `_SUCCESS.json` is written.

Useful overrides include `GPU_IDS`, `NPROC`, `LOADER_THREADS`,
`VERIFY_SAMPLES`, `OVERWRITE`, and `NWM_LATENT_ROOT`. Do not change
`VAE_BATCH_SIZE=128`, `COMPUTE_DTYPE=bfloat16`, or
`STORAGE_DTYPE=bfloat16` for the paper-compatible cache.

Verify an already completed cache on one GPU:

```bash
VERIFY_ONLY=1 \
GPU_IDS=0 \
VERIFY_SAMPLES=96 \
./nwm_precompute_latents.sh
```

The current NAS population should finish with 12,229 trajectories and 754,701
frames. The 173 unavailable SACSoN/HuRoN training trajectories are recorded in
metadata and are not presented as cached data.

## Training smoke tests

After the full cache has produced `_SUCCESS.json`, validate the actual cached
training path for five optimizer steps:

```bash
GPU_IDS=0 \
NPROC=1 \
BATCH_SIZE=16 \
USE_PRECOMPUTED_LATENTS=true \
NUM_WORKERS=4 \
MAX_TRAIN_STEPS=5 \
EPOCHS=1 \
EVAL_AT_FIRST_STEP=false \
LOG_EVERY=1 \
WANDB_ENABLED=false \
./nwm_train.sh
```

Then validate DDP before committing to the full run:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
BATCH_SIZE=16 \
USE_PRECOMPUTED_LATENTS=true \
MAX_TRAIN_STEPS=2 \
EPOCHS=1 \
EVAL_AT_FIRST_STEP=false \
LOG_EVERY=1 \
WANDB_ENABLED=false \
./nwm_train.sh
```

For comparison, the original online-VAE path can be smoke-tested with a much
smaller observation batch:

```bash
GPU_IDS=2 \
NPROC=1 \
BATCH_SIZE=1 \
USE_PRECOMPUTED_LATENTS=false \
NUM_WORKERS=0 \
MAX_TRAIN_STEPS=2 \
EVAL_AT_FIRST_STEP=false \
LOG_EVERY=1 \
WANDB_MODE=offline \
./nwm_train.sh
```

The script activates the existing `nwm` Conda environment. `WANDB_MODE=offline`
stores a local W&B run without requiring authentication; omit it for the
default online mode.

## Paper training shape

The CompACT navigation world model uses global observation batch 128 for 200K
steps. This NAS launcher defaults to eight GPUs with batch 16 per GPU, so the
full run is simply (after the latent cache has completed):

```bash
./nwm_train.sh
```

`nwm_train.sh` enables the validated precomputed cache by default. It refuses
to train if `_SUCCESS.json`, manifests, VAE/transform/software fingerprints,
source coverage, BF16 settings, or the per-GPU batch do not match. To compare
against the original online VAE path explicitly use:

```bash
USE_PRECOMPUTED_LATENTS=false ./nwm_train.sh
```

Its released setup used four RTX 6000 Ada GPUs. The equivalent four-GPU
override is:

```bash
GPU_IDS=0,1,2,3 \
NPROC=4 \
BATCH_SIZE=32 \
USE_PRECOMPUTED_LATENTS=false \
./nwm_train.sh
```

The precomputed cache deliberately rejects batch 32 because its fixed VAE
encoding shape targets batch 16 per GPU. The cached-latent paper run therefore
uses eight GPUs and batch 16.

Do not start a full run on shared or occupied GPUs. The launcher checks W&B
authentication in online mode and prints the resolved training settings.
Launcher logs, Hydra logs, W&B local files, visualizations, and checkpoints are
all stored under `/file_system/nas/algorithm/dujun.nie/nwm/compact/`.

## Data indexes

Dataset indexes are cached under
`/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/dataset_indices` because
rescanning NAS storage is slow. If the data population changes, run once with
`REBUILD_INDEX=1`; return it to zero after the indexes have been rebuilt.
