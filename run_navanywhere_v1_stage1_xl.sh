#!/usr/bin/env bash
set -euo pipefail

# Opt-in 8x80GB CDiT-XL LatentPT recipe. The original Stage-1 launcher keeps
# all of its defaults; this wrapper selects only the v1 13-source split/cache.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Pin both launch layers to the same environment by absolute path.  Dedicated
# variables avoid inheriting a generic CONDA_ENV=nwm from the caller.  This is
# opt-in to this XL wrapper and does not change the original Stage-1 defaults.
export NWM_CONDA_ACTIVATE="${NWM_CONDA_ACTIVATE:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/activate}"
export NWM_CONDA_ENV_PATH="${NWM_CONDA_ENV_PATH:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm}"
if [[ "${NWM_CONDA_ACTIVATE}" != /* || "${NWM_CONDA_ENV_PATH}" != /* ]]; then
    echo "ERROR: NWM_CONDA_ACTIVATE and NWM_CONDA_ENV_PATH must be absolute paths." >&2
    exit 2
fi
export CONDA_ACTIVATE="${NWM_CONDA_ACTIVATE}"
export CONDA_ENV="${NWM_CONDA_ENV_PATH}"

COMPACT_ROOT="${COMPACT_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
export STAGE1_MODE=latentpt
export MODEL_GENERATOR=cdit_xl
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
if [[ "${NPROC:-8}" != "8" ]]; then
    echo "ERROR: this launcher and its pair caches require exactly 8 ranks." >&2
    exit 2
fi
REQUESTED_BATCH_SIZE="${BATCH_SIZE:-16}"
if ! [[ "${REQUESTED_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BATCH_SIZE must be a positive integer." >&2
    exit 2
fi
REQUESTED_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
if ! [[ "${REQUESTED_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]] || \
   (( REQUESTED_TRAIN_STEPS > 200000 )); then
    echo "ERROR: MAX_TRAIN_STEPS must be a positive prefix of the prepared 200000-step cache." >&2
    exit 2
fi
export NPROC=8
export BATCH_SIZE="${REQUESTED_BATCH_SIZE}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_TRAIN_STEPS="${REQUESTED_TRAIN_STEPS}"
export EVAL_EVERY="${EVAL_EVERY:-5000}"
export EVAL_AT_FIRST_STEP="${EVAL_AT_FIRST_STEP:-true}"
export EVAL_OFFLOAD_MODELS="${EVAL_OFFLOAD_MODELS:-true}"
export MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-74000}"
export ALLOW_VAE_RECIPE_SUBSET=true
export DETACH=0

export SAMPLING_SEED="${SAMPLING_SEED:-20260901}"
export SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v1_13src_train_seed20260901.json}"
export VAE_LATENT_ROOT="${VAE_LATENT_ROOT:-${COMPACT_ROOT}/cache/navanywhere_vae_latents_sdvae224_seed20260901}"
export LATENT_PROXY_ROOT="${LATENT_PROXY_ROOT:-${COMPACT_ROOT}/cache/navanywhere_v1_13src_train_nav1_pixel_action_step100000_ws8_bs${BATCH_SIZE}_steps200000}"
export NWM_NAVANYWHERE_VAL_ENABLED=true
export NWM_NAVANYWHERE_VAL_RECIPE="${NWM_NAVANYWHERE_VAL_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v1_13src_val_seed20260901.json}"
export NWM_NAVANYWHERE_VAL_PROXY_ROOT="${NWM_NAVANYWHERE_VAL_PROXY_ROOT:-${COMPACT_ROOT}/cache/navanywhere_v1_13src_val_nav1_pixel_action_step100000_ws8_bs${BATCH_SIZE}_batches1}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-latentpt-xl-v1-$(date -u +%m%d-%H%M%S)}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-nwm-latentpt-xl-v1-seed${SAMPLING_SEED}-$(date -u +%Y%m%d_%H%M%S)}"
export WANDB_NOTES="${WANDB_NOTES:-NavAnywhere v1 13-source Stage-1 LatentPT; CDiT-XL; disjoint eval every 5K; reused v1 caches}"

exec "${SCRIPT_DIR}/run_navanywhere_stage1.sh"
