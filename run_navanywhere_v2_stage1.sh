#!/usr/bin/env bash
set -euo pipefail

# NavAnywhere v2 is fully opt-in. All paths are versioned, while the v1 launcher
# retains its original defaults and 13-source recipe.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

COMPACT_ROOT="${COMPACT_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
export DETACH=0
export STAGE1_MODE="${STAGE1_MODE:-latentpt}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-nwm-${STAGE1_MODE}-navanywhere-v2-15src}"
export WANDB_PROJECT="${WANDB_PROJECT:-compact-nwm-navanywhere-v2}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-nwm-${STAGE1_MODE}-navanywhere-v2-15src-seed20260901}"
export WANDB_NOTES="${WANDB_NOTES:-NavAnywhere v2; 15 uniform sources; trajectory-disjoint 15-source validation}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NPROC="${NPROC:-8}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
export SAMPLING_SEED="${SAMPLING_SEED:-20260901}"
export SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v2_15src_train_seed20260901.json}"
export LATENT_PROXY_ROOT="${LATENT_PROXY_ROOT:-${COMPACT_ROOT}/cache/navanywhere_v2_15src_train_nav1_pixel_action_step100000_ws8_bs16_steps200000}"
export NWM_NAVANYWHERE_VAL_ENABLED=true
export NWM_NAVANYWHERE_VAL_RECIPE="${NWM_NAVANYWHERE_VAL_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v2_15src_val_seed20260901.json}"
export NWM_NAVANYWHERE_VAL_PROXY_ROOT="${NWM_NAVANYWHERE_VAL_PROXY_ROOT:-${COMPACT_ROOT}/cache/navanywhere_v2_15src_val_nav1_pixel_action_step100000_ws8_bs16_batches1}"
export EVAL_EVERY="${EVAL_EVERY:-5000}"
export EVAL_AT_FIRST_STEP="${EVAL_AT_FIRST_STEP:-true}"

# A v2 SD-VAE posterior cache is optional. Online VAE encoding is the safe
# default until such a recipe-bound cache has been completed.
export USE_PRECOMPUTED_LATENTS="${USE_PRECOMPUTED_LATENTS:-false}"
export VAE_LATENT_ROOT="${VAE_LATENT_ROOT:-${COMPACT_ROOT}/cache/navanywhere_v2_15src_train_vae_latents_sdvae224_seed20260901}"

for required in "${SAMPLING_RECIPE}" "${NWM_NAVANYWHERE_VAL_RECIPE}"; do
    [[ -r "${required}" ]] || { echo "ERROR: recipe is missing: ${required}" >&2; exit 2; }
done
if [[ "${DRY_RUN:-0}" != "1" && ( "${STAGE1_MODE}" == "latentpt" || "${STAGE1_MODE}" == "latentonlypt" ) ]]; then
    for cache_root in "${LATENT_PROXY_ROOT}" "${NWM_NAVANYWHERE_VAL_PROXY_ROOT}"; do
        for marker in metadata.json _SUCCESS.json; do
            [[ -r "${cache_root}/${marker}" ]] || {
                echo "ERROR: exact v2 latent cache is incomplete: ${cache_root}/${marker}" >&2
                echo "Run ./run_navanywhere_v2_latent_actions.sh first." >&2
                exit 2
            }
        done
    done
fi

exec "${SCRIPT_DIR}/run_navanywhere_stage1.sh"
