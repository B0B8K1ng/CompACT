#!/usr/bin/env bash
set -Eeuo pipefail

# prepare: freeze all visible sources, extract VAE and all local action pairs.
# train:   CDiT-B LatentPT -> latent_reset, using the completed caches.
# all:     prepare and then train; failures stop the chain.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
MODE="${1:-all}"
case "${MODE}" in prepare|train|all) ;; *) echo "Usage: $0 [prepare|train|all]"; exit 2 ;; esac
export COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
export NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere}"
export SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_NAS_ROOT}/recipes/navanywhere_full_seed20260901.json}"
export CHECKPOINT="${CHECKPOINT:-/file_system/nas/algorithm/dujun.nie/nwm/weights/navigation_lam/variant_4_pixel_action/nav15-pixel-action/checkpoints/step=60000.ckpt}"
export LATENT_CHECKPOINT_STEP="${LATENT_CHECKPOINT_STEP:-60000}"
export NAV1_PROXY_ROOT="${NAV1_PROXY_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_full_final_lam_allpairs}"
export STAGE1_VAE_LATENT_ROOT="${STAGE1_VAE_LATENT_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_full_vae_sd224_seed20260901}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export RUN_ID="${RUN_ID:-nav15_full_80gb_$(date -u +%Y%m%d_%H%M%S)}"
export RESULTS_ROOT="${RESULTS_ROOT:-${COMPACT_NAS_ROOT}/runs/nav15_full_80gb}"
export PYTHONUNBUFFERED=1
export NAVANYWHERE_RECIPE_PROGRESS="${NAVANYWHERE_RECIPE_PROGRESS:-1}"
CONDA="${CONDA:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda}"
NWM_PYTHON="${NWM_PYTHON:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python}"
trap 'status=$?; echo "Pipeline exit status: ${status}"' EXIT
run() {
    printf 'Exact command (cwd=%s): ' "${SCRIPT_DIR}"
    printf '%q ' "$@"
    printf '\n'
    "$@"
}
echo "cwd: ${SCRIPT_DIR}; PID: $$; mode: ${MODE}; RUN_ID: ${RUN_ID}"
echo "Recipe: ${SAMPLING_RECIPE}"
df -h "${COMPACT_NAS_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv

if [[ "${MODE}" != train ]]; then
    if [[ ! -f "${SAMPLING_RECIPE}" ]]; then
        # Exclude hidden staging trees. Freeze the inventory once; retries use it.
        SOURCES=()
        for source in "${NAVANYWHERE_ROOT}"/*; do
            [[ -d "${source}" && ! -L "${source}" ]] || continue
            SOURCES+=(--source-id "${source##*/}")
        done
        run "${NWM_PYTHON}" scripts/build_navanywhere_sampling_recipe.py \
            --root "${NAVANYWHERE_ROOT}" --output "${SAMPLING_RECIPE}" \
            --seed 20260901 "${SOURCES[@]}"
    fi
    run env DETACH=0 NPROC=8 VAE_LATENT_ROOT="${STAGE1_VAE_LATENT_ROOT}" \
        VAE_REUSE_ROOT="${VAE_REUSE_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_vae_latents_sdvae224_seed20260901}" \
        VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}" LOADER_THREADS="${LOADER_THREADS:-16}" \
        LOG_EVERY_TRAJECTORIES=10 OVERWRITE=0 \
        bash ./precompute_navanywhere_vae_latents_8gpu.sh
    run env DETACH=0 FULL_PAIRS=1 NPROC=8 OUTPUT_ROOT="${NAV1_PROXY_ROOT}" \
        BATCH_SIZE="${ACTION_BATCH_SIZE:-128}" LOADER_THREADS="${LOADER_THREADS:-16}" \
        PRECISION=bf16-mixed OVERWRITE=0 LOG_EVERY_TRAJECTORIES=10 \
        REUSE_ROOT="" \
        bash ./precompute_navanywhere_nav1_latent_actions_8gpu.sh
fi
if [[ "${MODE}" != prepare ]]; then
    run env GPU_MEMORY_GB=80 BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}" \
        BACKBONE_LR="${BACKBONE_LR:-1e-4}" \
        STAGE2_EVAL_EVERY="${STAGE2_EVAL_EVERY:-1000}" \
        STAGE2_EVAL_BATCH_SIZE="${STAGE2_EVAL_BATCH_SIZE:-16}" \
        STAGE2_EVAL_NUM_BATCHES="${STAGE2_EVAL_NUM_BATCHES:-1}" \
        LATENT_CHECKPOINT="${CHECKPOINT}" LATENT_CHECKPOINT_STEP="${LATENT_CHECKPOINT_STEP}" \
        WANDB_ENABLED=true WANDB_MODE=online LOG_EVERY="${LOG_EVERY:-20}" \
        "${CONDA}" run --no-capture-output -n nwm \
        bash ./run_nwm_latentpt_nav1_80gb.sh
fi
