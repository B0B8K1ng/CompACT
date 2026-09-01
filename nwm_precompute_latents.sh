#!/usr/bin/env bash
set -euo pipefail

# Precompute SD-VAE posterior statistics for the CompACT NWM training split.
# Defaults: 8 GPUs, BF16 posterior stats, fixed 128-image VAE batches, NAS output.
# Common overrides:
#   MAX_TRAJECTORIES=16 VERIFY_SAMPLES=8 ./nwm_precompute_latents.sh
#   VERIFY_ONLY=1 GPU_IDS=0 ./nwm_precompute_latents.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-nwm}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/activate}"
NWM_DATA_ROOT="${NWM_DATA_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data}"
COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
NWM_LATENT_ROOT="${NWM_LATENT_ROOT:-${COMPACT_NAS_ROOT}/cache/vae_latents_sd_vae_ft_ema_224}"
NWM_HF_HOME="${NWM_HF_HOME:-/file_system/nas/algorithm/dujun.nie/huggingface}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-}"
DATASET_CONFIG="${DATASET_CONFIG:-${SCRIPT_DIR}/conf/dataset/nwm.yaml}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
LOADER_THREADS="${LOADER_THREADS:-4}"
DISCOVERY_THREADS="${DISCOVERY_THREADS:-32}"
STORAGE_DTYPE="${STORAGE_DTYPE:-bfloat16}"
COMPUTE_DTYPE="${COMPUTE_DTYPE:-bfloat16}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}"
VERIFY_SAMPLES="${VERIFY_SAMPLES:-32}"
VERIFY_ATOL="${VERIFY_ATOL:-0.02}"
VERIFY_RTOL="${VERIFY_RTOL:-0.005}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"
FAIL_ON_MISSING="${FAIL_ON_MISSING:-0}"
LOG_EVERY_TRAJECTORIES="${LOG_EVERY_TRAJECTORIES:-25}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LOG_DIR="${LOG_DIR:-${COMPACT_NAS_ROOT}/logs/precompute_latents}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ -z "${NPROC}" ]]; then
    NPROC="${#GPU_ARRAY[@]}"
fi
if (( NPROC < 1 || NPROC > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: NPROC=${NPROC} must be between 1 and the number of GPU_IDS (${#GPU_ARRAY[@]})." >&2
    exit 2
fi
if [[ "${VERIFY_ONLY}" == "1" && "${NPROC}" != "1" ]]; then
    NPROC=1
fi

if [[ ! -x "${CONDA_ACTIVATE}" ]]; then
    echo "ERROR: conda activate script is not executable: ${CONDA_ACTIVATE}" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

export HF_HOME="${NWM_HF_HOME}"
export HF_HUB_CACHE="${NWM_HF_HOME}/hub"
if [[ -z "${VAE_MODEL_PATH}" ]]; then
    VAE_CACHE_DIR="${HF_HUB_CACHE}/models--stabilityai--sd-vae-ft-ema"
    VAE_REF_FILE="${VAE_CACHE_DIR}/refs/main"
    if [[ -r "${VAE_REF_FILE}" ]]; then
        VAE_REVISION="$(<"${VAE_REF_FILE}")"
        VAE_SNAPSHOT="${VAE_CACHE_DIR}/snapshots/${VAE_REVISION}"
        if [[ -r "${VAE_SNAPSHOT}/config.json" && -r "${VAE_SNAPSHOT}/diffusion_pytorch_model.bin" ]]; then
            VAE_MODEL_PATH="${VAE_SNAPSHOT}"
        fi
    fi
fi
if [[ -z "${VAE_MODEL_PATH}" || ! -r "${VAE_MODEL_PATH}/config.json" ]]; then
    echo "ERROR: a complete local SD-VAE snapshot was not found; set VAE_MODEL_PATH." >&2
    exit 2
fi

mkdir -p "${NWM_LATENT_ROOT}" "${LOG_DIR}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/nwm_precompute_latents_${TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NWM_DATA_ROOT NWM_LATENT_ROOT VAE_MODEL_PATH HF_HUB_OFFLINE
export PYTHONUNBUFFERED=1

ARGS=(
    --data-root "${NWM_DATA_ROOT}"
    --output-root "${NWM_LATENT_ROOT}"
    --dataset-config "${DATASET_CONFIG}"
    --vae-model-path "${VAE_MODEL_PATH}"
    --vae-batch-size "${VAE_BATCH_SIZE}"
    --loader-threads "${LOADER_THREADS}"
    --discovery-threads "${DISCOVERY_THREADS}"
    --storage-dtype "${STORAGE_DTYPE}"
    --compute-dtype "${COMPUTE_DTYPE}"
    --max-trajectories "${MAX_TRAJECTORIES}"
    --verify-samples "${VERIFY_SAMPLES}"
    --verify-atol "${VERIFY_ATOL}"
    --verify-rtol "${VERIFY_RTOL}"
    --log-every-trajectories "${LOG_EVERY_TRAJECTORIES}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "${FAIL_ON_MISSING}" == "1" ]]; then
    ARGS+=(--fail-on-missing-split-trajectories)
fi

echo "CompACT SD-VAE latent precompute"
echo "  repo:             ${SCRIPT_DIR}"
echo "  conda env:        ${CONDA_ENV}"
echo "  data root:        ${NWM_DATA_ROOT}"
echo "  dataset config:   ${DATASET_CONFIG}"
echo "  latent root:      ${NWM_LATENT_ROOT}"
echo "  VAE:              ${VAE_MODEL_PATH}"
echo "  visible GPUs:     ${GPU_IDS}"
echo "  processes:        ${NPROC}"
echo "  VAE batch/rank:   ${VAE_BATCH_SIZE} (fixed/padded)"
echo "  compute/storage:  ${COMPUTE_DTYPE}/${STORAGE_DTYPE}"
echo "  loader threads:   ${LOADER_THREADS}/rank"
echo "  discovery threads:${DISCOVERY_THREADS} (rank 0)"
echo "  max trajectories: ${MAX_TRAJECTORIES} (0=all)"
echo "  verify samples:   ${VERIFY_SAMPLES}"
echo "  verify only:      ${VERIFY_ONLY}"
echo "  log:              ${LOG_FILE}"

if [[ "${VERIFY_ONLY}" == "1" ]]; then
    python precompute_vae_latents.py --verify-only "${ARGS[@]}"
else
    torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
        precompute_vae_latents.py "${ARGS[@]}"
fi
