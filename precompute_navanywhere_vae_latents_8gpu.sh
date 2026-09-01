#!/usr/bin/env bash
set -euo pipefail

# Fast, resumable NavAnywhere SD-VAE posterior extraction.  Edit/override the
# variables in this block; the default launch detaches through codex-exp.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

DETACH="${DETACH:-1}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-navvae8-$(date -u +%m%d-%H%M%S)}"
if [[ "${DETACH}" == "1" && "${RUN_UNDER_CODEX_EXP:-0}" != "1" ]]; then
    if ! command -v codex-exp >/dev/null 2>&1; then
        echo "ERROR: codex-exp is required for detached precompute; set DETACH=0 to run attached." >&2
        exit 2
    fi
    echo "Starting experiment ${EXPERIMENT_NAME} in ${SCRIPT_DIR}"
    echo "Monitor: codex-exp status ${EXPERIMENT_NAME}"
    echo "Logs:    codex-exp logs ${EXPERIMENT_NAME}"
    echo "Run log: ${LOG_DIR:-/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_vae_latents}/precompute_${RUN_TIMESTAMP}.log"
    exec codex-exp start "${EXPERIMENT_NAME}" -- \
        env RUN_UNDER_CODEX_EXP=1 DETACH=0 \
        EXPERIMENT_NAME="${EXPERIMENT_NAME}" RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
        "${SCRIPT_PATH}"
fi

CONDA_ENV="${CONDA_ENV:-nwm}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/activate}"
NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere}"
COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
SAMPLING_SEED="${SAMPLING_SEED:-20260901}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-0}"
SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_NAS_ROOT}/recipes/navanywhere_balanced_seed${SAMPLING_SEED}.json}"
VAE_LATENT_ROOT="${VAE_LATENT_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_vae_latents_sdvae224_seed${SAMPLING_SEED}}"
NWM_HF_HOME="${NWM_HF_HOME:-/file_system/nas/algorithm/dujun.nie/huggingface}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
LOADER_THREADS="${LOADER_THREADS:-8}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}"
OVERWRITE="${OVERWRITE:-0}"
LOG_EVERY_TRAJECTORIES="${LOG_EVERY_TRAJECTORIES:-10}"
LOG_DIR="${LOG_DIR:-${COMPACT_NAS_ROOT}/logs/navanywhere_vae_latents}"

if [[ ! -r "${CONDA_ACTIVATE}" ]]; then
    echo "ERROR: conda activation script is not readable: ${CONDA_ACTIVATE}" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

if [[ ! -d "${NAVANYWHERE_ROOT}" ]]; then
    echo "ERROR: NavAnywhere root does not exist: ${NAVANYWHERE_ROOT}" >&2
    exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( NPROC < 1 || NPROC > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: NPROC=${NPROC} must fit GPU_IDS=${GPU_IDS}." >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required for the capacity check." >&2
    exit 2
fi

export HF_HOME="${NWM_HF_HOME}"
export HF_HUB_CACHE="${NWM_HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -z "${VAE_MODEL_PATH}" ]]; then
    VAE_CACHE_DIR="${HF_HUB_CACHE}/models--stabilityai--sd-vae-ft-ema"
    if [[ -r "${VAE_CACHE_DIR}/refs/main" ]]; then
        VAE_REVISION="$(<"${VAE_CACHE_DIR}/refs/main")"
        VAE_MODEL_PATH="${VAE_CACHE_DIR}/snapshots/${VAE_REVISION}"
    fi
fi
if [[ ! -r "${VAE_MODEL_PATH}/config.json" ]] || \
   [[ ! -r "${VAE_MODEL_PATH}/diffusion_pytorch_model.bin" && \
      ! -r "${VAE_MODEL_PATH}/diffusion_pytorch_model.safetensors" ]]; then
    echo "ERROR: set VAE_MODEL_PATH to a complete local stabilityai/sd-vae-ft-ema snapshot." >&2
    exit 2
fi

mkdir -p "$(dirname -- "${SAMPLING_RECIPE}")" "${VAE_LATENT_ROOT}" "${LOG_DIR}"
if [[ ! -f "${SAMPLING_RECIPE}" ]]; then
    python scripts/build_navanywhere_sampling_recipe.py \
        --root "${NAVANYWHERE_ROOT}" \
        --output "${SAMPLING_RECIPE}" \
        --seed "${SAMPLING_SEED}" \
        --context-size 4 \
        --goals-per-obs 4 \
        --samples-per-epoch "${SAMPLES_PER_EPOCH}"
else
    python scripts/build_navanywhere_sampling_recipe.py \
        --root "${NAVANYWHERE_ROOT}" \
        --output "${SAMPLING_RECIPE}" \
        --seed "${SAMPLING_SEED}" \
        --context-size 4 \
        --goals-per-obs 4 \
        --validate-existing
fi

LOG_FILE="${LOG_DIR}/precompute_${RUN_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1

echo "NavAnywhere VAE latent precompute"
echo "  experiment:        ${EXPERIMENT_NAME}"
echo "  working directory: ${SCRIPT_DIR}"
echo "  log:               ${LOG_FILE}"
echo "  conda env:         ${CONDA_ENV}"
echo "  NavAnywhere:       ${NAVANYWHERE_ROOT}"
echo "  sampling recipe:   ${SAMPLING_RECIPE}"
echo "  output:            ${VAE_LATENT_ROOT}"
echo "  GPUs/processes:    ${GPU_IDS} / ${NPROC}"
echo "  VAE batch/rank:    ${VAE_BATCH_SIZE}"
echo "GPU capacity before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
if [[ "${REQUIRE_IDLE_GPUS}" == "1" ]]; then
    GPU_CAPACITY="$({ nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits; } 2>&1)"
    for gpu_id in "${GPU_ARRAY[@]:0:${NPROC}}"; do
        read -r free_mb utilization < <(
            awk -F',' -v wanted="${gpu_id}" '
                { for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
                $1 == wanted { print $2, $3; found=1 }
                END { if (!found) exit 1 }
            ' <<< "${GPU_CAPACITY}"
        ) || {
            echo "ERROR: selected GPU ${gpu_id} was not reported by nvidia-smi." >&2
            exit 2
        }
        if (( free_mb < MIN_FREE_GPU_MB || utilization > MAX_GPU_UTILIZATION )); then
            echo "ERROR: GPU ${gpu_id} is busy (free=${free_mb} MiB, util=${utilization}%)." >&2
            echo "Choose idle GPU_IDS or set REQUIRE_IDLE_GPUS=0 only if sharing is intentional." >&2
            exit 2
        fi
    done
fi
echo "Output filesystem capacity:"
df -h "${COMPACT_NAS_ROOT}"

ARGS=(
    --data-root "${NAVANYWHERE_ROOT}"
    --sampling-recipe "${SAMPLING_RECIPE}"
    --output-root "${VAE_LATENT_ROOT}"
    --dataset-config "${SCRIPT_DIR}/conf/dataset/navanywhere.yaml"
    --vae-model-path "${VAE_MODEL_PATH}"
    --vae-batch-size "${VAE_BATCH_SIZE}"
    --loader-threads "${LOADER_THREADS}"
    --compute-dtype bfloat16
    --storage-dtype bfloat16
    --max-trajectories "${MAX_TRAJECTORIES}"
    --log-every-trajectories "${LOG_EVERY_TRAJECTORIES}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    precompute_navanywhere_vae_latents.py "${ARGS[@]}"

echo "Finished successfully. Cache completion marker: ${VAE_LATENT_ROOT}/_SUCCESS.json"
