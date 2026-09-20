#!/usr/bin/env bash
set -euo pipefail

# One configurable entry point for TimePT (default), GeoPT, IDMPT, and
# LatentPT, and LatentOnlyPT. All modes consume the same immutable sampling
# recipe.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

STAGE1_MODE="${STAGE1_MODE:-timept}" # timept | geopt | idmpt | latentpt | latentonlypt
DRY_RUN="${DRY_RUN:-0}"
DETACH="${DETACH:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${STAGE1_MODE}-$(date -u +%m%d-%H%M%S)}"
if [[ "${DRY_RUN}" == "1" ]]; then
    DETACH=0
fi
if [[ "${DETACH}" == "1" && "${RUN_UNDER_CODEX_EXP:-0}" != "1" ]]; then
    if ! command -v codex-exp >/dev/null 2>&1; then
        echo "ERROR: codex-exp is required for detached training; set DETACH=0 to run attached." >&2
        exit 2
    fi
    echo "Starting experiment ${EXPERIMENT_NAME} in ${SCRIPT_DIR}"
    echo "Monitor: codex-exp status ${EXPERIMENT_NAME}"
    echo "Logs:    codex-exp logs ${EXPERIMENT_NAME}"
    echo "Run log: ${LOG_DIR:-/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_stage1}/train_${STAGE1_MODE}_${RUN_TIMESTAMP}.log"
    exec codex-exp start "${EXPERIMENT_NAME}" -- \
        env RUN_UNDER_CODEX_EXP=1 DETACH=0 \
        EXPERIMENT_NAME="${EXPERIMENT_NAME}" RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
        "${SCRIPT_PATH}"
fi

case "${STAGE1_MODE}" in
    timept|geopt|idmpt|latentpt|latentonlypt) ;;
    *) echo "ERROR: STAGE1_MODE must be timept, geopt, idmpt, latentpt, or latentonlypt." >&2; exit 2 ;;
esac

# ---- Frequently edited experiment settings ---------------------------------
CONDA_ENV="${CONDA_ENV:-nwm}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/activate}"
NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere}"
COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
SAMPLING_SEED="${SAMPLING_SEED:-20260901}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-0}"
SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_NAS_ROOT}/recipes/navanywhere_balanced_seed${SAMPLING_SEED}.json}"
VAE_LATENT_ROOT="${VAE_LATENT_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_vae_latents_sdvae224_seed${SAMPLING_SEED}}"
RESULTS_DIR="${RESULTS_DIR:-${COMPACT_NAS_ROOT}/runs/navanywhere_stage1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MODEL_GENERATOR="${MODEL_GENERATOR:-cdit_b}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LOG_EVERY="${LOG_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
EVAL_AT_FIRST_STEP="${EVAL_AT_FIRST_STEP:-false}"
EVAL_OFFLOAD_MODELS="${EVAL_OFFLOAD_MODELS:-false}"
LATENT_LRU_SIZE="${LATENT_LRU_SIZE:-8}"
USE_PRECOMPUTED_LATENTS="${USE_PRECOMPUTED_LATENTS:-true}"
ALLOW_VAE_RECIPE_SUBSET="${ALLOW_VAE_RECIPE_SUBSET:-false}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# W&B online logging is the default. WANDB_API_KEY or an existing login is
# required; use WANDB_MODE=offline only for local smoke runs.
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-compact-nwm-navanywhere}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-nwm-${STAGE1_MODE}-seed${SAMPLING_SEED}-$(date -u +%Y%m%d_%H%M%S)}"
WANDB_NOTES="${WANDB_NOTES:-NavAnywhere-only ${STAGE1_MODE}; shared balanced recipe; precomputed SD-VAE posterior}"

# Proxy stores are read only for local pairs. TimePT ignores all three.
GEOMETRY_PROXY_ROOT="${GEOMETRY_PROXY_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_geometry_proxy}"
IDM_PROXY_ROOT="${IDM_PROXY_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_idm_proxy}"
LATENT_PROXY_ROOT="${LATENT_PROXY_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_nav1_pixel_action_step100000}"

NWM_HF_HOME="${NWM_HF_HOME:-/file_system/nas/algorithm/dujun.nie/huggingface}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-}"
WANDB_CACHE_ROOT="${WANDB_CACHE_ROOT:-${COMPACT_NAS_ROOT}/cache/wandb}"
TORCH_CACHE_ROOT="${TORCH_CACHE_ROOT:-${COMPACT_NAS_ROOT}/cache/torch}"
LOG_DIR="${LOG_DIR:-${COMPACT_NAS_ROOT}/logs/navanywhere_stage1}"
# -----------------------------------------------------------------------------

if [[ ! -r "${CONDA_ACTIVATE}" ]]; then
    echo "ERROR: conda activation script is not readable: ${CONDA_ACTIVATE}" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ -z "${NPROC}" ]]; then
    NPROC="${#GPU_ARRAY[@]}"
fi
if (( NPROC < 1 || NPROC > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: NPROC=${NPROC} must fit GPU_IDS=${GPU_IDS}." >&2
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

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -d "${NAVANYWHERE_ROOT}" ]]; then
        echo "ERROR: NavAnywhere root does not exist: ${NAVANYWHERE_ROOT}" >&2
        exit 2
    fi
    if [[ "${USE_PRECOMPUTED_LATENTS}" != "true" && ! -r "${VAE_MODEL_PATH}/config.json" ]]; then
        echo "ERROR: a complete local SD-VAE snapshot was not found; set VAE_MODEL_PATH." >&2
        exit 2
    fi
    mkdir -p "$(dirname -- "${SAMPLING_RECIPE}")" "${RESULTS_DIR}" "${LOG_DIR}" \
        "${WANDB_CACHE_ROOT}/cache" "${WANDB_CACHE_ROOT}/data" \
        "${WANDB_CACHE_ROOT}/artifacts" "${TORCH_CACHE_ROOT}"
    if [[ ! -f "${SAMPLING_RECIPE}" ]]; then
        python scripts/build_navanywhere_sampling_recipe.py \
            --root "${NAVANYWHERE_ROOT}" --output "${SAMPLING_RECIPE}" \
            --seed "${SAMPLING_SEED}" --context-size 4 --goals-per-obs 4 \
            --samples-per-epoch "${SAMPLES_PER_EPOCH}"
    else
        python scripts/build_navanywhere_sampling_recipe.py \
            --root "${NAVANYWHERE_ROOT}" --output "${SAMPLING_RECIPE}" \
            --seed "${SAMPLING_SEED}" --context-size 4 --goals-per-obs 4 \
            --validate-existing
    fi
    if [[ "${USE_PRECOMPUTED_LATENTS}" == "true" ]]; then
        for marker in "${VAE_LATENT_ROOT}/metadata.json" "${VAE_LATENT_ROOT}/_SUCCESS.json"; do
            if [[ ! -r "${marker}" ]]; then
                echo "ERROR: precomputed VAE cache is incomplete: ${marker}" >&2
                echo "Run ./precompute_navanywhere_vae_latents_8gpu.sh first." >&2
                exit 2
            fi
        done
    fi
    case "${STAGE1_MODE}" in
        geopt)
            for marker in "${GEOMETRY_PROXY_ROOT}/metadata.json" "${GEOMETRY_PROXY_ROOT}/_SUCCESS.json"; do
                if [[ ! -r "${marker}" ]]; then
                    echo "ERROR: precomputed geometry cache is incomplete: ${marker}" >&2
                    echo "Wait for ./precompute_navanywhere_geometry_actions_7gpu.sh to finish." >&2
                    exit 2
                fi
            done
            ;;
        idmpt) [[ -d "${IDM_PROXY_ROOT}" ]] || { echo "ERROR: ${IDM_PROXY_ROOT} missing" >&2; exit 2; } ;;
        latentpt|latentonlypt)
            for marker in "${LATENT_PROXY_ROOT}/metadata.json" "${LATENT_PROXY_ROOT}/_SUCCESS.json"; do
                if [[ ! -r "${marker}" ]]; then
                    echo "ERROR: precomputed nav1 latent-action cache is incomplete: ${marker}" >&2
                    echo "Run ./precompute_navanywhere_nav1_latent_actions_8gpu.sh first." >&2
                    exit 2
                fi
            done
            ;;
    esac
    if [[ "${NWM_NAVANYWHERE_VAL_ENABLED:-false}" == "true" ]]; then
        if [[ ! -r "${NWM_NAVANYWHERE_VAL_RECIPE:-}" ]]; then
            echo "ERROR: Stage-1 validation recipe is missing: ${NWM_NAVANYWHERE_VAL_RECIPE:-<unset>}" >&2
            exit 2
        fi
        if [[ "${STAGE1_MODE}" == "latentpt" || "${STAGE1_MODE}" == "latentonlypt" ]]; then
            for marker in \
                "${NWM_NAVANYWHERE_VAL_PROXY_ROOT:-}/metadata.json" \
                "${NWM_NAVANYWHERE_VAL_PROXY_ROOT:-}/_SUCCESS.json"; do
                if [[ ! -r "${marker}" ]]; then
                    echo "ERROR: validation latent-action cache is incomplete: ${marker}" >&2
                    echo "Run ./prepare_navanywhere_v1_stage1_xl.sh cache first." >&2
                    exit 2
                fi
            done
        fi
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required for the capacity check." >&2
        exit 2
    fi
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
    if [[ "${WANDB_ENABLED}" == "true" && "${WANDB_MODE}" == "online" ]]; then
        wandb login --verify || {
            echo "ERROR: W&B is not authenticated; run wandb login or set WANDB_MODE=offline." >&2
            exit 2
        }
    fi
fi

if [[ -f "${SAMPLING_RECIPE}" ]]; then
    RECIPE_SHA256="$(sha256sum "${SAMPLING_RECIPE}" | awk '{print $1}')"
else
    RECIPE_SHA256="dry-run-recipe-sha256"
fi
VAE_INPUT_TAG="$([[ "${USE_PRECOMPUTED_LATENTS}" == "true" ]] && echo precomputed-VAE || echo online-VAE)"
LAUNCH_LOG="${LOG_DIR}/train_${STAGE1_MODE}_${RUN_TIMESTAMP}.log"
if [[ "${DRY_RUN}" != "1" ]]; then
    exec > >(tee -a "${LAUNCH_LOG}") 2>&1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NWM_NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT}"
export NWM_NAVANYWHERE_RECIPE="${SAMPLING_RECIPE}"
export NWM_NAVANYWHERE_VAE_LATENT_ROOT="${VAE_LATENT_ROOT}"
export NWM_RESULTS_DIR="${RESULTS_DIR}"
export NWM_GEOMETRY_PROXY_ROOT="${GEOMETRY_PROXY_ROOT}"
export NWM_IDM_PROXY_ROOT="${IDM_PROXY_ROOT}"
export NWM_LATENT_PROXY_ROOT="${LATENT_PROXY_ROOT}"
export WANDB_MODE
export WANDB_CACHE_DIR="${WANDB_CACHE_ROOT}/cache"
export WANDB_DATA_DIR="${WANDB_CACHE_ROOT}/data"
export WANDB_ARTIFACT_DIR="${WANDB_CACHE_ROOT}/artifacts"
export WANDB_DIR="${RESULTS_DIR}"
export TORCH_HOME="${TORCH_CACHE_ROOT}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "NavAnywhere Stage-1 launch"
echo "  experiment:        ${EXPERIMENT_NAME}"
echo "  working directory: ${SCRIPT_DIR}"
echo "  launcher log:      ${LAUNCH_LOG}"
echo "  mode:              ${STAGE1_MODE}"
echo "  conda env:         ${CONDA_ENV}"
echo "  NavAnywhere:       ${NAVANYWHERE_ROOT}"
echo "  recipe:            ${SAMPLING_RECIPE}"
echo "  recipe sha256:     ${RECIPE_SHA256}"
echo "  VAE input:         $([[ "${USE_PRECOMPUTED_LATENTS}" == "true" ]] && echo precomputed-posterior || echo online-pixels)"
echo "  VAE cache:         ${VAE_LATENT_ROOT}"
if [[ "${STAGE1_MODE}" == "latentpt" || "${STAGE1_MODE}" == "latentonlypt" ]]; then
    echo "  latent cache:      ${LATENT_PROXY_ROOT}"
fi
if [[ "${NWM_NAVANYWHERE_VAL_ENABLED:-false}" == "true" ]]; then
    echo "  validation recipe:${NWM_NAVANYWHERE_VAL_RECIPE:-<unset>}"
    echo "  validation cache: ${NWM_NAVANYWHERE_VAL_PROXY_ROOT:-<unset>}"
    echo "  evaluation:       step1=${EVAL_AT_FIRST_STEP}, every=${EVAL_EVERY} steps"
fi
echo "  GPUs/processes:    ${GPU_IDS} / ${NPROC}"
echo "  generator:         ${MODEL_GENERATOR}"
echo "  batch/GPU:         ${BATCH_SIZE}"
echo "  max steps:         ${MAX_TRAIN_STEPS}"
echo "  W&B:               ${WANDB_ENABLED} ${WANDB_MODE} ${WANDB_PROJECT}/${WANDB_RUN_NAME}"

HYDRA_ARGS=(
    "model/generator=${MODEL_GENERATOR}"
    "seed=${SAMPLING_SEED}"
    "dataset.sampling_recipe.path=${SAMPLING_RECIPE}"
    "dataset.sampling_recipe.sha256=${RECIPE_SHA256}"
    "dataset.precomputed_latents.enabled=${USE_PRECOMPUTED_LATENTS}"
    "dataset.precomputed_latents.root=${VAE_LATENT_ROOT}"
    "dataset.precomputed_latents.cache_size=${LATENT_LRU_SIZE}"
    "dataset.precomputed_latents.allow_recipe_subset=${ALLOW_VAE_RECIPE_SUBSET}"
    "training.batch_size=${BATCH_SIZE}"
    "training.num_workers=${NUM_WORKERS}"
    "training.optimizer.lr=${LEARNING_RATE}"
    "training.optimizer.weight_decay=${WEIGHT_DECAY}"
    "training.results_dir=${RESULTS_DIR}"
    "training.run_name=${WANDB_RUN_NAME}"
    "training.notes=\"${WANDB_NOTES}\""
    "training.wandb_enabled=${WANDB_ENABLED}"
    "training.wandb_project=${WANDB_PROJECT}"
    "training.wandb_tags=[NavAnywhere,Stage1,${STAGE1_MODE},shared-recipe,${VAE_INPUT_TAG}]"
    "model.tokenizer.model_path=${VAE_MODEL_PATH:-stabilityai/sd-vae-ft-ema}"
    "max_train_steps=${MAX_TRAIN_STEPS}"
    "log_every=${LOG_EVERY}"
    "ckpt_every=${CKPT_EVERY}"
    "eval_every=${EVAL_EVERY}"
    "eval_at_first_step=${EVAL_AT_FIRST_STEP}"
    "eval_offload_models=${EVAL_OFFLOAD_MODELS}"
    "log_cuda_memory=true"
    "bfloat16=1"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
    HYDRA_ARGS+=("training.wandb_entity=${WANDB_ENTITY}")
fi

LAUNCH=(bash two_stage_nwm.sh stage1 "${STAGE1_MODE}" "--nproc=${NPROC}" "--gpus=${GPU_IDS}")
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    LAUNCH+=("--resume=${RESUME_CHECKPOINT}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    LAUNCH+=(--dry-run)
fi
LAUNCH+=(-- "${HYDRA_ARGS[@]}")
exec "${LAUNCH[@]}"
