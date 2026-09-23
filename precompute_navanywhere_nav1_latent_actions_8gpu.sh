#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; echo "Launcher exit status: ${status}"' EXIT

# Resumable 8-GPU extraction of raw 32-D navigation-LAM posterior means.
# The default launch uses nohup; logs and PID are printed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

DETACH="${DETACH:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ "${PLAN_ONLY}" == "1" ]]; then
    DEFAULT_EXPERIMENT_NAME="nav1latent-plan-$(date -u +%m%d-%H%M%S)"
else
    DEFAULT_EXPERIMENT_NAME="nav1latent8-$(date -u +%m%d-%H%M%S)"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
if [[ "${DETACH}" == "1" ]]; then
    DETACHED_LOG_DIR="${LOG_DIR:-/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_nav1_latent_actions}"
    mkdir -p "${DETACHED_LOG_DIR}"
    DETACHED_LOG="${DETACHED_LOG_DIR}/background_${RUN_TIMESTAMP}.log"
    nohup env DETACH=0 RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
        bash "${SCRIPT_PATH}" > "${DETACHED_LOG}" 2>&1 < /dev/null &
    pid=$!
    echo "${pid}" > "${DETACHED_LOG}.pid"
    echo "cwd: ${SCRIPT_DIR}"
    echo "PID: ${pid}"
    echo "Log: ${DETACHED_LOG}"
    exit 0
fi

VENV_PYTHON="${VENV_PYTHON:-/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python}"
PLAN_PYTHON="${PLAN_PYTHON:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python}"
LAM_PROJECT_ROOT="${LAM_PROJECT_ROOT:-/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/external/lam_project}"
NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere}"
COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_NAS_ROOT}/recipes/navanywhere_balanced_seed20260901.json}"
CHECKPOINT="${CHECKPOINT:-/file_system/nas/algorithm/dujun.nie/nwm/weights/navigation_lam/variant_4_pixel_action/nav1-pixel-action/checkpoints/step=100000.ckpt}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_nav1_pixel_action_step100000}"
LOG_DIR="${LOG_DIR:-${COMPACT_NAS_ROOT}/logs/navanywhere_nav1_latent_actions}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOADER_THREADS="${LOADER_THREADS:-8}"
DINO_FRAME_BATCH_SIZE="${DINO_FRAME_BATCH_SIZE:-32}"
DINO_LAM_BATCH_SIZE="${DINO_LAM_BATCH_SIZE:-16}"
PRECISION="${PRECISION:-bf16-mixed}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}"
TRAJECTORIES="${TRAJECTORIES:-}"
OVERWRITE="${OVERWRITE:-0}"
LOG_EVERY_TRAJECTORIES="${LOG_EVERY_TRAJECTORIES:-10}"
PLAN_WORLD_SIZE="${PLAN_WORLD_SIZE:-8}"
PLAN_TRAIN_BATCH_SIZE="${PLAN_TRAIN_BATCH_SIZE:-16}"
PLAN_MAX_TRAIN_STEPS="${PLAN_MAX_TRAIN_STEPS:-200000}"
PLAN_WORKERS="${PLAN_WORKERS:-16}"
PLAN_CHUNK_SIZE="${PLAN_CHUNK_SIZE:-50000}"
PLAN_OUTPUT="${PLAN_OUTPUT:-${COMPACT_NAS_ROOT}/plans/navanywhere_latent_action_seed20260901_ws${PLAN_WORLD_SIZE}_bs${PLAN_TRAIN_BATCH_SIZE}_steps${PLAN_MAX_TRAIN_STEPS}.json}"
PLAN_PAIR_BITMAP="${PLAN_PAIR_BITMAP:-${PLAN_OUTPUT%.json}.pairs.bin}"
TRAINING_PAIR_PLAN="${TRAINING_PAIR_PLAN:-${PLAN_OUTPUT}}"
FULL_PAIRS="${FULL_PAIRS:-0}"
REUSE_ROOT="${REUSE_ROOT:-}"
[[ "${FULL_PAIRS}" == 0 || "${FULL_PAIRS}" == 1 ]] || { echo "ERROR: FULL_PAIRS must be 0 or 1"; exit 2; }
TIMEPT_REFERENCE_LOG="${TIMEPT_REFERENCE_LOG:-${COMPACT_NAS_ROOT}/logs/navanywhere_stage1/train_timept_20260902_130035.log}"
GEOPT_REFERENCE_LOG="${GEOPT_REFERENCE_LOG:-${COMPACT_NAS_ROOT}/logs/navanywhere_stage1/train_geopt_20260905_020847.log}"

if [[ "${PLAN_ONLY}" == "1" ]]; then
    for required in "${PLAN_PYTHON}" "${SAMPLING_RECIPE}" \
        "${TIMEPT_REFERENCE_LOG}" "${GEOPT_REFERENCE_LOG}"; do
        [[ -r "${required}" ]] || { echo "ERROR: required plan input is missing: ${required}" >&2; exit 2; }
    done
    mkdir -p "$(dirname -- "${PLAN_OUTPUT}")" "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/plan_${RUN_TIMESTAMP}.log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    RECIPE_SHA256="$(sha256sum "${SAMPLING_RECIPE}" | awk '{print $1}')"
    echo "NavAnywhere nav1 latent-action plan-only replay"
    echo "  experiment:          ${EXPERIMENT_NAME}"
    echo "  working directory:   ${SCRIPT_DIR}"
    echo "  exact command:       ${PLAN_PYTHON} plan_navanywhere_latent_actions.py --sampling-recipe ${SAMPLING_RECIPE} --output ${PLAN_OUTPUT} --pair-bitmap-output ${PLAN_PAIR_BITMAP} --world-size ${PLAN_WORLD_SIZE} --batch-size ${PLAN_TRAIN_BATCH_SIZE} --max-train-steps ${PLAN_MAX_TRAIN_STEPS} --workers ${PLAN_WORKERS} --chunk-size ${PLAN_CHUNK_SIZE} --reference-log ${TIMEPT_REFERENCE_LOG} --reference-log ${GEOPT_REFERENCE_LOG} --expected-recipe-sha256 ${RECIPE_SHA256}"
    echo "  log:                 ${LOG_FILE}"
    echo "  output:              ${PLAN_OUTPUT}"
    echo "  pair bitmap:         ${PLAN_PAIR_BITMAP}"
    exec "${PLAN_PYTHON}" plan_navanywhere_latent_actions.py \
        --sampling-recipe "${SAMPLING_RECIPE}" \
        --output "${PLAN_OUTPUT}" \
        --pair-bitmap-output "${PLAN_PAIR_BITMAP}" \
        --world-size "${PLAN_WORLD_SIZE}" \
        --batch-size "${PLAN_TRAIN_BATCH_SIZE}" \
        --max-train-steps "${PLAN_MAX_TRAIN_STEPS}" \
        --workers "${PLAN_WORKERS}" \
        --chunk-size "${PLAN_CHUNK_SIZE}" \
        --reference-log "${TIMEPT_REFERENCE_LOG}" \
        --reference-log "${GEOPT_REFERENCE_LOG}" \
        --expected-recipe-sha256 "${RECIPE_SHA256}"
fi

REQUIRED_INPUTS=("${VENV_PYTHON}" "${SAMPLING_RECIPE}" "${CHECKPOINT}")
if [[ "${FULL_PAIRS}" == 0 ]]; then REQUIRED_INPUTS+=("${TRAINING_PAIR_PLAN}"); fi
for required in "${REQUIRED_INPUTS[@]}"; do
    [[ -r "${required}" ]] || { echo "ERROR: required input is missing: ${required}" >&2; exit 2; }
done
[[ -d "${LAM_PROJECT_ROOT}/lam" ]] || { echo "ERROR: navigation LAM project missing: ${LAM_PROJECT_ROOT}" >&2; exit 2; }
[[ -d "${NAVANYWHERE_ROOT}" ]] || { echo "ERROR: NavAnywhere root missing: ${NAVANYWHERE_ROOT}" >&2; exit 2; }
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required for the capacity check." >&2
    exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( NPROC < 1 || NPROC > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: NPROC=${NPROC} must fit GPU_IDS=${GPU_IDS}." >&2
    exit 2
fi

echo "NavAnywhere nav1 latent-action precompute"
echo "  experiment:        ${EXPERIMENT_NAME}"
echo "  working directory: ${SCRIPT_DIR}"
echo "  NavAnywhere:       ${NAVANYWHERE_ROOT}"
echo "  sampling recipe:   ${SAMPLING_RECIPE}"
echo "  training pair plan:${TRAINING_PAIR_PLAN}"
echo "  checkpoint:        ${CHECKPOINT}"
echo "  output:            ${OUTPUT_ROOT}"
echo "  GPUs/processes:    ${GPU_IDS} / ${NPROC}"
echo "  batch/precision:   ${BATCH_SIZE} / ${PRECISION}"
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
            exit 2
        fi
    done
fi
echo "Output filesystem capacity:"
df -h "${COMPACT_NAS_ROOT}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/precompute_${RUN_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "  log:               ${LOG_FILE}"

if [[ -z "${CHECKPOINT_SHA256}" ]]; then
    echo "Computing checkpoint SHA-256 once before distributed model loading..."
    CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
fi
if [[ ! "${CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: CHECKPOINT_SHA256 must be a lowercase SHA-256 digest." >&2
    exit 2
fi
echo "  checkpoint SHA-256: ${CHECKPOINT_SHA256}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=0

ARGS=(
    --data-root "${NAVANYWHERE_ROOT}"
    --sampling-recipe "${SAMPLING_RECIPE}"
    --output-root "${OUTPUT_ROOT}"
    --lam-project-root "${LAM_PROJECT_ROOT}"
    --checkpoint "${CHECKPOINT}"
    --checkpoint-sha256 "${CHECKPOINT_SHA256}"
    --precision "${PRECISION}"
    --batch-size "${BATCH_SIZE}"
    --loader-threads "${LOADER_THREADS}"
    --dino-frame-batch-size "${DINO_FRAME_BATCH_SIZE}"
    --dino-lam-batch-size "${DINO_LAM_BATCH_SIZE}"
    --image-height 240
    --image-width 320
    --context-size 4
    --max-abs-frame-offset 8
    --max-trajectories "${MAX_TRAJECTORIES}"
    --log-every-trajectories "${LOG_EVERY_TRAJECTORIES}"
)
if [[ "${FULL_PAIRS}" == 0 ]]; then ARGS+=(--training-pair-plan "${TRAINING_PAIR_PLAN}"); fi
if [[ -n "${REUSE_ROOT}" ]]; then ARGS+=(--reuse-root "${REUSE_ROOT}"); fi
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi
if [[ -n "${TRAJECTORIES}" ]]; then
    IFS=',' read -r -a TRAJECTORY_ARRAY <<< "${TRAJECTORIES}"
    for trajectory in "${TRAJECTORY_ARRAY[@]}"; do
        ARGS+=(--trajectory "${trajectory}")
    done
fi

echo "  PID: $$; full pairs: ${FULL_PAIRS}"
printf "Exact command: "; printf "%q " "${VENV_PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NPROC}" precompute_navanywhere_nav1_latent_actions.py "${ARGS[@]}"; printf "\n"
"${VENV_PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    precompute_navanywhere_nav1_latent_actions.py "${ARGS[@]}"

if [[ "${MAX_TRAJECTORIES}" == "0" && -z "${TRAJECTORIES}" ]]; then
    echo "Finished successfully. Cache marker: ${OUTPUT_ROOT}/_SUCCESS.json"
else
    echo "Partial smoke extraction finished. Metadata: ${OUTPUT_ROOT}/metadata.json"
fi
