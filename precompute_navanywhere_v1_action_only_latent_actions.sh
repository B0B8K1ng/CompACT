#!/usr/bin/env bash
set -euo pipefail

# Extract the exact NavAnywhere-v1 validation and training pair caches with the
# completed navigation ActionOnlyLAM. Validation is completed first so Stage-1
# evaluation has a fully published cache before the larger training pass.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NWM_ROOT="${NWM_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm}"
COMPACT_ROOT="${COMPACT_ROOT:-${NWM_ROOT}/compact}"
PYTHON="${EXTRACT_PYTHON:-/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/scripts/run_ego4d_nav1_latent_actions.py"
LAM_PROJECT_ROOT="${LAM_PROJECT_ROOT:-${SCRIPT_DIR}/../DreamDojo/external/lam_project}"
DATA_ROOT="${DATA_ROOT:-${NWM_ROOT}/data/NavAnywhere}"
CHECKPOINT="${CHECKPOINT:-${NWM_ROOT}/weights/navigation_lam/variant_1_action/action/checkpoints/step=100000.ckpt}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-3f0c270a8a117013b694254c8f5bfde6239baf7a1e446b96bbcbae5f2d43b43b}"

VAL_RECIPE="${VAL_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v1_13src_val_seed20260901.json}"
VAL_PLAN="${VAL_PLAN:-${COMPACT_ROOT}/plans/navanywhere_v1_13src_val_latent_action_seed20260901_ws8_bs16_batches1.json}"
VAL_CACHE="${VAL_CACHE:-${COMPACT_ROOT}/cache/navanywhere_v1_13src_val_nav1_action_only_step100000_ws8_bs16_batches1}"
TRAIN_RECIPE="${TRAIN_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v1_13src_train_seed20260901.json}"
TRAIN_PLAN="${TRAIN_PLAN:-${COMPACT_ROOT}/plans/navanywhere_v1_13src_train_latent_action_seed20260901_ws8_bs16_steps200000.json}"
TRAIN_CACHE="${TRAIN_CACHE:-${COMPACT_ROOT}/cache/navanywhere_v1_13src_train_nav1_action_only_step100000_ws8_bs16_steps200000}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOADER_THREADS="${LOADER_THREADS:-16}"

for required in \
    "${PYTHON}" "${RUNNER}" "${CHECKPOINT}" \
    "${VAL_RECIPE}" "${VAL_PLAN}" "${TRAIN_RECIPE}" "${TRAIN_PLAN}"; do
    [[ -r "${required}" ]] || {
        echo "ERROR: required input is missing: ${required}" >&2
        exit 2
    }
done
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: data root is missing: ${DATA_ROOT}" >&2; exit 2; }
[[ -d "${LAM_PROJECT_ROOT}/lam" ]] || {
    echo "ERROR: navigation LAM project is missing: ${LAM_PROJECT_ROOT}" >&2
    exit 2
}

echo "NavAnywhere-v1 ActionOnlyLAM latent extraction"
echo "  working directory: ${SCRIPT_DIR}"
echo "  checkpoint:        ${CHECKPOINT}"
echo "  validation cache:  ${VAL_CACHE}"
echo "  training cache:    ${TRAIN_CACHE}"
echo "  GPUs:              ${GPU_IDS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
df -h "${COMPACT_ROOT}"

common_args=(
    --data-root "${DATA_ROOT}"
    --lam-project-root "${LAM_PROJECT_ROOT}"
    --checkpoint "${CHECKPOINT}"
    --checkpoint-sha256 "${CHECKPOINT_SHA256}"
    --gpu-ids "${GPU_IDS}"
    --min-free-mib "${MIN_FREE_MIB}"
    --max-utilization "${MAX_UTILIZATION}"
    --batch-size "${BATCH_SIZE}"
    --loader-threads "${LOADER_THREADS}"
    --stream-pair-chunk-size 4096
    --poll-seconds 30
    --progress-seconds 120
)

"${PYTHON}" -u "${RUNNER}" run \
    --sampling-recipe "${VAL_RECIPE}" \
    --training-pair-plan "${VAL_PLAN}" \
    --output-root "${VAL_CACHE}" \
    "${common_args[@]}"

"${PYTHON}" -u "${RUNNER}" run \
    --sampling-recipe "${TRAIN_RECIPE}" \
    --training-pair-plan "${TRAIN_PLAN}" \
    --output-root "${TRAIN_CACHE}" \
    "${common_args[@]}"
