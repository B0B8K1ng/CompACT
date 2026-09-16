#!/usr/bin/env bash
set -euo pipefail

# Foreground, resumable extraction of only the exact NavAnywhere v2 train and
# validation frame pairs. The dynamic supervisor uses every currently idle GPU
# from GPU_IDS and waits safely when none are available.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NWM_ROOT="${NWM_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm}"
DATA_ROOT="${DATA_ROOT:-${NWM_ROOT}/data/NavAnywhere}"
COMPACT_ROOT="${COMPACT_ROOT:-${NWM_ROOT}/compact}"
VENV_PYTHON="${VENV_PYTHON:-/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/scripts/run_ego4d_nav1_latent_actions.py"
CHECKPOINT="${CHECKPOINT:-${NWM_ROOT}/weights/navigation_lam/variant_4_pixel_action/nav1-pixel-action/checkpoints/step=100000.ckpt}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-ec7d4c159a0bcd661167b35ea88a1c61ac42d73a771a0de5de660cced4325ac1}"

TRAIN_RECIPE="${TRAIN_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v2_15src_train_seed20260901.json}"
VAL_RECIPE="${VAL_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v2_15src_val_seed20260901.json}"
TRAIN_PLAN="${TRAIN_PLAN:-${COMPACT_ROOT}/plans/navanywhere_v2_15src_train_latent_action_seed20260901_ws8_bs16_steps200000.json}"
VAL_PLAN="${VAL_PLAN:-${COMPACT_ROOT}/plans/navanywhere_v2_15src_val_latent_action_seed20260901_ws8_bs16_batches1.json}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${COMPACT_ROOT}/cache/navanywhere_v2_15src_train_nav1_pixel_action_step100000_ws8_bs16_steps200000}"
VAL_OUTPUT="${VAL_OUTPUT:-${COMPACT_ROOT}/cache/navanywhere_v2_15src_val_nav1_pixel_action_step100000_ws8_bs16_batches1}"

V1_REUSE_ROOT="${V1_REUSE_ROOT:-${COMPACT_ROOT}/cache/navanywhere_nav1_pixel_action_step100000}"
EGO4D_REUSE_ROOT="${EGO4D_REUSE_ROOT:-${COMPACT_ROOT}/cache/ego4d_nav1_pixel_action_step100000_ws8_bs16_steps200000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOADER_THREADS="${LOADER_THREADS:-16}"
STREAM_PAIR_CHUNK_SIZE="${STREAM_PAIR_CHUNK_SIZE:-4096}"

for required in "${VENV_PYTHON}" "${RUNNER}" "${CHECKPOINT}" \
    "${TRAIN_RECIPE}" "${VAL_RECIPE}" "${TRAIN_PLAN}" "${VAL_PLAN}"; do
    [[ -r "${required}" ]] || { echo "ERROR: required input is missing: ${required}" >&2; exit 2; }
done
for reuse_root in "${V1_REUSE_ROOT}" "${EGO4D_REUSE_ROOT}"; do
    [[ -d "${reuse_root}" ]] || { echo "ERROR: reuse cache is missing: ${reuse_root}" >&2; exit 2; }
done

echo "NavAnywhere v2 exact latent-action extraction"
echo "  working directory: ${SCRIPT_DIR}"
echo "  GPUs (dynamic):     ${GPU_IDS}"
echo "  validation output: ${VAL_OUTPUT}"
echo "  training output:   ${TRAIN_OUTPUT}"
echo "  row reuse:         ${V1_REUSE_ROOT}, ${EGO4D_REUSE_ROOT}"
echo "  order:             validation first, then training"

COMMON_ARGS=(
    --data-root "${DATA_ROOT}"
    --lam-project-root "${SCRIPT_DIR}/../DreamDojo/external/lam_project"
    --checkpoint "${CHECKPOINT}"
    --checkpoint-sha256 "${CHECKPOINT_SHA256}"
    --reuse-root "${V1_REUSE_ROOT}"
    --reuse-root "${EGO4D_REUSE_ROOT}"
    --gpu-ids "${GPU_IDS}"
    --min-free-mib "${MIN_FREE_MIB}"
    --max-utilization "${MAX_UTILIZATION}"
    --batch-size "${BATCH_SIZE}"
    --loader-threads "${LOADER_THREADS}"
    --stream-pair-chunk-size "${STREAM_PAIR_CHUNK_SIZE}"
    --poll-seconds 30
    --progress-seconds 120
)

"${VENV_PYTHON}" -u "${RUNNER}" run \
    --sampling-recipe "${VAL_RECIPE}" \
    --training-pair-plan "${VAL_PLAN}" \
    --output-root "${VAL_OUTPUT}" \
    "${COMMON_ARGS[@]}"

"${VENV_PYTHON}" -u "${RUNNER}" run \
    --sampling-recipe "${TRAIN_RECIPE}" \
    --training-pair-plan "${TRAIN_PLAN}" \
    --output-root "${TRAIN_OUTPUT}" \
    "${COMMON_ARGS[@]}"

echo "NavAnywhere v2 train and validation latent-action caches are complete."
