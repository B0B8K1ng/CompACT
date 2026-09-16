#!/usr/bin/env bash
set -euo pipefail

# Prepare the deterministic v1 train/validation recipes and exact LatentPT
# pair caches. Cache extraction copies matching rows from the completed v1
# cache and runs PixelActionLAM only for missing rows.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ACTION="${1:-recipes}" # recipes | plans | cache | all
case "${ACTION}" in
    recipes|plans|cache|all) ;;
    *) echo "ERROR: action must be recipes, plans, cache, or all." >&2; exit 2 ;;
esac

NWM_ROOT="${NWM_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm}"
COMPACT_ROOT="${COMPACT_ROOT:-${NWM_ROOT}/compact}"
DATA_ROOT="${DATA_ROOT:-${NWM_ROOT}/data/NavAnywhere}"
ORIGINAL_RECIPE="${ORIGINAL_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_balanced_seed20260901.json}"
TRAIN_RECIPE="${TRAIN_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v1_13src_train_seed20260901.json}"
VAL_RECIPE="${VAL_RECIPE:-${COMPACT_ROOT}/recipes/navanywhere_v1_13src_val_seed20260901.json}"
SPLIT_REPORT="${SPLIT_REPORT:-${COMPACT_ROOT}/recipes/navanywhere_v1_13src_split_seed20260901.json}"

WORLD_SIZE="${WORLD_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
TRAIN_PLAN="${TRAIN_PLAN:-${COMPACT_ROOT}/plans/navanywhere_v1_13src_train_latent_action_seed20260901_ws${WORLD_SIZE}_bs${TRAIN_BATCH_SIZE}_steps${MAX_TRAIN_STEPS}.json}"
VAL_PLAN="${VAL_PLAN:-${COMPACT_ROOT}/plans/navanywhere_v1_13src_val_latent_action_seed20260901_ws${WORLD_SIZE}_bs${TRAIN_BATCH_SIZE}_batches1.json}"
TRAIN_CACHE="${TRAIN_CACHE:-${COMPACT_ROOT}/cache/navanywhere_v1_13src_train_nav1_pixel_action_step100000_ws${WORLD_SIZE}_bs${TRAIN_BATCH_SIZE}_steps${MAX_TRAIN_STEPS}}"
VAL_CACHE="${VAL_CACHE:-${COMPACT_ROOT}/cache/navanywhere_v1_13src_val_nav1_pixel_action_step100000_ws${WORLD_SIZE}_bs${TRAIN_BATCH_SIZE}_batches1}"
V1_REUSE_ROOT="${V1_REUSE_ROOT:-${COMPACT_ROOT}/cache/navanywhere_nav1_pixel_action_step100000}"

NWM_CONDA_ACTIVATE="${NWM_CONDA_ACTIVATE:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/activate}"
NWM_CONDA_ENV_PATH="${NWM_CONDA_ENV_PATH:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm}"
PLAN_PYTHON="${NWM_CONDA_ENV_PATH}/bin/python"
EXTRACT_PYTHON="${EXTRACT_PYTHON:-/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/scripts/run_ego4d_nav1_latent_actions.py"
CHECKPOINT="${CHECKPOINT:-${NWM_ROOT}/weights/navigation_lam/variant_4_pixel_action/nav1-pixel-action/checkpoints/step=100000.ckpt}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-ec7d4c159a0bcd661167b35ea88a1c61ac42d73a771a0de5de660cced4325ac1}"

PLAN_WORKERS="${PLAN_WORKERS:-16}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-20}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-64}"
LOADER_THREADS="${LOADER_THREADS:-16}"

if [[ "${NWM_CONDA_ACTIVATE}" != /* || "${NWM_CONDA_ENV_PATH}" != /* ]]; then
    echo "ERROR: NWM_CONDA_ACTIVATE and NWM_CONDA_ENV_PATH must be absolute paths." >&2
    exit 2
fi
for required in "${ORIGINAL_RECIPE}" "${NWM_CONDA_ACTIVATE}" "${PLAN_PYTHON}"; do
    [[ -r "${required}" ]] || { echo "ERROR: required input is missing: ${required}" >&2; exit 2; }
done
[[ -d "${NWM_CONDA_ENV_PATH}" ]] || {
    echo "ERROR: nwm conda environment does not exist: ${NWM_CONDA_ENV_PATH}" >&2
    exit 2
}

# Activate the environment by absolute path so an inherited Conda setup cannot
# redirect plan generation to a same-named environment from another install.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
# shellcheck disable=SC1090
source "${NWM_CONDA_ACTIVATE}" "${NWM_CONDA_ENV_PATH}"
echo "Conda environment=${CONDA_PREFIX}"

build_recipes() {
    if [[ -e "${TRAIN_RECIPE}" || -e "${VAL_RECIPE}" || -e "${SPLIT_REPORT}" ]]; then
        if [[ ! -r "${TRAIN_RECIPE}" || ! -r "${VAL_RECIPE}" || ! -r "${SPLIT_REPORT}" ]]; then
            echo "ERROR: v1 split outputs are only partially present." >&2
            exit 2
        fi
        echo "Reusing existing v1 split: ${SPLIT_REPORT}"
        return
    fi
    "${PLAN_PYTHON}" \
        scripts/build_navanywhere_v1_split.py \
        --inventory-recipe "${ORIGINAL_RECIPE}" \
        --train-output "${TRAIN_RECIPE}" \
        --val-output "${VAL_RECIPE}" \
        --split-output "${SPLIT_REPORT}"
}

build_plans() {
    build_recipes
    mkdir -p "$(dirname -- "${TRAIN_PLAN}")"
    if [[ ! -r "${TRAIN_PLAN}" || ! -r "${TRAIN_PLAN%.json}.pairs.bin" ]]; then
        "${PLAN_PYTHON}" \
            plan_navanywhere_latent_actions.py \
            --sampling-recipe "${TRAIN_RECIPE}" \
            --output "${TRAIN_PLAN}" \
            --pair-bitmap-output "${TRAIN_PLAN%.json}.pairs.bin" \
            --world-size "${WORLD_SIZE}" \
            --batch-size "${TRAIN_BATCH_SIZE}" \
            --max-train-steps "${MAX_TRAIN_STEPS}" \
            --workers "${PLAN_WORKERS}"
    else
        echo "Reusing existing training pair plan: ${TRAIN_PLAN}"
    fi
    if [[ ! -r "${VAL_PLAN}" || ! -r "${VAL_PLAN%.json}.pairs.bin" ]]; then
        "${PLAN_PYTHON}" \
            plan_navanywhere_latent_actions.py \
            --sampling-recipe "${VAL_RECIPE}" \
            --output "${VAL_PLAN}" \
            --pair-bitmap-output "${VAL_PLAN%.json}.pairs.bin" \
            --usage validation \
            --world-size "${WORLD_SIZE}" \
            --batch-size "${TRAIN_BATCH_SIZE}" \
            --max-train-steps 1 \
            --workers 0
    else
        echo "Reusing existing validation pair plan: ${VAL_PLAN}"
    fi
}

extract_cache() {
    build_plans
    for required in "${EXTRACT_PYTHON}" "${RUNNER}" "${CHECKPOINT}" "${V1_REUSE_ROOT}"; do
        [[ -r "${required}" ]] || { echo "ERROR: required cache input is missing: ${required}" >&2; exit 2; }
    done
    command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi is required." >&2; exit 2; }
    echo "GPU capacity before cache supplementation:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
    echo "Cache filesystem capacity:"
    df -h "${COMPACT_ROOT}"

    common_args=(
        --data-root "${DATA_ROOT}"
        --lam-project-root "${SCRIPT_DIR}/../DreamDojo/external/lam_project"
        --checkpoint "${CHECKPOINT}"
        --checkpoint-sha256 "${CHECKPOINT_SHA256}"
        --reuse-root "${V1_REUSE_ROOT}"
        --gpu-ids "${GPU_IDS}"
        --min-free-mib "${MIN_FREE_MIB}"
        --max-utilization "${MAX_UTILIZATION}"
        --batch-size "${EXTRACT_BATCH_SIZE}"
        --loader-threads "${LOADER_THREADS}"
        --stream-pair-chunk-size 4096
        --poll-seconds 30
        --progress-seconds 120
    )
    "${EXTRACT_PYTHON}" -u "${RUNNER}" run \
        --sampling-recipe "${VAL_RECIPE}" \
        --training-pair-plan "${VAL_PLAN}" \
        --output-root "${VAL_CACHE}" \
        "${common_args[@]}"
    "${EXTRACT_PYTHON}" -u "${RUNNER}" run \
        --sampling-recipe "${TRAIN_RECIPE}" \
        --training-pair-plan "${TRAIN_PLAN}" \
        --output-root "${TRAIN_CACHE}" \
        "${common_args[@]}"
}

case "${ACTION}" in
    recipes) build_recipes ;;
    plans) build_plans ;;
    cache) extract_cache ;;
    all) extract_cache ;;
esac
