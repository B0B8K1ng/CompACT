#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_PYTHON="${VENV_PYTHON:-${SCRIPT_DIR}/.venv-vggt-omega/bin/python}"
NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere}"
COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_NAS_ROOT}/recipes/navanywhere_balanced_seed20260901.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_geometry_proxy}"
VGGT_ROOT="${VGGT_ROOT:-${SCRIPT_DIR}/third_party/vggt_omega}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/file_system/nas/algorithm/dujun.nie/models/VGGT-Omega-1B-512}"
CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_ROOT}/vggt_omega_1b_512.pt}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-${CHECKPOINT_ROOT}/checkpoint_manifest.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
NPROC="${NPROC:-7}"
RESOLUTION="${RESOLUTION:-384}"
WINDOW_SIZE="${WINDOW_SIZE:-128}"
OVERLAP="${OVERLAP:-32}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}"
OVERWRITE="${OVERWRITE:-0}"

for required in "${VENV_PYTHON}" "${SAMPLING_RECIPE}" \
    "${CHECKPOINT}" "${CHECKPOINT_MANIFEST}"; do
    [[ -r "${required}" ]] || { echo "ERROR: required input is missing: ${required}" >&2; exit 2; }
done
[[ -d "${NAVANYWHERE_ROOT}" ]] || { echo "ERROR: ${NAVANYWHERE_ROOT} missing" >&2; exit 2; }
[[ -d "${VGGT_ROOT}" ]] || { echo "ERROR: ${VGGT_ROOT} missing" >&2; exit 2; }

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( NPROC < 1 || NPROC > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: NPROC=${NPROC} must fit GPU_IDS=${GPU_IDS}" >&2
    exit 2
fi

echo "NavAnywhere VGGT-Omega geometry precompute"
echo "  working directory: ${SCRIPT_DIR}"
echo "  input recipe:      ${SAMPLING_RECIPE}"
echo "  output cache:      ${OUTPUT_ROOT}"
echo "  GPUs/processes:    ${GPU_IDS} / ${NPROC}"
echo "  resolution/window: ${RESOLUTION} / ${WINDOW_SIZE} (overlap ${OVERLAP})"
echo "  preprocess workers: ${PREPROCESS_WORKERS} per GPU process"
echo "GPU capacity before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
if [[ "${REQUIRE_IDLE_GPUS}" == "1" ]]; then
    GPU_CAPACITY="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)"
    for gpu_id in "${GPU_ARRAY[@]:0:${NPROC}}"; do
        read -r free_mb utilization < <(
            awk -F',' -v wanted="${gpu_id}" '
                { for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
                $1 == wanted { print $2, $3; found=1 }
                END { if (!found) exit 1 }
            ' <<< "${GPU_CAPACITY}"
        )
        if (( free_mb < MIN_FREE_GPU_MB || utilization > MAX_GPU_UTILIZATION )); then
            echo "ERROR: GPU ${gpu_id} is busy (free=${free_mb} MiB, util=${utilization}%)" >&2
            exit 2
        fi
    done
fi
echo "Output filesystem capacity:"
df -h "${COMPACT_NAS_ROOT}"

mkdir -p "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Ranks can legitimately wait a long time at the final gather when another
# rank is processing an unusually long trajectory.  The default 8-minute
# watchdog heartbeat can mistake that for a hang even though the process group
# timeout is 72 hours.
export TORCH_NCCL_ENABLE_MONITORING=0

ARGS=(
    --data-root "${NAVANYWHERE_ROOT}"
    --sampling-recipe "${SAMPLING_RECIPE}"
    --output-root "${OUTPUT_ROOT}"
    --third-party-root "${VGGT_ROOT}"
    --checkpoint "${CHECKPOINT}"
    --checkpoint-manifest "${CHECKPOINT_MANIFEST}"
    --resolution "${RESOLUTION}"
    --resize-mode max_size
    --window-size "${WINDOW_SIZE}"
    --overlap "${OVERLAP}"
    --preprocess-workers "${PREPROCESS_WORKERS}"
    --context-size 4
    --max-abs-frame-offset 8
    --max-trajectories "${MAX_TRAJECTORIES}"
    --log-every-trajectories 10
)
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

exec "${VENV_PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    precompute_navanywhere_geometry_actions.py "${ARGS[@]}"
