#!/usr/bin/env bash
set -euo pipefail

# Keep the current four-rank extraction running until GPUs 0-3 are genuinely
# idle, then atomically hand the resumable cache over to an eight-rank launch.
OLD_EXPERIMENT="${OLD_EXPERIMENT:-nav1-latent-full-fast64-0906}"
NEW_EXPERIMENT="${NEW_EXPERIMENT:-nav1-latent-full-8gpu-resume-0906}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:?RUN_TIMESTAMP must be set by the caller}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/precompute_navanywhere_nav1_latent_actions_8gpu.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/navanywhere_nav1_pixel_action_step100000}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"

gpu_range_idle() {
    local last_gpu="$1"
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits | awk -F',' \
        -v last_gpu="${last_gpu}" \
        -v min_free="${MIN_FREE_GPU_MB}" \
        -v max_util="${MAX_GPU_UTILIZATION}" '
            {
                for (i = 1; i <= NF; i++) {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                }
            }
            $1 >= 0 && $1 <= last_gpu {
                seen++
                if ($2 >= min_free && $3 <= max_util) idle++
            }
            END { exit !(seen == last_gpu + 1 && idle == seen) }
        '
}

echo "Waiting for GPUs 0-3 to remain idle for three consecutive checks."
stable_checks=0
while (( stable_checks < 3 )); do
    if gpu_range_idle 3; then
        stable_checks=$((stable_checks + 1))
        echo "$(date -u +%FT%TZ) idle check ${stable_checks}/3"
    else
        stable_checks=0
    fi
    sleep 20
done

echo "$(date -u +%FT%TZ) stopping ${OLD_EXPERIMENT} at an atomic-cache boundary"
codex-exp stop "${OLD_EXPERIMENT}" || true

echo "Waiting for all eight GPUs and the output lock to be released."
until gpu_range_idle 7; do
    sleep 5
done

cd "${SCRIPT_DIR}"
echo "$(date -u +%FT%TZ) resuming ${OUTPUT_ROOT} on GPUs 0-7"
exec env \
    RUN_UNDER_CODEX_EXP=1 \
    DETACH=0 \
    EXPERIMENT_NAME="${NEW_EXPERIMENT}" \
    RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    GPU_IDS=0,1,2,3,4,5,6,7 \
    NPROC=8 \
    BATCH_SIZE=64 \
    LOADER_THREADS=16 \
    PRECISION=bf16-mixed \
    REQUIRE_IDLE_GPUS=1 \
    "${LAUNCHER}"
