#!/usr/bin/env bash
# Run nwm-latent in the requested strict order:
# time, rollout, Go Stanford unseen, RECON nav, SCAND nav.
set -euo pipefail

CONDA_ROOT="/file_system/vepfs/algorithm/dujun.nie/miniconda3"
source "${CONDA_ROOT}/bin/activate" nwm

MODEL="nwm-latent"
EXP_DIR="/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260824_163328_nwm-latent-badlam_bs16"
CHECKPOINT="${EXP_DIR}/checkpoints/0200000.pth.tar"
BENCHMARK_ROOT="/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark"
PREDICTION_ROOT="${BENCHMARK_ROOT}/predictions/${MODEL}"
REGISTRY="${BENCHMARK_ROOT}/benchmark_results.json"
COMPARISON="${BENCHMARK_ROOT}/benchmark_comparison.md"
GT_ROOT="/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/gt/recon"
DREAMSIM_CACHE="/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models"
GPUS="${NWM_BENCHMARK_GPUS:-0,1,2,3}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"

export NWM_DATA_ROOT="${NWM_DATA_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data}"
export NWM_INDEX_ROOT="${NWM_INDEX_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices}"
export TORCH_HOME="${TORCH_HOME:-/file_system/vepfs/algorithm/dujun.nie/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${PREDICTION_ROOT}"
exec 9>"${BENCHMARK_ROOT}/nwm-latent.lock"
if ! flock -n 9; then
    echo "Another nwm-latent benchmark pipeline already holds the lock." >&2
    exit 1
fi

if [[ ! -s "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

metric() {
    local evaluation="$1"
    local gt_dir="$2"
    local pred_dir="$3"
    local audit="$4"
    local frames="$5"
    local rollout_fps="${6:-}"

    if [[ ! -s "${audit}" ]]; then
        local command=(
            python scripts/evaluate_nwm_predictions.py
            --gt-dir "${gt_dir}"
            --pred-dir "${pred_dir}"
            --output "${audit}"
            --frames "${frames}"
            --dataset recon
            --eval-type "${evaluation%%_*}"
            --eval-name "${evaluation}"
            --batch-size 32
            --device cuda
            --dreamsim-cache "${DREAMSIM_CACHE}"
        )
        if [[ -n "${rollout_fps}" ]]; then
            command+=(--rollout-fps "${rollout_fps}")
        fi
        CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "${command[@]}"
    fi

    python scripts/nwm_benchmark_registry.py \
        --registry "${REGISTRY}" \
        import-prediction \
        --model "${MODEL}" \
        --dataset recon \
        --evaluation "${evaluation}" \
        --audit "${audit}"
}

echo "[$(date --iso-8601=seconds)] Phase 1/5: RECON time inference"
TIME_AUDIT="${PREDICTION_ROOT}/recon_time_audit.json"
if [[ ! -s "${TIME_AUDIT}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc-per-node="${NPROC}" isolated_nwm_infer.py \
        "exp_dir=${EXP_DIR}" \
        ckp=0200000 \
        "output_dir=${PREDICTION_ROOT}" \
        "prediction_dir=${PREDICTION_ROOT}" \
        'datasets_to_eval=[recon]' \
        eval_type=time \
        batch_size=64 \
        num_workers=8 \
        pin_memory=false \
        seed=0
fi
metric \
    time \
    "${GT_ROOT}/time" \
    "${PREDICTION_ROOT}/recon/time" \
    "${TIME_AUDIT}" \
    '1s:1,2s:2,4s:4,8s:8,16s:16'
echo "[$(date --iso-8601=seconds)] Phase 1/5 complete"

echo "[$(date --iso-8601=seconds)] Phase 2/5: RECON rollout inference"
ROLLOUT_1_AUDIT="${PREDICTION_ROOT}/recon_rollout_1fps_audit.json"
ROLLOUT_4_AUDIT="${PREDICTION_ROOT}/recon_rollout_4fps_audit.json"
if [[ ! -s "${ROLLOUT_1_AUDIT}" || ! -s "${ROLLOUT_4_AUDIT}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc-per-node="${NPROC}" isolated_nwm_infer.py \
        "exp_dir=${EXP_DIR}" \
        ckp=0200000 \
        "output_dir=${PREDICTION_ROOT}" \
        "prediction_dir=${PREDICTION_ROOT}" \
        'datasets_to_eval=[recon]' \
        eval_type=rollout \
        'rollout_fps_values=[1,4]' \
        use_efficient_rollout=true \
        batch_size=64 \
        num_workers=4 \
        pin_memory=false \
        seed=0
fi
metric \
    rollout_1fps \
    "${GT_ROOT}/rollout_1fps" \
    "${PREDICTION_ROOT}/recon/rollout_1fps" \
    "${ROLLOUT_1_AUDIT}" \
    '1s:0,2s:1,4s:3,8s:7,16s:15' \
    1
metric \
    rollout_4fps \
    "${GT_ROOT}/rollout_4fps" \
    "${PREDICTION_ROOT}/recon/rollout_4fps" \
    "${ROLLOUT_4_AUDIT}" \
    '1s:3,2s:7,4s:15,8s:31,16s:63' \
    4
echo "[$(date --iso-8601=seconds)] Phase 2/5 complete"

echo "[$(date --iso-8601=seconds)] Phase 3/5: Go Stanford unseen evaluation"
python scripts/run_nwm_benchmark.py \
    --models "${MODEL}" \
    --tasks unseen \
    --gpus "${GPUS}"
echo "[$(date --iso-8601=seconds)] Phase 3/5 complete"

echo "[$(date --iso-8601=seconds)] Phase 4/5: RECON navigation"
python scripts/run_nwm_benchmark.py \
    --models "${MODEL}" \
    --tasks navigation \
    --gpus "${GPUS}" \
    --navigation-datasets recon
echo "[$(date --iso-8601=seconds)] Phase 4/5 complete"

echo "[$(date --iso-8601=seconds)] Phase 5/5: SCAND navigation"
python scripts/run_nwm_benchmark.py \
    --models "${MODEL}" \
    --tasks navigation \
    --gpus "${GPUS}" \
    --navigation-datasets scand
echo "[$(date --iso-8601=seconds)] Phase 5/5 complete"

python scripts/nwm_benchmark_registry.py \
    --registry "${REGISTRY}" \
    render \
    --output "${COMPARISON}"
echo "[$(date --iso-8601=seconds)] All nwm-latent evaluations complete: ${COMPARISON}"
