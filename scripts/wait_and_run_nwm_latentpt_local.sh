#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the three local navigation-LAM caches and an actually idle 8xL20
# node, probe the largest safe per-GPU batch, then run LatentPT -> reset.

REPO_ROOT="/home/user/ndj/code/CompACT"
TRAIN_LAUNCHER="${REPO_ROOT}/run_nwm_latentpt_nav1_80gb.sh"
NWM_ENV="/home/user/ndj/miniconda3/envs/nwm"
OUTPUT_PARENT="/data1/ndj/nwm_runs/nav1_latentpt_l20"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${OUTPUT_PARENT}/supervisor.log}"
STATUS_FILE="${OUTPUT_PARENT}/supervisor.status"
LOCK_FILE="${OUTPUT_PARENT}/supervisor.lock"
EXTRACTION_POLL_SECONDS="${EXTRACTION_POLL_SECONDS:-30}"
GPU_IDLE_POLL_SECONDS="${GPU_IDLE_POLL_SECONDS:-20}"

mkdir -p "${OUTPUT_PARENT}"
exec >>"${SUPERVISOR_LOG}" 2>&1
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "$(date -u +%FT%TZ) another supervisor owns ${LOCK_FILE}"
    exit 2
fi

set_status() {
    local state="$1"
    local detail="$2"
    local temporary="${STATUS_FILE}.tmp.$$"
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${state}" "${detail}" >"${temporary}"
    mv -f -- "${temporary}" "${STATUS_FILE}"
    echo "$(date -u +%FT%TZ) status=${state} detail=${detail}"
}

fail() {
    set_status FAILED "$*"
    exit 1
}

trap 'fail "line=${LINENO} command=${BASH_COMMAND} exit=$?"' ERR

EXTRACTION_ROOTS=(
    /data1/ndj/LAM-Data/navv1_latent_actions_u25l100_step40000
    /data1/ndj/LAM-Data/navv1_latent_actions_u50l100_step40000
    /data1/ndj/LAM-Data/navv1_latent_actions_u75l100_step40000
)
EXPERIMENT_TAGS=(u25l100 u50l100 u75l100)
CHECKPOINT_SHA256S=(
    ce287808ac08ee030cda3af27e8ae96e5bd070fd2ccc5615494abec1921295f3
    8286c4371a9e8e5f71643ead8b8b84c607a01509d3c32b674d042ecf3002381d
    9cc743d86ceb8436bb85fbaaa96b1c9f075f21171e2b95b0b533c8af060643fe
)

validate_extractions() {
    "${NWM_ENV}/bin/python" - "${EXTRACTION_ROOTS[@]}" <<'PY'
import json
import pathlib
import sys

for raw in sys.argv[1:]:
    root = pathlib.Path(raw)
    for name in ("plan.json", "metadata.json", "_SUCCESS.json"):
        path = root / name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    metadata = json.loads((root / "metadata.json").read_text())
    success = json.loads((root / "_SUCCESS.json").read_text())
    if metadata.get("complete") is not True or metadata.get("status") != "complete":
        raise SystemExit(f"incomplete metadata: {root}")
    if success.get("complete") is not True:
        raise SystemExit(f"incomplete success marker: {root}")
    failures = metadata.get("failures", [])
    if failures:
        raise SystemExit(f"recorded extraction failures under {root}: {len(failures)}")
print("validated latent-action caches:", ", ".join(sys.argv[1:]))
PY
}

set_status WAITING_EXTRACTIONS "u25l100,u50l100,u75l100"
missing_process_checks=0
while :; do
    complete=1
    for root in "${EXTRACTION_ROOTS[@]}"; do
        [[ -f "${root}/_SUCCESS.json" && -f "${root}/metadata.json" ]] || complete=0
    done
    if (( complete == 1 )); then
        validate_extractions
        break
    fi
    if ! pgrep -f '[r]un_navv1_ablation_latent_actions.sh' >/dev/null && \
       ! pgrep -f '[p]recompute_finetune_nav1_actions.py' >/dev/null; then
        missing_process_checks=$((missing_process_checks + 1))
        echo "$(date -u +%FT%TZ) extraction process absent (${missing_process_checks}/3 grace checks)"
        if (( missing_process_checks >= 3 )); then
            fail "latent-action extraction remained absent before all three success markers appeared"
        fi
    else
        missing_process_checks=0
    fi
    echo "$(date -u +%FT%TZ) extraction still running"
    sleep "${EXTRACTION_POLL_SECONDS}"
done

gpu_snapshot() {
    nvidia-smi \
        --query-gpu=index,memory.total,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
}

gpus_idle() {
    local snapshot="$1"
    local count
    count="$(awk -F',' '
        {for(i=1;i<=NF;i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)}
        $2 >= 45000 && $3 >= 44000 && $4 <= 10 {ok++}
        END {print ok+0}
    ' <<<"${snapshot}")"
    [[ "${count}" == "8" ]]
}

wait_for_idle_gpus() {
    local stable=0
    local snapshot
    set_status WAITING_GPUS "require 8 GPUs free>=44000MiB util<=10% for three checks"
    while (( stable < 3 )); do
        snapshot="$(gpu_snapshot)"
        if gpus_idle "${snapshot}"; then
            stable=$((stable + 1))
            echo "$(date -u +%FT%TZ) idle GPU check ${stable}/3"
        else
            stable=0
            echo "$(date -u +%FT%TZ) GPUs not exclusively available"
            echo "${snapshot}"
        fi
        (( stable == 3 )) || sleep "${GPU_IDLE_POLL_SECONDS}"
    done
}

source /home/user/ndj/miniconda3/bin/activate "${NWM_ENV}"
cd "${REPO_ROOT}"

export GPU_MEMORY_GB=48
export GPU_IDS=0,1,2,3,4,5,6,7
export RESULTS_ROOT="${OUTPUT_PARENT}/runs"
export NAVANYWHERE_ROOT=/data1/ndj/LAM-Data/NavAnywhere
export NWM_DATA_ROOT=/data1/ndj/LAM-Data
export SAMPLING_RECIPE=/data1/ndj/datasets/hongyu/navanywhere_v1_13src_train_seed20260901.json
export RECIPE_SHA256=7bdaca48d10df941aef367703ef3a867b8ceef85b991d1eb487ea95511b6ccf4
export STAGE2_VAE_LATENT_ROOT=/data1/ndj/LAM-Data/vae_latents_sd_vae_ft_ema_224_four_datasets
export STAGE1_VAE_LATENT_ROOT="${STAGE2_VAE_LATENT_ROOT}"
export STAGE1_DATASET_PROFILE=nwm_real
export NWM_CONDA_ACTIVATE=/home/user/ndj/miniconda3/bin/activate
export NWM_CONDA_ENV_PATH="${NWM_ENV}"
export NWM_HF_HOME=/data1/ndj/nwm_benchmark/models
# DreamSim is read independently by all eight ranks. Keep the already-verified
# read-only weights on tmpfs so low-frequency eval cannot be dominated by NAS
# contention; training data and all outputs remain persistent under /data1.
export NWM_MODEL_CACHE=/dev/shm/compact_nwm_dreamsim
for eval_asset in \
    dino_vitb16_pretrain.pth \
    open_clip_vitb16_pretrain.pth.tar \
    clip_vitb16_pretrain.pth.tar \
    ensemble_lora/adapter_config.json \
    ensemble_lora/adapter_model.safetensors \
    facebookresearch_dino_main/hubconf.py \
    checkpoints/dino_vitbase16_pretrain.pth; do
    [[ -r "${NWM_MODEL_CACHE}/${eval_asset}" ]] || \
        fail "missing RAM-cached DreamSim asset: ${NWM_MODEL_CACHE}/${eval_asset}"
done
export VAE_MODEL_PATH=/data1/ndj/nwm_benchmark/models/sd-vae-ft-ema
export REFERENCE_SAMPLES_PER_RANK_PER_EPOCH=40680
export EXPECTED_PLAN_BATCH_SIZE=16
export EXPECTED_PLAN_DATASET_LENGTH=4114668
export EXPECTED_PLAN_MAX_STEPS=200000
export EXPECTED_PLAN_STEPS_PER_EPOCH=32145
export EXPECTED_PLAN_WORLD_SIZE=8
export STAGE1_ALLOW_RECIPE_SUBSET=true
# The Stage-2 cache was generated by an older software stack. Its immutable
# posterior tensors and all generation/file fingerprints remain validated.
export STAGE2_ALLOW_SOFTWARE_MISMATCH=true
# The same four-dataset tree is mounted under /data1 on this host instead of
# the extraction host's NAS prefix; manifests and cache files stay validated.
export STAGE2_ALLOW_DATA_ROOT_RELOCATION=true
export NWM_NAVANYWHERE_VAL_ENABLED=false
export NWM_NAVANYWHERE_VAL_RECIPE=/data1/ndj/datasets/hongyu/navanywhere_v1_13src_val_seed20260901.json
export NWM_NAVANYWHERE_VAL_PROXY_ROOT=/data1/ndj/datasets/hongyu/navanywhere_v1_13src_val_nav1_pixel_action_step100000_ws8_bs16_batches1
export WANDB_ENABLED=true
export WANDB_MODE=online
export WANDB_PROJECT=compact-nwm
export LOG_EVERY=50
export CKPT_EVERY=5000
export EVAL_EVERY=10000
export EVAL_BATCH_SIZE=16
export EVAL_AT_FIRST_STEP=false
export EVAL_OFFLOAD_MODELS=true
export STAGE1_NUM_WORKERS=16
export STAGE2_NUM_WORKERS=8
export MIN_FREE_GPU_MB=44000
export MAX_GPU_UTILIZATION=10
export MIN_FREE_DISK_GB=500

# Re-check the largest viable L20 batch with the actual four-dataset cache.
# Two optimizer steps per phase cover initialization, forward/backward,
# optimizer, checkpoint transition, and the Stage-2 adapter reset.
chosen_batch="${REUSE_VALIDATED_BATCH:-}"
if [[ -n "${chosen_batch}" ]]; then
    [[ "${chosen_batch}" == "48" || "${chosen_batch}" == "32" ]] || \
        fail "REUSE_VALIDATED_BATCH must be 48 or 32"
    echo "$(date -u +%FT%TZ) reusing previously validated batch ${chosen_batch}"
fi
for candidate in 48 32; do
    [[ -z "${chosen_batch}" ]] || break
    while :; do
        wait_for_idle_gpus
        probe_id="probe_$(date -u +%Y%m%d_%H%M%S)_bs${candidate}"
        probe_log="${OUTPUT_PARENT}/${probe_id}.log"
        set_status PROBING "batch=${candidate} log=${probe_log}"
        if RUN_ID="${probe_id}" BATCH_SIZE="${candidate}" \
            EXPERIMENT_TAG="probe_${EXPERIMENT_TAGS[0]}" \
            NAV1_PROXY_ROOT="${EXTRACTION_ROOTS[0]}" \
            LATENT_ACTION_CHECKPOINT_SHA256="${CHECKPOINT_SHA256S[0]}" \
            STAGE1_SAMPLES_PER_RANK="$((candidate * 2))" \
            WARMUP_SAMPLES_PER_RANK="$((candidate * 2))" \
            JOINT_SAMPLES_PER_RANK="$((candidate * 2))" \
            WANDB_ENABLED=false EVAL_EVERY=999999 EVAL_AT_FIRST_STEP=false \
            EVAL_OFFLOAD_MODELS=true LOG_EVERY=1 CKPT_EVERY=999999 \
            NWM_NAVANYWHERE_VAL_ENABLED=true \
            STAGE1_NUM_WORKERS=2 STAGE2_NUM_WORKERS=2 \
                bash "${TRAIN_LAUNCHER}" >"${probe_log}" 2>&1; then
            probe_status=0
        else
            probe_status=$?
        fi
        if (( probe_status == 0 )); then
            chosen_batch="${candidate}"
            echo "$(date -u +%FT%TZ) batch ${candidate} passed both stages"
            break
        fi
        if grep -Eqi 'GPU [0-9]+ is busy|GPUs not exclusively available' "${probe_log}"; then
            echo "$(date -u +%FT%TZ) another job won the GPU launch race; retrying batch ${candidate} only after full-node idle"
            continue
        fi
        if grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${probe_log}"; then
            echo "$(date -u +%FT%TZ) batch ${candidate} exceeded stable memory; trying next candidate"
            break
        fi
        fail "non-OOM memory probe failure for batch ${candidate}; see ${probe_log}"
    done
    [[ -z "${chosen_batch}" ]] || break
done
[[ -n "${chosen_batch}" ]] || fail "no safe batch among 48,32 after batches 96,80,64 exceeded memory"

wait_for_idle_gpus
# This local time-budget profile retains the original optimizer/LR contract but
# caps total samples so Stage 1 + Stage 2 fits comfortably inside twelve hours
# on 8xL20 at the selected large batch.
export STAGE1_SAMPLES_PER_RANK=1000000
export WARMUP_SAMPLES_PER_RANK=50000
export JOINT_SAMPLES_PER_RANK=500000

completed_runs=()
for index in "${!EXTRACTION_ROOTS[@]}"; do
    tag="${EXPERIMENT_TAGS[index]}"
    cache_root="${EXTRACTION_ROOTS[index]}"
    checkpoint_sha="${CHECKPOINT_SHA256S[index]}"
    resume_existing=0
    if (( index == 0 )) && [[ -n "${RESUME_U25_RUN_ID:-}" ]]; then
        run_id="${RESUME_U25_RUN_ID}"
        [[ -d "${RESULTS_ROOT}/${run_id}" ]] || \
            fail "requested u25 resume directory is missing: ${RESULTS_ROOT}/${run_id}"
        resume_existing=1
    else
        run_id="nav1_latentpt_${tag}_l20_$(date -u +%Y%m%d_%H%M%S)_bs${chosen_batch}"
    fi
    run_log="${OUTPUT_PARENT}/${run_id}.log"
    set_status TRAINING "experiment=$((index + 1))/3 tag=${tag} run_id=${run_id} batch=${chosen_batch} log=${run_log}"

    attempt=0
    train_status=1
    while (( attempt < 3 )); do
        attempt=$((attempt + 1))
        if (( attempt == 1 && resume_existing == 0 )); then
            if RUN_ID="${run_id}" BATCH_SIZE="${chosen_batch}" \
                EXPERIMENT_TAG="${tag}" NAV1_PROXY_ROOT="${cache_root}" \
                LATENT_ACTION_CHECKPOINT_SHA256="${checkpoint_sha}" \
                bash "${TRAIN_LAUNCHER}" >>"${run_log}" 2>&1; then
                train_status=0
            else
                train_status=$?
            fi
        else
            if RUN_ID="${run_id}" BATCH_SIZE="${chosen_batch}" RESUME=1 \
                EXPERIMENT_TAG="${tag}" NAV1_PROXY_ROOT="${cache_root}" \
                LATENT_ACTION_CHECKPOINT_SHA256="${checkpoint_sha}" \
                bash "${TRAIN_LAUNCHER}" --resume >>"${run_log}" 2>&1; then
                train_status=0
            else
                train_status=$?
            fi
        fi
        (( train_status != 0 )) || break
        if grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${run_log}"; then
            fail "production OOM for ${tag} after a successful two-stage probe; see ${run_log}"
        fi
        echo "$(date -u +%FT%TZ) ${tag} attempt ${attempt}/3 failed; waiting for idle GPUs before checkpoint resume"
        (( attempt < 3 )) || break
        wait_for_idle_gpus
    done
    (( train_status == 0 )) || fail "${tag} failed after three resumable attempts; see ${run_log}"
    completed_runs+=("${run_id}")
    echo "$(date -u +%FT%TZ) completed ${tag}: ${run_id}"
    if (( index + 1 < ${#EXTRACTION_ROOTS[@]} )); then
        wait_for_idle_gpus
    fi
done

set_status COMPLETE "runs=${completed_runs[*]} batch=${chosen_batch}"
