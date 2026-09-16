#!/usr/bin/env bash
# Run the missing stochastic evaluation seeds serially with isolated outputs.
set -euo pipefail

PROJECT_ROOT="/file_system/vepfs/algorithm/dujun.nie/code/CompACT"
BENCHMARK_BASE="/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark"
LOG_ROOT="${BENCHMARK_BASE}/evaluation_seeds/logs"
MODELS="nwm-real,nwm-no-pretrain,nwm-timept-ft,nwm-geopt-ft,nwm-latentpt-ft,nwm-latentpt-ft-align,nwm-latentpt-ft-action2latent"
TASKS="recon_prediction,unseen,unseen_rollout,navigation"

if [[ "${1:-}" == "--worker" ]]; then
    status_file="$2"
    log_file="$3"
    gpus="$4"
    shift 4
    cd "${PROJECT_ROOT}"
    exit_code=0
    failed_seed=""
    for seed in "$@"; do
        seed_root="${BENCHMARK_BASE}/evaluation_seeds/seed${seed}"
        command=(
            "${PROJECT_ROOT}/scripts/run_nwm_benchmark.sh"
            --models "${MODELS}"
            --tasks "${TASKS}"
            --gpus "${gpus}"
            --eval-seed "${seed}"
            --benchmark-root "${seed_root}"
            --shared-benchmark-root "${BENCHMARK_BASE}"
            --registry "${seed_root}/benchmark_results.json"
        )
        {
            printf '[seed-queue][%s] starting eval_seed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${seed}"
            printf '[seed-queue] command='
            printf '%q ' "${command[@]}"
            printf '\n'
        } >>"${log_file}"
        set +e
        "${command[@]}" >>"${log_file}" 2>&1
        exit_code=$?
        set -e
        if [[ ${exit_code} -ne 0 ]]; then
            failed_seed="${seed}"
            break
        fi
        printf '[seed-queue][%s] completed eval_seed=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${seed}" >>"${log_file}"
    done
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ ${exit_code} -eq 0 ]]; then
        state="complete"
    else
        state="failed"
    fi
    printf 'state=%s\nexit_code=%s\nfailed_seed=%s\nfinished_at=%s\nlog=%s\n' \
        "${state}" "${exit_code}" "${failed_seed}" "${finished_at}" "${log_file}" >"${status_file}"
    printf '[seed-queue][%s] finished state=%s exit_code=%s failed_seed=%s\n' \
        "${finished_at}" "${state}" "${exit_code}" "${failed_seed}" >>"${log_file}"
    exit "${exit_code}"
fi

gpus="${1:-0,1,2,4}"
shift || true
if [[ $# -gt 0 ]]; then
    seeds=("$@")
else
    seeds=(1 2)
fi

IFS=',' read -r -a selected_gpus <<<"${gpus}"
for gpu in "${selected_gpus[@]}"; do
    memory_used="$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -F',' -v target="${gpu}" '$1 + 0 == target {gsub(/ /, "", $2); print $2}'
    )"
    if [[ ! "${memory_used}" =~ ^[0-9]+$ ]] || (( memory_used >= 1000 )); then
        printf 'GPU %s is not free: memory.used=%s MiB\n' "${gpu}" "${memory_used:-unknown}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_ROOT}"
pid_file="${LOG_ROOT}/seed1_seed2_queue.pid"
if [[ -s "${pid_file}" ]]; then
    existing_pid="$(awk -F= '$1 == "pid" {print $2}' "${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        printf 'Evaluation seed queue is already running: PID=%s\n' "${existing_pid}" >&2
        exit 1
    fi
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${LOG_ROOT}/seed1_seed2_queue_${timestamp}.log"
status_file="${LOG_ROOT}/seed1_seed2_queue_${timestamp}.status"
{
    printf '[seed-queue][%s] cwd=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PROJECT_ROOT}"
    printf '[seed-queue] seeds=%s gpus=%s models=%s tasks=%s\n' \
        "${seeds[*]}" "${gpus}" "${MODELS}" "${TASKS}"
    printf '[seed-queue] GPU snapshot before launch:\n'
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    printf '[seed-queue] Disk snapshot before launch:\n'
    df -h "${BENCHMARK_BASE}"
} >"${log_file}"

nohup setsid bash "$0" --worker "${status_file}" "${log_file}" "${gpus}" \
    "${seeds[@]}" </dev/null >/dev/null 2>&1 &
pid=$!
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'state=running\npid=%s\nstarted_at=%s\nlog=%s\n' \
    "${pid}" "${started_at}" "${log_file}" >"${status_file}"
printf 'pid=%s\ngpus=%s\nseeds=%s\nlog=%s\nstatus=%s\n' \
    "${pid}" "${gpus}" "${seeds[*]}" "${log_file}" "${status_file}" >"${pid_file}"
printf '[seed-queue][%s] started pid=%s status=%s\n' \
    "${started_at}" "${pid}" "${status_file}" >>"${log_file}"

printf 'Started evaluation seed queue\nPID: %s\nGPUs: %s\nSeeds: %s\nLog: %s\nStatus: %s\n' \
    "${pid}" "${gpus}" "${seeds[*]}" "${log_file}" "${status_file}"
