#!/usr/bin/env bash
# Start the complete, resumable NWM benchmark in the background with a durable log.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_ROOT="${NWM_BENCHMARK_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark}"
LOG_ROOT="${BENCHMARK_ROOT}/logs"
STANDARD_TASKS="recon_prediction,unseen,unseen_rollout,navigation"

if [[ "${1:-}" == "--worker" ]]; then
    status_file="$2"
    log_file="$3"
    shift 3
    cd "${PROJECT_ROOT}"
    set +e
    "$@" >>"${log_file}" 2>&1
    exit_code=$?
    set -e
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'state=%s\nexit_code=%s\nfinished_at=%s\nlog=%s\n' \
        "$([[ ${exit_code} -eq 0 ]] && printf complete || printf failed)" \
        "${exit_code}" "${finished_at}" "${log_file}" >"${status_file}"
    printf '[launcher][%s] finished exit_code=%s status=%s\n' \
        "${finished_at}" "${exit_code}" "${status_file}" >>"${log_file}"
    exit "${exit_code}"
fi

usage() {
    printf 'Usage: %s MODEL [GPU_IDS] [extra run_nwm_benchmark.py arguments]\n' "$0"
    printf 'Example: %s nwm-latentpt-ft\n' "$0"
    printf 'Example: %s nwm-latentpt-ft 1,2,3,4 --planning-microbatch-size 40\n' "$0"
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0 || exit 2
fi

model="$1"
shift
gpus=""
if [[ $# -gt 0 && "$1" != --* ]]; then
    gpus="$1"
    shift
fi

required_gpu_count="${NWM_BENCHMARK_GPU_COUNT:-4}"
if [[ -z "${gpus}" ]]; then
    mapfile -t free_gpus < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk '$2 < 1000 {print $1}' \
            | head -n "${required_gpu_count}"
    )
    if [[ ${#free_gpus[@]} -lt ${required_gpu_count} ]]; then
        printf 'Need %s GPUs with <1 GiB used, but found %s. Pass GPU_IDS explicitly if appropriate.\n' \
            "${required_gpu_count}" "${#free_gpus[@]}" >&2
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader >&2
        exit 1
    fi
    gpus="$(IFS=,; printf '%s' "${free_gpus[*]}")"
fi

mkdir -p "${LOG_ROOT}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
safe_model="${model//[^[:alnum:]_.-]/_}"
log_file="${LOG_ROOT}/${safe_model}_standard_${timestamp}.log"
status_file="${LOG_ROOT}/${safe_model}_standard_${timestamp}.status"
pid_file="${LOG_ROOT}/${safe_model}_standard.pid"

if [[ -s "${pid_file}" ]]; then
    existing_pid="$(awk -F= '$1 == "pid" {print $2}' "${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        existing_log="$(awk -F= '$1 == "log" {sub(/^log=/, ""); print}' "${pid_file}")"
        printf 'Benchmark for %s is already running: PID=%s LOG=%s\n' \
            "${model}" "${existing_pid}" "${existing_log}" >&2
        exit 1
    fi
fi

command=(
    "${PROJECT_ROOT}/scripts/run_nwm_benchmark.sh"
    --models "${model}"
    --tasks "${NWM_BENCHMARK_TASKS:-${STANDARD_TASKS}}"
    --gpus "${gpus}"
    "$@"
)

{
    printf '[launcher][%s] cwd=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PROJECT_ROOT}"
    printf '[launcher] model=%s gpus=%s tasks=%s\n' \
        "${model}" "${gpus}" "${NWM_BENCHMARK_TASKS:-${STANDARD_TASKS}}"
    printf '[launcher] command='
    printf '%q ' "${command[@]}"
    printf '\n[launcher] GPU snapshot before launch:\n'
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    printf '[launcher] Disk snapshot before launch:\n'
    df -h "${BENCHMARK_ROOT}"
} >"${log_file}"

nohup setsid bash "$0" --worker "${status_file}" "${log_file}" \
    "${command[@]}" </dev/null >/dev/null 2>&1 &
pid=$!
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'state=running\npid=%s\nstarted_at=%s\nlog=%s\n' \
    "${pid}" "${started_at}" "${log_file}" >"${status_file}"
printf 'pid=%s\nmodel=%s\ngpus=%s\nlog=%s\nstatus=%s\n' \
    "${pid}" "${model}" "${gpus}" "${log_file}" "${status_file}" >"${pid_file}"
printf '[launcher][%s] started pid=%s status=%s\n' \
    "${started_at}" "${pid}" "${status_file}" >>"${log_file}"

printf 'Started %s standard benchmark\nPID: %s\nGPUs: %s\nLog: %s\nStatus: %s\n' \
    "${model}" "${pid}" "${gpus}" "${log_file}" "${status_file}"
printf 'Progress: tail -f %q\n' "${log_file}"
