#!/usr/bin/env bash
set -euo pipefail

readonly RAE_PROJECT_ROOT="/file_system/vepfs/algorithm/dujun.nie/code/CompACT"
readonly RAE_CONDA="/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda"
readonly RAE_PYTHON="/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python"
readonly RAE_REQUIRED_GPUS="${1:-4}"
readonly RAE_MEMORY_LIMIT_MIB=2000
readonly RAE_UTIL_LIMIT_PERCENT=5
readonly RAE_POLL_SECONDS=30

if ! [[ "${RAE_REQUIRED_GPUS}" =~ ^[1-8]$ ]]; then
  echo "GPU count must be an integer from 1 through 8" >&2
  exit 2
fi

cd "${RAE_PROJECT_ROOT}"
if [[ -n "${RAE_LAUNCHER_PID_PATH:-}" ]]; then
  printf '%s\n' "$$" > "${RAE_LAUNCHER_PID_PATH}"
fi
echo "[rae-nwm-launcher] waiting for ${RAE_REQUIRED_GPUS} free GPUs"
echo "[rae-nwm-launcher] free means memory < ${RAE_MEMORY_LIMIT_MIB} MiB and utilization < ${RAE_UTIL_LIMIT_PERCENT}%"

RAE_LAST_SELECTION=""
RAE_STABLE_PROBES=0
while true; do
  mapfile -t RAE_FREE_GPUS < <(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      | awk -F, -v memory_limit="${RAE_MEMORY_LIMIT_MIB}" -v util_limit="${RAE_UTIL_LIMIT_PERCENT}" '
          {
            gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
            if (($2 + 0) < memory_limit && ($3 + 0) < util_limit) print $1
          }
        '
  )

  if (( ${#RAE_FREE_GPUS[@]} >= RAE_REQUIRED_GPUS )); then
    RAE_SELECTED_GPUS=("${RAE_FREE_GPUS[@]:0:RAE_REQUIRED_GPUS}")
    RAE_GPU_CSV="$(IFS=,; echo "${RAE_SELECTED_GPUS[*]}")"
    if [[ "${RAE_GPU_CSV}" == "${RAE_LAST_SELECTION}" ]]; then
      RAE_STABLE_PROBES=$((RAE_STABLE_PROBES + 1))
    else
      RAE_LAST_SELECTION="${RAE_GPU_CSV}"
      RAE_STABLE_PROBES=1
    fi
  else
    RAE_LAST_SELECTION=""
    RAE_STABLE_PROBES=0
  fi

  printf '[rae-nwm-launcher] %(%Y-%m-%dT%H:%M:%SZ)T free=%s stable=%d/2\n' \
    -1 "${RAE_FREE_GPUS[*]:-none}" "${RAE_STABLE_PROBES}"

  if (( RAE_STABLE_PROBES >= 2 )); then
    RAE_COMMAND=(
      "${RAE_CONDA}" run --no-capture-output -n nwm
      python scripts/run_nwm_benchmark.py
      --models rae-nwm
      --tasks navigation
      --navigation-datasets recon,scand
      --gpus "${RAE_GPU_CSV}"
      --planning-microbatch-size 80
      --raenwm-planning-steps 50
      --raenwm-python "${RAE_PYTHON}"
    )
    printf '[rae-nwm-launcher] starting:'
    printf ' %q' "${RAE_COMMAND[@]}"
    printf '\n'
    if [[ -n "${RAE_LAUNCHER_STATUS_PATH:-}" ]]; then
      printf 'running\n' > "${RAE_LAUNCHER_STATUS_PATH}"
    fi
    set +e
    "${RAE_COMMAND[@]}"
    RAE_EXIT_STATUS=$?
    set -e
    printf '[rae-nwm-launcher] command exited with status %d\n' "${RAE_EXIT_STATUS}"
    if [[ -n "${RAE_LAUNCHER_STATUS_PATH:-}" ]]; then
      printf '%d\n' "${RAE_EXIT_STATUS}" > "${RAE_LAUNCHER_STATUS_PATH}"
    fi
    exit "${RAE_EXIT_STATUS}"
  fi
  sleep "${RAE_POLL_SECONDS}"
done
