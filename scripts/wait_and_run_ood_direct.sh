#!/usr/bin/env bash
set -euo pipefail

readonly OOD_PROJECT_ROOT="/file_system/vepfs/algorithm/dujun.nie/code/CompACT"
readonly OOD_CONDA="/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda"
readonly OOD_RAENWM_PYTHON="/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python"
readonly OOD_REQUIRED_GPUS=4
readonly OOD_MEMORY_LIMIT_MIB=2000
readonly OOD_UTIL_LIMIT_PERCENT=5
readonly OOD_POLL_SECONDS=30

cd "${OOD_PROJECT_ROOT}"
export NWM_DATA_ROOT="/file_system/nas/algorithm/dujun.nie/nwm/data"
export NWM_INDEX_ROOT="/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices"
export PYTHONUNBUFFERED=1

if [[ -n "${OOD_LAUNCHER_PID_PATH:-}" ]]; then
  printf '%s\n' "$$" > "${OOD_LAUNCHER_PID_PATH}"
fi
if [[ -n "${OOD_LAUNCHER_STATUS_PATH:-}" ]]; then
  printf 'waiting\n' > "${OOD_LAUNCHER_STATUS_PATH}"
fi

# A previously queued benchmark gets first access to newly free GPUs. Once it
# has started and allocated memory, this launcher can safely use a disjoint set
# of four GPUs if one exists; otherwise it naturally waits for completion.
if [[ -n "${OOD_PREDECESSOR_PID:-}" ]]; then
  while kill -0 "${OOD_PREDECESSOR_PID}" 2>/dev/null; do
    OOD_PREDECESSOR_STATUS=""
    if [[ -n "${OOD_PREDECESSOR_STATUS_PATH:-}" && -f "${OOD_PREDECESSOR_STATUS_PATH}" ]]; then
      OOD_PREDECESSOR_STATUS="$(<"${OOD_PREDECESSOR_STATUS_PATH}")"
    fi
    if [[ "${OOD_PREDECESSOR_STATUS}" == "running" ]]; then
      echo "[ood-launcher] predecessor is running; waiting 60 seconds for GPU allocation"
      sleep "${OOD_POLL_SECONDS}"
      sleep "${OOD_POLL_SECONDS}"
      break
    fi
    printf '[ood-launcher] %(%Y-%m-%dT%H:%M:%SZ)T waiting for predecessor pid=%s\n' \
      -1 "${OOD_PREDECESSOR_PID}"
    sleep "${OOD_POLL_SECONDS}"
  done
fi

echo "[ood-launcher] waiting for ${OOD_REQUIRED_GPUS} free GPUs"
echo "[ood-launcher] free means memory < ${OOD_MEMORY_LIMIT_MIB} MiB and utilization < ${OOD_UTIL_LIMIT_PERCENT}%"

OOD_LAST_SELECTION=""
OOD_STABLE_PROBES=0
while true; do
  mapfile -t OOD_FREE_GPUS < <(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      | awk -F, -v memory_limit="${OOD_MEMORY_LIMIT_MIB}" -v util_limit="${OOD_UTIL_LIMIT_PERCENT}" '
          {
            gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
            if (($2 + 0) < memory_limit && ($3 + 0) < util_limit) print $1
          }
        '
  )

  if (( ${#OOD_FREE_GPUS[@]} >= OOD_REQUIRED_GPUS )); then
    OOD_SELECTED_GPUS=("${OOD_FREE_GPUS[@]:0:OOD_REQUIRED_GPUS}")
    OOD_GPU_CSV="$(IFS=,; echo "${OOD_SELECTED_GPUS[*]}")"
    if [[ "${OOD_GPU_CSV}" == "${OOD_LAST_SELECTION}" ]]; then
      OOD_STABLE_PROBES=$((OOD_STABLE_PROBES + 1))
    else
      OOD_LAST_SELECTION="${OOD_GPU_CSV}"
      OOD_STABLE_PROBES=1
    fi
  else
    OOD_LAST_SELECTION=""
    OOD_STABLE_PROBES=0
  fi

  printf '[ood-launcher] %(%Y-%m-%dT%H:%M:%SZ)T free=%s stable=%d/2\n' \
    -1 "${OOD_FREE_GPUS[*]:-none}" "${OOD_STABLE_PROBES}"

  if (( OOD_STABLE_PROBES >= 2 )); then
    OOD_COMMAND=(
      "${OOD_CONDA}" run --no-capture-output -n nwm
      python scripts/run_nwm_benchmark.py
      --models nwm-real,rae-nwm,nwm-latentpt-reset-nwm-real-recipe-180k
      --tasks ood_direct_4s
      --ood-datasets planetary_rover,unitree_go2,tum_rgbd,uzh_fpv
      --gpus "${OOD_GPU_CSV}"
      --raenwm-python "${OOD_RAENWM_PYTHON}"
    )
    printf '[ood-launcher] starting:'
    printf ' %q' "${OOD_COMMAND[@]}"
    printf '\n'
    if [[ -n "${OOD_LAUNCHER_STATUS_PATH:-}" ]]; then
      printf 'running\n' > "${OOD_LAUNCHER_STATUS_PATH}"
    fi
    set +e
    "${OOD_COMMAND[@]}"
    OOD_EXIT_STATUS=$?
    set -e
    printf '[ood-launcher] command exited with status %d\n' "${OOD_EXIT_STATUS}"
    if [[ -n "${OOD_LAUNCHER_STATUS_PATH:-}" ]]; then
      printf '%d\n' "${OOD_EXIT_STATUS}" > "${OOD_LAUNCHER_STATUS_PATH}"
    fi
    exit "${OOD_EXIT_STATUS}"
  fi
  sleep "${OOD_POLL_SECONDS}"
done
