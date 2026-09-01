#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="/file_system/vepfs/algorithm/dujun.nie/miniconda3"
source "${CONDA_ROOT}/bin/activate" nwm

export NWM_DATA_ROOT="${NWM_DATA_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data}"
export NWM_INDEX_ROOT="${NWM_INDEX_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices}"
export TORCH_HOME="${TORCH_HOME:-/file_system/vepfs/algorithm/dujun.nie/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec python scripts/run_nwm_benchmark.py "$@"
