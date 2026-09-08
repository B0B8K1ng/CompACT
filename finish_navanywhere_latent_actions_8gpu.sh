#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
VENV_PYTHON="${VENV_PYTHON:-/file_system/vepfs/algorithm/dujun.nie/code/DreamDojo/.venv/bin/python}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${VENV_PYTHON}" -u "${SCRIPT_DIR}/scripts/finish_navanywhere_latent_actions.py" "$@"
