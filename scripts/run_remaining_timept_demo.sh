#!/usr/bin/env bash
set -euo pipefail

while codex-exp status nwm-real-matrix | grep -q '^status: running'; do
  sleep 30
done

env CUDA_VISIBLE_DEVICES=0 \
  /file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm \
  python scripts/run_nwm_demo_matrix.py \
  --model nwm-timept-ft \
  --device cuda \
  --scope all
