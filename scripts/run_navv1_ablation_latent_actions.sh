#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/user/ndj/code/DreamDojo/.venv/bin/python
SCRIPT=/home/user/ndj/code/CompACT/scripts/precompute_finetune_nav1_actions.py
DATA_ROOT=/data1/ndj/LAM-Data
CHECKPOINT_ROOT=/data1/ndj/checkpoints/hongyu
SPLIT_ROOT=/home/user/ndj/code/CompACT/data_splits
LAM_ROOT=/home/user/ndj/code/DreamDojo/external/lam_project

export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

for tag in u25l100 u50l100 u75l100; do
    case "${tag}" in
        u25l100) checkpoint="${CHECKPOINT_ROOT}/LAM-U25L100.ckpt" ;;
        u50l100) checkpoint="${CHECKPOINT_ROOT}/LAM-U50L100.ckpt" ;;
        u75l100) checkpoint="${CHECKPOINT_ROOT}/LAM-U75L100.ckpt" ;;
    esac
    output_root="${DATA_ROOT}/navv1_latent_actions_${tag}_step40000"
    if [[ -f "${output_root}/_SUCCESS.json" && -f "${output_root}/metadata.json" ]]; then
        echo "$(date -u +%FT%TZ) skip completed extraction: ${tag} (${output_root})"
        continue
    fi
    "${PYTHON}" -u "${SCRIPT}" run \
        --output-root "${output_root}" \
        --checkpoint "${checkpoint}" \
        --data-root "${DATA_ROOT}" \
        --split-root "${SPLIT_ROOT}" \
        --lam-root "${LAM_ROOT}" \
        --gpus 0,1,2,3,4,5,6,7 \
        --workers-per-gpu 3 \
        --batch-size 24 \
        --compile-encoder \
        --loader-threads 4
done
