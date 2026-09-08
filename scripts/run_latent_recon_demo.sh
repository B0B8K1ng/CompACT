#!/usr/bin/env bash
set -euo pipefail

for experiment in nwm-real-matrix nwm-nopre-matrix nwm-timept-matrix; do
  while codex-exp status "$experiment" | grep -q '^status: running'; do
    sleep 30
  done
  if ! codex-exp status "$experiment" | grep -q '^exit_code: 0'; then
    echo "Required experiment did not complete successfully: $experiment" >&2
    exit 1
  fi
done

env CUDA_VISIBLE_DEVICES=0 \
  /file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm \
  python demo_nwm_rollout.py \
  --checkpoint /file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_timept_ft/nwm-nav1-timept-finetune/checkpoints/joint_0100000.pth.tar \
  --first-image /file_system/nas/algorithm/dujun.nie/nwm/data/recon/jackal_2019-10-24-13-35-07_2_r02/123.jpg \
  --actions /file_system/nas/algorithm/dujun.nie/nwm/demo_outputs/figure11_15_4datasets_20260907/inputs/recon/actions.json \
  --gt-frames-dir /file_system/nas/algorithm/dujun.nie/nwm/data/recon/jackal_2019-10-24-13-35-07_2_r02 \
  --gt-start-index 124 \
  --feedback-mode latent \
  --device cuda \
  --output-dir /file_system/nas/algorithm/dujun.nie/nwm/demo_outputs/nwm_rollout_matrix_20260907_v2/latent_only/nwm-timept-ft/recon
