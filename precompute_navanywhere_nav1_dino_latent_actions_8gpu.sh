#!/usr/bin/env bash
set -euo pipefail

# Extract the exact NavAnywhere local pairs needed by the frozen 200k-step
# NWM-LatentPT sampling plan using the completed nav1 DINOFeatureLAM.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DETACH="${DETACH:-0}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NPROC="${NPROC:-8}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export LOADER_THREADS="${LOADER_THREADS:-8}"
# The generic BATCH_SIZE remains the online compatibility setting. The
# optimized DINO path automatically decouples unique-frame and LAM batches.
export DINO_FRAME_BATCH_SIZE="${DINO_FRAME_BATCH_SIZE:-32}"
export DINO_LAM_BATCH_SIZE="${DINO_LAM_BATCH_SIZE:-16}"
export PRECISION="${PRECISION:-bf16-mixed}"
export CHECKPOINT="${CHECKPOINT:-/file_system/nas/algorithm/dujun.nie/nwm/weights/navigation_lam/variant_3_dino/nav1-dino/checkpoints/step=100000.ckpt}"
export CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-6b4ff679d79d877da30d8e07eb50b197e485933bfd9cb8cf4b190c0fd63fadaa}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/navanywhere_nav1_dino_step100000_ws8_bs16_steps200000}"
export LOG_DIR="${LOG_DIR:-/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_nav1_dino_latent_actions}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-nav1-dino-latent-$(date -u +%m%d-%H%M%S)}"

exec "${SCRIPT_DIR}/precompute_navanywhere_nav1_latent_actions_8gpu.sh"
