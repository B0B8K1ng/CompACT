#!/usr/bin/env bash
set -euo pipefail

# CDiT-B + SD-VAE baseline from the CompACT navigation experiments.
# Override any setting by prefixing the command, for example:
#   ./nwm_train.sh  # default: 8 GPUs x batch 16 = global batch 128
#   GPU_IDS=2 BATCH_SIZE=16 MAX_TRAIN_STEPS=2 WANDB_MODE=offline ./nwm_train.sh
#   NWM_VARIANT=real NWM_MOTION_VARIANT=latent_tartan ./nwm_train.sh
#   NWM_VARIANT=real NWM_MOTION_VARIANT=latent_tartan_scand ./nwm_train.sh
#   NWM_VARIANT=real NWM_MOTION_VARIANT=geometry_tartan ./nwm_train.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_ENV="${CONDA_ENV:-nwm}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/activate}"
NWM_DATA_ROOT="${NWM_DATA_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data}"
COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
NWM_VARIANT="${NWM_VARIANT:-base}"
NWM_MOTION_VARIANT="${NWM_MOTION_VARIANT:-real}"
NWM_LAM_ROOT="${NWM_LAM_ROOT:-}"
NWM_LAM_CHECKPOINT_SHA256="${NWM_LAM_CHECKPOINT_SHA256:-}"
NWM_GEOMETRY_ROOT="${NWM_GEOMETRY_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/vggt_omega_tartan/geometry_motion}"
case "${NWM_VARIANT}" in
    base)
        NWM_DATASET_CONFIG_NAME="${NWM_DATASET_CONFIG_NAME:-nwm}"
        DEFAULT_LATENT_ROOT="${COMPACT_NAS_ROOT}/cache/vae_latents_sd_vae_ft_ema_224"
        DEFAULT_RUN_NOTES="compact_nwm_cdit_b_sdvae"
        ;;
    real)
        NWM_DATASET_CONFIG_NAME="${NWM_DATASET_CONFIG_NAME:-nwm_real}"
        DEFAULT_LATENT_ROOT="${COMPACT_NAS_ROOT}/cache/vae_latents_sd_vae_ft_ema_224_four_datasets"
        DEFAULT_RUN_NOTES="nwm_real_recon_scand_tartan_huron"
        ;;
    *)
        echo "ERROR: NWM_VARIANT must be base or real, got ${NWM_VARIANT}." >&2
        exit 2
        ;;
esac
case "${NWM_MOTION_VARIANT}" in
    real)
        DEFAULT_USE_PRECOMPUTED_LATENTS=true
        ;;
    latent_tartan)
        if [[ "${NWM_VARIANT}" != "real" ]]; then
            echo "ERROR: NWM_MOTION_VARIANT=latent_tartan requires NWM_VARIANT=real." >&2
            exit 2
        fi
        # The existing four-dataset VAE cache may not be readable by all NAS
        # users. Online VAE encoding is the safe default for this mode.
        DEFAULT_USE_PRECOMPUTED_LATENTS=false
        DEFAULT_RUN_NOTES="nwm_latent_tartan_lam_step125000"
        NWM_LAM_ROOT="${NWM_LAM_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/latent_actions/navigation_lam/variant_2_pixel/step_125000}"
        NWM_LAM_CHECKPOINT_SHA256="${NWM_LAM_CHECKPOINT_SHA256:-f4ad8a272786a5040da164ab5733d3abcb04b1b6851bfb296dfe7a2e934c0ad7}"
        ;;
    latent_tartan_scand)
        if [[ "${NWM_VARIANT}" != "real" ]]; then
            echo "ERROR: NWM_MOTION_VARIANT=latent_tartan_scand requires NWM_VARIANT=real." >&2
            exit 2
        fi
        DEFAULT_USE_PRECOMPUTED_LATENTS=false
        DEFAULT_RUN_NOTES="nwm_latent_tartan_scand_pixel_step100000_offset8"
        NWM_LAM_ROOT="${NWM_LAM_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/latent_actions/navigation_lam/variant_2_pixel/step_100000_offset8}"
        NWM_LAM_CHECKPOINT_SHA256="${NWM_LAM_CHECKPOINT_SHA256:-a7ad05765998e55421013d411c9f16d403b90ed39336f7a352d80410829fbda1}"
        ;;
    geometry_tartan)
        if [[ "${NWM_VARIANT}" != "real" ]]; then
            echo "ERROR: NWM_MOTION_VARIANT=geometry_tartan requires NWM_VARIANT=real." >&2
            exit 2
        fi
        # The current four-dataset VAE cache contains root-owned TartanDrive
        # files that are not readable by this user. Keep online VAE encoding as
        # the safe default; callers may opt in after fixing cache permissions.
        DEFAULT_USE_PRECOMPUTED_LATENTS=false
        DEFAULT_RUN_NOTES="nwm_geometry_tartan_vggt_omega"
        ;;
    *)
        echo "ERROR: NWM_MOTION_VARIANT must be real, latent_tartan, latent_tartan_scand, or geometry_tartan, got ${NWM_MOTION_VARIANT}." >&2
        exit 2
        ;;
esac
# CompACT's global observation batch is 128: 8 GPUs x 16 samples/GPU.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LOG_EVERY="${LOG_EVERY:-100}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
EVAL_AT_FIRST_STEP="${EVAL_AT_FIRST_STEP:-true}"
BFLOAT16="${BFLOAT16:-1}"
TORCH_COMPILE="${TORCH_COMPILE:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-compact-nwm}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RUN_NOTES="${RUN_NOTES:-${DEFAULT_RUN_NOTES}}"
RESULTS_DIR="${RESULTS_DIR:-${COMPACT_NAS_ROOT}/runs}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${COMPACT_NAS_ROOT}/logs/launcher}"
NWM_INDEX_ROOT="${NWM_INDEX_ROOT:-${COMPACT_NAS_ROOT}/cache/dataset_indices}"
NWM_MODEL_CACHE="${NWM_MODEL_CACHE:-${COMPACT_NAS_ROOT}/cache/models}"
USE_PRECOMPUTED_LATENTS="${USE_PRECOMPUTED_LATENTS:-${DEFAULT_USE_PRECOMPUTED_LATENTS}}"
NWM_LATENT_ROOT="${NWM_LATENT_ROOT:-${DEFAULT_LATENT_ROOT}}"
LATENT_LRU_SIZE="${LATENT_LRU_SIZE:-8}"
TORCH_CACHE_ROOT="${TORCH_CACHE_ROOT:-${COMPACT_NAS_ROOT}/cache/torch}"
WANDB_CACHE_ROOT="${WANDB_CACHE_ROOT:-${COMPACT_NAS_ROOT}/cache/wandb}"
FROM_CHECKPOINT="${FROM_CHECKPOINT:-}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
NWM_HF_HOME="${NWM_HF_HOME:-/file_system/nas/algorithm/dujun.nie/huggingface}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ -z "${NPROC}" ]]; then
    NPROC="${#GPU_ARRAY[@]}"
fi
if (( NPROC < 1 || NPROC > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: NPROC=${NPROC} must be between 1 and the number of GPU_IDS (${#GPU_ARRAY[@]})." >&2
    exit 2
fi

case "${USE_PRECOMPUTED_LATENTS}" in
    true|false) ;;
    *)
        echo "ERROR: USE_PRECOMPUTED_LATENTS must be true or false, got ${USE_PRECOMPUTED_LATENTS}." >&2
        exit 2
        ;;
esac
if ! [[ "${LATENT_LRU_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: LATENT_LRU_SIZE must be a positive integer, got ${LATENT_LRU_SIZE}." >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "ERROR: DRY_RUN must be 0 or 1, got ${DRY_RUN}." >&2
    exit 2
fi
if [[ "${USE_PRECOMPUTED_LATENTS}" == "true" && "${BFLOAT16}" != "1" ]]; then
    echo "ERROR: the precomputed posterior cache requires BFLOAT16=1." >&2
    exit 2
fi

# Four GPUs x 32 or eight GPUs x 16 reproduces global observation batch 128.
# On one occupied/shared L20, batch 16 is the measured safe default.
if [[ -z "${BATCH_SIZE}" ]]; then
    case "${NPROC}" in
        4) BATCH_SIZE=32 ;;
        8) BATCH_SIZE=16 ;;
        *) BATCH_SIZE=16 ;;
    esac
fi
if [[ "${USE_PRECOMPUTED_LATENTS}" == "true" && "${BATCH_SIZE}" != "16" ]]; then
    echo "ERROR: precomputed BF16 posteriors require BATCH_SIZE=16 per GPU" >&2
    echo "       (the cache was encoded with the matching flattened VAE batch 16 x 8 = 128)." >&2
    echo "       Set USE_PRECOMPUTED_LATENTS=false to train another batch size online." >&2
    exit 2
fi

if [[ ! -x "${CONDA_ACTIVATE}" ]]; then
    echo "ERROR: conda activate script is not executable: ${CONDA_ACTIVATE}" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

# Prefer the verified NAS snapshot over resolving the repository ID through a
# shell-dependent Hugging Face cache. VAE_MODEL_PATH can still override it.
export HF_HOME="${NWM_HF_HOME}"
export HF_HUB_CACHE="${NWM_HF_HOME}/hub"
if [[ -z "${VAE_MODEL_PATH}" ]]; then
    VAE_CACHE_DIR="${HF_HUB_CACHE}/models--stabilityai--sd-vae-ft-ema"
    VAE_REF_FILE="${VAE_CACHE_DIR}/refs/main"
    if [[ -r "${VAE_REF_FILE}" ]]; then
        VAE_REVISION="$(<"${VAE_REF_FILE}")"
        VAE_SNAPSHOT="${VAE_CACHE_DIR}/snapshots/${VAE_REVISION}"
        if [[ -r "${VAE_SNAPSHOT}/config.json" && \
              -r "${VAE_SNAPSHOT}/diffusion_pytorch_model.bin" ]]; then
            VAE_MODEL_PATH="${VAE_SNAPSHOT}"
        fi
    fi
    VAE_MODEL_PATH="${VAE_MODEL_PATH:-stabilityai/sd-vae-ft-ema}"
fi

mkdir -p \
    "${RESULTS_DIR}" \
    "${LAUNCH_LOG_DIR}" \
    "${NWM_INDEX_ROOT}" \
    "${NWM_MODEL_CACHE}" \
    "${TORCH_CACHE_ROOT}" \
    "${WANDB_CACHE_ROOT}/cache" \
    "${WANDB_CACHE_ROOT}/data" \
    "${WANDB_CACHE_ROOT}/artifacts"
LAUNCH_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/nwm_train_${LAUNCH_TIMESTAMP}.log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NWM_DATA_ROOT
export NWM_INDEX_ROOT
export NWM_MODEL_CACHE
export NWM_LATENT_ROOT
export NWM_GEOMETRY_ROOT
export NWM_REBUILD_INDEX="${REBUILD_INDEX}"
export HF_HUB_OFFLINE
export TORCH_HOME="${TORCH_CACHE_ROOT}"
export WANDB_CACHE_DIR="${WANDB_CACHE_ROOT}/cache"
export WANDB_DATA_DIR="${WANDB_CACHE_ROOT}/data"
export WANDB_ARTIFACT_DIR="${WANDB_CACHE_ROOT}/artifacts"
export WANDB_DIR="${RESULTS_DIR}"
export PYTORCH_ALLOC_CONF
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE

if [[ "${USE_PRECOMPUTED_LATENTS}" == "true" ]]; then
    echo "Checking completed precomputed latent cache: ${NWM_LATENT_ROOT}"
    python - "${NWM_LATENT_ROOT}" <<'PY'
import hashlib
import json
import os
import sys

root = os.path.realpath(sys.argv[1])
metadata_path = os.path.join(root, "metadata.json")
success_path = os.path.join(root, "_SUCCESS.json")
for path in (metadata_path, success_path):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: required latent cache marker is missing: {path}")
with open(metadata_path, "rb") as handle:
    metadata_bytes = handle.read()
with open(success_path, "r", encoding="utf-8") as handle:
    success = json.load(handle)
metadata = json.loads(metadata_bytes)
expected = {
    "schema_version": 1,
    "format": "sd_vae_posterior_stats",
    "status": "complete",
    "complete": True,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(
            f"ERROR: latent metadata {key}={metadata.get(key)!r}, expected {value!r}"
        )
if success.get("schema_version") != 1 or success.get("complete") is not True:
    raise SystemExit("ERROR: latent _SUCCESS.json is invalid or incomplete")
digest = hashlib.sha256(metadata_bytes).hexdigest()
if success.get("metadata_sha256") != digest:
    raise SystemExit("ERROR: latent metadata hash does not match _SUCCESS.json")
if metadata.get("storage", {}).get("dtype") != "bfloat16":
    raise SystemExit("ERROR: latent cache storage dtype must be bfloat16")
if float(metadata.get("vae", {}).get("scaling_factor", -1)) != 0.18215:
    raise SystemExit("ERROR: latent cache has the wrong SD-VAE scaling factor")
encoding = metadata.get("encoding", {})
if encoding.get("compute_dtype") != "bfloat16" or encoding.get("storage_dtype") != "bfloat16":
    raise SystemExit("ERROR: latent cache encoding must use BF16 compute/storage")
if encoding.get("vae_batch_size") != 128 or encoding.get("scaling_applied") is not False:
    raise SystemExit("ERROR: latent cache does not match the fixed-128 unscaled posterior contract")
for dataset_name in ("recon", "sacson", "scand"):
    manifest = os.path.join(root, dataset_name, "manifest.jsonl")
    if not os.path.isfile(manifest):
        raise SystemExit(f"ERROR: latent manifest is missing: {manifest}")
print(f"Precomputed latent cache completion marker is valid: {root}")
PY
fi

if [[ "${NWM_MOTION_VARIANT}" == latent_tartan* ]]; then
    echo "Checking completed LAM action cache: ${NWM_LAM_ROOT}"
    python - "${NWM_LAM_ROOT}" "${NWM_LAM_CHECKPOINT_SHA256}" "${NWM_MOTION_VARIANT}" <<'PY'
import hashlib
import json
import os
import sys

root = os.path.realpath(sys.argv[1])
expected_checkpoint_sha256 = sys.argv[2]
variant = sys.argv[3]
datasets = {
    "tartan_drive": {"num_pairs": 2_903_727, "num_trajectories": 1_251},
}
if variant == "latent_tartan_scand":
    datasets = {
        "tartan_drive": {"num_pairs": 451_472, "num_trajectories": 1_251},
        "scand": {"num_pairs": 1_403_645, "num_trajectories": 604},
    }
base_expected = {
    "schema_version": 1,
    "format": "compact_offline_lam_action",
    "status": "complete",
    "complete": True,
    "motion_type": "latent",
    "latent_dim": 32,
    "pair_direction": "current_to_goal",
    "normalization": "raw",
}
for dataset_name, counts in datasets.items():
    suffix = "" if dataset_name == "tartan_drive" else f".{dataset_name}"
    metadata_path = os.path.join(root, f"metadata{suffix}.json")
    success_path = os.path.join(root, f"_SUCCESS{suffix}.json")
    for path in (metadata_path, success_path):
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: required LAM action cache marker is missing: {path}")
    with open(metadata_path, "rb") as handle:
        metadata_bytes = handle.read()
    with open(success_path, "r", encoding="utf-8") as handle:
        success = json.load(handle)
    metadata = json.loads(metadata_bytes)
    expected = {**base_expected, "dataset_name": dataset_name, **counts}
    if variant == "latent_tartan_scand":
        expected.update({"checkpoint_global_step": 100_000, "min_offset": -8, "max_offset": 8})
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(
                f"ERROR: {dataset_name} LAM metadata {key}={metadata.get(key)!r}, expected {value!r}"
            )
    if success.get("schema_version") != 1 or success.get("complete") is not True:
        raise SystemExit(f"ERROR: {dataset_name} LAM _SUCCESS is invalid or incomplete")
    if success.get("format") != base_expected["format"]:
        raise SystemExit(f"ERROR: {dataset_name} LAM _SUCCESS has the wrong format")
    digest = hashlib.sha256(metadata_bytes).hexdigest()
    if success.get("metadata_sha256") != digest:
        raise SystemExit(f"ERROR: {dataset_name} metadata hash does not match _SUCCESS")
    if metadata.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise SystemExit(f"ERROR: {dataset_name} cache used an unexpected checkpoint")
    for key in ("checkpoint_sha256", "num_pairs", "num_trajectories", "trajectory_files_sha256"):
        if success.get(key) != metadata.get(key):
            raise SystemExit(f"ERROR: {dataset_name} metadata/_SUCCESS mismatch for {key}")
    print(
        f"{dataset_name} LAM action cache is valid: "
        f"{metadata['num_trajectories']} trajectories, {metadata['num_pairs']} pairs"
    )
PY
fi

if [[ "${NWM_MOTION_VARIANT}" == "geometry_tartan" ]]; then
    echo "Checking validated TartanDrive geometry action cache: ${NWM_GEOMETRY_ROOT}"
    python - "${NWM_GEOMETRY_ROOT}" "${SCRIPT_DIR}/data_splits/tartan_drive" <<'PY'
import json
import os
import sys
from pathlib import Path

geometry_root = Path(os.path.realpath(sys.argv[1]))
split_root = Path(os.path.realpath(sys.argv[2]))
artifact_root = geometry_root.parent

receipts = (
    (artifact_root / "validation" / "report.json", ("summary", "status")),
    (artifact_root / "validation" / "provenance_receipt.json", ("status",)),
    (artifact_root / "provenance" / "formal_run_receipt.json", ("status",)),
)
for path, keys in receipts:
    if not path.is_file():
        raise SystemExit(f"ERROR: required geometry validation receipt is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: cannot read geometry validation receipt {path}: {exc}")
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise SystemExit(f"ERROR: geometry receipt has no {'.'.join(keys)}: {path}")
        value = value[key]
    if value != "pass":
        raise SystemExit(f"ERROR: geometry receipt did not pass ({value!r}): {path}")

expected = set()
for split in ("train", "test"):
    path = split_root / split / "traj_names.txt"
    if not path.is_file():
        raise SystemExit(f"ERROR: TartanDrive split file is missing: {path}")
    expected.update(line.strip() for line in path.read_text().splitlines() if line.strip())

cache_dir = geometry_root / "tartan_drive"
actual = {path.stem for path in cache_dir.glob("*.pt")} if cache_dir.is_dir() else set()
missing = expected - actual
unexpected = actual - expected
if len(expected) != 1_251 or missing or unexpected:
    raise SystemExit(
        "ERROR: geometry cache does not exactly cover the 1,251 TartanDrive trajectories "
        f"(expected={len(expected)}, actual={len(actual)}, missing={len(missing)}, "
        f"unexpected={len(unexpected)})"
    )
print(
    "TartanDrive geometry action cache is valid: "
    f"{geometry_root} ({len(actual)} trajectories, all validation receipts pass)"
)
PY
fi

# Resolve/download the SD-VAE once before torchrun. This prevents multiple DDP
# ranks from observing a partially populated Hugging Face cache on first use.
echo "Checking SD-VAE checkpoint: ${VAE_MODEL_PATH}"
if ! python - "${VAE_MODEL_PATH}" <<'PY'
import sys

from diffusers.models import AutoencoderKL

model_path = sys.argv[1]
vae = AutoencoderKL.from_pretrained(model_path, use_safetensors=False)
print(f"SD-VAE checkpoint is ready: {model_path}")
del vae
PY
then
    echo "ERROR: SD-VAE checkpoint is unavailable or incomplete: ${VAE_MODEL_PATH}" >&2
    if [[ "${VAE_MODEL_PATH}" == /* ]]; then
        echo "       Check that this NAS snapshot is mounted and readable." >&2
    elif [[ "${HF_HUB_OFFLINE}" == "1" ]]; then
        echo "       The launcher is offline. Retry once with HF_HUB_OFFLINE=0 to download it." >&2
    fi
    exit 2
fi

if [[ "${WANDB_ENABLED}" == "true" && "${WANDB_MODE}" == "online" ]]; then
    echo "Checking W&B authentication (set WANDB_API_KEY or log in interactively)..."
    if ! wandb login --verify; then
        echo "ERROR: W&B authentication failed. Run 'source ${CONDA_ACTIVATE} ${CONDA_ENV} && wandb login'," >&2
        echo "       or use WANDB_MODE=offline for a local smoke test." >&2
        exit 2
    fi
fi

HYDRA_ARGS=(
    --config-name nwm
    "dataset=${NWM_DATASET_CONFIG_NAME}"
    "training.batch_size=${BATCH_SIZE}"
    "training.num_workers=${NUM_WORKERS}"
    "training.optimizer.lr=${LEARNING_RATE}"
    "training.optimizer.weight_decay=${WEIGHT_DECAY}"
    "training.wandb_enabled=${WANDB_ENABLED}"
    "training.wandb_project=${WANDB_PROJECT}"
    "training.notes=${RUN_NOTES}"
    "training.results_dir=${RESULTS_DIR}"
    "model.tokenizer.model_path=${VAE_MODEL_PATH}"
    "dataset.precomputed_latents.enabled=${USE_PRECOMPUTED_LATENTS}"
    "dataset.precomputed_latents.root=${NWM_LATENT_ROOT}"
    "dataset.precomputed_latents.cache_size=${LATENT_LRU_SIZE}"
    "max_train_steps=${MAX_TRAIN_STEPS}"
    "epochs=${EPOCHS}"
    "log_every=${LOG_EVERY}"
    "ckpt_every=${CKPT_EVERY}"
    "eval_every=${EVAL_EVERY}"
    "eval_at_first_step=${EVAL_AT_FIRST_STEP}"
    "bfloat16=${BFLOAT16}"
    "torch_compile=${TORCH_COMPILE}"
)

if [[ "${NWM_VARIANT}" == "real" && "${NWM_MOTION_VARIANT}" == "real" ]]; then
    HYDRA_ARGS+=(
        "motion_condition.train_types=[real]"
        "motion_condition.dataset_motion_types={recon:real,sacson:real,scand:real,tartan_drive:real}"
    )
fi
if [[ "${NWM_MOTION_VARIANT}" == "latent_tartan" ]]; then
    HYDRA_ARGS+=(
        "motion_condition.train_types=[real,latent]"
        "motion_condition.dataset_motion_types={recon:real,sacson:real,scand:real,tartan_drive:latent}"
        "motion_condition.latent.latent_dim=32"
        "motion_condition.latent.offline.root=${NWM_LAM_ROOT}"
    )
fi
if [[ "${NWM_MOTION_VARIANT}" == "latent_tartan_scand" ]]; then
    HYDRA_ARGS+=(
        "motion_condition.train_types=[real,latent]"
        "motion_condition.dataset_motion_types={recon:real,sacson:real,scand:latent,tartan_drive:latent}"
        "motion_condition.latent.latent_dim=32"
        "motion_condition.latent.max_frame_offset=8"
        "motion_condition.latent.offline.root=${NWM_LAM_ROOT}"
    )
fi
if [[ "${NWM_MOTION_VARIANT}" == "geometry_tartan" ]]; then
    HYDRA_ARGS+=(
        "motion_condition.train_types=[real,geometry]"
        "motion_condition.dataset_motion_types={recon:real,sacson:real,scand:real,tartan_drive:geometry}"
        "motion_condition.geometry.geometry_dim=3"
        "motion_condition.geometry.offline.root=${NWM_GEOMETRY_ROOT}"
    )
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
    HYDRA_ARGS+=("training.wandb_entity=${WANDB_ENTITY}")
fi
if [[ -n "${FROM_CHECKPOINT}" ]]; then
    HYDRA_ARGS+=("training.from_checkpoint=${FROM_CHECKPOINT}")
fi

echo "CompACT NWM training launch"
echo "  repo:              ${SCRIPT_DIR}"
echo "  conda env:         ${CONDA_ENV}"
echo "  data root:         ${NWM_DATA_ROOT}"
echo "  dataset indexes:   ${NWM_INDEX_ROOT}"
echo "  eval model cache:  ${NWM_MODEL_CACHE}"
echo "  motion variant:    ${NWM_MOTION_VARIANT}"
if [[ "${NWM_MOTION_VARIANT}" == latent_tartan* ]]; then
    echo "  LAM action root:   ${NWM_LAM_ROOT}"
fi
if [[ "${NWM_MOTION_VARIANT}" == "geometry_tartan" ]]; then
    echo "  geometry root:     ${NWM_GEOMETRY_ROOT}"
fi
echo "  train input:       $([[ "${USE_PRECOMPUTED_LATENTS}" == "true" ]] && echo 'precomputed BF16 SD-VAE posteriors' || echo 'pixels + online VAE')"
echo "  latent root:       ${NWM_LATENT_ROOT}"
echo "  latent worker LRU: ${LATENT_LRU_SIZE} trajectories"
echo "  SD-VAE:            ${VAE_MODEL_PATH}"
echo "  visible GPUs:      ${GPU_IDS}"
echo "  processes:         ${NPROC}"
echo "  batch/GPU:         ${BATCH_SIZE}"
echo "  global obs batch:  $((NPROC * BATCH_SIZE))"
echo "  max steps:         ${MAX_TRAIN_STEPS}"
echo "  W&B:               ${WANDB_ENABLED} (${WANDB_MODE}), ${WANDB_PROJECT}"
echo "  run/checkpoint dir:${RESULTS_DIR}"
echo "  torch cache:       ${TORCH_HOME}"
echo "  launcher log:      ${LAUNCH_LOG}"
if (( NPROC * BATCH_SIZE != 128 )); then
    echo "  WARNING: global observation batch is not the paper value 128."
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Validated dry-run command:'
    printf ' %q' bash scripts/train.sh "--nproc=${NPROC}" -- "${HYDRA_ARGS[@]}"
    printf '\n'
    exit 0
fi

bash scripts/train.sh --nproc="${NPROC}" -- "${HYDRA_ARGS[@]}"
