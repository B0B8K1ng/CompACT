#!/usr/bin/env bash
set -Eeuo pipefail

# One attached, resumable 8x80GB Nav1 run: CDiT-B LatentPT -> latent_reset.
# Invoke through an explicit environment, for example:
#   conda run --no-capture-output -n nwm bash ./run_nwm_latentpt_nav1_80gb.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

show_help() {
    cat <<'EOF'
Usage: bash ./run_nwm_latentpt_nav1_80gb.sh [--dry-run] [--resume]

Main overrides (environment variables):
  RUN_ID                    Unique run id (default: UTC timestamp)
  GPU_IDS                   Exactly eight physical GPU indices (default: 0..7)
  GPU_MEMORY_GB             GPU memory profile: 80 (default) or 48
  BATCH_SIZE                Per-GPU batch: 16/32/48/64/80/96 (default: 96 on 80GB, 48 on 48GB)
  RESULTS_ROOT              NAS parent for outputs
  EVAL_EVERY                Low-frequency evaluation interval (default: 10000)
  EVAL_BATCH_SIZE           Per-GPU evaluation batch (default: 16)
  EVAL_OFFLOAD_MODELS       Move VAE/DreamSim to CPU between evals (default: true)
  WANDB_ENABLED             true/false (default: true)
  WANDB_MODE                online/offline/disabled (default: online)
  STAGE2_ALLOW_SOFTWARE_MISMATCH
                            Explicitly allow reading immutable Stage-2 latent
                            tensors produced by another software stack
  STAGE2_ALLOW_DATA_ROOT_RELOCATION
                            Explicitly allow a relocated Stage-2 source tree
  RESUME                    1 resumes this RUN_ID from latest checkpoints
  KEEP_INTERMEDIATE         1 keeps periodic checkpoints (default: 0)

Default full-contract sample budgets (environment-overridable for a timed run):
  Stage 1: 3,200,000/rank (25.6M global)
  Warmup:    160,000/rank (1.28M global)
  Joint:   1,600,000/rank (12.8M global)
EOF
}

DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
while (( $# > 0 )); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --resume) RESUME=1 ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; show_help >&2; exit 2 ;;
    esac
    shift
done

RUN_ID="${RUN_ID:-nav1_latentpt_80gb_$(date -u +%Y%m%d_%H%M%S)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
STAGE1_DATASET_PROFILE="${STAGE1_DATASET_PROFILE:-navanywhere}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPU_MEMORY_GB="${GPU_MEMORY_GB:-80}"
case "${GPU_MEMORY_GB}" in
    80) DEFAULT_BATCH_SIZE=96; DEFAULT_TOTAL_GPU_MB=80000; DEFAULT_FREE_GPU_MB=79000 ;;
    48) DEFAULT_BATCH_SIZE=48; DEFAULT_TOTAL_GPU_MB=45000; DEFAULT_FREE_GPU_MB=44000 ;;
    *) echo "ERROR: GPU_MEMORY_GB must be 80 or 48." >&2; exit 2 ;;
esac
HARDWARE_LABEL="8x${GPU_MEMORY_GB}GB"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
REFERENCE_BATCH_SIZE=16
WORLD_SIZE=8
SEED=20260901
RECIPE_SHA256="${RECIPE_SHA256:-604fd1e3ad4dc541198cf41b0a430adb586c07719e7994089872d0639b85bc1b}"
NAV1_CHECKPOINT_SHA256="${NAV1_CHECKPOINT_SHA256:-ec7d4c159a0bcd661167b35ea88a1c61ac42d73a771a0de5de660cced4325ac1}"

# Original batch16 budgets, expressed per rank so the global sample counts stay
# identical on the required eight-GPU topology.
STAGE1_SAMPLES_PER_RANK="${STAGE1_SAMPLES_PER_RANK:-3200000}"
WARMUP_SAMPLES_PER_RANK="${WARMUP_SAMPLES_PER_RANK:-160000}"
JOINT_SAMPLES_PER_RANK="${JOINT_SAMPLES_PER_RANK:-1600000}"
REFERENCE_SAMPLES_PER_RANK_PER_EPOCH="${REFERENCE_SAMPLES_PER_RANK_PER_EPOCH:-516544}"
EXPECTED_PLAN_BATCH_SIZE="${EXPECTED_PLAN_BATCH_SIZE:-16}"
EXPECTED_PLAN_DATASET_LENGTH="${EXPECTED_PLAN_DATASET_LENGTH:-4132468}"
EXPECTED_PLAN_MAX_STEPS="${EXPECTED_PLAN_MAX_STEPS:-200000}"
EXPECTED_PLAN_STEPS_PER_EPOCH="${EXPECTED_PLAN_STEPS_PER_EPOCH:-32284}"
EXPECTED_PLAN_WORLD_SIZE="${EXPECTED_PLAN_WORLD_SIZE:-8}"

STAGE1_LR="${STAGE1_LR:-2e-4}"
ADAPTER_LR="${ADAPTER_LR:-2e-4}"
BACKBONE_LR="${BACKBONE_LR:-2e-5}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-16}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-1667}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_AT_FIRST_STEP="${EVAL_AT_FIRST_STEP:-false}"
EVAL_OFFLOAD_MODELS="${EVAL_OFFLOAD_MODELS:-true}"
STAGE1_ALLOW_RECIPE_SUBSET="${STAGE1_ALLOW_RECIPE_SUBSET:-false}"
STAGE2_ALLOW_SOFTWARE_MISMATCH="${STAGE2_ALLOW_SOFTWARE_MISMATCH:-false}"
STAGE2_ALLOW_DATA_ROOT_RELOCATION="${STAGE2_ALLOW_DATA_ROOT_RELOCATION:-false}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-compact-nwm}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
CLEAN_WANDB_DEBUG_LOGS="${CLEAN_WANDB_DEBUG_LOGS:-1}"

COMPACT_NAS_ROOT="${COMPACT_NAS_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/compact}"
RESULTS_ROOT="${RESULTS_ROOT:-${COMPACT_NAS_ROOT}/runs/nav1_latentpt_80gb}"
NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere}"
NWM_DATA_ROOT="${NWM_DATA_ROOT:-/file_system/nas/algorithm/dujun.nie/nwm/data}"
SAMPLING_RECIPE="${SAMPLING_RECIPE:-${COMPACT_NAS_ROOT}/recipes/navanywhere_balanced_seed20260901.json}"
NAV1_PROXY_ROOT="${NAV1_PROXY_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_nav1_pixel_action_step100000}"
STAGE1_VAE_LATENT_ROOT="${STAGE1_VAE_LATENT_ROOT:-${COMPACT_NAS_ROOT}/cache/navanywhere_vae_latents_sdvae224_seed20260901}"
STAGE2_VAE_LATENT_ROOT="${STAGE2_VAE_LATENT_ROOT:-${COMPACT_NAS_ROOT}/cache/vae_latents_sd_vae_ft_ema_224_four_datasets}"
NWM_CONDA_ENV_PATH="${NWM_CONDA_ENV_PATH:-/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm}"

MIN_TOTAL_GPU_MB="${MIN_TOTAL_GPU_MB:-${DEFAULT_TOTAL_GPU_MB}}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-${DEFAULT_FREE_GPU_MB}}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-500}"

if ! [[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
if ! [[ "${EXPERIMENT_TAG}" =~ ^[A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: EXPERIMENT_TAG may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
if [[ "${STAGE1_DATASET_PROFILE}" != "navanywhere" && "${STAGE1_DATASET_PROFILE}" != "nwm_real" ]]; then
    echo "ERROR: STAGE1_DATASET_PROFILE must be navanywhere or nwm_real." >&2
    exit 2
fi
for value_name in BATCH_SIZE STAGE1_NUM_WORKERS STAGE2_NUM_WORKERS LOG_EVERY CKPT_EVERY EVAL_EVERY EVAL_BATCH_SIZE STAGE1_SAMPLES_PER_RANK WARMUP_SAMPLES_PER_RANK JOINT_SAMPLES_PER_RANK REFERENCE_SAMPLES_PER_RANK_PER_EPOCH EXPECTED_PLAN_BATCH_SIZE EXPECTED_PLAN_DATASET_LENGTH EXPECTED_PLAN_MAX_STEPS EXPECTED_PLAN_STEPS_PER_EPOCH EXPECTED_PLAN_WORLD_SIZE; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${value_name} must be a positive integer, got ${value}." >&2
        exit 2
    fi
done
for flag_name in DRY_RUN RESUME KEEP_INTERMEDIATE CLEAN_WANDB_DEBUG_LOGS; do
    flag="${!flag_name}"
    if [[ "${flag}" != "0" && "${flag}" != "1" ]]; then
        echo "ERROR: ${flag_name} must be 0 or 1." >&2
        exit 2
    fi
done
if [[ "${BATCH_SIZE}" != "96" && "${BATCH_SIZE}" != "80" && "${BATCH_SIZE}" != "64" && "${BATCH_SIZE}" != "48" && "${BATCH_SIZE}" != "32" && "${BATCH_SIZE}" != "16" ]]; then
    echo "ERROR: BATCH_SIZE must be 16, 32, 48, 64, 80, or 96." >&2
    exit 2
fi
if (( BATCH_SIZE % REFERENCE_BATCH_SIZE != 0 )); then
    echo "ERROR: BATCH_SIZE must be a multiple of ${REFERENCE_BATCH_SIZE}." >&2
    exit 2
fi
if [[ "${RESULTS_ROOT}" != /* || "${RESULTS_ROOT}" == "/" ]]; then
    echo "ERROR: RESULTS_ROOT must be an absolute non-root path." >&2
    exit 2
fi
if [[ "${EVAL_AT_FIRST_STEP}" != "true" && "${EVAL_AT_FIRST_STEP}" != "false" ]]; then
    echo "ERROR: EVAL_AT_FIRST_STEP must be true or false." >&2
    exit 2
fi
if [[ "${EVAL_OFFLOAD_MODELS}" != "true" && "${EVAL_OFFLOAD_MODELS}" != "false" ]]; then
    echo "ERROR: EVAL_OFFLOAD_MODELS must be true or false." >&2
    exit 2
fi
if [[ "${STAGE1_ALLOW_RECIPE_SUBSET}" != "true" && "${STAGE1_ALLOW_RECIPE_SUBSET}" != "false" ]]; then
    echo "ERROR: STAGE1_ALLOW_RECIPE_SUBSET must be true or false." >&2
    exit 2
fi
if [[ "${STAGE2_ALLOW_SOFTWARE_MISMATCH}" != "true" && "${STAGE2_ALLOW_SOFTWARE_MISMATCH}" != "false" ]]; then
    echo "ERROR: STAGE2_ALLOW_SOFTWARE_MISMATCH must be true or false." >&2
    exit 2
fi
if [[ "${STAGE2_ALLOW_DATA_ROOT_RELOCATION}" != "true" && "${STAGE2_ALLOW_DATA_ROOT_RELOCATION}" != "false" ]]; then
    echo "ERROR: STAGE2_ALLOW_DATA_ROOT_RELOCATION must be true or false." >&2
    exit 2
fi

ceil_div() {
    local numerator="$1"
    local denominator="$2"
    echo $(( (numerator + denominator - 1) / denominator ))
}

# Reference rebatching emits one smaller, 16-aligned tail at every NavAnywhere
# epoch, so Stage 1 needs two more optimizer steps than a plain ceil at batch96.
full_reference_epochs=$(( STAGE1_SAMPLES_PER_RANK / REFERENCE_SAMPLES_PER_RANK_PER_EPOCH ))
remaining_stage1_samples=$(( STAGE1_SAMPLES_PER_RANK % REFERENCE_SAMPLES_PER_RANK_PER_EPOCH ))
steps_per_reference_epoch="$(ceil_div "${REFERENCE_SAMPLES_PER_RANK_PER_EPOCH}" "${BATCH_SIZE}")"
remaining_stage1_steps="$(ceil_div "${remaining_stage1_samples}" "${BATCH_SIZE}")"
STAGE1_STEPS=$(( full_reference_epochs * steps_per_reference_epoch + remaining_stage1_steps ))
WARMUP_STEPS="$(ceil_div "${WARMUP_SAMPLES_PER_RANK}" "${BATCH_SIZE}")"
JOINT_STEPS="$(ceil_div "${JOINT_SAMPLES_PER_RANK}" "${BATCH_SIZE}")"

RUN_ROOT="${RESULTS_ROOT}/${RUN_ID}"
if [[ -n "${EXPERIMENT_TAG}" ]]; then
    STAGE1_RUN_NAME="stage1_latentpt_${EXPERIMENT_TAG}_bs${BATCH_SIZE}"
    STAGE2_RUN_NAME="stage2_latent_reset_${EXPERIMENT_TAG}_bs${BATCH_SIZE}"
else
    STAGE1_RUN_NAME="stage1_latentpt_nav1_bs${BATCH_SIZE}"
    STAGE2_RUN_NAME="stage2_latent_reset_bs${BATCH_SIZE}"
fi
STAGE1_DIR="${RUN_ROOT}/${STAGE1_RUN_NAME}"
STAGE2_DIR="${RUN_ROOT}/${STAGE2_RUN_NAME}"
printf -v STAGE1_FINAL_NAME 'pretrain_%07d.pth.tar' "${STAGE1_STEPS}"
printf -v STAGE2_FINAL_NAME 'joint_%07d.pth.tar' "${JOINT_STEPS}"
STAGE1_FINAL="${STAGE1_DIR}/checkpoints/${STAGE1_FINAL_NAME}"
STAGE2_FINAL="${STAGE2_DIR}/checkpoints/${STAGE2_FINAL_NAME}"
LAUNCH_LOG="${RUN_ROOT}/logs/launcher.log"

export NWM_NAVANYWHERE_ROOT="${NAVANYWHERE_ROOT}"
export NWM_NAVANYWHERE_RECIPE="${SAMPLING_RECIPE}"
export NWM_NAVANYWHERE_VAE_LATENT_ROOT="${STAGE1_VAE_LATENT_ROOT}"
export NWM_LATENT_PROXY_ROOT="${NAV1_PROXY_ROOT}"
export NWM_DATA_ROOT
export NWM_FINETUNE_VAE_LATENT_ROOT="${STAGE2_VAE_LATENT_ROOT}"
export NWM_RESULTS_DIR="${RUN_ROOT}"
export WANDB_MODE
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_OVERRIDES=(
    "model/generator=cdit_b"
    "training.batch_size=${BATCH_SIZE}"
    "training.eval_batch_size=${EVAL_BATCH_SIZE}"
    "training.wandb_enabled=${WANDB_ENABLED}"
    "training.wandb_project=${WANDB_PROJECT}"
    "training.results_dir=${RUN_ROOT}"
    "log_every=${LOG_EVERY}"
    "ckpt_every=${CKPT_EVERY}"
    "eval_every=${EVAL_EVERY}"
    "eval_at_first_step=${EVAL_AT_FIRST_STEP}"
    "eval_offload_models=${EVAL_OFFLOAD_MODELS}"
    "log_cuda_memory=true"
    "bfloat16=1"
    "torch_compile=0"
)

STAGE1_OVERRIDES=(
    "${COMMON_OVERRIDES[@]}"
    "seed=${SEED}"
    "dataset.precomputed_latents.enabled=true"
    "dataset.precomputed_latents.root=${STAGE1_VAE_LATENT_ROOT}"
    "+dataset.precomputed_latents.allow_training_hardware_mismatch=true"
    "training.num_workers=${STAGE1_NUM_WORKERS}"
    "training.optimizer.lr=${STAGE1_LR}"
    "training.run_name=${STAGE1_RUN_NAME}"
    "training.notes=LatentPT_${EXPERIMENT_TAG:-Nav1}_${HARDWARE_LABEL}_bs${BATCH_SIZE}"
    "training.wandb_tags=[LatentPT,${EXPERIMENT_TAG:-Nav1},CDiT-B,${HARDWARE_LABEL},bs${BATCH_SIZE}]"
    "+training.reference_batch_size=${REFERENCE_BATCH_SIZE}"
    "+training.target_samples_per_rank=${STAGE1_SAMPLES_PER_RANK}"
    "max_train_steps=${STAGE1_STEPS}"
)
if [[ "${STAGE1_DATASET_PROFILE}" == "navanywhere" ]]; then
    STAGE1_OVERRIDES+=(
        "dataset.sampling_recipe.path=${SAMPLING_RECIPE}"
        "dataset.sampling_recipe.sha256=${RECIPE_SHA256}"
        "dataset.precomputed_latents.allow_recipe_subset=${STAGE1_ALLOW_RECIPE_SUBSET}"
    )
else
    STAGE1_OVERRIDES+=(
        "dataset=nwm_real"
        "+dataset.precomputed_latents.allow_training_batch_resize=true"
        "+dataset.precomputed_latents.allow_training_software_mismatch=${STAGE2_ALLOW_SOFTWARE_MISMATCH}"
        "+dataset.precomputed_latents.allow_training_data_root_relocation=${STAGE2_ALLOW_DATA_ROOT_RELOCATION}"
        "dataset_selection.pretrain=[recon,sacson,scand,tartan_drive]"
        "dataset_selection.excluded_from_training=[navanywhere,go_stanford]"
        "motion_condition.train_types=[latent]"
        "~motion_condition.dataset_motion_types"
        "+motion_condition.dataset_motion_types={recon:latent,sacson:latent,scand:latent,tartan_drive:latent}"
        "motion_condition.eval_type=latent"
    )
fi

STAGE2_OVERRIDES=(
    "${COMMON_OVERRIDES[@]}"
    "seed=${SEED}"
    "dataset.precomputed_latents.enabled=true"
    "dataset.precomputed_latents.root=${STAGE2_VAE_LATENT_ROOT}"
    "+dataset.precomputed_latents.allow_training_batch_resize=true"
    "+dataset.precomputed_latents.allow_training_hardware_mismatch=true"
    "+dataset.precomputed_latents.allow_training_software_mismatch=${STAGE2_ALLOW_SOFTWARE_MISMATCH}"
    "+dataset.precomputed_latents.allow_training_data_root_relocation=${STAGE2_ALLOW_DATA_ROOT_RELOCATION}"
    "training.num_workers=${STAGE2_NUM_WORKERS}"
    "training.optimizer.lr=${ADAPTER_LR}"
    "training.run_name=${STAGE2_RUN_NAME}"
    "training.notes=LatentPT_${EXPERIMENT_TAG:-Nav1}_latent_reset_${HARDWARE_LABEL}_bs${BATCH_SIZE}"
    "training.wandb_tags=[LatentPT,${EXPERIMENT_TAG:-Nav1},latent-reset,CDiT-B,${HARDWARE_LABEL},bs${BATCH_SIZE}]"
    "finetune.adapter_lr=${ADAPTER_LR}"
    "finetune.backbone_lr=${BACKBONE_LR}"
    "finetune.warmup_steps=${WARMUP_STEPS}"
    "finetune.joint_steps=${JOINT_STEPS}"
    "+finetune.warmup_samples_per_rank=${WARMUP_SAMPLES_PER_RANK}"
    "+finetune.joint_samples_per_rank=${JOINT_SAMPLES_PER_RANK}"
)

STAGE1_COMMAND=(
    bash "${SCRIPT_DIR}/two_stage_nwm.sh" stage1 latentpt
    "--nproc=${WORLD_SIZE}" "--gpus=${GPU_IDS}" --
    "${STAGE1_OVERRIDES[@]}"
)
STAGE2_COMMAND=(
    bash "${SCRIPT_DIR}/two_stage_nwm.sh" stage2 latent_reset
    "--nproc=${WORLD_SIZE}" "--gpus=${GPU_IDS}"
    "--stage1-checkpoint=${STAGE1_FINAL}" --
    "${STAGE2_OVERRIDES[@]}"
)

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

echo "Nav1 ${HARDWARE_LABEL} training contract"
echo "  cwd:                         ${SCRIPT_DIR}"
echo "  run id:                      ${RUN_ID}"
echo "  output:                      ${RUN_ROOT}"
echo "  GPU ids/world size:          ${GPU_IDS} / ${WORLD_SIZE}"
echo "  batch/GPU:                   ${BATCH_SIZE}"
echo "  experiment tag/profile:      ${EXPERIMENT_TAG:-Nav1} / ${STAGE1_DATASET_PROFILE}"
echo "  eval interval/batch:         ${EVAL_EVERY} / ${EVAL_BATCH_SIZE}"
echo "  Stage-1 reference batch:     ${REFERENCE_BATCH_SIZE}"
echo "  Stage-1 samples/rank, steps: ${STAGE1_SAMPLES_PER_RANK}, ${STAGE1_STEPS}"
echo "  warmup samples/rank, steps:  ${WARMUP_SAMPLES_PER_RANK}, ${WARMUP_STEPS}"
echo "  joint samples/rank, steps:   ${JOINT_SAMPLES_PER_RANK}, ${JOINT_STEPS}"
echo "  Stage-1 LR:                  ${STAGE1_LR}"
echo "  Stage-2 adapter/backbone LR: ${ADAPTER_LR} / ${BACKBONE_LR}"
echo "  Nav1 cache:                  ${NAV1_PROXY_ROOT}"
echo "  recipe SHA-256:              ${RECIPE_SHA256}"
echo "Stage 1 command:"
print_command "${STAGE1_COMMAND[@]}"
echo "Stage 2 command:"
print_command "${STAGE2_COMMAND[@]}"

if (( DRY_RUN == 1 )); then
    echo "DRY_RUN=1: preflight, writes, and training were skipped."
    exit 0
fi

if [[ "${CONDA_PREFIX:-}" != "${NWM_CONDA_ENV_PATH}" ]]; then
    echo "ERROR: run this script in the explicit nwm environment:" >&2
    echo "  /file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm bash ${SCRIPT_DIR}/run_nwm_latentpt_nav1_80gb.sh" >&2
    exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != WORLD_SIZE )); then
    echo "ERROR: exactly ${WORLD_SIZE} GPU_IDS are required to preserve global samples." >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "ERROR: GPU_IDS must contain eight distinct numeric indices." >&2
        exit 2
    fi
    SEEN_GPUS["${gpu_id}"]=1
done

require_dir() {
    [[ -d "$1" ]] || { echo "ERROR: required directory is missing: $1" >&2; exit 2; }
}
require_file() {
    [[ -r "$1" ]] || { echo "ERROR: required file is unreadable: $1" >&2; exit 2; }
}
require_completed_cache() {
    require_dir "$1"
    require_file "$1/metadata.json"
    require_file "$1/_SUCCESS.json"
}

require_dir "${NAVANYWHERE_ROOT}"
require_dir "${NWM_DATA_ROOT}"
for dataset_dir in recon sacson scand tartan; do
    require_dir "${NWM_DATA_ROOT}/${dataset_dir}"
done
require_file "${SAMPLING_RECIPE}"
actual_recipe_sha="$(sha256sum "${SAMPLING_RECIPE}" | awk '{print $1}')"
if [[ "${actual_recipe_sha}" != "${RECIPE_SHA256}" ]]; then
    echo "ERROR: Nav1 recipe SHA-256 changed: ${actual_recipe_sha}" >&2
    exit 2
fi
require_completed_cache "${NAV1_PROXY_ROOT}"
require_completed_cache "${STAGE1_VAE_LATENT_ROOT}"
require_completed_cache "${STAGE2_VAE_LATENT_ROOT}"

if [[ "${STAGE1_DATASET_PROFILE}" == "navanywhere" ]]; then
python - "${NAV1_PROXY_ROOT}/metadata.json" "${RECIPE_SHA256}" \
    "${NAV1_CHECKPOINT_SHA256}" "${EXPECTED_PLAN_BATCH_SIZE}" \
    "${EXPECTED_PLAN_DATASET_LENGTH}" "${EXPECTED_PLAN_MAX_STEPS}" \
    "${EXPECTED_PLAN_STEPS_PER_EPOCH}" "${EXPECTED_PLAN_WORLD_SIZE}" <<'PY'
import json
import sys

(
    path,
    expected_recipe,
    expected_checkpoint,
    expected_batch_size,
    expected_dataset_length,
    expected_max_steps,
    expected_steps_per_epoch,
    expected_world_size,
) = sys.argv[1:]
with open(path, "r", encoding="utf-8") as stream:
    metadata = json.load(stream)
actual = metadata.get("sampling_recipe", {}).get("sha256")
if metadata.get("complete") is not True or metadata.get("status") != "complete":
    raise SystemExit(f"ERROR: Nav1 cache is not complete: {path}")
if actual != expected_recipe:
    raise SystemExit(
        "ERROR: Nav1 cache recipe SHA-256 mismatch: "
        f"{actual!r} != {expected_recipe!r}"
    )
checkpoint = metadata.get("checkpoint", {})
if (
    checkpoint.get("class_path") != "lam.navigation_variants.PixelActionLAM"
    or checkpoint.get("global_step") != 100000
    or checkpoint.get("sha256") != expected_checkpoint
):
    raise SystemExit("ERROR: proxy cache is not the pinned Nav1 step-100000 cache")
contract = metadata.get("training_pair_plan", {}).get("training_contract", {})
expected_contract = {
    "batch_size_per_rank": int(expected_batch_size),
    "dataset_length": int(expected_dataset_length),
    "max_train_steps": int(expected_max_steps),
    "seed": 20260901,
    "steps_per_epoch": int(expected_steps_per_epoch),
    "world_size": int(expected_world_size),
}
changed = {
    key: (contract.get(key), value)
    for key, value in expected_contract.items()
    if contract.get(key) != value
}
if changed:
    raise SystemExit(f"ERROR: Nav1 training-pair plan contract changed: {changed}")
PY
else
python - "${NAV1_PROXY_ROOT}" "${LATENT_ACTION_CHECKPOINT_SHA256:-}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_checkpoint = sys.argv[2]
metadata_bytes = (root / "metadata.json").read_bytes()
metadata = json.loads(metadata_bytes)
success = json.loads((root / "_SUCCESS.json").read_text())
if metadata.get("complete") is not True or success.get("complete") is not True:
    raise SystemExit(f"ERROR: incomplete four-dataset latent-action cache: {root}")
if success.get("metadata_sha256") != hashlib.sha256(metadata_bytes).hexdigest():
    raise SystemExit(f"ERROR: latent-action metadata digest mismatch: {root}")
contract = metadata.get("contract", {})
checkpoint = contract.get("checkpoint", {})
if checkpoint.get("class_path") != "lam.navigation_variants.PixelActionLAM":
    raise SystemExit("ERROR: unexpected latent-action checkpoint class")
if checkpoint.get("global_step") != 40000 or checkpoint.get("latent_dim") != 32:
    raise SystemExit("ERROR: expected a step-40000, 32-D latent-action cache")
if expected_checkpoint and checkpoint.get("sha256") != expected_checkpoint:
    raise SystemExit("ERROR: latent-action checkpoint SHA-256 mismatch")
counts = metadata.get("counts", {})
required = {"recon", "sacson", "scand", "tartan_drive"}
if set(counts) != required:
    raise SystemExit(f"ERROR: four-dataset cache coverage mismatch: {sorted(counts)}")
for dataset in sorted(required):
    if int(counts[dataset].get("pairs", 0)) <= 0:
        raise SystemExit(f"ERROR: no cached pairs for {dataset}")
print(f"Validated four-dataset latent-action cache: {root}")
PY
fi

if [[ -e "${STAGE1_DIR}" || -e "${STAGE2_DIR}" ]] && (( RESUME == 0 )); then
    echo "ERROR: run directories already exist; choose a new RUN_ID or pass --resume." >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}/logs"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
echo "  PID:                         $$"
echo "  launcher log:                ${LAUNCH_LOG}"
echo "Preflight filesystem capacity:"
df -h "${RESULTS_ROOT}"
available_kb="$(df -Pk "${RESULTS_ROOT}" | awk 'NR==2 {print $4}')"
if (( available_kb < MIN_FREE_DISK_GB * 1024 * 1024 )); then
    echo "ERROR: less than ${MIN_FREE_DISK_GB} GiB is free under RESULTS_ROOT." >&2
    exit 2
fi

if [[ -f "${STAGE2_FINAL}" ]]; then
    echo "Stage 2 final checkpoint already exists; no GPU work is needed."
else
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "ERROR: nvidia-smi is required for the GPU preflight." >&2
        exit 2
    }
    echo "GPU capacity before launch:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
    gpu_capacity="$(nvidia-smi --query-gpu=index,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits)"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        read -r total_mb free_mb utilization < <(
            awk -F',' -v wanted="${gpu_id}" '
                { for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
                $1 == wanted { print $2, $3, $4; found=1 }
                END { if (!found) exit 1 }
            ' <<< "${gpu_capacity}"
        ) || { echo "ERROR: selected GPU ${gpu_id} was not reported." >&2; exit 2; }
        if (( total_mb < MIN_TOTAL_GPU_MB )); then
            echo "ERROR: GPU ${gpu_id} has ${total_mb} MiB, below the ${MIN_TOTAL_GPU_MB} MiB threshold." >&2
            exit 2
        fi
        if (( free_mb < MIN_FREE_GPU_MB || utilization > MAX_GPU_UTILIZATION )); then
            echo "ERROR: GPU ${gpu_id} is busy (free=${free_mb} MiB, util=${utilization}%)." >&2
            echo "This launcher never stops or replaces an active experiment." >&2
            exit 2
        fi
    done
fi

run_stage() {
    local label="$1"
    shift
    echo "Starting ${label} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Exact ${label} command (cwd=${SCRIPT_DIR}):"
    print_command "$@"
    set +e
    "$@"
    local status=$?
    set -e
    if (( status != 0 )); then
        echo "ERROR: ${label} exited with status ${status}." >&2
        echo "If this was CUDA OOM, start a new RUN_ID with a smaller BATCH_SIZE (96 -> 80 -> 64 -> 48 -> 32 -> 16)." >&2
        return "${status}"
    fi
    echo "Completed ${label} at $(date -u +%Y-%m-%dT%H:%M:%SZ), exit status 0"
}

resolve_completed_checkpoint() {
    local latest_link="$1"
    local phase_prefix="$2"
    local minimum_steps="$3"
    local checkpoint filename checkpoint_steps

    [[ -e "${latest_link}" ]] || return 1
    checkpoint="$(realpath -e "${latest_link}")" || return 1
    filename="${checkpoint##*/}"
    if [[ ! "${filename}" =~ ^${phase_prefix}_([0-9]+)\.pth\.tar$ ]]; then
        return 1
    fi
    checkpoint_steps=$((10#${BASH_REMATCH[1]}))
    (( checkpoint_steps >= minimum_steps )) || return 1
    printf '%s\n' "${checkpoint}"
}

# Sample-budgeted training can take a few more optimizer steps than the static
# ceil estimate when the final dataloader batch crosses an epoch boundary.  The
# phase's latest checkpoint is authoritative once it has reached the estimated
# minimum; using only the predicted filename can strand a successful run at the
# Stage-1 -> Stage-2 transition.
stage1_completed_checkpoint=""
if stage1_completed_checkpoint="$(resolve_completed_checkpoint \
    "${STAGE1_DIR}/checkpoints/latest.pth.tar" pretrain "${STAGE1_STEPS}")"; then
    STAGE1_FINAL="${stage1_completed_checkpoint}"
    echo "Stage 1 sample budget is already complete: ${STAGE1_FINAL}"
fi

if [[ -z "${stage1_completed_checkpoint}" ]]; then
    if (( RESUME == 1 )) && [[ -e "${STAGE1_DIR}/checkpoints/latest.pth.tar" ]]; then
        stage1_resume="$(realpath -e "${STAGE1_DIR}/checkpoints/latest.pth.tar")"
        STAGE1_COMMAND=(
            bash "${SCRIPT_DIR}/two_stage_nwm.sh" stage1 latentpt
            "--nproc=${WORLD_SIZE}" "--gpus=${GPU_IDS}"
            "--resume=${stage1_resume}" -- "${STAGE1_OVERRIDES[@]}"
        )
    fi
    run_stage "Stage 1 LatentPT" "${STAGE1_COMMAND[@]}"
fi
if ! STAGE1_FINAL="$(resolve_completed_checkpoint \
    "${STAGE1_DIR}/checkpoints/latest.pth.tar" pretrain "${STAGE1_STEPS}")"; then
    echo "ERROR: Stage 1 latest checkpoint did not reach ${STAGE1_STEPS} steps." >&2
    exit 2
fi

stage2_completed_checkpoint=""
if stage2_completed_checkpoint="$(resolve_completed_checkpoint \
    "${STAGE2_DIR}/checkpoints/latest.pth.tar" joint "${JOINT_STEPS}")"; then
    STAGE2_FINAL="${stage2_completed_checkpoint}"
    echo "Stage 2 sample budget is already complete: ${STAGE2_FINAL}"
fi

if [[ -z "${stage2_completed_checkpoint}" ]]; then
    if (( RESUME == 1 )) && [[ -e "${STAGE2_DIR}/checkpoints/latest.pth.tar" ]]; then
        stage2_resume="$(realpath -e "${STAGE2_DIR}/checkpoints/latest.pth.tar")"
        STAGE2_COMMAND=(
            bash "${SCRIPT_DIR}/two_stage_nwm.sh" stage2 latent_reset
            "--nproc=${WORLD_SIZE}" "--gpus=${GPU_IDS}"
            "--resume=${stage2_resume}" -- "${STAGE2_OVERRIDES[@]}"
        )
    else
        STAGE2_COMMAND=(
            bash "${SCRIPT_DIR}/two_stage_nwm.sh" stage2 latent_reset
            "--nproc=${WORLD_SIZE}" "--gpus=${GPU_IDS}"
            "--stage1-checkpoint=${STAGE1_FINAL}" --
            "${STAGE2_OVERRIDES[@]}"
        )
    fi
    run_stage "Stage 2 latent_reset" "${STAGE2_COMMAND[@]}"
fi
if ! STAGE2_FINAL="$(resolve_completed_checkpoint \
    "${STAGE2_DIR}/checkpoints/latest.pth.tar" joint "${JOINT_STEPS}")"; then
    echo "ERROR: Stage 2 latest checkpoint did not reach ${JOINT_STEPS} joint steps." >&2
    exit 2
fi

if (( KEEP_INTERMEDIATE == 0 )); then
    echo "Removing intermediate checkpoints from this completed run only."
    shopt -s nullglob
    for checkpoint in "${STAGE1_DIR}/checkpoints"/pretrain_*.pth.tar; do
        [[ "${checkpoint}" == "${STAGE1_FINAL}" ]] || rm -f -- "${checkpoint}"
    done
    for checkpoint in "${STAGE2_DIR}/checkpoints"/warmup_*.pth.tar; do
        rm -f -- "${checkpoint}"
    done
    for checkpoint in "${STAGE2_DIR}/checkpoints"/joint_*.pth.tar; do
        [[ "${checkpoint}" == "${STAGE2_FINAL}" ]] || rm -f -- "${checkpoint}"
    done
    rm -f -- "${STAGE2_DIR}/checkpoints/transition.pth.tar"
    shopt -u nullglob
fi

# Keep the launcher/Hydra training logs, but discard W&B's verbose internal
# debug logs and orphaned atomic temp files after a fully successful run.
if (( CLEAN_WANDB_DEBUG_LOGS == 1 )); then
    while IFS= read -r -d '' debug_log; do
        rm -f -- "${debug_log}"
    done < <(
        find "${RUN_ROOT}" -type f \
            \( -name debug.log -o -name debug-internal.log \) -print0
    )
fi
while IFS= read -r -d '' temporary_file; do
    rm -f -- "${temporary_file}"
done < <(find "${RUN_ROOT}" -type f -name '*.tmp' -print0)

echo "Nav1 LatentPT -> latent_reset training completed successfully."
echo "  Stage 1 final: ${STAGE1_FINAL}"
echo "  Stage 2 final: ${STAGE2_FINAL}"
echo "  Preserved log: ${LAUNCH_LOG}"
