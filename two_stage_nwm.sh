#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

show_help() {
    cat <<'EOF'
Usage:
  ./two_stage_nwm.sh stage1 {timept|geopt|idmpt|latentpt} [options] [-- HYDRA_OVERRIDES...]
  ./two_stage_nwm.sh stage2 {latent_reset|latent_align|latent_real_to_latent|time_reset|geo_reset|idm_reset} [options] [-- HYDRA_OVERRIDES...]

Options:
  --nproc=N                    Processes per node (default: one, or GPU list size)
  --gpus=LIST                  CUDA_VISIBLE_DEVICES, for example 0 or 0,1,2,3
  --nodes=N                    Number of nodes (default: 1)
  --host=HOST                  Rendezvous host (default: localhost)
  --rank=N                     Current node rank (default: 0)
  --port=PORT                  Rendezvous port (default: 0 for one node)
  --stage1-checkpoint=PATH     Stage-1 initialization checkpoint for stage 2
  --resume=PATH                Resume a warm-up or joint checkpoint in the same run
  --dry-run                    Print the exact scripts/train.sh command only
  --help                       Show this message

Required data environment:
  stage 1: NWM_NAVANYWHERE_ROOT
  stage 2: NWM_DATA_ROOT
  all runs: NWM_RESULTS_DIR (logs and checkpoints; not needed by --dry-run)

Optional stage-1 split selection:
  NWM_NAVANYWHERE_MANIFEST (JSON, JSONL, or text trajectory manifest)

Additional cache environment:
  geopt:        NWM_GEOMETRY_PROXY_ROOT
  idmpt:        NWM_IDM_PROXY_ROOT
  latentpt:     NWM_LATENT_PROXY_ROOT
  latent_align: NWM_FINETUNE_LATENT_ROOT
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi
if (( $# < 2 )); then
    show_help >&2
    exit 2
fi

STAGE="$1"
VARIANT="$2"
shift 2

NPROC=""
GPU_LIST="${CUDA_VISIBLE_DEVICES:-}"
NUM_NODES=1
HOST_NODE_ADDR=localhost
CURR_NODE_RANK=0
PORT=0
STAGE1_CHECKPOINT=""
RESUME_CHECKPOINT=""
DRY_RUN=0
HYDRA_EXTRA=()

while (( $# > 0 )); do
    case "$1" in
        --nproc=*) NPROC="${1#*=}" ;;
        --gpus=*) GPU_LIST="${1#*=}" ;;
        --nodes=*) NUM_NODES="${1#*=}" ;;
        --host=*) HOST_NODE_ADDR="${1#*=}" ;;
        --rank=*) CURR_NODE_RANK="${1#*=}" ;;
        --port=*) PORT="${1#*=}" ;;
        --stage1-checkpoint=*) STAGE1_CHECKPOINT="${1#*=}" ;;
        --resume=*) RESUME_CHECKPOINT="${1#*=}" ;;
        --dry-run) DRY_RUN=1 ;;
        --help|-h) show_help; exit 0 ;;
        --)
            shift
            HYDRA_EXTRA=("$@")
            break
            ;;
        *)
            echo "ERROR: unknown option $1 (put Hydra overrides after --)." >&2
            exit 2
            ;;
    esac
    shift
done

case "${STAGE}" in
    stage1)
        case "${VARIANT}" in
            timept|geopt|idmpt|latentpt) ;;
            *) echo "ERROR: unsupported stage-1 variant ${VARIANT}." >&2; exit 2 ;;
        esac
        ;;
    stage2)
        case "${VARIANT}" in
            latent_reset|latent_align|latent_real_to_latent|time_reset|geo_reset|idm_reset) ;;
            *) echo "ERROR: unsupported stage-2 variant ${VARIANT}." >&2; exit 2 ;;
        esac
        ;;
    *)
        echo "ERROR: stage must be stage1 or stage2, got ${STAGE}." >&2
        exit 2
        ;;
esac

for integer_value in "${NUM_NODES}" "${CURR_NODE_RANK}" "${PORT}"; do
    if ! [[ "${integer_value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: node, rank, and port values must be non-negative integers." >&2
        exit 2
    fi
done
if (( NUM_NODES < 1 )); then
    echo "ERROR: --nodes must be at least 1." >&2
    exit 2
fi
if (( NUM_NODES > 1 && PORT == 0 )); then
    echo "ERROR: multi-node launch requires a non-zero --port." >&2
    exit 2
fi

if [[ -n "${GPU_LIST}" ]]; then
    IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
    if [[ -z "${NPROC}" ]]; then
        NPROC="${#GPU_ARRAY[@]}"
    fi
else
    NPROC="${NPROC:-1}"
fi
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --nproc must be a positive integer." >&2
    exit 2
fi
if [[ -n "${GPU_LIST}" ]] && (( NPROC > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: --nproc=${NPROC} exceeds the ${#GPU_ARRAY[@]} entries in --gpus." >&2
    exit 2
fi
if [[ -n "${STAGE1_CHECKPOINT}" && -n "${RESUME_CHECKPOINT}" ]]; then
    echo "ERROR: use either --stage1-checkpoint for a new stage 2 run or --resume, not both." >&2
    exit 2
fi
if [[ "${STAGE}" == "stage1" && -n "${STAGE1_CHECKPOINT}" ]]; then
    echo "ERROR: --stage1-checkpoint is valid only for stage2." >&2
    exit 2
fi
if [[ "${STAGE}" == "stage2" && -z "${STAGE1_CHECKPOINT}" && -z "${RESUME_CHECKPOINT}" && "${DRY_RUN}" == 0 ]]; then
    echo "ERROR: a new stage-2 run needs --stage1-checkpoint; use --resume for an existing stage-2 checkpoint." >&2
    exit 2
fi

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} must be set for ${STAGE}/${VARIANT}." >&2
        exit 2
    fi
}

canonicalize_existing_dir_env() {
    local name="$1"
    local value="${!name}"
    local resolved
    if ! resolved="$(realpath -e -- "${value}")" || [[ ! -d "${resolved}" ]]; then
        echo "ERROR: ${name} must name an existing directory: ${value}" >&2
        exit 2
    fi
    printf -v "${name}" '%s' "${resolved}"
    export "${name}"
}

canonicalize_existing_file_env() {
    local name="$1"
    local value="${!name}"
    local resolved
    if ! resolved="$(realpath -e -- "${value}")" || [[ ! -f "${resolved}" ]]; then
        echo "ERROR: ${name} must name an existing file: ${value}" >&2
        exit 2
    fi
    printf -v "${name}" '%s' "${resolved}"
    export "${name}"
}

if (( DRY_RUN == 0 )); then
    require_env NWM_RESULTS_DIR
    NWM_RESULTS_DIR="$(realpath -m -- "${NWM_RESULTS_DIR}")"
    export NWM_RESULTS_DIR
    if [[ "${STAGE}" == "stage1" ]]; then
        require_env NWM_NAVANYWHERE_ROOT
        canonicalize_existing_dir_env NWM_NAVANYWHERE_ROOT
        if [[ -n "${NWM_NAVANYWHERE_MANIFEST:-}" ]]; then
            canonicalize_existing_file_env NWM_NAVANYWHERE_MANIFEST
        fi
        case "${VARIANT}" in
            geopt)
                require_env NWM_GEOMETRY_PROXY_ROOT
                canonicalize_existing_dir_env NWM_GEOMETRY_PROXY_ROOT
                ;;
            idmpt)
                require_env NWM_IDM_PROXY_ROOT
                canonicalize_existing_dir_env NWM_IDM_PROXY_ROOT
                ;;
            latentpt)
                require_env NWM_LATENT_PROXY_ROOT
                canonicalize_existing_dir_env NWM_LATENT_PROXY_ROOT
                ;;
        esac
    else
        require_env NWM_DATA_ROOT
        canonicalize_existing_dir_env NWM_DATA_ROOT
        if [[ "${VARIANT}" == "latent_align" ]]; then
            require_env NWM_FINETUNE_LATENT_ROOT
            canonicalize_existing_dir_env NWM_FINETUNE_LATENT_ROOT
        fi
    fi

    for checkpoint_path in "${STAGE1_CHECKPOINT}" "${RESUME_CHECKPOINT}"; do
        if [[ -n "${checkpoint_path}" && ! -f "${checkpoint_path}" ]]; then
            echo "ERROR: checkpoint is not a readable file: ${checkpoint_path}" >&2
            exit 2
        fi
    done
    if [[ -n "${STAGE1_CHECKPOINT}" ]]; then
        STAGE1_CHECKPOINT="$(realpath -e -- "${STAGE1_CHECKPOINT}")"
    fi
    if [[ -n "${RESUME_CHECKPOINT}" ]]; then
        RESUME_CHECKPOINT="$(realpath -e -- "${RESUME_CHECKPOINT}")"
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is unavailable; GPU capacity cannot be checked." >&2
        exit 2
    fi
    echo "GPU capacity before launch:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    RESULTS_PARENT="${NWM_RESULTS_DIR}"
    while [[ ! -e "${RESULTS_PARENT}" && "${RESULTS_PARENT}" != "/" ]]; do
        RESULTS_PARENT="$(dirname -- "${RESULTS_PARENT}")"
    done
    echo "Output filesystem capacity (${RESULTS_PARENT}):"
    df -h "${RESULTS_PARENT}"
fi

HYDRA_ARGS=(
    --config-name nwm
    "two_stage=${VARIANT}"
)
if [[ -n "${STAGE1_CHECKPOINT}" ]]; then
    HYDRA_ARGS+=("finetune.stage1_checkpoint=${STAGE1_CHECKPOINT}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    HYDRA_ARGS+=("training.from_checkpoint=${RESUME_CHECKPOINT}")
fi
if [[ -n "${NWM_RESULTS_DIR:-}" ]]; then
    HYDRA_ARGS+=("training.results_dir=${NWM_RESULTS_DIR}")
fi
HYDRA_ARGS+=("${HYDRA_EXTRA[@]}")

LAUNCH=(
    bash scripts/train.sh
    "--nproc=${NPROC}"
    "--nodes=${NUM_NODES}"
    "--host=${HOST_NODE_ADDR}"
    "--rank=${CURR_NODE_RANK}"
    "--port=${PORT}"
    --
    "${HYDRA_ARGS[@]}"
)

echo "Two-stage NWM launch: stage=${STAGE}, variant=${VARIANT}, nproc/node=${NPROC}, nodes=${NUM_NODES}"
if [[ -n "${GPU_LIST}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
if (( DRY_RUN == 1 )); then
    printf 'Dry-run command:'
    printf ' %q' "${LAUNCH[@]}"
    printf '\n'
    exit 0
fi

exec "${LAUNCH[@]}"
