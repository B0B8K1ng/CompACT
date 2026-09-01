#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${DREAMSIM_CACHE_DIR:-/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models}"
DOWNLOAD_URL="https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip"
TOTAL_BYTES=1261181618
NUM_PARTS="${DREAMSIM_DOWNLOAD_CONNECTIONS:-16}"
CHUNK_BYTES=$(( (TOTAL_BYTES + NUM_PARTS - 1) / NUM_PARTS ))
PARTS_DIR="${CACHE_DIR}/.dreamsim_ensemble_parts"
ARCHIVE="${CACHE_DIR}/pretrained.zip"

mkdir -p "${CACHE_DIR}" "${PARTS_DIR}"

if [[ -f "${ARCHIVE}" ]]; then
    archive_size="$(stat -c '%s' "${ARCHIVE}")"
    if [[ "${archive_size}" -eq "${TOTAL_BYTES}" ]] && unzip -tq "${ARCHIVE}"; then
        echo "DreamSim archive is already complete: ${ARCHIVE}"
    else
        backup="${ARCHIVE}.incomplete.$(date -u +%Y%m%d_%H%M%S)"
        mv "${ARCHIVE}" "${backup}"
        echo "Moved incomplete archive to ${backup}"
    fi
fi

if [[ ! -f "${ARCHIVE}" ]]; then
    export CACHE_DIR DOWNLOAD_URL TOTAL_BYTES NUM_PARTS CHUNK_BYTES PARTS_DIR
    seq 0 $((NUM_PARTS - 1)) | xargs -P "${NUM_PARTS}" -I '{}' bash -c '
        set -euo pipefail
        idx="$1"
        start=$((idx * CHUNK_BYTES))
        end=$((start + CHUNK_BYTES - 1))
        if (( end >= TOTAL_BYTES )); then
            end=$((TOTAL_BYTES - 1))
        fi
        expected=$((end - start + 1))
        part="$(printf "%s/part_%02d" "${PARTS_DIR}" "${idx}")"

        if [[ -f "${part}" ]] && [[ "$(stat -c "%s" "${part}")" -eq "${expected}" ]]; then
            echo "part ${idx} already complete (${expected} bytes)"
            exit 0
        fi

        tmp="${part}.downloading"
        if [[ -f "${tmp}" ]]; then
            stale="${tmp}.stale.$(date -u +%Y%m%d_%H%M%S)"
            mv "${tmp}" "${stale}"
        fi

        success=0
        for attempt in $(seq 1 50); do
            if curl -L --fail --silent --show-error \
                --retry 10 --retry-delay 1 \
                --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
                --range "${start}-${end}" \
                -o "${tmp}" "${DOWNLOAD_URL}"; then
                success=1
                break
            fi
            echo "part ${idx} retry ${attempt}/50" >&2
            sleep 2
        done
        if [[ "${success}" -ne 1 ]]; then
            echo "part ${idx} failed after 50 attempts" >&2
            exit 1
        fi

        actual="$(stat -c "%s" "${tmp}")"
        if [[ "${actual}" -ne "${expected}" ]]; then
            echo "part ${idx} size mismatch: expected=${expected} actual=${actual}" >&2
            exit 1
        fi
        mv "${tmp}" "${part}"
        echo "part ${idx} complete (${actual} bytes)"
    ' _ '{}'

    archive_tmp="$(mktemp "${CACHE_DIR}/.pretrained.zip.multipart.XXXXXX")"
    for idx in $(seq 0 $((NUM_PARTS - 1))); do
        part="$(printf '%s/part_%02d' "${PARTS_DIR}" "${idx}")"
        cat "${part}" >> "${archive_tmp}"
    done

    actual_total="$(stat -c '%s' "${archive_tmp}")"
    if [[ "${actual_total}" -ne "${TOTAL_BYTES}" ]]; then
        echo "archive size mismatch: expected=${TOTAL_BYTES} actual=${actual_total}" >&2
        exit 1
    fi
    unzip -tq "${archive_tmp}"
    mv "${archive_tmp}" "${ARCHIVE}"
fi

unzip -oq "${ARCHIVE}" -d "${CACHE_DIR}"

for required in \
    dino_vitb16_pretrain.pth \
    open_clip_vitb16_pretrain.pth.tar \
    clip_vitb16_pretrain.pth.tar; do
    if [[ ! -s "${CACHE_DIR}/${required}" ]]; then
        echo "Missing required DreamSim weight: ${required}" >&2
        exit 1
    fi
done
if [[ ! -d "${CACHE_DIR}/ensemble_lora" ]]; then
    echo "Missing required DreamSim LoRA directory: ensemble_lora" >&2
    exit 1
fi

# The assembled archive has passed both exact-size and ZIP integrity checks.
# Remove only the downloader's private temporary directory.
find "${PARTS_DIR}" -type f -delete
rmdir "${PARTS_DIR}"

echo "DreamSim ensemble is ready in ${CACHE_DIR}"
