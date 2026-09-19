#!/usr/bin/env python3
"""Pin and download the external source and weights required by RAE-NWM."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "third_party/raenwm"
DEFAULT_ASSETS = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/rae-nwm"
)

RAENWM_REPOSITORY = "https://github.com/20robo/raenwm.git"
RAENWM_REVISION = "0219ce41c44d515f86719dd763c1efe7c7f72519"
RAENWM_MODEL_REPOSITORY = "zmkun20/raenwm"
RAENWM_MODEL_REVISION = "3d21560bdbdbc8cc3d4a796e1e110d60e920d273"
RAE_COLLECTION_REPOSITORY = "nyu-visionx/RAE-collections"
RAE_COLLECTION_REVISION = "1be4f03273523431f099a934da4cf1940dc6039f"
DINO_REPOSITORY = "facebook/dinov2-with-registers-base"
DINO_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"

CHECKPOINT_RELATIVE = Path("checkpoints/raenwm_b.pth.tar")
CHECKPOINT_SHA256 = "97244579618eb0e376355157a5c1df7a83e534cb1ca35d8ff4e43c4fb4f6a9d0"
DECODER_RELATIVE = Path("models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt")
STATS_RELATIVE = Path("models/stats/dinov2/wReg_base/imagenet1k/stat.pt")
DINO_RELATIVE = Path("models/dinov2-with-registers-base")
MANIFEST_RELATIVE = Path("assets_manifest.json")

# The three large upstream files total about 7.5 GB. Keep enough headroom for
# Hugging Face's temporary files and an interrupted/resumed download.
MINIMUM_FREE_BYTES = 16 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str, cwd: Path | None = None) -> str:
    command = [
        "git",
        "-c",
        "http.proxy=",
        "-c",
        "https.proxy=",
        *arguments,
    ]
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def prepare_source(source: Path) -> None:
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        git("clone", RAENWM_REPOSITORY, str(source))
    if not (source / ".git").exists():
        raise RuntimeError(f"RAE-NWM source is not a Git checkout: {source}")
    remote = git("remote", "get-url", "origin", cwd=source)
    if remote.rstrip("/").removesuffix(".git") != RAENWM_REPOSITORY.rstrip(
        "/"
    ).removesuffix(".git"):
        raise RuntimeError(f"Unexpected RAE-NWM origin at {source}: {remote}")
    try:
        git("cat-file", "-e", f"{RAENWM_REVISION}^{{commit}}", cwd=source)
    except subprocess.CalledProcessError:
        git("fetch", "origin", RAENWM_REVISION, cwd=source)
    git("checkout", "--detach", RAENWM_REVISION, cwd=source)
    actual = git("rev-parse", "HEAD", cwd=source)
    if actual != RAENWM_REVISION:
        raise RuntimeError(
            f"RAE-NWM revision mismatch: expected {RAENWM_REVISION}, got {actual}"
        )


def resolve_huggingface_file(
    repository: str, revision: str, remote_path: str
) -> tuple[str, int | None]:
    url = (
        f"https://huggingface.co/{repository}/resolve/{revision}/"
        f"{quote(remote_path, safe='/')}"
    )
    command = [
        "curl",
        "-fsSI",
        "--retry",
        "5",
        "--retry-connrefused",
        "--connect-timeout",
        "10",
        "--max-time",
        "120",
        url,
    ]
    for attempt in range(5):
        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            break
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            time.sleep(3)
    headers: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", maxsplit=1)
        headers[name.strip().lower()] = value.strip()
    resolved = urljoin(url, headers.get("location", url))
    size_value = headers.get("x-linked-size")
    return resolved, int(size_value) if size_value is not None else None


def download_huggingface_file(
    repository: str,
    revision: str,
    remote_path: str,
    destination: Path,
) -> None:
    resolved, expected_size = resolve_huggingface_file(
        repository, revision, remote_path
    )
    if destination.is_file() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        print(f"Using complete asset: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".incomplete")
    command = [
        "curl",
        "-fL",
        "--retry",
        "10",
        "--retry-connrefused",
        "--connect-timeout",
        "20",
        "--continue-at",
        "-",
        "--output",
        str(temporary),
    ]
    hostname = (urlparse(resolved).hostname or "").lower()
    if hostname != "huggingface.co":
        # The Hugging Face API is reachable through the configured proxy, while
        # its signed AWS CDN URLs are much faster over direct IPv4.
        command.extend(["--noproxy", "*", "-4"])
    print(
        f"Downloading {repository}@{revision}:{remote_path} -> {destination}",
        flush=True,
    )
    # The cluster image ships an older curl without --retry-all-errors, and its
    # built-in retry policy does not retry error 18 (a prematurely closed CDN
    # transfer). Re-run curl here so --continue-at resumes the partial file.
    for attempt in range(50):
        result = subprocess.run([*command, resolved], check=False)
        if result.returncode == 0:
            break
        if attempt == 49:
            result.check_returncode()
        partial_size = temporary.stat().st_size if temporary.exists() else 0
        print(
            f"curl exited {result.returncode}; retrying partial download "
            f"from {partial_size} bytes ({attempt + 2}/50)",
            flush=True,
        )
        time.sleep(3)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {destination}: expected {expected_size}, "
            f"got {temporary.stat().st_size}"
        )
    temporary.replace(destination)


def download_assets(assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(assets).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"At least {MINIMUM_FREE_BYTES / 1024**3:.0f} GiB free is required under "
            f"{assets}; only {free_bytes / 1024**3:.1f} GiB is available"
        )

    download_huggingface_file(
        RAENWM_MODEL_REPOSITORY,
        RAENWM_MODEL_REVISION,
        "raenwm_b.pth.tar",
        assets / CHECKPOINT_RELATIVE,
    )
    download_huggingface_file(
        RAE_COLLECTION_REPOSITORY,
        RAE_COLLECTION_REVISION,
        "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        assets / DECODER_RELATIVE,
    )
    download_huggingface_file(
        RAE_COLLECTION_REPOSITORY,
        RAE_COLLECTION_REVISION,
        "stats/dinov2/wReg_base/imagenet1k/stat.pt",
        assets / STATS_RELATIVE,
    )
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json"):
        download_huggingface_file(
            DINO_REPOSITORY,
            DINO_REVISION,
            filename,
            assets / DINO_RELATIVE / filename,
        )


def validate_and_manifest(source: Path, assets: Path) -> Path:
    files = {
        "checkpoint": assets / CHECKPOINT_RELATIVE,
        "decoder": assets / DECODER_RELATIVE,
        "normalization_stats": assets / STATS_RELATIVE,
        "dinov2_config": assets / DINO_RELATIVE / "config.json",
        "dinov2_weights": assets / DINO_RELATIVE / "model.safetensors",
        "dinov2_preprocessor": assets / DINO_RELATIVE / "preprocessor_config.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"RAE-NWM assets are incomplete: {missing}")
    checkpoint_sha256 = sha256_file(files["checkpoint"])
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            "RAE-NWM checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": RAENWM_REPOSITORY,
            "revision": RAENWM_REVISION,
            "path": str(source.resolve()),
        },
        "upstream_repositories": {
            RAENWM_MODEL_REPOSITORY: RAENWM_MODEL_REVISION,
            RAE_COLLECTION_REPOSITORY: RAE_COLLECTION_REVISION,
            DINO_REPOSITORY: DINO_REVISION,
        },
        "files": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": (
                    checkpoint_sha256 if name == "checkpoint" else sha256_file(path)
                ),
            }
            for name, path in files.items()
        },
    }
    output = assets / MANIFEST_RELATIVE
    temporary = output.with_suffix(f".tmp.{output.suffix.lstrip('.')}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not access the network; validate the pinned checkout and downloaded files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        actual = git("rev-parse", "HEAD", cwd=args.source)
        if actual != RAENWM_REVISION:
            raise RuntimeError(
                f"RAE-NWM revision mismatch: expected {RAENWM_REVISION}, got {actual}"
            )
    else:
        prepare_source(args.source)
        download_assets(args.assets)
    print(validate_and_manifest(args.source, args.assets))


if __name__ == "__main__":
    main()
