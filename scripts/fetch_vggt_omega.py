#!/usr/bin/env python3
"""Fetch and verify the pinned gated VGGT-Omega checkpoint on NAS.

The access token is read from an environment variable and is never written to
the checkpoint manifest.  Files are downloaded into a sibling staging
directory, checked, and atomically published one file at a time; the manifest
is published last and therefore acts as the completion marker.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MODEL_NAME = "VGGT-Omega-1B-512"
DEFAULT_REPO_ID = "facebook/VGGT-Omega"
DEFAULT_REVISION = "ba9db085d6b7349b738fa2e37d198bb4dd077954"
DEFAULT_WEIGHT = "vggt_omega_1b_512.pt"
DEFAULT_WEIGHT_SHA256 = (
    "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
)
DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_OUTPUT_DIR = Path(
    "/file_system/nas/algorithm/dujun.nie/models/VGGT-Omega-1B-512"
)
MANIFEST_NAME = "checkpoint_manifest.json"
METADATA_FILES = ("LICENSE.txt", "README.md")
OFFICIAL_CODE_REPO = "https://github.com/facebookresearch/vggt-omega.git"
OFFICIAL_CODE_COMMIT = "282ec70363edeff59424bf43731658092fba3d37"
MIN_WEIGHT_BYTES = 1_000_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLOCKED_GATED_ENDPOINT_HOSTS = {"hf-mirror.com", "www.hf-mirror.com"}


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def expected_files(weight_filename: str) -> tuple[str, ...]:
    return (weight_filename, *METADATA_FILES)


def validate_filename(filename: str) -> None:
    path = Path(filename)
    if path.is_absolute() or len(path.parts) != 1 or filename in {"", ".", ".."}:
        raise ValueError(f"Checkpoint filename must be a basename, got {filename!r}")


def normalize_endpoint(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Hugging Face endpoint must be an HTTPS origin without a path")
    hostname = parsed.hostname.lower()
    if hostname in BLOCKED_GATED_ENDPOINT_HOSTS or hostname.endswith(".hf-mirror.com"):
        raise ValueError(
            "hf-mirror.com is rejected for this gated model because its redirect "
            "drops the Authorization header; use https://huggingface.co"
        )
    return normalized


def validate_weight(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    size = path.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        raise ValueError(
            f"{path} is only {size} bytes; refusing a probable Git-LFS/Xet pointer"
        )
    with path.open("rb") as handle:
        prefix = handle.read(256)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"{path} is a Git-LFS pointer, not the checkpoint payload")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Weight SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return {"path": path.name, "bytes": size, "sha256": digest}


def validate_regular_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required file is missing or empty: {path}")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_install(
    output_dir: Path,
    *,
    repo_id: str,
    revision: str,
    endpoint: str,
    weight_filename: str,
    expected_weight_sha256: str | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return False, [f"missing completion manifest: {manifest_path}"]
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"invalid completion manifest: {exc}"]

    source = manifest.get("source", {})
    code = manifest.get("official_code", {})
    if manifest.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    if manifest.get("model_name") != MODEL_NAME:
        errors.append(f"manifest model_name is not {MODEL_NAME}")
    if source.get("repo_id") != repo_id:
        errors.append(f"manifest repo_id is not {repo_id}")
    if source.get("endpoint") != endpoint:
        errors.append(f"manifest endpoint is not {endpoint}")
    if source.get("requested_revision") != revision:
        errors.append(f"manifest requested_revision is not {revision}")
    if source.get("resolved_revision") != revision:
        errors.append(f"manifest resolved_revision is not pinned revision {revision}")
    if code.get("repository") != OFFICIAL_CODE_REPO:
        errors.append("manifest official code repository is unexpected")
    if code.get("commit") != OFFICIAL_CODE_COMMIT:
        errors.append(f"manifest official code commit is not {OFFICIAL_CODE_COMMIT}")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        return False, errors + ["manifest files must be a list"]
    by_path = {entry.get("path"): entry for entry in entries if isinstance(entry, dict)}
    for filename in expected_files(weight_filename):
        entry = by_path.get(filename)
        path = output_dir / filename
        if entry is None:
            errors.append(f"manifest has no entry for {filename}")
            continue
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        recorded_hash = entry.get("sha256")
        recorded_bytes = entry.get("bytes")
        if not isinstance(recorded_hash, str) or not SHA256_RE.fullmatch(recorded_hash):
            errors.append(f"invalid manifest SHA-256 for {filename}")
            continue
        if path.stat().st_size != recorded_bytes:
            errors.append(f"size mismatch for {filename}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != recorded_hash:
            errors.append(f"SHA-256 mismatch for {filename}")
        if filename == weight_filename:
            if path.stat().st_size < MIN_WEIGHT_BYTES:
                errors.append(
                    f"checkpoint is unexpectedly small: {path.stat().st_size} bytes"
                )
            if expected_weight_sha256 and actual_hash != expected_weight_sha256:
                errors.append("checkpoint does not match --expected-weight-sha256")
    return not errors, errors


@contextmanager
def exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_publish(source: Path, destination: Path) -> None:
    """Publish a regular file atomically without preserving a staging symlink."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".publishing", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=16 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    else:
        os.replace(source, destination)
    fsync_directory(destination.parent)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def download_and_publish(args: argparse.Namespace, token: str) -> None:
    # huggingface_hub reads HF_ENDPOINT while importing constants. Override a
    # machine-wide mirror before the first import, then also pass the endpoint
    # explicitly so gated redirects retain their Authorization header.
    os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install the project environment first"
        ) from exc

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Keep an interrupted Xet/LFS partial download for the next locked retry.
    # The name is fully resolved under output_dir.parent and never user-expanded.
    staging = output_dir.parent / (
        f".{output_dir.name}.download.{args.revision}.{args.weight_filename}"
    )
    staging.mkdir(parents=True, exist_ok=True)
    info = HfApi(endpoint=args.endpoint, token=token).model_info(
        args.repo_id,
        revision=args.revision,
        files_metadata=True,
    )
    resolved_revision = info.sha
    if resolved_revision != args.revision:
        raise RuntimeError(
            "The requested revision did not resolve to the exact pinned commit: "
            f"requested={args.revision}, resolved={resolved_revision}"
        )

    for filename in expected_files(args.weight_filename):
        hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            revision=args.revision,
            token=token,
            local_dir=staging,
            force_download=args.force_download,
            endpoint=args.endpoint,
        )

    files = [
        validate_weight(
            staging / args.weight_filename,
            expected_sha256=args.expected_weight_sha256,
        )
    ]
    files.extend(validate_regular_file(staging / name) for name in METADATA_FILES)

    manifest = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "created_at_utc": utc_now(),
        "source": {
            "provider": "huggingface",
            "gated": True,
            "endpoint": args.endpoint,
            "repo_id": args.repo_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "independent_expected_weight_sha256": args.expected_weight_sha256,
        },
        "official_code": {
            "repository": OFFICIAL_CODE_REPO,
            "commit": OFFICIAL_CODE_COMMIT,
        },
        "license": {
            "name": "FAIR Noncommercial Research License v1",
            "file": "LICENSE.txt",
            "acceptance_is_managed_out_of_band": True,
        },
        "files": files,
        "downloader": {
            "script": "scripts/fetch_vggt_omega.py",
            "python": sys.version.split()[0],
            "huggingface_hub": package_version("huggingface-hub"),
        },
    }
    manifest_in_staging = staging / MANIFEST_NAME
    write_json(manifest_in_staging, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    # A manifest is a commit marker.  Remove an old marker before replacing
    # payloads, then publish the new marker only after every payload is ready.
    (output_dir / MANIFEST_NAME).unlink(missing_ok=True)
    for filename in expected_files(args.weight_filename):
        atomic_publish(staging / filename, output_dir / filename)
    atomic_publish(manifest_in_staging, output_dir / MANIFEST_NAME)
    # Only downloader-owned cache metadata remains after every selected file was
    # atomically moved. Keep the directory on failures to resume partial data.
    shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned gated VGGT-Omega-1B-512 checkpoint to NAS and "
            "publish a SHA-256 manifest."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=(
            "Hugging Face HTTPS origin. Defaults to the official service even when "
            "the machine-wide HF_ENDPOINT points at a mirror."
        ),
    )
    parser.add_argument("--weight-filename", default=DEFAULT_WEIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the gated Hugging Face token.",
    )
    parser.add_argument(
        "--expected-weight-sha256",
        default=DEFAULT_WEIGHT_SHA256,
        help=(
            "Independently recorded immutable weight SHA-256 required before "
            f"publication (default: {DEFAULT_WEIGHT_SHA256})."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Bypass any Hugging Face cache after taking the process lock.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify local files against the manifest without network access or a token.",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print immutable source and target information without downloading.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_filename(args.weight_filename)
        args.endpoint = normalize_endpoint(args.endpoint)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        print(
            "ERROR: --revision must be a full 40-character commit SHA", file=sys.stderr
        )
        return 2
    if args.expected_weight_sha256:
        args.expected_weight_sha256 = args.expected_weight_sha256.lower()
        if not SHA256_RE.fullmatch(args.expected_weight_sha256):
            print(
                "ERROR: --expected-weight-sha256 must contain 64 hex digits",
                file=sys.stderr,
            )
            return 2

    print(f"model={MODEL_NAME}")
    print(f"repo_id={args.repo_id}")
    print(f"revision={args.revision}")
    print(f"endpoint={args.endpoint}")
    print(f"expected_weight_sha256={args.expected_weight_sha256}")
    print(f"official_code_commit={OFFICIAL_CODE_COMMIT}")
    print(f"output_dir={args.output_dir.resolve()}")
    if args.show_plan:
        return 0

    if args.verify_only:
        valid, errors = verify_install(
            args.output_dir.resolve(),
            repo_id=args.repo_id,
            revision=args.revision,
            endpoint=args.endpoint,
            weight_filename=args.weight_filename,
            expected_weight_sha256=args.expected_weight_sha256,
        )
        if valid:
            print("verification=ok")
            return 0
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    lock_path = args.output_dir.resolve().parent / f".{args.output_dir.name}.fetch.lock"
    with exclusive_lock(lock_path):
        valid, _ = verify_install(
            args.output_dir.resolve(),
            repo_id=args.repo_id,
            revision=args.revision,
            endpoint=args.endpoint,
            weight_filename=args.weight_filename,
            expected_weight_sha256=args.expected_weight_sha256,
        )
        if valid and not args.force_download:
            print("status=already_verified")
            return 0
        token = os.environ.get(args.token_env)
        if not token:
            print(
                f"ERROR: set {args.token_env} after access to {args.repo_id} is approved; "
                "do not pass or store the token in command-line arguments",
                file=sys.stderr,
            )
            return 2
        try:
            download_and_publish(args, token)
        except Exception as exc:  # noqa: BLE001
            # Deliberately omit exception text: an HTTP client's diagnostic may
            # contain request details.  No credential is ever written or echoed.
            print(
                f"ERROR: {type(exc).__name__} while fetching or validating the "
                "checkpoint; credentials and request details were suppressed",
                file=sys.stderr,
            )
            return 1

    valid, errors = verify_install(
        args.output_dir.resolve(),
        repo_id=args.repo_id,
        revision=args.revision,
        endpoint=args.endpoint,
        weight_filename=args.weight_filename,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    if not valid:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("status=downloaded_and_verified")
    print(f"manifest={args.output_dir.resolve() / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
