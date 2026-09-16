#!/usr/bin/env python3
"""Extract navigation-LAM posterior means for NavAnywhere LatentPT.

The output is one strict ``OfflineProxyStore``-compatible file per trajectory:
``{output_root}/{source_id}/{trajectory_id}.pt``.  Every existing
current-to-goal pair with ``abs(frame_offset) <= max_abs_frame_offset`` is
stored using the numeric frame IDs parsed from the NavAnywhere JPEG names.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from navanywhere_recipe import (
    canonical_json,
    frame_indices_sha256,
    load_sampling_recipe,
    scan_trajectory_frames,
)


SCHEMA_VERSION = 1
FORMAT_NAME = "navanywhere_navigation_lam_latent_proxy"
EXPECTED_CHECKPOINT_CLASS = "lam.navigation_variants.PixelActionLAM"
DINO_CHECKPOINT_CLASS = "lam.navigation_variants.DINOFeatureLAM"
SUPPORTED_CHECKPOINT_CLASSES = frozenset(
    {
        EXPECTED_CHECKPOINT_CLASS,
        DINO_CHECKPOINT_CLASS,
        "lam.navigation_variants.PixelLAM",
    }
)
EXPECTED_LATENT_DIM = 32
EXPECTED_PATCH_SIZE = 16
IMAGE_PREPROCESSING = "center_crop_4:3_then_bilinear_resize"
IMAGE_READ_ATTEMPTS = 3
IMAGE_READ_RETRY_DELAY_SECONDS = 0.25
GPU_FRAME_BANK_LIMIT_BYTES = 16 * 1024**3
PAIR_BANK_CHUNK_SIZE = 8192
DINO_FRAME_BATCH_SIZE = 32
DINO_LAM_PAIR_BATCH_SIZE = 16


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def log(rank: int, message: str) -> None:
    print(f"[{utc_now()}][rank {rank}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o640)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o640)
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"cache payload is not a mapping: {path}")
    return value


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sampling-recipe", required=True)
    parser.add_argument(
        "--training-pair-plan",
        help=(
            "Completed plan-only JSON whose packed bitmap restricts extraction "
            "to unique pairs requested by the target training run."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--reuse-root",
        action="append",
        default=None,
        help=(
            "Compatible latent-action cache used as a row-level source. "
            "Repeat to reuse several earlier recipes; only exact frame-pair "
            "matches are copied and all missing rows are inferred normally."
        ),
    )
    parser.add_argument(
        "--lam-project-root",
        default=str(repo.parent / "DreamDojo" / "external" / "lam_project"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        help="Precomputed lowercase SHA-256; when omitted rank 0 hashes the checkpoint once.",
    )
    parser.add_argument(
        "--precision", choices=("32", "16-mixed", "bf16-mixed"), default="bf16-mixed"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--loader-threads", type=int, default=8)
    parser.add_argument(
        "--dino-frame-batch-size",
        type=int,
        default=DINO_FRAME_BATCH_SIZE,
        help=(
            "Unique RGB frames per frozen DINO forward. DINO checkpoints "
            "automatically extract each referenced trajectory frame once."
        ),
    )
    parser.add_argument(
        "--dino-lam-batch-size",
        type=int,
        default=DINO_LAM_PAIR_BATCH_SIZE,
        help="Cached DINO feature pairs per navigation-LAM encoder forward.",
    )
    parser.add_argument(
        "--stream-frames",
        action="store_true",
        help=(
            "Decode only the frames needed by each pair chunk and overlap the "
            "next chunk's image reads with GPU inference. This is intended for "
            "long trajectories that cannot keep a full resized frame bank."
        ),
    )
    parser.add_argument(
        "--stream-pair-chunk-size",
        type=int,
        default=4096,
        help="Number of frame pairs per streamed read/inference chunk.",
    )
    parser.add_argument(
        "--allow-future-training-plan",
        action="store_true",
        help=(
            "Accept a cryptographically self-consistent pair plan whose "
            "reference_validation status is not_requested. Use this only when "
            "the plan defines a future training run rather than replaying an "
            "already completed reference run."
        ),
    )
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--max-abs-frame-offset", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument(
        "--trajectory",
        action="append",
        default=None,
        help="Restrict to source_id/trajectory_id (repeatable; use a separate output root).",
    )
    parser.add_argument("--log-every-trajectories", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=72))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for navigation-LAM latent extraction")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def safe_path(root: Path, source_id: str, trajectory_id: str, suffix: str) -> Path:
    resolved_root = root.resolve()
    source_root = (resolved_root / source_id).resolve()
    path = (source_root / f"{trajectory_id}{suffix}").resolve()
    if (
        source_root == resolved_root
        or resolved_root not in source_root.parents
        or path == source_root
        or source_root not in path.parents
    ):
        raise ValueError(f"Unsafe identity {source_id!r}/{trajectory_id!r}")
    return path


def build_navanywhere_frame_pairs(
    frame_indices: Sequence[int] | np.ndarray,
    *,
    context_size: int = 4,
    max_abs_frame_offset: int = 8,
) -> np.ndarray:
    """Enumerate every real-frame-ID pair strict LatentPT can request."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("frame_indices must be one-dimensional")
    if context_size < 1 or max_abs_frame_offset < 0:
        raise ValueError("context_size must be positive and offset non-negative")
    if len(indices) < context_size:
        return np.empty((0, 2), dtype=np.int64)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    available = set(map(int, indices.tolist()))
    rows = [
        (int(current), int(target))
        for current in indices[context_size - 1 :]
        for target in range(
            int(current) - max_abs_frame_offset,
            int(current) + max_abs_frame_offset + 1,
        )
        if target in available
    ]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def position_pairs(frame_indices: np.ndarray, frame_pairs: np.ndarray) -> np.ndarray:
    """Map numeric frame IDs to positions used for tensor indexing."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    pairs = np.asarray(frame_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise ValueError("frame_pairs must have shape [N,2]")
    positions = np.searchsorted(indices, pairs)
    if np.any(positions >= len(indices)):
        raise ValueError("frame pair refers to an absent frame")
    if not np.array_equal(indices[positions], pairs):
        raise ValueError("frame pair refers to an absent frame")
    return positions.astype(np.int64, copy=False)


def _lam_api(project_root: str | Path) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    root = Path(project_root).expanduser().resolve()
    if not (root / "lam").is_dir():
        raise FileNotFoundError(f"navigation LAM project is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from lam.navigation_evaluation.adapters import (  # type: ignore[import-not-found]
        inspect_evaluation_checkpoint,
        load_evaluation_adapter,
    )
    from lam.navigation_evaluation.catalog import (  # type: ignore[import-not-found]
        load_navigation_frame,
    )

    return inspect_evaluation_checkpoint, load_evaluation_adapter, load_navigation_frame


def validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    image_height: int,
    image_width: int,
    max_abs_frame_offset: int,
) -> None:
    checkpoint_class = metadata.get("class_path")
    if checkpoint_class not in SUPPORTED_CHECKPOINT_CLASSES:
        raise ValueError(
            f"checkpoint class={checkpoint_class!r}, expected one of "
            f"{sorted(SUPPORTED_CHECKPOINT_CLASSES)!r}"
        )
    hparams = metadata.get("hparams")
    data_hparams = metadata.get("datamodule_hparams")
    if not isinstance(hparams, Mapping) or not isinstance(data_hparams, Mapping):
        raise TypeError("checkpoint is missing model or datamodule hyperparameters")
    for key, expected in (
        ("lam_latent_dim", EXPECTED_LATENT_DIM),
        ("lam_patch_size", EXPECTED_PATCH_SIZE),
    ):
        if hparams.get(key) != expected:
            raise ValueError(f"checkpoint {key}={hparams.get(key)!r}, expected {expected}")
    for key, expected in (
        ("image_height", image_height),
        ("image_width", image_width),
        ("max_frame_offset", max_abs_frame_offset),
    ):
        if data_hparams.get(key) != expected:
            raise ValueError(
                f"checkpoint data contract {key}={data_hparams.get(key)!r}, expected {expected}"
            )
    for key in ("global_step", "epoch"):
        if not isinstance(metadata.get(key), int):
            raise TypeError(f"checkpoint {key} is missing or malformed")


def _checkpoint_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"navigation LAM checkpoint is missing: {checkpoint}")
    inspect_checkpoint, _, _ = _lam_api(args.lam_project_root)
    metadata = inspect_checkpoint(checkpoint)
    validate_checkpoint_metadata(
        metadata,
        image_height=int(args.image_height),
        image_width=int(args.image_width),
        max_abs_frame_offset=int(args.max_abs_frame_offset),
    )
    digest = args.checkpoint_sha256
    if digest is None:
        digest = sha256_file(checkpoint)
    digest = str(digest).lower()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("--checkpoint-sha256 must be a lowercase SHA-256 digest")
    return {
        "path": str(checkpoint),
        "sha256": digest,
        "size_bytes": checkpoint.stat().st_size,
        "class_path": metadata["class_path"],
        "global_step": int(metadata["global_step"]),
        "epoch": int(metadata["epoch"]),
        "latent_dim": int(metadata["hparams"]["lam_latent_dim"]),
        "patch_size": int(metadata["hparams"]["lam_patch_size"]),
        "data_contract": {
            "image_height": int(metadata["datamodule_hparams"]["image_height"]),
            "image_width": int(metadata["datamodule_hparams"]["image_width"]),
            "max_frame_offset": int(
                metadata["datamodule_hparams"]["max_frame_offset"]
            ),
        },
    }


def _full_pair_policy_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pair_domain": "NavAnywhere_observations_and_existing_local_targets_v1",
        "context_size": int(args.context_size),
        "max_abs_frame_offset": int(args.max_abs_frame_offset),
        "pair_direction": "current_to_goal",
        "latent_value": "deterministic_posterior_mean_z_mu",
        "normalization": "raw",
    }


def _policy(
    args: argparse.Namespace, training_plan: Mapping[str, Any] | None
) -> dict[str, Any]:
    if training_plan is None:
        configuration = _full_pair_policy_configuration(args)
    else:
        configuration = {
            "pair_domain": "NavAnywhere_training_plan_unique_local_pairs_v1",
            "context_size": int(args.context_size),
            "max_abs_frame_offset": int(args.max_abs_frame_offset),
            "pair_direction": "current_to_goal",
            "latent_value": "deterministic_posterior_mean_z_mu",
            "normalization": "raw",
            "training_plan_sha256": training_plan["plan_sha256"],
            "pair_bitmap_sha256": training_plan["pair_bitmap_sha256"],
            "full_pair_cache_is_valid_superset": True,
        }
    return {"configuration": configuration, "fingerprint": fingerprint(configuration)}


def _extraction(
    args: argparse.Namespace, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    is_dino = checkpoint.get("class_path") == DINO_CHECKPOINT_CLASS
    if is_dino:
        frame_bank_strategy = "trajectory_unique_dino_features_with_read_ahead_v1"
    elif args.stream_frames:
        frame_bank_strategy = "stream_unique_frames_with_one_chunk_read_ahead_v1"
    else:
        frame_bank_strategy = "full_gpu_up_to_16gib_else_compact_8192_pair_chunks_v1"
    configuration = {
        "precision": str(args.precision),
        "batch_size": int(args.batch_size),
        "fused_attention": True,
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "image_preprocessing": IMAGE_PREPROCESSING,
        "image_value_range": [0.0, 1.0],
        "output_dtype": "float32",
        "frame_bank_strategy": frame_bank_strategy,
    }
    if is_dino:
        configuration.update(
            {
                "dino_feature_strategy": "each_referenced_trajectory_frame_once_v1",
                "dino_feature_storage": "per_trajectory_cpu_mixed_precision",
                "dino_frame_batch_size": int(args.dino_frame_batch_size),
                "dino_lam_pair_batch_size": int(args.dino_lam_batch_size),
                "loader_threads": int(args.loader_threads),
            }
        )
    elif args.stream_frames:
        configuration["stream_pair_chunk_size"] = int(args.stream_pair_chunk_size)
        configuration["loader_threads"] = int(args.loader_threads)
    return {"configuration": configuration, "fingerprint": fingerprint(configuration)}


_NUMERICAL_EXTRACTION_FIELDS = (
    "precision",
    "fused_attention",
    "image_height",
    "image_width",
    "image_preprocessing",
    "image_value_range",
    "output_dtype",
    "dino_feature_strategy",
)


def _reuse_descriptors(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    extraction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate cache-level numerical compatibility for optional row reuse."""

    descriptors = []
    target_configuration = extraction["configuration"]
    roots = list(getattr(args, "reuse_root", None) or ())
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Latent-action reuse root does not exist: {root}")
        metadata_path = root / "metadata.json"
        state_path = root / "extraction_state.json"
        descriptor_path = metadata_path if metadata_path.is_file() else state_path
        if not descriptor_path.is_file():
            raise FileNotFoundError(
                f"Reuse root has neither metadata.json nor extraction_state.json: {root}"
            )
        document = json.loads(descriptor_path.read_text(encoding="utf-8"))
        source_checkpoint = document.get("checkpoint")
        source_extraction = document.get("extraction")
        if not isinstance(source_checkpoint, Mapping) or not isinstance(
            source_extraction, Mapping
        ):
            raise TypeError(f"Reuse descriptor is incomplete: {descriptor_path}")
        if source_checkpoint.get("sha256") != checkpoint.get("sha256"):
            raise ValueError(f"Reuse checkpoint differs from target: {root}")
        source_configuration = source_extraction.get("configuration")
        if not isinstance(source_configuration, Mapping):
            raise TypeError(f"Reuse extraction configuration is invalid: {root}")
        mismatches = [
            field
            for field in _NUMERICAL_EXTRACTION_FIELDS
            if source_configuration.get(field) != target_configuration.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"Reuse extraction is numerically incompatible for {root}: {mismatches}"
            )
        source_fingerprint = source_extraction.get("fingerprint")
        if not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64:
            raise ValueError(f"Reuse extraction fingerprint is invalid: {root}")
        descriptors.append(
            {
                "root": str(root),
                "descriptor_path": str(descriptor_path),
                "checkpoint_sha256": source_checkpoint["sha256"],
                "extraction_fingerprint": source_fingerprint,
                "numerical_configuration": {
                    field: source_configuration[field]
                    for field in _NUMERICAL_EXTRACTION_FIELDS
                },
            }
        )
    if len({item["root"] for item in descriptors}) != len(descriptors):
        raise ValueError("--reuse-root contains duplicates")
    return descriptors


def _strictly_sorted_pairs(pairs: np.ndarray) -> bool:
    if len(pairs) < 2:
        return True
    current_increases = pairs[1:, 0] > pairs[:-1, 0]
    same_current_target_increases = (
        (pairs[1:, 0] == pairs[:-1, 0])
        & (pairs[1:, 1] > pairs[:-1, 1])
    )
    return bool(np.all(current_increases | same_current_target_increases))


def _matching_pair_rows(
    requested_pairs: np.ndarray, available_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned requested/available row numbers for sorted unique pairs."""

    requested = np.ascontiguousarray(requested_pairs, dtype="<i8")
    available = np.ascontiguousarray(available_pairs, dtype="<i8")
    if requested.ndim != 2 or requested.shape[1:] != (2,):
        raise ValueError("Requested reuse pairs must have shape [N,2]")
    if available.ndim != 2 or available.shape[1:] != (2,):
        raise ValueError("Available reuse pairs must have shape [N,2]")
    if not _strictly_sorted_pairs(requested) or not _strictly_sorted_pairs(available):
        raise ValueError("Latent-action reuse pairs must be sorted and unique")
    pair_dtype = np.dtype([("current", "<i8"), ("target", "<i8")])
    requested_keys = requested.view(pair_dtype).reshape(-1)
    available_keys = available.view(pair_dtype).reshape(-1)
    positions = np.searchsorted(available_keys, requested_keys)
    valid = positions < len(available_keys)
    valid[valid] &= available_keys[positions[valid]] == requested_keys[valid]
    requested_rows = np.flatnonzero(valid).astype(np.int64, copy=False)
    return requested_rows, positions[valid].astype(np.int64, copy=False)


def _reuse_task_motion(
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    indices: np.ndarray,
    source_fingerprint: str,
    frame_pairs: np.ndarray,
) -> dict[str, Any]:
    """Fill exact target rows from compatible prior caches, without fallback."""

    latent_dim = int(state["checkpoint"]["latent_dim"])
    motion = torch.empty((len(frame_pairs), latent_dim), dtype=torch.float32)
    filled = np.zeros(len(frame_pairs), dtype=np.bool_)
    provenance: list[dict[str, Any]] = []
    for descriptor in state.get("reuse_roots", ()):
        path = safe_path(
            Path(descriptor["root"]),
            str(task["source_id"]),
            str(task["trajectory_id"]),
            ".pt",
        )
        if not path.is_file() or bool(np.all(filled)):
            continue
        try:
            payload = safe_torch_load(path)
            if not (
                payload.get("schema_version") == SCHEMA_VERSION
                and payload.get("format") == FORMAT_NAME
                and payload.get("proxy_type") == "latent"
                and payload.get("source_id") == task["source_id"]
                and payload.get("trajectory_id") == task["trajectory_id"]
                and payload.get("complete") is True
            ):
                raise ValueError("cache identity or format mismatch")
            metadata = payload.get("metadata")
            if not isinstance(metadata, Mapping):
                raise TypeError("cache metadata is not a mapping")
            required_metadata = {
                "frame_indices_sha256": task["frame_indices_sha256"],
                "source_fingerprint": source_fingerprint,
                "checkpoint_sha256": state["checkpoint"]["sha256"],
                "extraction_fingerprint": descriptor["extraction_fingerprint"],
            }
            for key, expected in required_metadata.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"cache metadata.{key} mismatch")
            stored_indices = torch.as_tensor(payload["frame_indices"])
            if stored_indices.dtype != torch.int64 or not torch.equal(
                stored_indices, torch.from_numpy(indices.copy())
            ):
                raise ValueError("cache frame inventory mismatch")
            available_pairs_tensor = torch.as_tensor(payload["frame_pairs"])
            available_motion = torch.as_tensor(payload["motion"])
            if available_pairs_tensor.dtype != torch.int64:
                raise TypeError("cache frame-pair dtype is invalid")
            if available_motion.dtype != torch.float32 or tuple(
                available_motion.shape
            ) != (len(available_pairs_tensor), latent_dim):
                raise ValueError("cache motion shape or dtype is invalid")
            requested_rows, available_rows = _matching_pair_rows(
                frame_pairs, available_pairs_tensor.numpy()
            )
            if len(requested_rows):
                keep = ~filled[requested_rows]
                requested_rows = requested_rows[keep]
                available_rows = available_rows[keep]
            if not len(requested_rows):
                continue
            reused = available_motion.index_select(
                0, torch.from_numpy(available_rows)
            ).contiguous()
            if not torch.isfinite(reused).all():
                raise ValueError("selected cache motion contains non-finite values")
            motion[torch.from_numpy(requested_rows)] = reused
            filled[requested_rows] = True
            provenance.append(
                {
                    "root": descriptor["root"],
                    "path": str(path),
                    "rows": int(len(requested_rows)),
                }
            )
        except Exception as exc:
            provenance.append(
                {
                    "root": descriptor["root"],
                    "path": str(path),
                    "rows": 0,
                    "ignored_error": f"{type(exc).__name__}: {exc}",
                }
            )
    missing_rows = np.flatnonzero(~filled).astype(np.int64, copy=False)
    return {
        "motion": motion,
        "missing_rows": missing_rows,
        "reused_rows": int(np.count_nonzero(filled)),
        "provenance": provenance,
    }


def _bitmap_bits(
    packed: np.ndarray, *, start_bit: int, bit_count: int
) -> np.ndarray:
    if start_bit < 0 or bit_count < 0:
        raise ValueError("Pair bitmap range must be non-negative")
    if bit_count == 0:
        return np.empty(0, dtype=np.bool_)
    end_bit = start_bit + bit_count
    byte_start = start_bit // 8
    byte_end = (end_bit + 7) // 8
    if byte_end > len(packed):
        raise ValueError("Pair bitmap is shorter than its declared key domain")
    unpacked = np.unpackbits(
        np.asarray(packed[byte_start:byte_end], dtype=np.uint8), bitorder="little"
    )
    offset = start_bit - byte_start * 8
    return unpacked[offset : offset + bit_count].astype(np.bool_, copy=False)


def _load_training_pair_plan(
    path: str | os.PathLike[str] | None,
    *,
    recipe_sha256: str,
    max_abs_frame_offset: int,
    allow_future_training_plan: bool = False,
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    if path is None:
        return None, None
    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Training pair plan does not exist: {plan_path}")
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise TypeError("Training pair plan must be a JSON object")
    if not (
        report.get("format") == "navanywhere_latent_action_training_pair_plan"
        and report.get("complete") is True
    ):
        raise ValueError("Training pair plan is incomplete or has the wrong format")
    reference_status = report.get("reference_validation", {}).get("status")
    allowed_reference_statuses = {"matched"}
    if allow_future_training_plan:
        allowed_reference_statuses.add("not_requested")
    if reference_status not in allowed_reference_statuses:
        raise ValueError(
            "Training pair plan reference validation status is not allowed: "
            f"{reference_status!r}; allowed={sorted(allowed_reference_statuses)}"
        )
    if report.get("sampling_recipe", {}).get("sha256") != recipe_sha256:
        raise ValueError("Training pair plan sampling recipe SHA-256 mismatch")
    contract = report.get("training_contract")
    if not isinstance(contract, Mapping) or int(
        contract.get("max_abs_frame_offset", -1)
    ) != int(max_abs_frame_offset):
        raise ValueError("Training pair plan local-offset contract mismatch")
    identity = report.get("pair_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Training pair plan has no pair identity")
    expected_plan_sha = fingerprint(identity)
    if report.get("plan_sha256") != expected_plan_sha:
        raise ValueError("Training pair plan identity SHA-256 mismatch")
    bitmap_record = report.get("pair_bitmap")
    if not isinstance(bitmap_record, Mapping):
        raise ValueError("Training pair plan has no packed pair bitmap")
    if not (
        bitmap_record.get("encoding") == "numpy.packbits"
        and bitmap_record.get("bitorder") == "little"
    ):
        raise ValueError("Unsupported training pair bitmap encoding")
    bitmap_path = Path(str(bitmap_record.get("path", ""))).expanduser()
    if not bitmap_path.is_absolute():
        bitmap_path = plan_path.parent / bitmap_path
    bitmap_path = bitmap_path.resolve()
    if not bitmap_path.is_file():
        raise FileNotFoundError(f"Training pair bitmap does not exist: {bitmap_path}")
    bitmap_sha = sha256_file(bitmap_path)
    if not (
        bitmap_sha == bitmap_record.get("sha256")
        and bitmap_sha == identity.get("pair_bitmap_sha256")
    ):
        raise ValueError("Training pair bitmap SHA-256 mismatch")
    pair_key_count = int(identity.get("pair_key_count", -1))
    packed_byte_count = (pair_key_count + 7) // 8
    if pair_key_count < 1 or bitmap_path.stat().st_size != packed_byte_count:
        raise ValueError("Training pair bitmap size does not match its key domain")
    bitmap = np.memmap(bitmap_path, dtype=np.uint8, mode="r")
    selected_count = int(
        np.count_nonzero(_bitmap_bits(bitmap, start_bit=0, bit_count=pair_key_count))
    )
    if selected_count != int(report.get("totals", {}).get("unique_local_pairs", -1)):
        raise ValueError("Training pair bitmap population count mismatch")
    descriptor = {
        "path": str(plan_path),
        "plan_sha256": expected_plan_sha,
        "pair_bitmap_path": str(bitmap_path),
        "pair_bitmap_sha256": bitmap_sha,
        "pair_key_count": pair_key_count,
        "unique_local_pairs": selected_count,
        "training_contract": dict(contract),
    }
    return descriptor, bitmap


def build_planned_navanywhere_frame_pairs(
    frame_indices: Sequence[int] | np.ndarray,
    *,
    observation_base: int,
    pair_bitmap: np.ndarray,
    context_size: int = 4,
    max_abs_frame_offset: int = 8,
) -> np.ndarray:
    """Decode one trajectory's exact training-plan pair subset."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    observation_count = max(0, len(indices) - int(context_size) + 1)
    pair_width = 2 * int(max_abs_frame_offset) + 1
    bits = _bitmap_bits(
        pair_bitmap,
        start_bit=int(observation_base) * pair_width,
        bit_count=observation_count * pair_width,
    ).reshape(observation_count, pair_width)
    observation_slots, offset_slots = np.nonzero(bits)
    current = indices[int(context_size) - 1 + observation_slots]
    target = current + offset_slots.astype(np.int64) - int(max_abs_frame_offset)
    if np.any(np.searchsorted(indices, target) >= len(indices)):
        raise ValueError("Training pair plan targets a frame outside the trajectory")
    positions = np.searchsorted(indices, target)
    if not np.array_equal(indices[positions], target):
        raise ValueError("Training pair plan targets an absent frame")
    return np.stack((current, target), axis=1).astype(np.int64, copy=False)


def _assign(tasks: list[dict[str, Any]], world_size: int, args: argparse.Namespace) -> None:
    loads = [0] * world_size
    for task in sorted(
        tasks,
        key=lambda item: (-int(item["frame_count"]), item["source_id"], item["trajectory_id"]),
    ):
        # Planned counts provide a tighter load balance; full extraction falls
        # back to the nearly 17 * frame_count local-pair domain.
        cost = int(task.get("planned_pair_count", 0)) or (
            max(1, int(task["frame_count"]) - int(args.context_size) + 1)
            * (2 * int(args.max_abs_frame_offset) + 1)
        )
        target = min(range(world_size), key=lambda value: (loads[value], value))
        task["assigned_rank"] = target
        loads[target] += cost


def _build_state(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"NavAnywhere root does not exist: {data_root}")
    recipe, recipe_path, recipe_sha = load_sampling_recipe(
        args.sampling_recipe, context_size=int(args.context_size)
    )
    training_plan, pair_bitmap = _load_training_pair_plan(
        args.training_pair_plan,
        recipe_sha256=recipe_sha,
        max_abs_frame_offset=int(args.max_abs_frame_offset),
        allow_future_training_plan=bool(args.allow_future_training_plan),
    )
    checkpoint = _checkpoint_descriptor(args)
    extraction = _extraction(args, checkpoint)
    reuse_roots = _reuse_descriptors(args, checkpoint, extraction)
    tasks = [dict(item) for item in recipe["trajectories"]]
    observation_base = 0
    pair_width = 2 * int(args.max_abs_frame_offset) + 1
    for task in tasks:
        task["observation_base"] = observation_base
        observation_count = int(task["observation_count"])
        if pair_bitmap is not None:
            task["planned_pair_count"] = int(
                np.count_nonzero(
                    _bitmap_bits(
                        pair_bitmap,
                        start_bit=observation_base * pair_width,
                        bit_count=observation_count * pair_width,
                    )
                )
            )
        observation_base += observation_count
    if training_plan is not None and (
        observation_base * pair_width != int(training_plan["pair_key_count"])
    ):
        raise ValueError("Training pair plan key domain does not match the recipe")
    if args.trajectory:
        requested = set(map(str, args.trajectory))
        tasks = [
            item
            for item in tasks
            if f"{item['source_id']}/{item['trajectory_id']}" in requested
        ]
        found = {f"{item['source_id']}/{item['trajectory_id']}" for item in tasks}
        if found != requested:
            raise ValueError(f"Unknown requested trajectories: {sorted(requested - found)}")
    if args.max_trajectories:
        tasks = tasks[: int(args.max_trajectories)]
    _assign(tasks, world_size, args)
    return {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "lam_project_root": str(Path(args.lam_project_root).expanduser().resolve()),
        "sampling_recipe_path": recipe_path,
        "sampling_recipe_sha256": recipe_sha,
        "recipe": recipe,
        "checkpoint": checkpoint,
        "policy": _policy(args, training_plan),
        "legacy_full_policy_fingerprint": fingerprint(
            _full_pair_policy_configuration(args)
        ),
        "extraction": extraction,
        "reuse_roots": reuse_roots,
        "training_pair_plan": training_plan,
        "tasks": tasks,
        "world_size": world_size,
        "selection_partial": bool(args.trajectory or args.max_trajectories),
    }


def _scan_task(
    data_root: Path, task: Mapping[str, Any]
) -> tuple[list[tuple[int, str]], np.ndarray, str]:
    trajectory_root = safe_path(
        data_root, str(task["source_id"]), str(task["trajectory_id"]), ""
    )
    frames = scan_trajectory_frames(trajectory_root)
    indices = np.asarray([item[0] for item in frames], dtype=np.int64)
    if len(indices) != int(task["frame_count"]):
        raise ValueError("trajectory frame count changed after recipe creation")
    if frame_indices_sha256(indices) != task["frame_indices_sha256"]:
        raise ValueError("trajectory frame indices changed after recipe creation")
    source_records = []
    for frame_index, frame_path in frames:
        path = Path(frame_path)
        stat = path.stat()
        source_records.append(
            [int(frame_index), path.name, int(stat.st_size), int(stat.st_mtime_ns)]
        )
    source_fingerprint = hashlib.sha256(canonical_json(source_records)).hexdigest()
    return frames, indices, source_fingerprint


def _expected_metadata(
    state: Mapping[str, Any], task: Mapping[str, Any], source_fingerprint: str
) -> dict[str, Any]:
    return {
        "sampling_recipe_sha256": state["sampling_recipe_sha256"],
        "frame_indices_sha256": task["frame_indices_sha256"],
        "source_fingerprint": source_fingerprint,
        "checkpoint_sha256": state["checkpoint"]["sha256"],
        "extraction_fingerprint": state["extraction"]["fingerprint"],
        "policy_fingerprint": state["policy"]["fingerprint"],
    }


def _task_frame_pairs(
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    indices: np.ndarray,
    pair_bitmap: np.ndarray | None,
) -> np.ndarray:
    policy = state["policy"]["configuration"]
    context_size = int(policy["context_size"])
    local_limit = int(policy["max_abs_frame_offset"])
    if state.get("training_pair_plan") is None:
        return build_navanywhere_frame_pairs(
            indices,
            context_size=context_size,
            max_abs_frame_offset=local_limit,
        )
    if pair_bitmap is None:
        raise RuntimeError("Training pair bitmap was not loaded on this rank")
    return build_planned_navanywhere_frame_pairs(
        indices,
        observation_base=int(task["observation_base"]),
        pair_bitmap=pair_bitmap,
        context_size=context_size,
        max_abs_frame_offset=local_limit,
    )


def _cache_record(
    path: Path,
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    indices: np.ndarray,
    source_fingerprint: str,
    pair_bitmap: np.ndarray | None,
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = safe_torch_load(path)
    required = {
        "schema_version",
        "format",
        "proxy_type",
        "source_id",
        "trajectory_id",
        "frame_indices",
        "frame_pairs",
        "motion",
        "metadata",
        "complete",
    }
    if required - set(payload):
        raise KeyError(f"cache misses keys {sorted(required - set(payload))}")
    if not (
        payload["schema_version"] == SCHEMA_VERSION
        and payload["format"] == FORMAT_NAME
        and payload["proxy_type"] == "latent"
        and payload["source_id"] == task["source_id"]
        and payload["trajectory_id"] == task["trajectory_id"]
        and payload["complete"] is True
    ):
        raise ValueError("cache identity or format mismatch")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("cache metadata is not a mapping")
    expected_metadata = _expected_metadata(state, task, source_fingerprint)
    for key, expected in expected_metadata.items():
        if key == "policy_fingerprint":
            continue
        if metadata.get(key) != expected:
            raise ValueError(f"cache metadata.{key} mismatch")
    stored_indices = torch.as_tensor(payload["frame_indices"])
    expected_indices = torch.from_numpy(indices.copy())
    if stored_indices.dtype != torch.int64 or not torch.equal(
        stored_indices, expected_indices
    ):
        raise ValueError("cache frame inventory differs from the sampling recipe")
    expected_pair_array = _task_frame_pairs(state, task, indices, pair_bitmap)
    expected_pairs = torch.from_numpy(expected_pair_array)
    pairs = torch.as_tensor(payload["frame_pairs"])
    motion = torch.as_tensor(payload["motion"])
    latent_dim = int(state["checkpoint"]["latent_dim"])
    if pairs.dtype != torch.int64:
        raise ValueError("cache frame-pair dtype is invalid")
    planned_domain = torch.equal(pairs, expected_pairs)
    full_superset = False
    if state.get("training_pair_plan") is not None:
        full_pairs = torch.from_numpy(
            build_navanywhere_frame_pairs(
                indices,
                context_size=int(state["policy"]["configuration"]["context_size"]),
                max_abs_frame_offset=int(
                    state["policy"]["configuration"]["max_abs_frame_offset"]
                ),
            )
        )
        full_superset = torch.equal(pairs, full_pairs)
    policy_fingerprint = metadata.get("policy_fingerprint")
    if planned_domain and policy_fingerprint == expected_metadata["policy_fingerprint"]:
        pair_coverage = "training_plan"
    elif (
        full_superset
        and policy_fingerprint == state["legacy_full_policy_fingerprint"]
    ):
        pair_coverage = "full_domain_superset"
    else:
        raise ValueError("cache frame-pair domain or policy is incompatible")
    if motion.dtype != torch.float32 or tuple(motion.shape) != (len(pairs), latent_dim):
        raise ValueError("cache motion shape or dtype mismatch")
    if not torch.isfinite(motion).all():
        raise ValueError("cache motion contains non-finite values")
    return {
        "source_id": task["source_id"],
        "trajectory_id": task["trajectory_id"],
        "frame_count": len(indices),
        "pair_count": len(pairs),
        "planned_pair_count": len(expected_pairs),
        "pair_coverage": pair_coverage,
        "file_size_bytes": path.stat().st_size,
        "invalid_frame_substitutions": payload.get(
            "invalid_frame_substitutions", []
        ),
    }


def _load_frame_with_fallback(
    frames: list[tuple[int, str]],
    position: int,
    load_frame: Callable[..., torch.Tensor],
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    original_index, original_path = frames[position]
    original_error: Exception | None = None
    for attempt in range(IMAGE_READ_ATTEMPTS):
        try:
            return (
                load_frame(
                    original_path,
                    image_height=image_height,
                    image_width=image_width,
                ),
                None,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            original_error = exc
            if attempt + 1 < IMAGE_READ_ATTEMPTS:
                time.sleep(IMAGE_READ_RETRY_DELAY_SECONDS)
    candidates = sorted(
        (candidate for candidate in range(len(frames)) if candidate != position),
        key=lambda candidate: (abs(candidate - position), candidate > position),
    )
    for candidate in candidates:
        replacement_index, replacement_path = frames[candidate]
        try:
            tensor = load_frame(
                replacement_path,
                image_height=image_height,
                image_width=image_width,
            )
        except (OSError, RuntimeError, ValueError):
            continue
        return tensor, {
            "frame_index": int(original_index),
            "frame_path": Path(original_path).name,
            "replacement_frame_index": int(replacement_index),
            "replacement_frame_path": Path(replacement_path).name,
            "reason": f"{type(original_error).__name__}: {original_error}",
        }
    raise RuntimeError(
        f"No readable frame remains near {original_index} ({original_path})"
    ) from original_error


def _load_frames(
    frames: list[tuple[int, str]],
    load_frame: Callable[..., torch.Tensor],
    *,
    image_height: int,
    image_width: int,
    loader_threads: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return _load_frame_positions(
        frames,
        range(len(frames)),
        load_frame,
        image_height=image_height,
        image_width=image_width,
        loader_threads=loader_threads,
    )


def _load_frame_positions(
    frames: list[tuple[int, str]],
    positions: Sequence[int] | np.ndarray,
    load_frame: Callable[..., torch.Tensor],
    *,
    image_height: int,
    image_width: int,
    loader_threads: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Load selected positions while retaining trajectory-wide fallbacks."""

    selected_positions = [int(position) for position in positions]
    if not selected_positions:
        raise ValueError("At least one frame position is required")
    if min(selected_positions) < 0 or max(selected_positions) >= len(frames):
        raise IndexError("Selected frame position is outside the trajectory")

    def load(position: int) -> tuple[torch.Tensor, dict[str, Any] | None]:
        return _load_frame_with_fallback(
            frames,
            position,
            load_frame,
            image_height=image_height,
            image_width=image_width,
        )

    if loader_threads > 0:
        with ThreadPoolExecutor(max_workers=loader_threads) as executor:
            loaded = list(executor.map(load, selected_positions))
    else:
        loaded = [load(position) for position in selected_positions]
    substitutions = [item[1] for item in loaded if item[1] is not None]
    return torch.stack([item[0] for item in loaded]).contiguous(), substitutions


def _autocast(device: torch.device, precision: str) -> Any:
    if precision == "32":
        return nullcontext()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def encode_pair_batches(
    adapter: Any,
    frames: torch.Tensor,
    pairs: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    precision: str,
    gpu_frame_bank_limit_bytes: int = GPU_FRAME_BANK_LIMIT_BYTES,
    pair_chunk_size: int = PAIR_BANK_CHUNK_SIZE,
) -> torch.Tensor:
    """Encode position pairs and return raw deterministic ``z_mu`` in FP32."""

    pair_tensor = torch.as_tensor(pairs, dtype=torch.int64)
    latent_dim = int(adapter.checkpoint_metadata["hparams"]["lam_latent_dim"])
    if len(pair_tensor) == 0:
        return torch.empty((0, latent_dim), dtype=torch.float32)
    if batch_size < 1 or gpu_frame_bank_limit_bytes < 0 or pair_chunk_size < 1:
        raise ValueError("Invalid batch or frame-bank chunk configuration")
    outputs: list[torch.Tensor] = []

    def encode_bank(frame_bank: torch.Tensor, pair_bank: torch.Tensor) -> None:
        for start in range(0, len(pair_bank), batch_size):
            selection = pair_bank[start : start + batch_size]
            current = frame_bank.index_select(0, selection[:, 0])
            goal = frame_bank.index_select(0, selection[:, 1])
            videos = torch.stack((current, goal), dim=1)
            with _autocast(adapter.device, precision):
                z_mu = adapter.encode(videos)
            expected = (len(selection), 1, 1, latent_dim)
            if tuple(z_mu.shape) != expected:
                raise RuntimeError(
                    f"navigation LAM returned z_mu {tuple(z_mu.shape)}, "
                    f"expected {expected}"
                )
            outputs.append(z_mu[:, 0, 0].detach().float().cpu())

    frame_bank_bytes = frames.numel() * frames.element_size()
    if frame_bank_bytes <= gpu_frame_bank_limit_bytes:
        encode_bank(
            frames.to(device=adapter.device),
            pair_tensor.to(device=adapter.device),
        )
    else:
        # Walking Tours contains trajectories with up to 42k resized frames.
        # Preserve exact float preprocessing while moving only the unique frames
        # needed by a compact group of local pairs onto the inference GPU.
        for start in range(0, len(pair_tensor), pair_chunk_size):
            pair_chunk = pair_tensor[start : start + pair_chunk_size]
            positions, inverse = torch.unique(
                pair_chunk.reshape(-1), sorted=True, return_inverse=True
            )
            compact_bank = frames.index_select(0, positions).to(device=adapter.device)
            compact_pairs = inverse.reshape(-1, 2).to(device=adapter.device)
            encode_bank(compact_bank, compact_pairs)
    return torch.cat(outputs, dim=0).contiguous()


def encode_pair_batches_streaming(
    adapter: Any,
    frames: list[tuple[int, str]],
    position_pairs_array: np.ndarray,
    load_frame: Callable[..., torch.Tensor],
    *,
    batch_size: int,
    precision: str,
    image_height: int,
    image_width: int,
    loader_threads: int,
    pair_chunk_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Encode a long trajectory without materializing its full RGB frame bank.

    Pair chunks retain canonical row-major order. Each chunk decodes its unique
    source frames once, while a one-item read-ahead overlaps NAS/JPEG work for
    the next chunk with current GPU inference.
    """

    position_pairs_array = np.asarray(position_pairs_array, dtype=np.int64)
    if position_pairs_array.ndim != 2 or position_pairs_array.shape[1:] != (2,):
        raise ValueError("position_pairs_array must have shape [N,2]")
    if pair_chunk_size < 1:
        raise ValueError("stream pair chunk size must be positive")
    latent_dim = int(adapter.checkpoint_metadata["hparams"]["lam_latent_dim"])
    if len(position_pairs_array) == 0:
        return torch.empty((0, latent_dim), dtype=torch.float32), []

    chunks = [
        position_pairs_array[start : start + pair_chunk_size]
        for start in range(0, len(position_pairs_array), pair_chunk_size)
    ]

    def prepare(
        pair_chunk: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]]]:
        unique_positions, inverse = np.unique(pair_chunk, return_inverse=True)
        selected = [frames[int(position)] for position in unique_positions]
        frame_tensors, substitutions = _load_frames(
            selected,
            load_frame,
            image_height=image_height,
            image_width=image_width,
            loader_threads=loader_threads,
        )
        return frame_tensors, inverse.reshape(-1, 2), substitutions

    outputs: list[torch.Tensor] = []
    substitutions_by_frame: dict[int, dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=1) as read_ahead:
        future = read_ahead.submit(prepare, chunks[0])
        for chunk_index, pair_chunk in enumerate(chunks):
            frame_tensors, compact_pairs, substitutions = future.result()
            if chunk_index + 1 < len(chunks):
                future = read_ahead.submit(prepare, chunks[chunk_index + 1])
            outputs.append(
                encode_pair_batches(
                    adapter,
                    frame_tensors,
                    compact_pairs,
                    batch_size=batch_size,
                    precision=precision,
                    gpu_frame_bank_limit_bytes=GPU_FRAME_BANK_LIMIT_BYTES,
                )
            )
            del frame_tensors
            for substitution in substitutions:
                substitutions_by_frame[int(substitution["frame_index"])] = substitution
            completed += len(pair_chunk)
            if progress is not None:
                progress(completed, len(position_pairs_array))
    return torch.cat(outputs, dim=0).contiguous(), list(substitutions_by_frame.values())


def encode_dino_pair_batches(
    adapter: Any,
    frames: list[tuple[int, str]],
    position_pairs_array: np.ndarray,
    load_frame: Callable[..., torch.Tensor],
    *,
    precision: str,
    image_height: int,
    image_width: int,
    loader_threads: int,
    frame_batch_size: int = DINO_FRAME_BATCH_SIZE,
    lam_batch_size: int = DINO_LAM_PAIR_BATCH_SIZE,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Encode DINO pairs after extracting every referenced frame only once.

    The feature bank exists only for the current trajectory and stays on CPU.
    JPEG reads for the next frame chunk overlap the current DINO forward, while
    long trajectories never materialize a full RGB tensor bank.
    """

    position_pairs_array = np.asarray(position_pairs_array, dtype=np.int64)
    if position_pairs_array.ndim != 2 or position_pairs_array.shape[1:] != (2,):
        raise ValueError("position_pairs_array must have shape [N,2]")
    if frame_batch_size < 1 or lam_batch_size < 1:
        raise ValueError("DINO frame and LAM pair batch sizes must be positive")
    latent_dim = int(adapter.checkpoint_metadata["hparams"]["lam_latent_dim"])
    if len(position_pairs_array) == 0:
        return torch.empty((0, latent_dim), dtype=torch.float32), []
    if np.min(position_pairs_array) < 0 or np.max(position_pairs_array) >= len(
        frames
    ):
        raise IndexError("DINO pair position is outside the trajectory")

    model = getattr(adapter, "model", None)
    extract_features = getattr(model, "extract_dino_features", None)
    lam = getattr(model, "lam", None)
    if not callable(extract_features) or lam is None or not callable(
        getattr(lam, "encode", None)
    ):
        raise TypeError("DINO feature-once extraction requires a DINOFeatureLAM adapter")

    unique_positions, inverse = np.unique(position_pairs_array, return_inverse=True)
    feature_bank: torch.Tensor | None = None
    substitutions_by_frame: dict[int, dict[str, Any]] = {}
    chunks = [
        unique_positions[start : start + frame_batch_size]
        for start in range(0, len(unique_positions), frame_batch_size)
    ]

    def prepare(position_chunk: np.ndarray) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        return _load_frame_positions(
            frames,
            position_chunk,
            load_frame,
            image_height=image_height,
            image_width=image_width,
            loader_threads=loader_threads,
        )

    completed_frames = 0
    with ThreadPoolExecutor(max_workers=1) as read_ahead:
        future = read_ahead.submit(prepare, chunks[0])
        for chunk_index, position_chunk in enumerate(chunks):
            frame_tensors, substitutions = future.result()
            if chunk_index + 1 < len(chunks):
                future = read_ahead.submit(prepare, chunks[chunk_index + 1])
            videos = frame_tensors.to(device=adapter.device).unsqueeze(1)
            with _autocast(adapter.device, precision):
                feature_sequence = extract_features(videos)
            expected_prefix = (len(position_chunk), 1)
            if (
                feature_sequence.ndim != 4
                or tuple(feature_sequence.shape[:2]) != expected_prefix
            ):
                raise RuntimeError(
                    "DINO encoder returned malformed features: "
                    f"got {tuple(feature_sequence.shape)}, expected prefix {expected_prefix}"
                )
            chunk_features = feature_sequence[:, 0].detach().cpu()
            if feature_bank is None:
                feature_bank = torch.empty(
                    (len(unique_positions), *chunk_features.shape[1:]),
                    dtype=chunk_features.dtype,
                )
            start = completed_frames
            completed_frames += len(position_chunk)
            feature_bank[start:completed_frames].copy_(chunk_features)
            for substitution in substitutions:
                substitutions_by_frame[int(substitution["frame_index"])] = substitution
            if progress is not None:
                progress("dino_frames", completed_frames, len(unique_positions))
            del frame_tensors, videos, feature_sequence, chunk_features

    assert feature_bank is not None
    compact_pairs = torch.from_numpy(inverse.reshape(-1, 2).astype(np.int64, copy=False))
    outputs: list[torch.Tensor] = []
    completed_pairs = 0
    for start in range(0, len(compact_pairs), lam_batch_size):
        selection = compact_pairs[start : start + lam_batch_size]
        flat_selection = selection.reshape(-1)
        pair_features = feature_bank.index_select(0, flat_selection).reshape(
            len(selection), 2, *feature_bank.shape[1:]
        )
        pair_features = pair_features.to(device=adapter.device)
        if hasattr(lam, "mu_record"):
            lam.mu_record = None
        try:
            with _autocast(adapter.device, precision):
                encoded = lam.encode(pair_features)
            z_mu = encoded["z_mu"]
            expected = (len(selection), latent_dim)
            if tuple(z_mu.shape) != expected:
                raise RuntimeError(
                    f"DINO navigation LAM returned z_mu {tuple(z_mu.shape)}, "
                    f"expected {expected}"
                )
            outputs.append(z_mu.detach().float().cpu())
        finally:
            if hasattr(lam, "mu_record"):
                lam.mu_record = None
        completed_pairs += len(selection)
        if progress is not None:
            progress("lam_pairs", completed_pairs, len(compact_pairs))
    return torch.cat(outputs, dim=0).contiguous(), list(substitutions_by_frame.values())


def _load_adapter(state: Mapping[str, Any], local_rank: int) -> Any:
    _, load_adapter, _ = _lam_api(state["lam_project_root"])
    adapter = load_adapter(
        state["checkpoint"]["path"],
        device=torch.device("cuda", local_rank),
        fused_attention=True,
    )
    runtime = adapter.checkpoint_metadata
    validate_checkpoint_metadata(
        runtime,
        image_height=int(state["extraction"]["configuration"]["image_height"]),
        image_width=int(state["extraction"]["configuration"]["image_width"]),
        max_abs_frame_offset=int(
            state["policy"]["configuration"]["max_abs_frame_offset"]
        ),
    )
    return adapter


def _compute_task(
    adapter: Any | None,
    load_frame: Callable[..., torch.Tensor],
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    frames: list[tuple[int, str]],
    indices: np.ndarray,
    source_fingerprint: str,
    pair_bitmap: np.ndarray | None,
    args: argparse.Namespace,
    rank: int,
    reuse_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frame_pairs = _task_frame_pairs(state, task, indices, pair_bitmap)
    reuse = (
        dict(reuse_result)
        if reuse_result is not None
        else _reuse_task_motion(
            state, task, indices, source_fingerprint, frame_pairs
        )
    )
    motion = torch.as_tensor(reuse["motion"])
    missing_rows = np.asarray(reuse["missing_rows"], dtype=np.int64)
    substitutions: list[dict[str, Any]] = []
    if int(reuse.get("reused_rows", 0)):
        log(
            rank,
            f"reused {int(reuse['reused_rows'])}/{len(frame_pairs)} exact rows "
            f"for {task['source_id']}/{task['trajectory_id']}",
        )
    if len(missing_rows):
        if adapter is None:
            raise RuntimeError("Navigation LAM adapter is required for missing reuse rows")
        missing_pairs = frame_pairs[missing_rows]
        position_pair_array = position_pairs(indices, missing_pairs)
        if state["checkpoint"]["class_path"] == DINO_CHECKPOINT_CLASS:
            last_logged = {"dino_frames": 0, "lam_pairs": 0}

            def report_dino_progress(phase: str, completed: int, total: int) -> None:
                if phase == "dino_frames":
                    interval = max(int(args.dino_frame_batch_size) * 20, 1)
                else:
                    interval = max(int(args.dino_lam_batch_size) * 20, 1)
                if total >= interval and (
                    completed == total or completed - last_logged[phase] >= interval
                ):
                    log(
                        rank,
                        f"DINO feature-once {task['source_id']}/{task['trajectory_id']} "
                        f"{phase}={completed}/{total}",
                    )
                    last_logged[phase] = completed

            inferred, substitutions = encode_dino_pair_batches(
                adapter,
                frames,
                position_pair_array,
                load_frame,
                precision=str(args.precision),
                image_height=int(args.image_height),
                image_width=int(args.image_width),
                loader_threads=int(args.loader_threads),
                frame_batch_size=int(args.dino_frame_batch_size),
                lam_batch_size=int(args.dino_lam_batch_size),
                progress=report_dino_progress,
            )
        elif args.stream_frames:
            last_logged = 0

            def report_progress(completed: int, total: int) -> None:
                nonlocal last_logged
                interval = max(int(args.stream_pair_chunk_size) * 20, 1)
                if completed == total or completed - last_logged >= interval:
                    log(
                        rank,
                        f"streaming {task['source_id']}/{task['trajectory_id']} "
                        f"missing_pairs={completed}/{total}",
                    )
                    last_logged = completed

            inferred, substitutions = encode_pair_batches_streaming(
                adapter,
                frames,
                position_pair_array,
                load_frame,
                batch_size=int(args.batch_size),
                precision=str(args.precision),
                image_height=int(args.image_height),
                image_width=int(args.image_width),
                loader_threads=int(args.loader_threads),
                pair_chunk_size=int(args.stream_pair_chunk_size),
                progress=report_progress,
            )
        else:
            frame_tensors, substitutions = _load_frames(
                frames,
                load_frame,
                image_height=int(args.image_height),
                image_width=int(args.image_width),
                loader_threads=int(args.loader_threads),
            )
            inferred = encode_pair_batches(
                adapter,
                frame_tensors,
                position_pair_array,
                batch_size=int(args.batch_size),
                precision=str(args.precision),
            )
            del frame_tensors
        motion[torch.from_numpy(missing_rows)] = inferred
    for item in substitutions:
        log(
            rank,
            f"replaced unreadable {task['source_id']}/{task['trajectory_id']}/"
            f"{item['frame_path']} with {item['replacement_frame_path']}",
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "proxy_type": "latent",
        "motion_type": "latent",
        "action_mode": "latent",
        "source_id": task["source_id"],
        "trajectory_id": task["trajectory_id"],
        "dataset_name": task["source_id"],
        "trajectory_name": task["trajectory_id"],
        "pair_direction": "current_to_goal",
        "normalization": "raw",
        "latent_value": "z_mu",
        "latent_dim": int(state["checkpoint"]["latent_dim"]),
        "frame_indices": torch.from_numpy(indices.copy()),
        "frame_pairs": torch.from_numpy(frame_pairs.copy()),
        "motion": motion,
        "invalid_frame_substitutions": substitutions,
        "reuse": {
            "reused_rows": int(reuse.get("reused_rows", 0)),
            "inferred_rows": int(len(missing_rows)),
            "sources": list(reuse.get("provenance", ())),
        },
        "metadata": _expected_metadata(state, task, source_fingerprint),
        "complete": True,
    }
    output = safe_path(
        Path(state["output_root"]),
        str(task["source_id"]),
        str(task["trajectory_id"]),
        ".pt",
    )
    atomic_torch_save(payload, output)
    return _cache_record(
        output, state, task, indices, source_fingerprint, pair_bitmap
    )


def _write_completion(
    state: Mapping[str, Any], records: list[dict[str, Any]], partial: bool
) -> None:
    output_root = Path(state["output_root"])
    records.sort(key=lambda item: (item["source_id"], item["trajectory_id"]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["source_id"]), []).append(record)
    sources: dict[str, Any] = {}
    for source_id, source_records in grouped.items():
        manifest_path = output_root / source_id / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in source_records
        )
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.chmod(0o640)
        os.replace(temporary, manifest_path)
        sources[source_id] = {
            "trajectories": len(source_records),
            "frames": sum(item["frame_count"] for item in source_records),
            "pairs": sum(item["pair_count"] for item in source_records),
            "planned_pairs": sum(
                item["planned_pair_count"] for item in source_records
            ),
            "full_domain_superset_trajectories": sum(
                item["pair_coverage"] == "full_domain_superset"
                for item in source_records
            ),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    complete = not partial and len(records) == len(state["recipe"]["trajectories"])
    totals = {
        "trajectories": len(records),
        "frames": sum(item["frame_count"] for item in records),
        "pairs": sum(item["pair_count"] for item in records),
        "planned_pairs": sum(item["planned_pair_count"] for item in records),
        "full_domain_superset_trajectories": sum(
            item["pair_coverage"] == "full_domain_superset" for item in records
        ),
        "bytes": sum(item["file_size_bytes"] for item in records),
        "invalid_frame_substitutions": sum(
            len(item["invalid_frame_substitutions"]) for item in records
        ),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "status": "complete" if complete else "partial",
        "complete": complete,
        "updated_at_utc": utc_now(),
        "sampling_recipe": {
            "path": state["sampling_recipe_path"],
            "sha256": state["sampling_recipe_sha256"],
            "inventory_sha256": state["recipe"]["inventory_sha256"],
        },
        "checkpoint": state["checkpoint"],
        "extraction": state["extraction"],
        "policy": state["policy"],
        "training_pair_plan": state.get("training_pair_plan"),
        "storage": {
            "file_pattern": "{source_id}/{trajectory_id}.pt",
            "pairs_key": "frame_pairs",
            "values_key": "motion",
            "motion_dtype": "float32",
        },
        "sources": sources,
        "totals": totals,
    }
    metadata_path = output_root / "metadata.json"
    atomic_json_dump(metadata, metadata_path)
    success_path = output_root / "_SUCCESS.json"
    if complete:
        atomic_json_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "format": FORMAT_NAME,
                "complete": True,
                "completed_at_utc": utc_now(),
                "metadata_sha256": sha256_file(metadata_path),
                "sampling_recipe_sha256": state["sampling_recipe_sha256"],
                "checkpoint_sha256": state["checkpoint"]["sha256"],
                "extraction_fingerprint": state["extraction"]["fingerprint"],
                "policy_fingerprint": state["policy"]["fingerprint"],
                "totals": totals,
            },
            success_path,
        )
    else:
        with contextlib.suppress(FileNotFoundError):
            success_path.unlink()


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.loader_threads < 0
        or args.stream_pair_chunk_size < 1
        or args.dino_frame_batch_size < 1
        or args.dino_lam_batch_size < 1
    ):
        raise ValueError("Invalid encoder batch size or loader thread count")
    if args.context_size < 1 or args.max_abs_frame_offset < 0:
        raise ValueError("Invalid pair-domain arguments")
    if args.max_trajectories < 0:
        raise ValueError("--max-trajectories cannot be negative")
    rank, world_size, local_rank = distributed_context()
    state_message: list[Any] = [None]
    lock_handle = None
    if rank == 0:
        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        lock_handle = (output_root / ".precompute.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            state_message[0] = {"ok": True, "state": _build_state(args, world_size)}
        except Exception as exc:
            state_message[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if world_size > 1:
        dist.broadcast_object_list(state_message, src=0)
    if not state_message[0]["ok"]:
        raise RuntimeError(state_message[0]["error"])
    state = state_message[0]["state"]
    pair_bitmap: np.ndarray | None = None
    if state.get("training_pair_plan") is not None:
        pair_bitmap = np.memmap(
            state["training_pair_plan"]["pair_bitmap_path"],
            dtype=np.uint8,
            mode="r",
        )
    assigned = [item for item in state["tasks"] if item["assigned_rank"] == rank]
    log(
        rank,
        f"assigned trajectories={len(assigned)}, frames="
        f"{sum(int(item['frame_count']) for item in assigned)}, planned_pairs="
        f"{sum(int(item.get('planned_pair_count', 0)) for item in assigned)}",
    )
    _, _, load_frame = _lam_api(state["lam_project_root"])
    adapter = None
    records: list[dict[str, Any]] = []
    pairs_done = 0
    started = time.perf_counter()
    for task_index, task in enumerate(assigned, 1):
        frames, indices, source_fingerprint = _scan_task(Path(state["data_root"]), task)
        output = safe_path(
            Path(state["output_root"]), task["source_id"], task["trajectory_id"], ".pt"
        )
        record = None
        if output.is_file() and not args.overwrite:
            try:
                record = _cache_record(
                    output,
                    state,
                    task,
                    indices,
                    source_fingerprint,
                    pair_bitmap,
                )
            except Exception as exc:
                log(rank, f"recomputing invalid {output}: {type(exc).__name__}: {exc}")
        if record is None:
            target_pairs = _task_frame_pairs(state, task, indices, pair_bitmap)
            reuse_result = _reuse_task_motion(
                state, task, indices, source_fingerprint, target_pairs
            )
            if len(reuse_result["missing_rows"]) and adapter is None:
                log(rank, "loading navigation LAM checkpoint")
                adapter = _load_adapter(state, local_rank)
                log(rank, "navigation LAM ready")
            record = _compute_task(
                adapter,
                load_frame,
                state,
                task,
                frames,
                indices,
                source_fingerprint,
                pair_bitmap,
                args,
                rank,
                reuse_result=reuse_result,
            )
        records.append(record)
        pairs_done += int(record["pair_count"])
        if task_index % max(1, args.log_every_trajectories) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            log(
                rank,
                f"completed {task_index}/{len(assigned)} trajectories, "
                f"{pairs_done / elapsed:.1f} pairs/s",
            )
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(records, gathered, dst=0)
    else:
        gathered = [records]
    if rank == 0:
        assert gathered is not None
        all_records = [item for rank_records in gathered for item in rank_records]
        _write_completion(
            state, all_records, partial=bool(state["selection_partial"])
        )
        log(
            rank,
            f"cache {'partial' if state['selection_partial'] else 'complete'}: "
            f"trajectories={len(all_records)}, output={state['output_root']}",
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if lock_handle is not None:
        lock_handle.close()


if __name__ == "__main__":
    main()
