#!/usr/bin/env python3
"""Extract nav1 PixelActionLAM posterior means for NavAnywhere LatentPT.

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
EXPECTED_LATENT_DIM = 32
EXPECTED_PATCH_SIZE = 16
IMAGE_PREPROCESSING = "center_crop_4:3_then_bilinear_resize"
IMAGE_READ_ATTEMPTS = 3
IMAGE_READ_RETRY_DELAY_SECONDS = 0.25
GPU_FRAME_BANK_LIMIT_BYTES = 16 * 1024**3
PAIR_BANK_CHUNK_SIZE = 8192


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
    if metadata.get("class_path") != EXPECTED_CHECKPOINT_CLASS:
        raise ValueError(
            f"checkpoint class={metadata.get('class_path')!r}, "
            f"expected {EXPECTED_CHECKPOINT_CLASS!r}"
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


def _extraction(args: argparse.Namespace) -> dict[str, Any]:
    configuration = {
        "precision": str(args.precision),
        "batch_size": int(args.batch_size),
        "fused_attention": True,
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "image_preprocessing": IMAGE_PREPROCESSING,
        "image_value_range": [0.0, 1.0],
        "output_dtype": "float32",
        "frame_bank_strategy": (
            "full_gpu_up_to_16gib_else_compact_8192_pair_chunks_v1"
        ),
    }
    return {"configuration": configuration, "fingerprint": fingerprint(configuration)}


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
    if report.get("reference_validation", {}).get("status") != "matched":
        raise ValueError("Training pair plan was not matched to TimePT/GeoPT references")
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
    )
    checkpoint = _checkpoint_descriptor(args)
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
        "extraction": _extraction(args),
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
            loaded = list(executor.map(load, range(len(frames))))
    else:
        loaded = [load(position) for position in range(len(frames))]
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
    adapter: Any,
    load_frame: Callable[..., torch.Tensor],
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    frames: list[tuple[int, str]],
    indices: np.ndarray,
    source_fingerprint: str,
    pair_bitmap: np.ndarray | None,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, Any]:
    frame_pairs = _task_frame_pairs(state, task, indices, pair_bitmap)
    position_pair_array = position_pairs(indices, frame_pairs)
    frame_tensors, substitutions = _load_frames(
        frames,
        load_frame,
        image_height=int(args.image_height),
        image_width=int(args.image_width),
        loader_threads=int(args.loader_threads),
    )
    motion = encode_pair_batches(
        adapter,
        frame_tensors,
        position_pair_array,
        batch_size=int(args.batch_size),
        precision=str(args.precision),
    )
    del frame_tensors
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
    if args.batch_size < 1 or args.loader_threads < 0:
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
            if adapter is None:
                log(rank, "loading nav1 PixelActionLAM checkpoint")
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
