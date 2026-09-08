#!/usr/bin/env python3
"""Distributed VGGT-Omega geometry-action precompute for NavAnywhere.

The output is one strict ``OfflineProxyStore``-compatible file per trajectory:
``{output_root}/{source_id}/{trajectory_id}.pt``.  Camera poses are inferred
from RGB only, long trajectories are joined with overlap-based Sim(3)
alignment, and translation is normalized so the median non-zero adjacent
predicted step in each trajectory is one waypoint unit.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from geometry_action.tartandrive_vggt_omega import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    PINNED_CODE_REVISION,
    VGGTOmegaCameraExtractor,
    atomic_json_dump,
    atomic_torch_save,
    geometry_actions_tartandrive_forward_camera,
    raw_extraction_descriptor,
    safe_torch_load,
    sha256_file,
    tartandrive_image_only_scale,
)
from navanywhere_recipe import (
    canonical_json,
    frame_indices_sha256,
    load_sampling_recipe,
    scan_trajectory_frames,
)


SCHEMA_VERSION = 1
FORMAT_NAME = "navanywhere_vggt_omega_geometry_proxy"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def log(rank: int, message: str) -> None:
    print(f"[{utc_now()}][rank {rank}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sampling-recipe", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--third-party-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--expected-code-revision", default=PINNED_CODE_REVISION)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument(
        "--resize-mode", choices=("balanced", "max_size"), default="max_size"
    )
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--max-abs-frame-offset", type=int, default=8)
    parser.add_argument("--nonzero-step-epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument(
        "--trajectory",
        action="append",
        default=None,
        help="Restrict to source_id/trajectory_id (repeatable; useful for smoke tests).",
    )
    parser.add_argument("--log-every-trajectories", type=int, default=10)
    parser.add_argument("--preprocess-workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-degenerate-window-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
        raise RuntimeError("CUDA is required for VGGT-Omega geometry extraction")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def safe_path(root: Path, source_id: str, trajectory_id: str, suffix: str) -> Path:
    source_root = (root / source_id).resolve()
    path = (source_root / f"{trajectory_id}{suffix}").resolve()
    if path == source_root or source_root not in path.parents:
        raise ValueError(f"Unsafe identity {source_id!r}/{trajectory_id!r}")
    return path


def build_navanywhere_frame_pairs(
    frame_indices: Sequence[int] | np.ndarray,
    *,
    context_size: int = 4,
    max_abs_frame_offset: int = 8,
) -> np.ndarray:
    """Enumerate every pair that strict local-proxy training can request."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < context_size:
        if indices.ndim != 1:
            raise ValueError("frame_indices must be one-dimensional")
        return np.empty((0, 2), dtype=np.int64)
    if context_size < 1 or max_abs_frame_offset < 0:
        raise ValueError("context_size must be positive and offset non-negative")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    available = set(map(int, indices.tolist()))
    rows = [
        (int(current), target)
        for current in indices[context_size - 1 :]
        for target in range(
            int(current) - max_abs_frame_offset,
            int(current) + max_abs_frame_offset + 1,
        )
        if target in available
    ]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def _position_pairs(frame_indices: np.ndarray, frame_pairs: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(frame_indices, frame_pairs)
    if np.any(positions >= len(frame_indices)):
        raise ValueError("frame pair refers to an absent frame")
    if not np.array_equal(frame_indices[positions], frame_pairs):
        raise ValueError("frame pair refers to an absent frame")
    return positions.astype(np.int64, copy=False)


def _checkpoint_receipt(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    receipt_path = Path(args.checkpoint_manifest).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint manifest")
    source = receipt.get("source", {})
    code = receipt.get("official_code", {})
    if source.get("resolved_revision") != args.model_revision:
        raise ValueError("VGGT-Omega model revision differs from receipt")
    if code.get("commit") != args.expected_code_revision:
        raise ValueError("VGGT-Omega code revision differs from receipt")
    entry = next(
        (
            item
            for item in receipt.get("files", [])
            if isinstance(item, Mapping) and item.get("path") == checkpoint.name
        ),
        None,
    )
    if entry is None or not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint or its receipt entry is missing")
    if checkpoint.stat().st_size != int(entry.get("bytes", -1)):
        raise ValueError("Checkpoint byte count differs from receipt")
    digest = str(entry.get("sha256", ""))
    if len(digest) != 64:
        raise ValueError("Checkpoint receipt has no valid SHA256")
    return {
        "path": str(checkpoint),
        "sha256": digest,
        "manifest_path": str(receipt_path),
        "manifest_sha256": sha256_file(receipt_path),
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": args.model_revision,
        "code_revision": args.expected_code_revision,
    }


def _policy(args: argparse.Namespace) -> dict[str, Any]:
    configuration = {
        "pair_domain": "NavAnywhere_observations_and_existing_targets_v1",
        "context_size": int(args.context_size),
        "max_abs_frame_offset": int(args.max_abs_frame_offset),
        "pair_direction": "current_to_goal",
        "camera_axes": {
            "x": "OpenCV_camera_z_forward",
            "y": "negative_OpenCV_camera_x_left",
            "yaw": "heading_delta_radians",
        },
        "projection": "first_frame_navigation_se2_then_inverse_current_times_goal",
        "translation_scale": "per_trajectory_median_nonzero_adjacent_step_to_one",
        "degenerate_trajectory_scale": "unit",
        "translation_unit": "waypoint_spacing_units",
        "ground_truth_usage": "none",
    }
    return {"configuration": configuration, "fingerprint": fingerprint(configuration)}


def _extraction(args: argparse.Namespace) -> dict[str, Any]:
    return raw_extraction_descriptor(
        resolution=args.resolution,
        resize_mode=args.resize_mode,
        window_size=args.window_size,
        overlap=args.overlap,
        inference_path="fast",
        dtype="bfloat16",
        allow_tf32=True,
        allow_degenerate_window_scale=args.allow_degenerate_window_scale,
        seed=args.seed,
    )


def _window_cost(num_frames: int, window_size: int, overlap: int) -> int:
    if window_size <= 0 or num_frames <= window_size:
        return max(16, num_frames) ** 2
    cost = 0
    start = 0
    while True:
        end = min(num_frames, start + window_size)
        cost += max(16, end - start) ** 2
        if end == num_frames:
            return cost
        start = end - overlap


def _assign(tasks: list[dict[str, Any]], world_size: int, args: argparse.Namespace) -> None:
    loads = [0] * world_size
    for task in sorted(
        tasks,
        key=lambda item: (
            -_window_cost(int(item["frame_count"]), args.window_size, args.overlap),
            item["source_id"],
            item["trajectory_id"],
        ),
    ):
        target = min(range(world_size), key=lambda value: (loads[value], value))
        task["assigned_rank"] = target
        loads[target] += _window_cost(
            int(task["frame_count"]), args.window_size, args.overlap
        )


def _scan_task(
    data_root: Path, task: Mapping[str, Any]
) -> tuple[list[tuple[int, str]], np.ndarray]:
    trajectory_root = safe_path(
        data_root, str(task["source_id"]), str(task["trajectory_id"]), ""
    )
    frames = scan_trajectory_frames(trajectory_root)
    indices = np.asarray([item[0] for item in frames], dtype=np.int64)
    if len(indices) != int(task["frame_count"]):
        raise ValueError("trajectory frame count changed after recipe creation")
    if frame_indices_sha256(indices) != task["frame_indices_sha256"]:
        raise ValueError("trajectory frame indices changed after recipe creation")
    return frames, indices


def _expected_metadata(state: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sampling_recipe_sha256": state["sampling_recipe_sha256"],
        "frame_indices_sha256": task["frame_indices_sha256"],
        "checkpoint_sha256": state["checkpoint"]["sha256"],
        "extraction_fingerprint": state["extraction"]["fingerprint"],
        "policy_fingerprint": state["policy"]["fingerprint"],
    }


def _cache_record(
    path: Path,
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    payload = safe_torch_load(path)
    required = {
        "schema_version",
        "format",
        "proxy_type",
        "source_id",
        "trajectory_id",
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
        and payload["proxy_type"] == "geometry"
        and payload["source_id"] == task["source_id"]
        and payload["trajectory_id"] == task["trajectory_id"]
        and payload["complete"] is True
    ):
        raise ValueError("cache identity or format mismatch")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("cache metadata is not a mapping")
    for key, expected in _expected_metadata(state, task).items():
        if metadata.get(key) != expected:
            raise ValueError(f"cache metadata.{key} mismatch")
    pairs = torch.as_tensor(payload["frame_pairs"])
    expected_pairs = torch.from_numpy(
        build_navanywhere_frame_pairs(
            indices,
            context_size=int(state["policy"]["configuration"]["context_size"]),
            max_abs_frame_offset=int(
                state["policy"]["configuration"]["max_abs_frame_offset"]
            ),
        )
    )
    motion = torch.as_tensor(payload["motion"])
    if pairs.dtype != torch.int64 or not torch.equal(pairs, expected_pairs):
        raise ValueError("cache frame-pair domain is incomplete")
    if motion.dtype != torch.float32 or motion.shape != (len(pairs), 3):
        raise ValueError("cache motion shape or dtype mismatch")
    if not torch.isfinite(motion).all():
        raise ValueError("cache motion contains non-finite values")
    return {
        "source_id": task["source_id"],
        "trajectory_id": task["trajectory_id"],
        "frame_count": len(indices),
        "pair_count": len(pairs),
        "file_size_bytes": path.stat().st_size,
        "scale_status": payload.get("scale_status"),
        "invalid_frame_substitutions": payload.get(
            "invalid_frame_substitutions", []
        ),
    }


def _readable(path: str) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _repair_unreadable(
    frames: list[tuple[int, str]], rank: int, identity: str
) -> tuple[list[str], list[dict[str, Any]]]:
    readable = [_readable(path) for _, path in frames]
    if all(readable):
        raise RuntimeError(
            f"VGGT preprocessing failed for {identity}, but every JPEG verifies"
        )
    replacements: list[str] = []
    records: list[dict[str, Any]] = []
    good = [position for position, valid in enumerate(readable) if valid]
    if not good:
        raise RuntimeError(f"No readable JPEG remains in {identity}")
    for position, (frame_index, path) in enumerate(frames):
        if readable[position]:
            replacements.append(path)
            continue
        replacement = min(good, key=lambda value: (abs(value - position), value > position))
        replacement_index, replacement_path = frames[replacement]
        replacements.append(replacement_path)
        record = {
            "frame_index": int(frame_index),
            "frame_path": Path(path).name,
            "replacement_frame_index": int(replacement_index),
            "replacement_frame_path": Path(replacement_path).name,
            "reason": "unreadable JPEG",
        }
        records.append(record)
        log(rank, f"{identity}: replaced unreadable {Path(path).name} with {Path(replacement_path).name}")
    return replacements, records


def _summarize_windows(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    center = [float(item["overlap_center_rmse"]) for item in records[1:]]
    rotation = [float(item["overlap_rotation_rmse_deg"]) for item in records[1:]]
    degenerate = sum(
        bool(item["alignment_to_global"]["degenerate_scale"])
        for item in records[1:]
    )
    return {
        "count": len(records),
        "degenerate_overlap_scales": int(degenerate),
        "max_overlap_center_rmse": max(center, default=0.0),
        "max_overlap_rotation_rmse_deg": max(rotation, default=0.0),
    }


def _compute_task(
    extractor: VGGTOmegaCameraExtractor,
    state: Mapping[str, Any],
    task: Mapping[str, Any],
    frames: list[tuple[int, str]],
    indices: np.ndarray,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, Any]:
    paths = [item[1] for item in frames]
    substitutions: list[dict[str, Any]] = []
    try:
        extrinsics, _, _, windows, alignment = extractor.extract_trajectory(
            image_paths=paths,
            window_size=args.window_size,
            overlap=args.overlap,
            resize_mode=args.resize_mode,
            resolution=args.resolution,
            full=False,
            allow_degenerate_window_scale=args.allow_degenerate_window_scale,
        )
    except OSError:
        paths, substitutions = _repair_unreadable(
            frames, rank, f"{task['source_id']}/{task['trajectory_id']}"
        )
        extrinsics, _, _, windows, alignment = extractor.extract_trajectory(
            image_paths=paths,
            window_size=args.window_size,
            overlap=args.overlap,
            resize_mode=args.resize_mode,
            resolution=args.resolution,
            full=False,
            allow_degenerate_window_scale=args.allow_degenerate_window_scale,
        )
    pairs = build_navanywhere_frame_pairs(
        indices,
        context_size=args.context_size,
        max_abs_frame_offset=args.max_abs_frame_offset,
    )
    position_pairs = _position_pairs(indices, pairs)
    scale, scale_diagnostics = tartandrive_image_only_scale(
        extrinsics, nonzero_epsilon=args.nonzero_step_epsilon
    )
    scale_status = "estimated"
    if scale is None:
        scale = 1.0
        scale_status = "fallback_unit_scale_no_nonzero_adjacent_translation"
    motion = geometry_actions_tartandrive_forward_camera(
        extrinsics, position_pairs, scale=scale
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "proxy_type": "geometry",
        "motion_type": "geometry",
        "action_mode": "geometry",
        "source_id": task["source_id"],
        "trajectory_id": task["trajectory_id"],
        "dataset_name": task["source_id"],
        "trajectory_name": task["trajectory_id"],
        "pair_direction": "current_to_goal",
        "normalization": "raw",
        "coordinate_frame": "current_navigation_frame",
        "translation_unit": "waypoint_spacing_units",
        "yaw_unit": "radians",
        "components": ["delta_x", "delta_y", "delta_yaw"],
        "frame_indices": torch.from_numpy(indices.copy()),
        "frame_pairs": torch.from_numpy(pairs.copy()),
        "motion": torch.from_numpy(motion),
        "scale": float(scale),
        "scale_status": scale_status,
        "scale_diagnostics": scale_diagnostics,
        "window_alignment": dict(alignment),
        "window_summary": _summarize_windows(windows),
        "invalid_frame_substitutions": substitutions,
        "metadata": _expected_metadata(state, task),
        "complete": True,
    }
    output = safe_path(
        Path(state["output_root"]),
        str(task["source_id"]),
        str(task["trajectory_id"]),
        ".pt",
    )
    atomic_torch_save(payload, output)
    return _cache_record(output, state, task, indices)


def _build_state(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"NavAnywhere root does not exist: {data_root}")
    recipe, recipe_path, recipe_sha = load_sampling_recipe(
        args.sampling_recipe, context_size=args.context_size
    )
    checkpoint = _checkpoint_receipt(args)
    extraction = _extraction(args)
    policy = _policy(args)
    tasks = [dict(item) for item in recipe["trajectories"]]
    if args.trajectory:
        requested = set(args.trajectory)
        tasks = [
            item
            for item in tasks
            if f"{item['source_id']}/{item['trajectory_id']}" in requested
        ]
        found = {f"{item['source_id']}/{item['trajectory_id']}" for item in tasks}
        if found != requested:
            raise ValueError(f"Unknown requested trajectories: {sorted(requested - found)}")
    if args.max_trajectories:
        tasks = tasks[: args.max_trajectories]
    _assign(tasks, world_size, args)
    return {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "sampling_recipe_path": recipe_path,
        "sampling_recipe_sha256": recipe_sha,
        "recipe": recipe,
        "tasks": tasks,
        "checkpoint": checkpoint,
        "extraction": extraction,
        "policy": policy,
        "world_size": world_size,
        "selection_partial": bool(args.trajectory or args.max_trajectories),
    }


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
        os.replace(temporary, manifest_path)
        sources[source_id] = {
            "trajectories": len(source_records),
            "frames": sum(item["frame_count"] for item in source_records),
            "pairs": sum(item["pair_count"] for item in source_records),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    complete = not partial and len(records) == len(state["recipe"]["trajectories"])
    totals = {
        "trajectories": len(records),
        "frames": sum(item["frame_count"] for item in records),
        "pairs": sum(item["pair_count"] for item in records),
        "bytes": sum(item["file_size_bytes"] for item in records),
        "degenerate_trajectory_scales": sum(
            item["scale_status"] != "estimated" for item in records
        ),
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
    if args.context_size < 1 or args.max_abs_frame_offset < 0:
        raise ValueError("Invalid pair-domain arguments")
    if args.window_size < 2 or not 3 <= args.overlap < args.window_size:
        raise ValueError("Require window_size >= 2 and 3 <= overlap < window_size")
    if args.resolution < 16 or args.resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
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
            state_message[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if world_size > 1:
        dist.broadcast_object_list(state_message, src=0)
    if not state_message[0]["ok"]:
        raise RuntimeError(state_message[0]["error"])
    state = state_message[0]["state"]
    assigned = [item for item in state["tasks"] if item["assigned_rank"] == rank]
    log(
        rank,
        f"assigned trajectories={len(assigned)}, frames="
        f"{sum(int(item['frame_count']) for item in assigned)}",
    )
    extractor: VGGTOmegaCameraExtractor | None = None
    records: list[dict[str, Any]] = []
    frames_done = 0
    started = time.perf_counter()
    for task_index, task in enumerate(assigned, 1):
        frames, indices = _scan_task(Path(state["data_root"]), task)
        output = safe_path(
            Path(state["output_root"]), task["source_id"], task["trajectory_id"], ".pt"
        )
        record = None
        if output.is_file() and not args.overwrite:
            try:
                record = _cache_record(output, state, task, indices)
            except Exception as exc:
                log(rank, f"recomputing invalid {output}: {type(exc).__name__}: {exc}")
        if record is None:
            if extractor is None:
                log(rank, "loading strict VGGT-Omega checkpoint")
                extractor = VGGTOmegaCameraExtractor(
                    third_party_root=args.third_party_root,
                    checkpoint_path=args.checkpoint,
                    device=f"cuda:{local_rank}",
                    dtype="bfloat16",
                    expected_code_revision=args.expected_code_revision,
                    retain_dense_head=False,
                    allow_tf32=True,
                    preprocess_workers=args.preprocess_workers,
                )
                log(rank, "VGGT-Omega ready")
            record = _compute_task(
                extractor, state, task, frames, indices, args, rank
            )
        records.append(record)
        frames_done += int(record["frame_count"])
        if task_index % max(1, args.log_every_trajectories) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            log(
                rank,
                f"completed {task_index}/{len(assigned)} trajectories, "
                f"{frames_done / elapsed:.1f} input frames/s",
            )
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(records, gathered, dst=0)
    else:
        gathered = [records]
    if rank == 0:
        all_records = [item for rank_records in gathered for item in rank_records]
        _write_completion(state, all_records, partial=bool(state["selection_partial"]))
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
