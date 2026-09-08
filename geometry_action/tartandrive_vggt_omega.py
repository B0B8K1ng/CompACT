"""Reproducible VGGT-Omega camera-pose and geometry-action extraction.

The raw camera artifact is deliberately independent from the navigation action
policy.  VGGT-Omega predicts OpenCV world-to-camera extrinsics in an arbitrary
monocular gauge; those poses are always retained so a later calibration policy
can regenerate actions without another 1B-model forward pass.

The default TartanDrive action policy is image-only and explicit: OpenCV camera
``z`` is navigation ``x`` (forward), camera ``-x`` is navigation ``y`` (left),
and the median non-zero adjacent planar displacement is one waypoint unit.  No
``traj_data.pkl`` field is read by this module.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

RAW_POSE_SCHEMA_VERSION = 1
GEOMETRY_MOTION_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
RAW_ARTIFACT_TYPE = "vggt_omega_camera_poses"
MANIFEST_ARTIFACT_TYPE = "tartandrive_vggt_omega_input_manifest"
DEFAULT_MODEL_ID = "facebook/VGGT-Omega-1B-512"
DEFAULT_MODEL_REVISION = "ba9db085d6b7349b738fa2e37d198bb4dd077954"
PINNED_CODE_REVISION = "282ec70363edeff59424bf43731658092fba3d37"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_torch_save(value: Any, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def safe_torch_load(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping in {path}, got {type(payload).__name__}")
    return payload


def discover_numbered_frames(trajectory_dir: str | os.PathLike[str]) -> list[Path]:
    """Return strictly contiguous ``0.jpg ... N-1.jpg`` paths."""

    directory = Path(trajectory_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Trajectory directory does not exist: {directory}")
    indexed: list[tuple[int, Path]] = []
    for path in directory.glob("*.jpg"):
        if path.stem.isdecimal():
            indexed.append((int(path.stem), path))
    indexed.sort(key=lambda item: item[0])
    if not indexed:
        raise FileNotFoundError(f"No numbered JPEG frames found under {directory}")
    actual = [item[0] for item in indexed]
    expected = list(range(len(indexed)))
    if actual != expected:
        missing = sorted(set(range(actual[-1] + 1)).difference(actual))
        raise ValueError(
            f"Frames must be contiguous from 0 in {directory}; missing={missing[:20]}"
        )
    return [item[1] for item in indexed]


def build_frame_records(
    frame_paths: Sequence[str | os.PathLike[str]],
    *,
    data_root: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], str]:
    """Content-address frames and return records plus their canonical hash.

    ``frame_list_sha256`` is SHA256 over canonical JSON of the returned list.
    Each record contains the image-content SHA, so touching files without
    changing their bytes does not invalidate a completed artifact.
    """

    root = Path(data_root).resolve()
    records: list[dict[str, Any]] = []
    for frame_id, raw_path in enumerate(frame_paths):
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Frame {path} is outside data root {root}") from exc
        stat = path.stat()
        records.append(
            {
                "frame_id": int(frame_id),
                "relative_path": relative,
                "size_bytes": int(stat.st_size),
                "sha256": sha256_file(path),
            }
        )
    fingerprint = hashlib.sha256(_canonical_json_bytes(records)).hexdigest()
    return records, fingerprint


def read_split_names(
    split_root: str | os.PathLike[str], splits: Sequence[str]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in splits:
        path = Path(split_root) / str(split) / "traj_names.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split trajectory list: {path}")
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        names = [name for name in names if name]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate trajectory name in {path}")
        result[str(split)] = names
    return result


def build_input_manifest(
    *,
    data_root: str | os.PathLike[str],
    split_root: str | os.PathLike[str],
    splits: Sequence[str],
    dataset_name: str = "tartan_drive",
    selected_trajectories: Sequence[str] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Build the expensive content-hash manifest once before GPU sharding."""

    root = Path(data_root).resolve()
    split_names = read_split_names(split_root, splits)
    membership: dict[str, list[str]] = {}
    for split, names in split_names.items():
        for name in names:
            membership.setdefault(name, []).append(split)
    duplicated = {name: value for name, value in membership.items() if len(value) > 1}
    if duplicated:
        preview = dict(sorted(duplicated.items())[:10])
        raise ValueError(f"Trajectories occur in multiple splits: {preview}")

    if selected_trajectories:
        requested = list(dict.fromkeys(str(name) for name in selected_trajectories))
        unknown = sorted(set(requested).difference(membership))
        if unknown:
            raise ValueError(f"Requested trajectories are absent from splits: {unknown}")
        names = sorted(requested)
    else:
        names = sorted(membership)

    if workers < 1:
        raise ValueError("Manifest workers must be at least one")

    def fingerprint_trajectory(name: str) -> dict[str, Any]:
        frame_paths = discover_numbered_frames(root / name)
        records, fingerprint = build_frame_records(frame_paths, data_root=root)
        return {
            "trajectory_name": name,
            "split": membership[name][0],
            "num_frames": len(records),
            "frame_records": records,
            "frame_list_sha256": fingerprint,
        }

    if workers == 1:
        trajectories = [fingerprint_trajectory(name) for name in names]
    else:
        # executor.map preserves input order, so worker count cannot perturb the
        # canonical JSON bytes or the manifest SHA256.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            trajectories = list(executor.map(fingerprint_trajectory, names))

    split_files = {}
    for split in splits:
        split_file = Path(split_root) / split / "traj_names.txt"
        split_files[str(split)] = {
            "path": str(split_file.resolve()),
            "sha256": sha256_file(split_file),
            "num_trajectories": len(split_names[str(split)]),
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "dataset_name": str(dataset_name),
        "data_root": str(root),
        "split_root": str(Path(split_root).resolve()),
        "splits": split_files,
        "num_trajectories": len(trajectories),
        "num_frames": sum(item["num_frames"] for item in trajectories),
        "trajectories": trajectories,
        "complete": True,
    }


def validate_input_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported input manifest schema")
    if manifest.get("artifact_type") != MANIFEST_ARTIFACT_TYPE:
        raise ValueError("Unexpected input manifest artifact_type")
    if manifest.get("complete") is not True:
        raise ValueError("Input manifest is not complete")
    trajectories = manifest.get("trajectories")
    if not isinstance(trajectories, list):
        raise TypeError("Input manifest trajectories must be a list")
    names: set[str] = set()
    for item in trajectories:
        if not isinstance(item, Mapping):
            raise TypeError("Each manifest trajectory must be a mapping")
        name = str(item.get("trajectory_name", ""))
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate trajectory name: {name!r}")
        names.add(name)
        records = item.get("frame_records")
        if not isinstance(records, list) or len(records) != item.get("num_frames"):
            raise ValueError(f"Invalid frame records for {name}")
        expected = hashlib.sha256(_canonical_json_bytes(records)).hexdigest()
        if expected != item.get("frame_list_sha256"):
            raise ValueError(f"frame_list_sha256 mismatch for {name}")


def deterministic_shard(
    items: Sequence[Mapping[str, Any]],
    rank: int,
    world_size: int,
    *,
    window_size: int = 0,
    overlap: int = 0,
    cost_power: float = 2.0,
) -> list[Mapping[str, Any]]:
    """Deterministic LPT sharding balanced by estimated attention cost.

    A simple name modulo is badly imbalanced for TartanDrive (trajectory lengths
    range from 1 to 484).  Longest-processing-time-first provides stable shards
    while approximating the aggregator's quadratic frame attention.  In window
    mode, the cost is summed over the exact overlapping slices.
    """

    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid shard rank/world_size: {rank}/{world_size}")
    if cost_power <= 0:
        raise ValueError("cost_power must be positive")

    def estimated_cost(item: Mapping[str, Any]) -> float:
        num_frames = int(item["num_frames"])
        slices = make_window_slices(num_frames, window_size, overlap)
        return float(sum((end - start) ** cost_power for start, end in slices))

    ordered = sorted(
        items,
        key=lambda item: (
            -estimated_cost(item),
            str(item["trajectory_name"]),
        ),
    )
    loads = [0.0] * world_size
    assignments: list[list[Mapping[str, Any]]] = [[] for _ in range(world_size)]
    for item in ordered:
        target = min(range(world_size), key=lambda shard: (loads[shard], shard))
        assignments[target].append(item)
        loads[target] += estimated_cost(item)
    return sorted(assignments[rank], key=lambda item: str(item["trajectory_name"]))


def make_window_slices(
    num_frames: int, window_size: int, overlap: int
) -> list[tuple[int, int]]:
    if num_frames < 1:
        raise ValueError("A trajectory must have at least one frame")
    if window_size <= 0 or window_size >= num_frames:
        return [(0, num_frames)]
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if overlap < 3 or overlap >= window_size:
        raise ValueError("Windowed inference requires 3 <= overlap < window_size")
    result: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + window_size, num_frames)
        result.append((start, end))
        if end == num_frames:
            break
        start = end - overlap
    return result


def _homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    values = np.asarray(extrinsics, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 4):
        raise ValueError(f"Extrinsics must have shape [N,3,4], got {values.shape}")
    output = np.broadcast_to(np.eye(4, dtype=np.float64), (len(values), 4, 4)).copy()
    output[:, :3, :] = values
    return output


def _project_so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def w2c_to_c2w(extrinsics_w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_homogeneous(extrinsics_w2c))


def c2w_to_w2c(poses_c2w: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Camera poses must have shape [N,4,4], got {poses.shape}")
    return np.linalg.inv(poses)[:, :3, :]


@dataclass(frozen=True)
class SimilarityAlignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    degenerate_scale: bool


def estimate_pose_similarity(
    local_c2w: np.ndarray,
    global_c2w: np.ndarray,
    *,
    scale_epsilon: float = 1e-10,
    allow_degenerate_scale: bool = False,
) -> SimilarityAlignment:
    """Align one window gauge to another from shared camera poses."""

    local = np.asarray(local_c2w, dtype=np.float64)
    target = np.asarray(global_c2w, dtype=np.float64)
    if local.shape != target.shape or local.ndim != 3 or local.shape[1:] != (4, 4):
        raise ValueError("Overlap camera poses must have identical [N,4,4] shapes")
    if len(local) < 3:
        raise ValueError("At least three overlap poses are required for window alignment")

    rotation_candidates = np.einsum(
        "nij,nkj->nik", target[:, :3, :3], local[:, :3, :3]
    )
    rotation = _project_so3(rotation_candidates.mean(axis=0))
    local_centers = local[:, :3, 3]
    target_centers = target[:, :3, 3]
    local_mean = local_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    centered_local = (rotation @ (local_centers - local_mean).T).T
    centered_target = target_centers - target_mean
    denominator = float(np.sum(centered_local * centered_local))
    degenerate = denominator <= scale_epsilon
    if degenerate:
        if not allow_degenerate_scale:
            raise ValueError("Window overlap has insufficient translation to estimate scale")
        scale = 1.0
    else:
        scale = float(np.sum(centered_local * centered_target) / denominator)
        if not math.isfinite(scale):
            raise ValueError(f"Invalid overlap similarity scale: {scale}")
        if scale <= 0:
            if not allow_degenerate_scale:
                raise ValueError(f"Invalid overlap similarity scale: {scale}")
            # A non-positive least-squares scale cannot represent a Sim(3).
            # Treat it like a translation-degenerate overlap: retain the
            # orientation alignment, use unit scale, and expose the fallback
            # through ``degenerate_scale`` for provenance and diagnostics.
            scale = 1.0
            degenerate = True
    translation = target_mean - scale * (rotation @ local_mean)
    return SimilarityAlignment(scale, rotation, translation, degenerate)


def apply_pose_similarity(
    local_c2w: np.ndarray, alignment: SimilarityAlignment
) -> np.ndarray:
    local = np.asarray(local_c2w, dtype=np.float64)
    output = local.copy()
    output[:, :3, :3] = np.einsum(
        "ij,njk->nik", alignment.rotation, local[:, :3, :3]
    )
    output[:, :3, 3] = (
        alignment.scale
        * np.einsum("ij,nj->ni", alignment.rotation, local[:, :3, 3])
        + alignment.translation
    )
    return output


def _rotation_errors_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "nji,njk->nik", first[:, :3, :3], second[:, :3, :3]
    )
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def stitch_pose_windows(
    window_predictions: Sequence[Mapping[str, Any]],
    *,
    num_frames: int,
    requested_overlap: int,
    allow_degenerate_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Sequentially Sim(3)-stitch independently inferred overlapping windows."""

    if not window_predictions:
        raise ValueError("No window predictions were supplied")
    global_c2w = np.full((num_frames, 4, 4), np.nan, dtype=np.float64)
    global_intrinsics = np.full((num_frames, 3, 3), np.nan, dtype=np.float64)
    occupied = np.zeros(num_frames, dtype=bool)
    records: list[dict[str, Any]] = []

    for window_index, prediction in enumerate(window_predictions):
        start = int(prediction["start_index"])
        end = int(prediction["end_index_exclusive"])
        local_w2c = np.asarray(prediction["extrinsics_w2c"], dtype=np.float64)
        local_intrinsics = np.asarray(prediction["intrinsics"], dtype=np.float64)
        if end <= start or end - start != len(local_w2c):
            raise ValueError(f"Invalid window bounds [{start},{end})")
        if local_intrinsics.shape != (end - start, 3, 3):
            raise ValueError("Window intrinsics have an invalid shape")
        local_c2w = w2c_to_c2w(local_w2c)
        frame_indices = np.arange(start, end, dtype=np.int64)
        overlap_mask = occupied[frame_indices]
        overlap_indices = frame_indices[overlap_mask]

        if window_index == 0:
            if start != 0:
                raise ValueError("The first window must start at frame zero")
            alignment = SimilarityAlignment(
                1.0, np.eye(3, dtype=np.float64), np.zeros(3), False
            )
            aligned_c2w = local_c2w
            center_rmse = 0.0
            rotation_rmse = 0.0
        else:
            if len(overlap_indices) < 3:
                raise ValueError("Consecutive windows must overlap by at least three frames")
            local_overlap = local_c2w[overlap_mask]
            target_overlap = global_c2w[overlap_indices]
            alignment = estimate_pose_similarity(
                local_overlap,
                target_overlap,
                allow_degenerate_scale=allow_degenerate_scale,
            )
            aligned_c2w = apply_pose_similarity(local_c2w, alignment)
            aligned_overlap = aligned_c2w[overlap_mask]
            center_error = (
                aligned_overlap[:, :3, 3] - target_overlap[:, :3, 3]
            )
            center_rmse = float(np.sqrt(np.mean(np.sum(center_error**2, axis=1))))
            rotation_error = _rotation_errors_degrees(aligned_overlap, target_overlap)
            rotation_rmse = float(np.sqrt(np.mean(rotation_error**2)))

        new_mask = ~occupied[frame_indices]
        if not np.any(new_mask):
            raise ValueError(f"Window {window_index} contributes no new frames")
        new_indices = frame_indices[new_mask]
        global_c2w[new_indices] = aligned_c2w[new_mask]
        global_intrinsics[new_indices] = local_intrinsics[new_mask]
        occupied[new_indices] = True
        records.append(
            {
                "window_index": int(window_index),
                "start_index": start,
                "end_index_exclusive": end,
                "frame_ids": torch.arange(start, end, dtype=torch.int64),
                "extrinsics_w2c_local": torch.as_tensor(
                    local_w2c, dtype=torch.float32
                ),
                "intrinsics_local": torch.as_tensor(
                    local_intrinsics, dtype=torch.float32
                ),
                "image_size_hw": torch.as_tensor(
                    prediction["image_size_hw"], dtype=torch.int64
                ),
                "alignment_to_global": {
                    "scale": float(alignment.scale),
                    "rotation": torch.as_tensor(
                        alignment.rotation, dtype=torch.float32
                    ),
                    "translation": torch.as_tensor(
                        alignment.translation, dtype=torch.float32
                    ),
                    "degenerate_scale": bool(alignment.degenerate_scale),
                },
                "overlap_frame_ids": torch.as_tensor(
                    overlap_indices, dtype=torch.int64
                ),
                "overlap_center_rmse": center_rmse,
                "overlap_rotation_rmse_deg": rotation_rmse,
            }
        )

    if not occupied.all() or not np.isfinite(global_c2w).all():
        missing = np.flatnonzero(~occupied).tolist()
        raise ValueError(f"Window stitching did not cover every frame: {missing[:20]}")
    policy = (
        "single_window_identity"
        if len(window_predictions) == 1
        else "sequential_overlap_sim3"
    )
    metadata = {
        "policy": policy,
        "overlap": int(requested_overlap if len(window_predictions) > 1 else 0),
        "min_overlap": 3,
        "pose_selection": "first_prediction_wins_overlap",
        "scale_fallback": (
            "one_when_degenerate" if allow_degenerate_scale else "error"
        ),
    }
    return c2w_to_w2c(global_c2w), global_intrinsics, records, metadata


def build_nwm_frame_pairs(
    num_frames: int,
    *,
    min_offset: int = -64,
    max_offset: int = 64,
    context_size: int = 4,
    len_traj_pred: int = 64,
) -> np.ndarray:
    """Enumerate exactly every pair the current NWM training index can sample."""

    if context_size < 1 or len_traj_pred < 0:
        raise ValueError("context_size must be positive and len_traj_pred non-negative")
    if min_offset > max_offset:
        raise ValueError("min_offset must not exceed max_offset")
    rows: list[tuple[int, int]] = []
    # Mirrors BaseDataset._build_index: range(context_size - 1, N - horizon).
    for current in range(context_size - 1, num_frames - len_traj_pred):
        lower = max(0, current + min_offset)
        upper = min(num_frames - 1, current + max_offset)
        rows.extend((current, target) for target in range(lower, upper + 1))
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _camera_relative_components(
    extrinsics_w2c: np.ndarray, frame_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    pairs = np.asarray(frame_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("frame_pairs must have shape [N,2]")
    if len(pairs) == 0:
        return np.empty((0, 3)), np.empty((0, 3, 3))
    homogeneous = _homogeneous(extrinsics)
    rotations_cw = homogeneous[:, :3, :3]
    centers_w = w2c_to_c2w(extrinsics)[:, :3, 3]
    current = pairs[:, 0]
    target = pairs[:, 1]
    translations_current_camera = np.einsum(
        "nij,nj->ni",
        rotations_cw[current],
        centers_w[target] - centers_w[current],
    )
    rotations_current_from_target = np.einsum(
        "nij,nkj->nik", rotations_cw[current], rotations_cw[target]
    )
    return translations_current_camera, rotations_current_from_target


def project_camera_poses_to_first_frame_navigation_se2(
    extrinsics_w2c: np.ndarray,
) -> np.ndarray:
    """Project absolute cameras to one common x-forward/y-left SE(2) gauge.

    Translation and camera forward vectors are first expressed in camera 0.
    Projecting each absolute pose before taking pair differences explicitly
    drops camera vertical motion, pitch, and roll.  Consequently pair actions
    form a genuine SE(2) group and obey inverse/composition identities.
    """

    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    homogeneous = _homogeneous(extrinsics)
    rotations_cw = homogeneous[:, :3, :3]
    centers_world = w2c_to_c2w(extrinsics)[:, :3, 3]
    rotation_first_from_world = rotations_cw[0]
    centers_first_camera = np.einsum(
        "ij,nj->ni",
        rotation_first_from_world,
        centers_world - centers_world[0],
    )
    rotations_first_from_camera = np.einsum(
        "ij,nkj->nik", rotation_first_from_world, rotations_cw
    )
    forward_first_camera = rotations_first_from_camera[:, :, 2]
    heading = np.arctan2(
        -forward_first_camera[:, 0], forward_first_camera[:, 2]
    )
    result = np.column_stack(
        [centers_first_camera[:, 2], -centers_first_camera[:, 0], heading]
    )
    if not np.isfinite(result).all():
        raise ValueError("Projected first-frame navigation poses are non-finite")
    return result


def relative_actions_from_absolute_se2(
    absolute_poses: np.ndarray, frame_pairs: np.ndarray
) -> np.ndarray:
    poses = np.asarray(absolute_poses, dtype=np.float64)
    pairs = np.asarray(frame_pairs, dtype=np.int64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("absolute_poses must have shape [N,3]")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("frame_pairs must have shape [M,2]")
    if len(pairs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    current = pairs[:, 0]
    target = pairs[:, 1]
    delta_global = poses[target, :2] - poses[current, :2]
    current_heading = poses[current, 2]
    cosine = np.cos(current_heading)
    sine = np.sin(current_heading)
    delta_local_x = cosine * delta_global[:, 0] + sine * delta_global[:, 1]
    delta_local_y = -sine * delta_global[:, 0] + cosine * delta_global[:, 1]
    delta_yaw = (poses[target, 2] - current_heading + np.pi) % (2 * np.pi) - np.pi
    result = np.column_stack([delta_local_x, delta_local_y, delta_yaw])
    if not np.isfinite(result).all():
        raise ValueError("Relative SE(2) actions are non-finite")
    return result.astype(np.float32)


def tartandrive_image_only_scale(
    extrinsics_w2c: np.ndarray,
    *,
    nonzero_epsilon: float = 1e-6,
) -> tuple[float | None, dict[str, Any]]:
    """Scale the median adjacent image-predicted planar step to one unit."""

    num_frames = len(extrinsics_w2c)
    if num_frames < 2:
        return None, {
            "num_adjacent_steps": 0,
            "num_nonzero_steps": 0,
            "nonzero_epsilon_model_units": float(nonzero_epsilon),
            "measurement_frame": "first_frame_navigation_se2",
        }
    absolute_se2 = project_camera_poses_to_first_frame_navigation_se2(
        extrinsics_w2c
    )
    lengths = np.linalg.norm(np.diff(absolute_se2[:, :2], axis=0), axis=1)
    valid = np.isfinite(lengths) & (lengths > nonzero_epsilon)
    diagnostics = {
        "num_adjacent_steps": len(lengths),
        "num_nonzero_steps": int(valid.sum()),
        "nonzero_epsilon_model_units": float(nonzero_epsilon),
        "measurement_frame": "first_frame_navigation_se2",
    }
    if not np.any(valid):
        return None, diagnostics
    median = float(np.median(lengths[valid]))
    diagnostics["median_adjacent_planar_step_model_units"] = median
    return 1.0 / median, diagnostics


def geometry_actions_tartandrive_forward_camera(
    extrinsics_w2c: np.ndarray,
    frame_pairs: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Convert OpenCV relative poses to x-forward/y-left planar actions."""

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    absolute_se2 = project_camera_poses_to_first_frame_navigation_se2(
        extrinsics_w2c
    )
    absolute_se2[:, :2] *= scale
    return relative_actions_from_absolute_se2(absolute_se2, frame_pairs)


def _validate_rigid_transform(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("camera_to_navigation must be a 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("camera_to_navigation has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera_to_navigation rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("camera_to_navigation rotation must have determinant +1")
    return matrix


def geometry_actions_fixed_calibration(
    extrinsics_w2c: np.ndarray,
    frame_pairs: np.ndarray,
    *,
    camera_to_navigation: np.ndarray,
    meters_per_model_unit: float,
    output_divisor_meters: float = 1.0,
) -> np.ndarray:
    """Future calibrated policy; every scale/axis parameter must be explicit."""

    if meters_per_model_unit <= 0 or output_divisor_meters <= 0:
        raise ValueError("Metric scale and output divisor must be positive")
    transform = _validate_rigid_transform(camera_to_navigation)
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    homogeneous = _homogeneous(extrinsics)
    rotations_cw = homogeneous[:, :3, :3]
    centers_world = w2c_to_c2w(extrinsics)[:, :3, 3]
    relative = np.broadcast_to(np.eye(4), (len(extrinsics), 4, 4)).copy()
    relative[:, :3, :3] = np.einsum(
        "ij,nkj->nik", rotations_cw[0], rotations_cw
    )
    relative[:, :3, 3] = (
        np.einsum(
            "ij,nj->ni", rotations_cw[0], centers_world - centers_world[0]
        )
        * meters_per_model_unit
    )
    inverse_transform = np.linalg.inv(transform)
    absolute_navigation_3d = np.einsum(
        "ij,njk,kl->nil", transform, relative, inverse_transform
    )
    absolute_navigation_se2 = np.column_stack(
        [
            absolute_navigation_3d[:, 0, 3] / output_divisor_meters,
            absolute_navigation_3d[:, 1, 3] / output_divisor_meters,
            np.arctan2(
                absolute_navigation_3d[:, 1, 0],
                absolute_navigation_3d[:, 0, 0],
            ),
        ]
    )
    return relative_actions_from_absolute_se2(absolute_navigation_se2, frame_pairs)


def geometry_policy_descriptor(
    *,
    alignment_policy: str,
    degenerate_scale_policy: str,
    nonzero_epsilon: float,
    camera_to_navigation: np.ndarray | None,
    meters_per_model_unit: float | None,
    translation_unit: str,
    waypoint_spacing_meters: float | None,
    min_offset: int,
    max_offset: int,
    context_size: int,
    len_traj_pred: int,
) -> dict[str, Any]:
    """Return a self-fingerprinting description of every action-policy input."""

    configuration: dict[str, Any] = {
        "alignment_policy": str(alignment_policy),
        "degenerate_scale_policy": str(degenerate_scale_policy),
        "nonzero_epsilon": float(nonzero_epsilon),
        "translation_unit": str(translation_unit),
        "min_offset": int(min_offset),
        "max_offset": int(max_offset),
        "context_size": int(context_size),
        "len_traj_pred": int(len_traj_pred),
        "meters_per_model_unit": (
            None if meters_per_model_unit is None else float(meters_per_model_unit)
        ),
        "waypoint_spacing_meters": (
            None
            if waypoint_spacing_meters is None
            else float(waypoint_spacing_meters)
        ),
        "camera_to_navigation": (
            None
            if camera_to_navigation is None
            else np.asarray(camera_to_navigation, dtype=np.float64).tolist()
        ),
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(configuration)).hexdigest()
    return {"configuration": configuration, "fingerprint": fingerprint}


def raw_extraction_descriptor(
    *,
    resolution: int,
    resize_mode: str,
    window_size: int,
    overlap: int,
    inference_path: str,
    dtype: str,
    allow_tf32: bool,
    allow_degenerate_window_scale: bool,
    seed: int,
) -> dict[str, Any]:
    """Fingerprint every setting that can change the raw camera-pose result.

    Policy/version strings are intentionally part of the fingerprint. A future
    preprocessing, window partitioning, or Sim(3)-stitching change therefore
    cannot silently reuse poses produced by the older algorithm.
    """

    if int(resolution) <= 0:
        raise ValueError("resolution must be positive")
    if resize_mode not in {"balanced", "max_size"}:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")
    if int(window_size) < 0 or int(window_size) == 1:
        raise ValueError("window_size must be 0 (whole trajectory) or at least 2")
    if int(window_size) > 0 and not 3 <= int(overlap) < int(window_size):
        raise ValueError("Windowed inference requires 3 <= overlap < window_size")
    if int(overlap) < 0:
        raise ValueError("overlap must be non-negative")
    if inference_path not in {"fast", "full"}:
        raise ValueError(f"Unsupported inference_path: {inference_path}")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError(f"Unsupported effective dtype: {dtype}")

    configuration: dict[str, Any] = {
        "preprocessing": {
            "implementation": "vggt_omega.utils.load_fn.load_and_preprocess_images",
            "resolution": int(resolution),
            "resize_mode": str(resize_mode),
            "patch_size": 16,
        },
        "inference": {
            "path": str(inference_path),
            "effective_dtype": str(dtype),
            "allow_tf32": bool(allow_tf32),
            "strict_checkpoint_load": True,
            "depth_head_executed": inference_path == "full",
            "aggregator_cached_layer_policy": (
                "all_dense_head_layers"
                if inference_path == "full"
                else "final_layer_only"
            ),
        },
        "windowing": {
            "window_size": int(window_size),
            "overlap": int(overlap),
            "partition_policy": "contiguous_trailing_overlap_v1",
            "alignment_policy": "single_identity_or_sequential_overlap_sim3_v1",
            "pose_selection": "first_prediction_wins_overlap",
            "allow_degenerate_scale": bool(allow_degenerate_window_scale),
            "scale_fallback": (
                "one_when_degenerate"
                if allow_degenerate_window_scale
                else "error"
            ),
        },
        "determinism": {"seed": int(seed)},
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(configuration)).hexdigest()
    return {
        "schema_version": 1,
        "configuration": configuration,
        "fingerprint": fingerprint,
    }


def _validated_raw_extraction_descriptor(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    if descriptor.get("schema_version") != 1:
        raise ValueError("Unsupported raw extraction descriptor schema")
    configuration = descriptor.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("Raw extraction descriptor has no configuration mapping")
    configuration_copy = dict(configuration)
    expected = hashlib.sha256(
        _canonical_json_bytes(configuration_copy)
    ).hexdigest()
    if descriptor.get("fingerprint") != expected:
        raise ValueError("Raw extraction descriptor fingerprint is invalid")
    return {
        "schema_version": 1,
        "configuration": configuration_copy,
        "fingerprint": expected,
    }


def build_geometry_payload(
    raw_pose: Mapping[str, Any],
    *,
    source_pose_sha256: str,
    alignment_policy: str = "tartandrive_forward_camera",
    degenerate_scale_policy: str = "empty_only",
    nonzero_epsilon: float = 1e-6,
    camera_to_navigation: np.ndarray | None = None,
    meters_per_model_unit: float | None = None,
    translation_unit: str = "waypoint_spacing_units",
    waypoint_spacing_meters: float | None = None,
    min_offset: int = -64,
    max_offset: int = 64,
    context_size: int = 4,
    len_traj_pred: int = 64,
) -> dict[str, Any]:
    """Create an OfflineMotionStore-compatible canonical geometry artifact."""

    if raw_pose.get("artifact_type") != RAW_ARTIFACT_TYPE or raw_pose.get("complete") is not True:
        raise ValueError("Raw pose artifact is incomplete or has the wrong type")
    extrinsics = torch.as_tensor(raw_pose["extrinsics_w2c"]).cpu().numpy()
    frame_ids = torch.as_tensor(raw_pose["frame_ids"], dtype=torch.int64)
    if not torch.equal(frame_ids, torch.arange(len(frame_ids), dtype=torch.int64)):
        raise ValueError("Raw pose frame_ids must be contiguous from zero")
    pairs = build_nwm_frame_pairs(
        len(frame_ids),
        min_offset=min_offset,
        max_offset=max_offset,
        context_size=context_size,
        len_traj_pred=len_traj_pred,
    )

    scale_metadata: dict[str, Any]
    if alignment_policy == "tartandrive_forward_camera":
        if translation_unit != "waypoint_spacing_units":
            raise ValueError(
                "The image-only median-step policy outputs waypoint_spacing_units"
            )
        scale, diagnostics = tartandrive_image_only_scale(
            extrinsics, nonzero_epsilon=nonzero_epsilon
        )
        if scale is None:
            if degenerate_scale_policy == "unit":
                scale = 1.0
                scale_status = "fallback_unit_scale"
            elif degenerate_scale_policy == "empty_only" and len(pairs) == 0:
                scale = 1.0
                scale_status = "unused_no_nwm_pairs"
            else:
                raise ValueError(
                    "Cannot estimate image-only scale from non-zero adjacent steps"
                )
        else:
            scale_status = "estimated"
        motion = geometry_actions_tartandrive_forward_camera(
            extrinsics, pairs, scale=scale
        )
        scale_metadata = {
            "camera_to_navigation_policy": "opencv_camera_poses_projected_to_first_frame_navigation_se2",
            "camera_to_navigation_axes": {
                "absolute_x": "first_camera_z",
                "absolute_y": "negative_first_camera_x",
                "absolute_heading": "atan2(-forward_x_in_first_camera,forward_z_in_first_camera)",
                "relative_action": "inverse_se2_current_times_se2_goal",
            },
            "scale_policy": "per_trajectory_median_adjacent_step_to_waypoint_unit",
            "scale": float(scale),
            "scale_status": scale_status,
            "scale_diagnostics": diagnostics,
            "ground_truth_usage": "none",
        }
    elif alignment_policy == "fixed":
        if camera_to_navigation is None or meters_per_model_unit is None:
            raise ValueError(
                "fixed policy requires camera_to_navigation and meters_per_model_unit"
            )
        if translation_unit == "meters":
            divisor = 1.0
        elif translation_unit == "waypoint_spacing_units":
            if waypoint_spacing_meters is None or waypoint_spacing_meters <= 0:
                raise ValueError(
                    "waypoint_spacing_meters is required for waypoint units"
                )
            divisor = float(waypoint_spacing_meters)
        else:
            raise ValueError(f"Unsupported translation_unit: {translation_unit}")
        calibration = _validate_rigid_transform(camera_to_navigation)
        motion = geometry_actions_fixed_calibration(
            extrinsics,
            pairs,
            camera_to_navigation=calibration,
            meters_per_model_unit=float(meters_per_model_unit),
            output_divisor_meters=divisor,
        )
        scale_metadata = {
            "camera_to_navigation_policy": "explicit_rigid_transform",
            "camera_to_navigation": torch.as_tensor(
                calibration, dtype=torch.float32
            ),
            "scale_policy": "fixed_meters_per_model_unit",
            "scale": float(meters_per_model_unit),
            "scale_status": "configured",
            "ground_truth_usage": "none",
        }
    else:
        raise ValueError(f"Unsupported alignment_policy: {alignment_policy}")

    policy_descriptor = geometry_policy_descriptor(
        alignment_policy=alignment_policy,
        degenerate_scale_policy=degenerate_scale_policy,
        nonzero_epsilon=nonzero_epsilon,
        camera_to_navigation=camera_to_navigation,
        meters_per_model_unit=meters_per_model_unit,
        translation_unit=translation_unit,
        waypoint_spacing_meters=waypoint_spacing_meters,
        min_offset=min_offset,
        max_offset=max_offset,
        context_size=context_size,
        len_traj_pred=len_traj_pred,
    )

    payload: dict[str, Any] = {
        "schema_version": GEOMETRY_MOTION_SCHEMA_VERSION,
        "motion_type": "geometry",
        "dataset_name": str(raw_pose["dataset_name"]),
        "trajectory_name": str(raw_pose["trajectory_name"]),
        "pair_direction": "current_to_goal",
        "normalization": "raw",
        "coordinate_frame": "current_navigation_frame",
        "translation_unit": translation_unit,
        "yaw_unit": "radians",
        "components": ["delta_x", "delta_y", "delta_yaw"],
        "frame_pairs": torch.as_tensor(pairs, dtype=torch.int64),
        "motion": torch.as_tensor(motion, dtype=torch.float32),
        "source_artifact_type": RAW_ARTIFACT_TYPE,
        "source_pose_sha256": str(source_pose_sha256),
        "frame_list_sha256": str(raw_pose["frame_list_sha256"]),
        "model_id": str(raw_pose["model_id"]),
        "checkpoint_sha256": str(raw_pose["checkpoint_sha256"]),
        "alignment_policy": str(alignment_policy),
        "policy_descriptor": policy_descriptor,
        "pair_generation": {
            "min_offset": int(min_offset),
            "max_offset": int(max_offset),
            "context_size": int(context_size),
            "len_traj_pred": int(len_traj_pred),
            "coverage": "exact_BaseDataset_training_index_domain",
        },
        "complete": True,
    }
    payload.update(scale_metadata)
    return payload


def _git_revision(directory: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class VGGTOmegaCameraExtractor:
    """Strict-loaded VGGT-Omega with a depth-free camera fast path."""

    def __init__(
        self,
        *,
        third_party_root: str | os.PathLike[str],
        checkpoint_path: str | os.PathLike[str],
        device: str = "cuda",
        dtype: str = "bfloat16",
        expected_code_revision: str = PINNED_CODE_REVISION,
        retain_dense_head: bool = False,
        allow_tf32: bool = True,
        preprocess_workers: int = 1,
    ) -> None:
        self.third_party_root = Path(third_party_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not (self.third_party_root / "vggt_omega").is_dir():
            raise FileNotFoundError(
                f"VGGT-Omega package not found under {self.third_party_root}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.code_revision = _git_revision(self.third_party_root)
        if expected_code_revision and self.code_revision != expected_code_revision:
            raise RuntimeError(
                f"VGGT-Omega code revision {self.code_revision} != expected "
                f"{expected_code_revision}"
            )
        if str(self.third_party_root) not in sys.path:
            sys.path.insert(0, str(self.third_party_root))
        models = importlib.import_module("vggt_omega.models")
        load_module = importlib.import_module("vggt_omega.utils.load_fn")
        pose_module = importlib.import_module("vggt_omega.utils.pose_enc")
        self._preprocess = load_module.load_and_preprocess_images
        self._pad_preprocessed = load_module._pad_images_to_common_size
        self._decode_camera = pose_module.encoding_to_camera
        self.preprocess_workers = int(preprocess_workers)
        if self.preprocess_workers < 1:
            raise ValueError("preprocess_workers must be at least one")
        self._preprocess_executor = (
            ThreadPoolExecutor(max_workers=self.preprocess_workers)
            if self.preprocess_workers > 1
            else None
        )

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("VGGT-Omega inference requires an available CUDA device")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        if dtype not in dtype_map:
            raise ValueError("dtype must be 'bfloat16' or 'float16'")
        self.amp_dtype = dtype_map[dtype]
        # The pinned official forward selects this dtype internally; reject a
        # contradictory flag instead of recording provenance that did not run.
        self.effective_dtype = (
            "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        )
        if dtype != self.effective_dtype:
            raise RuntimeError(
                f"Requested dtype={dtype}, but the pinned official forward will use "
                f"{self.effective_dtype} on this device"
            )
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

        model = models.VGGTOmega(
            enable_camera=True, enable_depth=True, enable_alignment=False
        ).eval()
        try:
            state_dict = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (TypeError, RuntimeError):
            state_dict = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise TypeError("Official checkpoint must be a non-empty raw state_dict")
        if not all(isinstance(key, str) for key in state_dict):
            raise TypeError("Checkpoint state_dict contains non-string keys")
        # Full model first: missing or unexpected dense-head keys are fatal.
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        self.retain_dense_head = bool(retain_dense_head)
        # Strict-load happened against the complete architecture.  Non-parity
        # ranks can now drop the dense head before the costly CPU->GPU transfer.
        if not self.retain_dense_head:
            model.dense_head = None
        self.model = model.to(self.device).eval()
        self._full_cached_layer_indices = set(
            self.model.aggregator.cached_layer_indices
        )
        self._camera_cached_layer_indices = {self.model.aggregator.depth - 1}
        self.strict_checkpoint_load = True
        if not self.retain_dense_head:
            self.model.aggregator.cached_layer_indices = set(
                self._camera_cached_layer_indices
            )
        self._dense_head_available = self.model.dense_head is not None

    def preprocess(
        self,
        image_paths: Sequence[str | os.PathLike[str]],
        *,
        resize_mode: str,
        resolution: int,
    ) -> torch.Tensor:
        paths = [str(path) for path in image_paths]
        if self._preprocess_executor is None or len(paths) < 2:
            images = self._preprocess(
                paths,
                mode=resize_mode,
                image_resolution=int(resolution),
                patch_size=16,
            )
        else:
            def preprocess_one(path: str) -> torch.Tensor:
                return self._preprocess(
                    [path],
                    mode=resize_mode,
                    image_resolution=int(resolution),
                    patch_size=16,
                )[0]

            image_list = list(self._preprocess_executor.map(preprocess_one, paths))
            shapes = {(int(image.shape[1]), int(image.shape[2])) for image in image_list}
            if len(shapes) > 1:
                image_list = self._pad_preprocessed(image_list, shapes)
            images = torch.stack(image_list)
        return images.pin_memory().to(self.device, non_blocking=True)

    def _forward(self, images: torch.Tensor, *, full: bool) -> Mapping[str, torch.Tensor]:
        if full:
            if self.model.dense_head is None:
                raise RuntimeError(
                    "Full inference requested after the dense head was discarded"
                )
            self.model.aggregator.cached_layer_indices = set(
                self._full_cached_layer_indices
            )
            return self.model(images)

        dense_head = self.model.dense_head
        cached_layer_indices = set(self.model.aggregator.cached_layer_indices)
        self.model.dense_head = None
        self.model.aggregator.cached_layer_indices = set(
            self._camera_cached_layer_indices
        )
        try:
            return self.model(images)
        finally:
            if self.retain_dense_head:
                self.model.dense_head = dense_head
                self.model.aggregator.cached_layer_indices = cached_layer_indices

    @torch.inference_mode()
    def infer_images(
        self,
        images: torch.Tensor,
        *,
        full: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        prediction = self._forward(images, full=full)
        if "pose_enc" not in prediction:
            raise RuntimeError("VGGT-Omega did not return pose_enc")
        image_size_hw = (int(images.shape[-2]), int(images.shape[-1]))
        extrinsics, intrinsics = self._decode_camera(
            prediction["pose_enc"], image_size_hw
        )
        if extrinsics.shape[0] != 1 or intrinsics.shape[0] != 1:
            raise RuntimeError("Only one trajectory per inference call is supported")
        extrinsics_np = extrinsics[0].float().cpu().numpy()
        intrinsics_np = intrinsics[0].float().cpu().numpy()
        if not np.isfinite(extrinsics_np).all() or not np.isfinite(intrinsics_np).all():
            raise ValueError("VGGT-Omega returned non-finite camera parameters")
        return extrinsics_np, intrinsics_np, image_size_hw

    @torch.inference_mode()
    def compare_full_and_fast(
        self,
        images: torch.Tensor,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> dict[str, Any]:
        if not self.retain_dense_head or self.model.dense_head is None:
            raise RuntimeError("Full-vs-fast comparison requires retain_dense_head=True")
        fast_extrinsics, fast_intrinsics, _ = self.infer_images(images, full=False)
        full_extrinsics, full_intrinsics, _ = self.infer_images(images, full=True)
        extrinsic_error = float(np.max(np.abs(fast_extrinsics - full_extrinsics)))
        intrinsic_error = float(np.max(np.abs(fast_intrinsics - full_intrinsics)))
        allclose = bool(
            np.allclose(fast_extrinsics, full_extrinsics, atol=atol, rtol=rtol)
            and np.allclose(fast_intrinsics, full_intrinsics, atol=atol, rtol=rtol)
        )
        return {
            "enabled": True,
            "num_frames": int(images.shape[0]),
            "atol": float(atol),
            "rtol": float(rtol),
            "max_abs_extrinsics": extrinsic_error,
            "max_abs_intrinsics": intrinsic_error,
            "allclose": allclose,
        }

    def discard_dense_head(self) -> None:
        """Release depth-head parameters after the one-time parity check."""

        self.model.dense_head = None
        self.model.aggregator.cached_layer_indices = set(
            self._camera_cached_layer_indices
        )
        self.retain_dense_head = False
        self._dense_head_available = False
        torch.cuda.empty_cache()

    def extract_trajectory(
        self,
        *,
        image_paths: Sequence[str | os.PathLike[str]],
        window_size: int,
        overlap: int,
        resize_mode: str,
        resolution: int,
        full: bool = False,
        allow_degenerate_window_scale: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[dict[str, Any]], dict[str, Any]]:
        slices = make_window_slices(len(image_paths), window_size, overlap)
        predictions: list[dict[str, Any]] = []
        common_size: tuple[int, int] | None = None
        for start, end in slices:
            images = self.preprocess(
                image_paths[start:end],
                resize_mode=resize_mode,
                resolution=resolution,
            )
            extrinsics, intrinsics, image_size_hw = self.infer_images(
                images, full=full
            )
            if common_size is None:
                common_size = image_size_hw
            elif image_size_hw != common_size:
                raise ValueError("Preprocessed image size changed between windows")
            predictions.append(
                {
                    "start_index": start,
                    "end_index_exclusive": end,
                    "extrinsics_w2c": extrinsics,
                    "intrinsics": intrinsics,
                    "image_size_hw": image_size_hw,
                }
            )
            del images
        stitched_extrinsics, stitched_intrinsics, records, alignment = stitch_pose_windows(
            predictions,
            num_frames=len(image_paths),
            requested_overlap=overlap,
            allow_degenerate_scale=allow_degenerate_window_scale,
        )
        assert common_size is not None
        return (
            stitched_extrinsics,
            stitched_intrinsics,
            common_size,
            records,
            alignment,
        )


def build_raw_pose_payload(
    *,
    dataset_name: str,
    trajectory_name: str,
    frame_list_sha256: str,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    image_size_hw: tuple[int, int],
    windows: list[dict[str, Any]],
    window_alignment: Mapping[str, Any],
    checkpoint_sha256: str,
    code_revision: str,
    model_revision: str,
    resize_mode: str,
    resolution: int,
    inference_path: str,
    dtype: str,
    fast_full_comparison: Mapping[str, Any] | None = None,
    input_manifest_sha256: str | None = None,
    checkpoint_manifest_sha256: str | None = None,
    extraction_descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    extrinsics = torch.as_tensor(extrinsics_w2c, dtype=torch.float32)
    intrinsics_tensor = torch.as_tensor(intrinsics, dtype=torch.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError("Raw extrinsics must have shape [N,3,4]")
    if intrinsics_tensor.shape != (len(extrinsics), 3, 3):
        raise ValueError("Raw intrinsics must have shape [N,3,3]")
    if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics_tensor).all():
        raise ValueError("Raw camera artifact contains non-finite values")
    validated_descriptor: dict[str, Any] | None = None
    if extraction_descriptor is not None:
        validated_descriptor = _validated_raw_extraction_descriptor(
            extraction_descriptor
        )
        configuration = validated_descriptor["configuration"]
        preprocessing = configuration.get("preprocessing")
        inference = configuration.get("inference")
        windowing = configuration.get("windowing")
        if not all(
            isinstance(value, Mapping)
            for value in (preprocessing, inference, windowing)
        ):
            raise ValueError("Raw extraction descriptor sections are invalid")
        if (
            preprocessing.get("resolution") != int(resolution)
            or preprocessing.get("resize_mode") != str(resize_mode)
            or inference.get("path") != str(inference_path)
            or inference.get("effective_dtype") != str(dtype)
        ):
            raise ValueError(
                "Raw extraction descriptor disagrees with payload provenance"
            )
        if window_alignment.get("pose_selection") != windowing.get(
            "pose_selection"
        ) or window_alignment.get("scale_fallback") != windowing.get(
            "scale_fallback"
        ):
            raise ValueError(
                "Window alignment metadata disagrees with extraction descriptor"
            )
        alignment_policy = window_alignment.get("policy")
        if alignment_policy not in {
            "single_window_identity",
            "sequential_overlap_sim3",
        }:
            raise ValueError(f"Unsupported window alignment policy: {alignment_policy}")
        expected_overlap = (
            0
            if alignment_policy == "single_window_identity"
            else int(windowing["overlap"])
        )
        if window_alignment.get("overlap") != expected_overlap:
            raise ValueError(
                "Window overlap metadata disagrees with extraction descriptor"
            )
    payload = {
        "schema_version": RAW_POSE_SCHEMA_VERSION,
        "artifact_type": RAW_ARTIFACT_TYPE,
        "dataset_name": str(dataset_name),
        "trajectory_name": str(trajectory_name),
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": str(model_revision),
        "checkpoint_sha256": str(checkpoint_sha256),
        "code_revision": str(code_revision),
        "pose_convention": "world_to_camera_opencv",
        "frame_ids": torch.arange(len(extrinsics), dtype=torch.int64),
        "frame_list_sha256": str(frame_list_sha256),
        "extrinsics_w2c": extrinsics.contiguous(),
        "intrinsics": intrinsics_tensor.contiguous(),
        "image_size_hw": torch.as_tensor(image_size_hw, dtype=torch.int64),
        "preprocessing": {
            "implementation": "vggt_omega.utils.load_fn.load_and_preprocess_images",
            "resize_mode": str(resize_mode),
            "image_resolution": int(resolution),
            "patch_size": 16,
        },
        "inference": {
            "path": str(inference_path),
            "effective_dtype": str(dtype),
            "strict_checkpoint_load": True,
            "depth_head_executed": inference_path == "full",
            "aggregator_cached_layer_policy": (
                "all_dense_head_layers" if inference_path == "full" else "final_layer_only"
            ),
        },
        "windows": windows,
        "window_alignment": dict(window_alignment),
        "fast_full_comparison": dict(fast_full_comparison or {"enabled": False}),
        "complete": True,
    }
    if validated_descriptor is not None:
        payload["extraction_descriptor"] = validated_descriptor
    if input_manifest_sha256 is not None:
        payload["input_manifest_sha256"] = str(input_manifest_sha256)
    if checkpoint_manifest_sha256 is not None:
        payload["checkpoint_manifest_sha256"] = str(checkpoint_manifest_sha256)
    return payload


def raw_pose_matches(
    path: str | os.PathLike[str],
    *,
    dataset_name: str,
    trajectory_name: str,
    frame_list_sha256: str,
    checkpoint_sha256: str,
    code_revision: str,
    checkpoint_manifest_sha256: str | None = None,
    model_revision: str | None = None,
    input_manifest_sha256: str | None = None,
    extraction_fingerprint: str | None = None,
) -> bool:
    if not Path(path).is_file():
        return False
    try:
        payload = safe_torch_load(path)
        metadata_matches = bool(
            payload.get("schema_version") == RAW_POSE_SCHEMA_VERSION
            and payload.get("artifact_type") == RAW_ARTIFACT_TYPE
            and payload.get("dataset_name") == dataset_name
            and payload.get("trajectory_name") == trajectory_name
            and payload.get("frame_list_sha256") == frame_list_sha256
            and payload.get("checkpoint_sha256") == checkpoint_sha256
            and payload.get("code_revision") == code_revision
            and payload.get("pose_convention") == "world_to_camera_opencv"
            and payload.get("complete") is True
        )
        if not metadata_matches:
            return False
        if (
            checkpoint_manifest_sha256 is not None
            and payload.get("checkpoint_manifest_sha256")
            != checkpoint_manifest_sha256
        ):
            return False
        if (
            model_revision is not None
            and payload.get("model_revision") != model_revision
        ):
            return False
        if (
            input_manifest_sha256 is not None
            and payload.get("input_manifest_sha256") != input_manifest_sha256
        ):
            return False
        if extraction_fingerprint is not None:
            descriptor = payload.get("extraction_descriptor")
            if not isinstance(descriptor, Mapping):
                return False
            validated_descriptor = _validated_raw_extraction_descriptor(descriptor)
            if validated_descriptor["fingerprint"] != extraction_fingerprint:
                return False
            configuration = validated_descriptor["configuration"]
            preprocessing = configuration.get("preprocessing")
            inference = configuration.get("inference")
            windowing = configuration.get("windowing")
            payload_preprocessing = payload.get("preprocessing")
            payload_inference = payload.get("inference")
            window_alignment = payload.get("window_alignment")
            if not all(
                isinstance(value, Mapping)
                for value in (
                    preprocessing,
                    inference,
                    windowing,
                    payload_preprocessing,
                    payload_inference,
                    window_alignment,
                )
            ):
                return False
            alignment_policy = window_alignment.get("policy")
            expected_overlap = (
                0
                if alignment_policy == "single_window_identity"
                else windowing.get("overlap")
            )
            if not (
                payload_preprocessing.get("image_resolution")
                == preprocessing.get("resolution")
                and payload_preprocessing.get("resize_mode")
                == preprocessing.get("resize_mode")
                and payload_preprocessing.get("implementation")
                == preprocessing.get("implementation")
                and payload_preprocessing.get("patch_size")
                == preprocessing.get("patch_size")
                and payload_inference.get("path") == inference.get("path")
                and payload_inference.get("effective_dtype")
                == inference.get("effective_dtype")
                and payload_inference.get("strict_checkpoint_load")
                == inference.get("strict_checkpoint_load")
                and payload_inference.get("depth_head_executed")
                == inference.get("depth_head_executed")
                and payload_inference.get("aggregator_cached_layer_policy")
                == inference.get("aggregator_cached_layer_policy")
                and alignment_policy
                in {"single_window_identity", "sequential_overlap_sim3"}
                and window_alignment.get("overlap") == expected_overlap
                and window_alignment.get("pose_selection")
                == windowing.get("pose_selection")
                and window_alignment.get("scale_fallback")
                == windowing.get("scale_fallback")
            ):
                return False
        frame_ids = torch.as_tensor(payload.get("frame_ids"), dtype=torch.int64)
        extrinsics = torch.as_tensor(payload.get("extrinsics_w2c"))
        intrinsics = torch.as_tensor(payload.get("intrinsics"))
        num_frames = frame_ids.numel()
        return bool(
            frame_ids.ndim == 1
            and torch.equal(frame_ids, torch.arange(num_frames))
            and extrinsics.shape == (num_frames, 3, 4)
            and intrinsics.shape == (num_frames, 3, 3)
            and torch.isfinite(extrinsics).all()
            and torch.isfinite(intrinsics).all()
            and isinstance(payload.get("windows"), list)
            and isinstance(payload.get("window_alignment"), Mapping)
        )
    # A resume probe must treat any corrupt or incompatible artifact as a miss;
    # the strict loader reports the concrete error if regeneration is disabled.
    except Exception:  # noqa: BLE001
        return False


def geometry_payload_matches(
    path: str | os.PathLike[str],
    *,
    dataset_name: str,
    trajectory_name: str,
    frame_list_sha256: str,
    checkpoint_sha256: str,
    alignment_policy: str,
    policy_fingerprint: str | None = None,
) -> bool:
    if not Path(path).is_file():
        return False
    try:
        payload = safe_torch_load(path)
        metadata_matches = bool(
            payload.get("schema_version") == GEOMETRY_MOTION_SCHEMA_VERSION
            and payload.get("motion_type") == "geometry"
            and payload.get("dataset_name") == dataset_name
            and payload.get("trajectory_name") == trajectory_name
            and payload.get("frame_list_sha256") == frame_list_sha256
            and payload.get("checkpoint_sha256") == checkpoint_sha256
            and payload.get("alignment_policy") == alignment_policy
            and payload.get("complete") is True
        )
        if not metadata_matches:
            return False
        if policy_fingerprint is not None:
            descriptor = payload.get("policy_descriptor")
            if not isinstance(descriptor, Mapping):
                return False
            if descriptor.get("fingerprint") != policy_fingerprint:
                return False
        pairs = torch.as_tensor(payload.get("frame_pairs"), dtype=torch.int64)
        motion = torch.as_tensor(payload.get("motion"), dtype=torch.float32)
        if pairs.ndim != 2 or pairs.shape[1:] != (2,):
            return False
        if motion.shape != (len(pairs), 3) or not torch.isfinite(motion).all():
            return False
        return len({tuple(row) for row in pairs.tolist()}) == len(pairs)
    # See raw_pose_matches: malformed caches are not eligible for resume.
    except Exception:  # noqa: BLE001
        return False


def frame_paths_from_manifest(
    trajectory: Mapping[str, Any], data_root: str | os.PathLike[str]
) -> list[Path]:
    root = Path(data_root).resolve()
    result: list[Path] = []
    for expected_id, record in enumerate(trajectory["frame_records"]):
        if int(record["frame_id"]) != expected_id:
            raise ValueError("Manifest frame records are not contiguous")
        path = (root / str(record["relative_path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe manifest frame path: {path}") from exc
        result.append(path)
    return result


def timed_checkpoint_sha256(
    checkpoint_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str] | None = None,
) -> tuple[str, float]:
    """Hash a multi-GB checkpoint once; reuse only when size and mtime match."""

    checkpoint = Path(checkpoint_path).resolve()
    stat = checkpoint.stat()
    if cache_path and Path(cache_path).is_file():
        try:
            cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            if (
                cached.get("path") == str(checkpoint)
                and cached.get("size_bytes") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
            ):
                return str(cached["sha256"]), 0.0
        except (OSError, ValueError, KeyError, TypeError):
            pass
    started = time.monotonic()
    digest = sha256_file(checkpoint)
    elapsed = time.monotonic() - started
    if cache_path:
        atomic_json_dump(
            {
                "path": str(checkpoint),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            },
            cache_path,
        )
    return digest, elapsed


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "GEOMETRY_MOTION_SCHEMA_VERSION",
    "MANIFEST_ARTIFACT_TYPE",
    "PINNED_CODE_REVISION",
    "RAW_ARTIFACT_TYPE",
    "RAW_POSE_SCHEMA_VERSION",
    "VGGTOmegaCameraExtractor",
    "atomic_json_dump",
    "atomic_torch_save",
    "build_frame_records",
    "build_geometry_payload",
    "build_input_manifest",
    "build_nwm_frame_pairs",
    "build_raw_pose_payload",
    "deterministic_shard",
    "discover_numbered_frames",
    "frame_paths_from_manifest",
    "geometry_actions_tartandrive_forward_camera",
    "geometry_payload_matches",
    "geometry_policy_descriptor",
    "make_window_slices",
    "project_camera_poses_to_first_frame_navigation_se2",
    "raw_pose_matches",
    "relative_actions_from_absolute_se2",
    "safe_torch_load",
    "sha256_file",
    "stitch_pose_windows",
    "tartandrive_image_only_scale",
    "timed_checkpoint_sha256",
    "validate_input_manifest",
]
