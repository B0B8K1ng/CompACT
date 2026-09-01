"""Validation for offline VGGT camera poses and canonical geometry actions.

The optional TartanDrive ground truth is used only to produce validation metrics.
No aligned pose or ground-truth-derived value is ever written back to an input
artifact, which keeps the image-only proxy-action pipeline isolated from labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

RAW_SCHEMA_VERSION = 1
RAW_ARTIFACT_TYPE = "vggt_omega_camera_poses"
RAW_POSE_CONVENTION = "world_to_camera_opencv"
GEOMETRY_SCHEMA_VERSION = 1
VALIDATION_SCHEMA_VERSION = 1
VALIDATOR_VERSION = "1.0"

_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ValidationThresholds:
    """Numerical limits used to turn validation metrics into pass/fail."""

    max_so3_orthogonality: float | None = 1e-3
    max_so3_det_error: float | None = 1e-3
    max_identity_translation: float | None = 1e-4
    max_identity_yaw_deg: float | None = 1e-3
    max_inverse_translation: float | None = 1e-3
    max_inverse_yaw_deg: float | None = 1e-2
    max_composition_translation: float | None = 1e-3
    max_composition_yaw_deg: float | None = 1e-2
    max_overlap_center_rmse: float | None = 1e-3
    max_overlap_rotation_rmse_deg: float | None = 1e-2
    max_pose_action_translation: float | None = 1e-4
    max_pose_action_yaw_deg: float | None = 1e-3

    # None means report the GT metric without using it as an exit criterion.
    max_gt_ate_rmse: float | None = None
    max_gt_rpe_translation_rmse: float | None = None
    max_gt_rpe_yaw_mae_deg: float | None = None
    max_gt_rpe_rotation_mae_deg: float | None = None
    max_action_translation_rmse: float | None = None
    max_action_yaw_mae_deg: float | None = None
    min_action_direction_cosine: float | None = None
    min_action_yaw_sign_agreement: float | None = None


@dataclass(frozen=True)
class ValidationOptions:
    strict: bool = True
    verify_frame_hashes: bool = True
    require_algebra_coverage: bool = False
    max_composition_checks: int = 10_000
    gt_rpe_delta: int = 1
    waypoint_spacing: float | None = None
    thresholds: ValidationThresholds = ValidationThresholds()


class _Report:
    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}
        self.issues: list[dict[str, Any]] = []

    def issue(self, code: str, message: str, *, error: bool = True) -> None:
        self.issues.append(
            {
                "code": code,
                "severity": "error" if error else "warning",
                "message": message,
            }
        )

    def require(self, condition: bool, code: str, message: str) -> bool:
        if not condition:
            self.issue(code, message)
            return False
        return True

    def threshold_max(self, metric: str, value: float, limit: float | None) -> None:
        self.metrics[metric] = _finite_float(value)
        if limit is not None and (not math.isfinite(value) or value > limit):
            self.issue(
                f"threshold.{metric}",
                f"{metric}={value:.8g} exceeds maximum {limit:.8g}",
            )

    def threshold_min(self, metric: str, value: float, limit: float | None) -> None:
        self.metrics[metric] = _finite_float(value)
        if limit is not None and (not math.isfinite(value) or value < limit):
            self.issue(
                f"threshold.{metric}",
                f"{metric}={value:.8g} is below minimum {limit:.8g}",
            )


def _finite_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _load_mapping(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = os.fspath(path)
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if not path.endswith((".pt", ".pth")):
        raise ValueError(f"Unsupported artifact extension: {path}")
    import torch  # Lazy: NumPy-only callers do not pay torch import cost.

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Artifact must contain a mapping: {path}")
    return dict(payload)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    value = _scalar(value)
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    )


def frame_list_records(
    trajectory_dir: str | os.PathLike[str],
    trajectory_name: str,
    frame_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Build the canonical content-hashed ordered frame manifest."""

    root = Path(trajectory_dir)
    records = []
    for frame_id in sorted(int(value) for value in frame_ids):
        path = root / f"{frame_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source frame: {path}")
        records.append(
            {
                "frame_id": frame_id,
                "relative_path": f"{trajectory_name}/{frame_id}.jpg",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def frame_list_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rotation_errors(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    identity = np.eye(3, dtype=np.float64)
    orthogonality = np.linalg.norm(
        np.swapaxes(rotations, -1, -2) @ rotations - identity, axis=(-2, -1)
    )
    determinant = np.abs(np.linalg.det(rotations) - 1.0)
    return orthogonality, determinant


def _rotation_angle_deg(rotations: np.ndarray) -> np.ndarray:
    # Serialized float32 rotations are only approximately orthogonal. Project
    # relative matrices back to SO(3) before acos so round-off in R.T @ R does
    # not appear as a false overlap/rotation residual.
    rotations = np.asarray(rotations, dtype=np.float64)
    left, _, right_t = np.linalg.svd(rotations)
    projected = left @ right_t
    reflected = np.linalg.det(projected) < 0
    if np.any(reflected):
        left = left.copy()
        left[reflected, :, -1] *= -1
        projected = left @ right_t
    trace = np.trace(projected, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _w2c_to_c2w(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations_w2c = extrinsics[..., :3]
    translations_w2c = extrinsics[..., 3]
    rotations_c2w = np.swapaxes(rotations_w2c, -1, -2)
    centers = -np.einsum("nij,nj->ni", rotations_c2w, translations_w2c)
    return rotations_c2w, centers


def _wrap_angle(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _se2(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    result = np.zeros((actions.shape[0], 3, 3), dtype=np.float64)
    cosine = np.cos(actions[:, 2])
    sine = np.sin(actions[:, 2])
    result[:, 0, 0] = cosine
    result[:, 0, 1] = -sine
    result[:, 1, 0] = sine
    result[:, 1, 1] = cosine
    result[:, :2, 2] = actions[:, :2]
    result[:, 2, 2] = 1.0
    return result


def _se2_error(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(expected) @ actual
    translation = float(np.linalg.norm(delta[:2, 2]))
    yaw = abs(float(math.atan2(delta[1, 0], delta[0, 0])))
    return translation, math.degrees(yaw)


def _canonical_raw_keys() -> tuple[str, ...]:
    return (
        "schema_version",
        "artifact_type",
        "dataset_name",
        "trajectory_name",
        "model_id",
        "model_revision",
        "checkpoint_sha256",
        "code_revision",
        "pose_convention",
        "frame_ids",
        "frame_list_sha256",
        "extrinsics_w2c",
        "intrinsics",
        "image_size_hw",
        "preprocessing",
        "windows",
        "window_alignment",
        "complete",
    )


def _validate_raw_pose(
    payload: Mapping[str, Any],
    *,
    dataset_name: str,
    trajectory_name: str,
    trajectory_dir: str | None,
    options: ValidationOptions,
    report: _Report,
) -> dict[str, Any] | None:
    missing = [key for key in _canonical_raw_keys() if key not in payload]
    if missing and options.strict:
        report.issue("raw.schema.missing", f"Missing canonical raw fields: {missing}")
        return None

    def value(canonical: str, *aliases: str) -> Any:
        for key in (canonical, *aliases):
            if key in payload:
                return payload[key]
        raise KeyError(canonical)

    try:
        frame_ids = _array(value("frame_ids", "frame_indices"), np.int64)
        extrinsics = _array(
            value("extrinsics_w2c", "camera_extrinsics", "extrinsics"), np.float64
        )
        intrinsics = _array(value("intrinsics", "camera_intrinsics"), np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        report.issue("raw.schema.decode", f"Cannot decode raw pose arrays: {exc}")
        return None

    expected_scalars = {
        "schema_version": RAW_SCHEMA_VERSION,
        "artifact_type": RAW_ARTIFACT_TYPE,
        "dataset_name": dataset_name,
        "trajectory_name": trajectory_name,
        "pose_convention": RAW_POSE_CONVENTION,
        "complete": True,
    }
    for key, expected in expected_scalars.items():
        if key not in payload:
            if options.strict:
                report.issue("raw.schema.missing", f"Missing raw field {key!r}")
            continue
        actual = _scalar(payload[key])
        report.require(
            actual == expected,
            f"raw.schema.{key}",
            f"raw {key} mismatch: {actual!r} != {expected!r}",
        )

    for key in ("model_id", "model_revision", "checkpoint_sha256", "code_revision"):
        if key in payload:
            actual = _scalar(payload[key])
            report.require(
                isinstance(actual, str) and bool(actual),
                f"raw.schema.{key}",
                f"raw {key} must be a non-empty string",
            )
    if "checkpoint_sha256" in payload:
        report.require(
            _is_sha256(payload["checkpoint_sha256"]),
            "raw.schema.checkpoint_sha256",
            "checkpoint_sha256 must contain 64 hexadecimal characters",
        )

    shape_ok = True
    shape_ok &= report.require(
        frame_ids.ndim == 1, "raw.frame_ids.shape", "frame_ids must have shape [N]"
    )
    if frame_ids.ndim != 1:
        return None
    num_frames = len(frame_ids)
    report.metrics["raw.num_frames"] = num_frames
    shape_ok &= report.require(
        extrinsics.shape == (num_frames, 3, 4),
        "raw.extrinsics.shape",
        f"extrinsics_w2c must have shape [{num_frames},3,4], got {extrinsics.shape}",
    )
    shape_ok &= report.require(
        intrinsics.shape == (num_frames, 3, 3),
        "raw.intrinsics.shape",
        f"intrinsics must have shape [{num_frames},3,3], got {intrinsics.shape}",
    )
    if not shape_ok:
        return None
    report.require(num_frames > 0, "raw.empty", "Raw pose artifact has no frames")
    expected_ids = np.arange(num_frames, dtype=np.int64)
    report.require(
        np.array_equal(frame_ids, expected_ids),
        "raw.frame_ids.contiguous",
        "frame_ids must be the ordered authoritative range 0..N-1",
    )
    report.require(
        np.isfinite(extrinsics).all(),
        "raw.extrinsics.finite",
        "extrinsics_w2c contains non-finite values",
    )
    report.require(
        np.isfinite(intrinsics).all(),
        "raw.intrinsics.finite",
        "intrinsics contains non-finite values",
    )
    report.require(
        bool(np.all(intrinsics[:, 0, 0] > 0) and np.all(intrinsics[:, 1, 1] > 0)),
        "raw.intrinsics.focal_length",
        "intrinsics focal lengths must be positive",
    )

    rotations = extrinsics[:, :3, :3]
    orthogonality, determinant = _rotation_errors(rotations)
    report.threshold_max(
        "raw.so3_orthogonality_max",
        float(np.max(orthogonality)),
        options.thresholds.max_so3_orthogonality,
    )
    report.threshold_max(
        "raw.so3_det_error_max",
        float(np.max(determinant)),
        options.thresholds.max_so3_det_error,
    )

    if "image_size_hw" in payload:
        size_hw = _array(payload["image_size_hw"], np.int64)
        report.require(
            size_hw.shape == (2,) and bool(np.all(size_hw > 0)),
            "raw.image_size_hw",
            "image_size_hw must contain two positive integers",
        )
    for key in ("preprocessing", "window_alignment"):
        if key in payload:
            report.require(
                isinstance(payload[key], Mapping),
                f"raw.{key}.type",
                f"{key} must be a mapping",
            )

    stored_frame_hash = _scalar(payload.get("frame_list_sha256"))
    report.require(
        _is_sha256(stored_frame_hash),
        "raw.frame_list_sha256.format",
        "frame_list_sha256 must contain 64 hexadecimal characters",
    )
    if options.verify_frame_hashes:
        if trajectory_dir is None:
            report.issue(
                "raw.frame_list_sha256.skipped",
                "Cannot recompute frame_list_sha256 without a data root",
                error=options.strict,
            )
        else:
            try:
                records = frame_list_records(trajectory_dir, trajectory_name, frame_ids)
                actual_hash = frame_list_sha256(records)
                report.metrics["raw.frame_hashes_verified"] = len(records)
                report.require(
                    actual_hash == stored_frame_hash,
                    "raw.frame_list_sha256.mismatch",
                    f"Source frame hash mismatch: {actual_hash} != {stored_frame_hash}",
                )
            except (OSError, ValueError) as exc:
                report.issue("raw.frame_list_sha256.error", str(exc))

    rotations_c2w, centers = _w2c_to_c2w(extrinsics)
    _validate_windows(
        payload,
        frame_ids=frame_ids,
        global_rotations_c2w=rotations_c2w,
        global_centers=centers,
        options=options,
        report=report,
    )
    return {
        "frame_ids": frame_ids,
        "extrinsics_w2c": extrinsics,
        "rotations_c2w": rotations_c2w,
        "centers": centers,
        "frame_list_sha256": stored_frame_hash,
        "alignment_policy": (
            _scalar(payload.get("window_alignment", {}).get("policy"))
            if isinstance(payload.get("window_alignment"), Mapping)
            else None
        ),
    }


def _validate_windows(
    payload: Mapping[str, Any],
    *,
    frame_ids: np.ndarray,
    global_rotations_c2w: np.ndarray,
    global_centers: np.ndarray,
    options: ValidationOptions,
    report: _Report,
) -> None:
    windows = payload.get("windows")
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        report.issue("raw.windows.type", "windows must be a sequence")
        return
    report.metrics["raw.num_windows"] = len(windows)
    if not windows:
        report.issue("raw.windows.empty", "windows cannot be empty")
        return

    alignment_metadata = payload.get("window_alignment")
    if not isinstance(alignment_metadata, Mapping):
        report.issue("raw.window_alignment.type", "window_alignment must be a mapping")
        return
    expected_policy = (
        "single_window_identity" if len(windows) == 1 else "sequential_overlap_sim3"
    )
    report.require(
        _scalar(alignment_metadata.get("policy")) == expected_policy,
        "raw.window_alignment.policy",
        f"window_alignment.policy must be {expected_policy!r}",
    )
    report.require(
        _scalar(alignment_metadata.get("pose_selection"))
        == "first_prediction_wins_overlap",
        "raw.window_alignment.pose_selection",
        "Unsupported pose selection policy",
    )
    report.require(
        _scalar(alignment_metadata.get("scale_fallback"))
        in {"error", "one_when_degenerate"},
        "raw.window_alignment.scale_fallback",
        "Unsupported scale fallback policy",
    )
    try:
        declared_overlap = int(_scalar(alignment_metadata["overlap"]))
        min_overlap = int(_scalar(alignment_metadata["min_overlap"]))
    except (KeyError, TypeError, ValueError) as exc:
        report.issue("raw.window_alignment.schema", f"Invalid overlap metadata: {exc}")
        declared_overlap, min_overlap = -1, -1
    report.require(
        (len(windows) == 1 and declared_overlap == 0)
        or (len(windows) > 1 and declared_overlap >= min_overlap >= 3),
        "raw.window_alignment.overlap",
        f"Invalid declared overlap/min_overlap: {declared_overlap}/{min_overlap}",
    )

    frame_to_row = {int(frame): row for row, frame in enumerate(frame_ids)}
    all_center_errors: list[float] = []
    all_rotation_errors: list[float] = []
    overlap_frames = 0
    covered = np.zeros(len(frame_ids), dtype=bool)
    global_image_size = _array(payload.get("image_size_hw"), np.int64)
    for expected_index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            report.issue(
                "raw.window.type", f"windows[{expected_index}] must be a mapping"
            )
            continue
        required = (
            "window_index",
            "start_index",
            "end_index_exclusive",
            "frame_ids",
            "extrinsics_w2c_local",
            "intrinsics_local",
            "image_size_hw",
            "alignment_to_global",
            "overlap_frame_ids",
            "overlap_center_rmse",
            "overlap_rotation_rmse_deg",
        )
        missing = [key for key in required if key not in window]
        if missing:
            report.issue(
                "raw.window.schema",
                f"windows[{expected_index}] missing fields: {missing}",
            )
            continue
        index = int(_scalar(window["window_index"]))
        start = int(_scalar(window["start_index"]))
        end = int(_scalar(window["end_index_exclusive"]))
        ids = _array(window["frame_ids"], np.int64)
        local = _array(window["extrinsics_w2c_local"], np.float64)
        local_intrinsics = _array(window["intrinsics_local"], np.float64)
        local_image_size = _array(window["image_size_hw"], np.int64)
        report.require(
            index == expected_index,
            "raw.window.index",
            f"windows[{expected_index}].window_index={index}",
        )
        report.require(
            local_image_size.shape == (2,)
            and np.array_equal(local_image_size, global_image_size),
            "raw.window.image_size_hw",
            f"windows[{expected_index}] image size differs from the global artifact",
        )
        report.require(
            0 <= start < end <= len(frame_ids),
            "raw.window.range",
            f"Invalid window range [{start},{end})",
        )
        range_valid = 0 <= start <= end <= len(frame_ids)
        expected_window_ids = (
            frame_ids[start:end] if range_valid else np.empty(0, dtype=np.int64)
        )
        expected_overlap_ids = (
            expected_window_ids[covered[start:end]]
            if range_valid
            else np.empty(0, dtype=np.int64)
        )
        contributes_new_frames = (
            bool(np.any(~covered[start:end])) if range_valid else False
        )
        report.require(
            ids.ndim == 1 and np.array_equal(ids, expected_window_ids),
            "raw.window.frame_ids",
            f"windows[{expected_index}] frame_ids do not match its range",
        )
        report.require(
            contributes_new_frames,
            "raw.window.new_frames",
            f"windows[{expected_index}] contributes no previously unseen frame",
        )
        if range_valid:
            covered[start:end] = True
        report.require(
            local.shape == (len(ids), 3, 4),
            "raw.window.extrinsics.shape",
            f"windows[{expected_index}] local extrinsics shape is {local.shape}",
        )
        report.require(
            local_intrinsics.shape == (len(ids), 3, 3),
            "raw.window.intrinsics.shape",
            f"windows[{expected_index}] local intrinsics shape is {local_intrinsics.shape}",
        )
        report.require(
            local_intrinsics.shape == (len(ids), 3, 3)
            and np.isfinite(local_intrinsics).all()
            and bool(np.all(local_intrinsics[:, 0, 0] > 0))
            and bool(np.all(local_intrinsics[:, 1, 1] > 0)),
            "raw.window.intrinsics.values",
            f"windows[{expected_index}] local intrinsics are non-finite or non-positive",
        )
        if local.shape != (len(ids), 3, 4) or not np.isfinite(local).all():
            report.issue(
                "raw.window.extrinsics.finite",
                f"windows[{expected_index}] local extrinsics are invalid",
            )
            continue
        local_orth, local_det = _rotation_errors(local[:, :3, :3])
        report.require(
            options.thresholds.max_so3_orthogonality is None
            or float(np.max(local_orth, initial=0.0))
            <= options.thresholds.max_so3_orthogonality,
            "raw.window.so3_orthogonality",
            f"windows[{expected_index}] contains a non-orthogonal rotation",
        )
        report.require(
            options.thresholds.max_so3_det_error is None
            or float(np.max(local_det, initial=0.0))
            <= options.thresholds.max_so3_det_error,
            "raw.window.so3_det",
            f"windows[{expected_index}] contains a rotation with determinant != 1",
        )

        alignment = window["alignment_to_global"]
        if not isinstance(alignment, Mapping):
            report.issue(
                "raw.window.alignment.type",
                f"windows[{expected_index}].alignment_to_global must be a mapping",
            )
            continue
        try:
            scale = float(_scalar(alignment["scale"]))
            align_rotation = _array(alignment["rotation"], np.float64)
            align_translation = _array(alignment["translation"], np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            report.issue(
                "raw.window.alignment.schema", f"Invalid window alignment: {exc}"
            )
            continue
        alignment_ok = (
            math.isfinite(scale)
            and scale > 0
            and align_rotation.shape == (3, 3)
            and align_translation.shape == (3,)
            and np.isfinite(align_rotation).all()
            and np.isfinite(align_translation).all()
        )
        report.require(
            alignment_ok,
            "raw.window.alignment.values",
            f"windows[{expected_index}] has invalid Sim(3) alignment",
        )
        if not alignment_ok:
            continue
        align_orth, align_det = _rotation_errors(align_rotation[None])
        report.require(
            (
                options.thresholds.max_so3_orthogonality is None
                or align_orth[0] <= options.thresholds.max_so3_orthogonality
            )
            and (
                options.thresholds.max_so3_det_error is None
                or align_det[0] <= options.thresholds.max_so3_det_error
            ),
            "raw.window.alignment.rotation",
            f"windows[{expected_index}] alignment rotation is not SO(3)",
        )

        overlap_ids = _array(window["overlap_frame_ids"], np.int64)
        report.require(
            overlap_ids.ndim == 1,
            "raw.window.overlap_frame_ids",
            f"windows[{expected_index}] overlap_frame_ids must have shape [O]",
        )
        if overlap_ids.ndim != 1:
            continue
        report.require(
            np.array_equal(overlap_ids, expected_overlap_ids),
            "raw.window.overlap.ids",
            f"windows[{expected_index}] overlap_frame_ids do not equal the previously "
            "occupied frames in its range",
        )
        if expected_index == 0:
            report.require(
                len(overlap_ids) == 0,
                "raw.window.first_overlap",
                "The first window must not declare overlap frames",
            )
        else:
            report.require(
                len(overlap_ids) >= min_overlap,
                "raw.window.overlap.empty",
                f"windows[{expected_index}] has {len(overlap_ids)} overlap frames; "
                f"at least {min_overlap} are required",
            )
        if len(overlap_ids) == 0:
            continue

        id_to_local = {int(frame): row for row, frame in enumerate(ids)}
        if any(
            int(frame) not in id_to_local or int(frame) not in frame_to_row
            for frame in overlap_ids
        ):
            report.issue(
                "raw.window.overlap.ids",
                f"windows[{expected_index}] overlap contains an unknown frame",
            )
            continue
        local_rotations, local_centers = _w2c_to_c2w(local)
        rows_local = np.array([id_to_local[int(frame)] for frame in overlap_ids])
        rows_global = np.array([frame_to_row[int(frame)] for frame in overlap_ids])
        aligned_centers = (
            scale * (align_rotation @ local_centers[rows_local].T).T + align_translation
        )
        aligned_rotations = align_rotation[None] @ local_rotations[rows_local]
        center_errors = np.linalg.norm(
            aligned_centers - global_centers[rows_global], axis=1
        )
        relative_rotations = (
            np.swapaxes(global_rotations_c2w[rows_global], -1, -2) @ aligned_rotations
        )
        rotation_errors = _rotation_angle_deg(relative_rotations)
        center_rmse = float(np.sqrt(np.mean(np.square(center_errors))))
        rotation_rmse = float(np.sqrt(np.mean(np.square(rotation_errors))))
        stored_center = float(_scalar(window["overlap_center_rmse"]))
        stored_rotation = float(_scalar(window["overlap_rotation_rmse_deg"]))
        report.require(
            math.isfinite(stored_center)
            and stored_center >= 0
            and math.isfinite(stored_rotation)
            and stored_rotation >= 0,
            "raw.window.overlap_record_values",
            f"windows[{expected_index}] stored overlap residuals must be finite/non-negative",
        )
        report.require(
            math.isclose(center_rmse, stored_center, rel_tol=1e-4, abs_tol=1e-6),
            "raw.window.overlap_center_record",
            f"windows[{expected_index}] stored/recomputed center RMSE differ: "
            f"{stored_center:.8g} vs {center_rmse:.8g}",
        )
        report.require(
            math.isclose(rotation_rmse, stored_rotation, rel_tol=1e-4, abs_tol=1e-5),
            "raw.window.overlap_rotation_record",
            f"windows[{expected_index}] stored/recomputed rotation RMSE differ: "
            f"{stored_rotation:.8g} vs {rotation_rmse:.8g}",
        )
        all_center_errors.extend(center_errors.tolist())
        all_rotation_errors.extend(rotation_errors.tolist())
        overlap_frames += len(overlap_ids)

    report.require(
        bool(np.all(covered)),
        "raw.windows.coverage",
        f"Window union does not cover all {len(frame_ids)} frames",
    )
    report.metrics["raw.overlap_frames_checked"] = overlap_frames
    if all_center_errors:
        report.threshold_max(
            "raw.overlap_center_rmse",
            float(np.sqrt(np.mean(np.square(all_center_errors)))),
            options.thresholds.max_overlap_center_rmse,
        )
        report.threshold_max(
            "raw.overlap_rotation_rmse_deg",
            float(np.sqrt(np.mean(np.square(all_rotation_errors)))),
            options.thresholds.max_overlap_rotation_rmse_deg,
        )


def _validate_geometry(
    payload: Mapping[str, Any],
    *,
    raw: Mapping[str, Any],
    raw_path: str,
    dataset_name: str,
    trajectory_name: str,
    options: ValidationOptions,
    report: _Report,
) -> dict[str, Any] | None:
    required = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "motion_type": "geometry",
        "dataset_name": dataset_name,
        "trajectory_name": trajectory_name,
        "pair_direction": "current_to_goal",
        "normalization": "raw",
        "coordinate_frame": "current_navigation_frame",
        "yaw_unit": "radians",
    }
    for key, expected in required.items():
        if key not in payload:
            report.issue("geometry.schema.missing", f"Missing geometry field {key!r}")
        else:
            actual = _scalar(payload[key])
            report.require(
                actual == expected,
                f"geometry.schema.{key}",
                f"geometry {key} mismatch: {actual!r} != {expected!r}",
            )
    for key, expected in (
        ("complete", True),
        ("source_artifact_type", RAW_ARTIFACT_TYPE),
        ("ground_truth_usage", "none"),
    ):
        if key in payload or options.strict:
            report.require(
                _scalar(payload.get(key)) == expected,
                f"geometry.schema.{key}",
                f"geometry {key} mismatch: {_scalar(payload.get(key))!r} != {expected!r}",
            )
    components = payload.get("components")
    if components is not None and hasattr(components, "tolist"):
        components = components.tolist()
    report.require(
        components == ["delta_x", "delta_y", "delta_yaw"],
        "geometry.schema.components",
        f"Unexpected geometry components: {components!r}",
    )
    translation_unit = _scalar(payload.get("translation_unit"))
    report.require(
        translation_unit in {"meters", "waypoint_spacing_units"},
        "geometry.schema.translation_unit",
        f"Unsupported translation_unit: {translation_unit!r}",
    )
    if options.strict:
        strict_extras = (
            "source_pose_sha256",
            "frame_list_sha256",
            "alignment_policy",
            "pair_generation",
            "scale_policy",
            "scale",
            "scale_status",
            "camera_to_navigation_policy",
            "camera_to_navigation_axes",
            "scale_diagnostics",
        )
        missing_extras = [key for key in strict_extras if key not in payload]
        if missing_extras:
            report.issue(
                "geometry.schema.provenance_missing",
                f"Missing geometry provenance fields: {missing_extras}",
            )
        report.require(
            _scalar(payload.get("scale_policy"))
            == "per_trajectory_median_adjacent_step_to_waypoint_unit",
            "geometry.scale_policy",
            "Unexpected or missing image-only scale policy",
        )
        report.require(
            _scalar(payload.get("camera_to_navigation_policy"))
            == "opencv_camera_poses_projected_to_first_frame_navigation_se2",
            "geometry.camera_to_navigation_policy",
            "Unexpected or missing camera-to-navigation policy",
        )
        report.require(
            payload.get("camera_to_navigation_axes")
            == {
                "absolute_x": "first_camera_z",
                "absolute_y": "negative_first_camera_x",
                "absolute_heading": (
                    "atan2(-forward_x_in_first_camera,forward_z_in_first_camera)"
                ),
                "relative_action": "inverse_se2_current_times_se2_goal",
            },
            "geometry.camera_to_navigation_axes",
            "Unexpected or missing common-gauge navigation-axis declaration",
        )
        scale_diagnostics = payload.get("scale_diagnostics")
        report.require(
            isinstance(scale_diagnostics, Mapping)
            and _scalar(scale_diagnostics.get("measurement_frame"))
            == "first_frame_navigation_se2",
            "geometry.scale_diagnostics.measurement_frame",
            "Scale must be estimated in the first-frame navigation SE(2) gauge",
        )
        report.require(
            isinstance(payload.get("pair_generation"), Mapping),
            "geometry.pair_generation",
            "pair_generation must be a mapping",
        )
        geometry_alignment = _scalar(payload.get("alignment_policy"))
        report.require(
            geometry_alignment == "tartandrive_forward_camera",
            "geometry.alignment_policy",
            f"Unexpected geometry action policy: {geometry_alignment!r}",
        )
    try:
        pairs = _array(payload["frame_pairs"], np.int64)
        motion = _array(payload["motion"], np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        report.issue("geometry.schema.decode", f"Cannot decode geometry arrays: {exc}")
        return None
    report.require(
        pairs.ndim == 2 and pairs.shape[1:] == (2,),
        "geometry.frame_pairs.shape",
        f"frame_pairs must have shape [M,2], got {pairs.shape}",
    )
    try:
        scale = float(_scalar(payload.get("scale")))
    except (TypeError, ValueError):
        scale = float("nan")
    report.require(
        math.isfinite(scale) and scale > 0,
        "geometry.scale",
        f"geometry scale must be positive and finite, got {scale}",
    )
    scale_status = _scalar(payload.get("scale_status"))
    report.require(
        isinstance(scale_status, str) and bool(scale_status),
        "geometry.scale_status",
        "scale_status must be a non-empty string",
    )
    if len(pairs):
        report.require(
            scale_status == "estimated",
            "geometry.scale_status",
            f"Non-empty image-only actions require estimated scale, got {scale_status!r}",
        )
    else:
        report.require(
            scale_status in {"estimated", "unused_no_nwm_pairs"},
            "geometry.scale_status",
            f"Unexpected empty-pair scale status {scale_status!r}",
        )
    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        return None
    report.require(
        motion.shape == (len(pairs), 3),
        "geometry.motion.shape",
        f"motion must have shape [{len(pairs)},3], got {motion.shape}",
    )
    if motion.shape != (len(pairs), 3):
        return None
    report.metrics["geometry.num_pairs"] = len(pairs)
    report.require(
        np.isfinite(motion).all(),
        "geometry.motion.finite",
        "geometry motion contains non-finite values",
    )
    pair_tuples = [tuple(map(int, pair)) for pair in pairs]
    report.require(
        len(set(pair_tuples)) == len(pair_tuples),
        "geometry.frame_pairs.duplicate",
        "geometry cache contains duplicate frame pairs",
    )
    valid_frames = set(map(int, raw["frame_ids"]))
    bad_pairs = [
        pair
        for pair in pair_tuples
        if pair[0] not in valid_frames or pair[1] not in valid_frames
    ]
    report.require(
        not bad_pairs,
        "geometry.frame_pairs.range",
        f"geometry cache references unknown frames; examples: {bad_pairs[:3]}",
    )
    pair_generation = payload.get("pair_generation")
    if isinstance(pair_generation, Mapping):
        try:
            expected_pairs = _expected_nwm_pairs(
                len(raw["frame_ids"]),
                min_offset=int(_scalar(pair_generation["min_offset"])),
                max_offset=int(_scalar(pair_generation["max_offset"])),
                context_size=int(_scalar(pair_generation["context_size"])),
                len_traj_pred=int(_scalar(pair_generation["len_traj_pred"])),
            )
            report.require(
                np.array_equal(pairs, expected_pairs),
                "geometry.frame_pairs.coverage",
                "frame_pairs do not exactly cover the declared BaseDataset index domain",
            )
            if "coverage" in pair_generation:
                report.require(
                    _scalar(pair_generation["coverage"])
                    == "exact_BaseDataset_training_index_domain",
                    "geometry.pair_generation.coverage",
                    "Unexpected pair_generation.coverage marker",
                )
        except (KeyError, TypeError, ValueError) as exc:
            report.issue("geometry.pair_generation.decode", str(exc))

    raw_frame_hash = raw["frame_list_sha256"]
    if "frame_list_sha256" in payload:
        actual = _scalar(payload["frame_list_sha256"])
        report.require(
            actual == raw_frame_hash,
            "geometry.frame_list_sha256",
            "geometry/raw frame_list_sha256 mismatch",
        )
    elif options.strict:
        report.issue(
            "geometry.frame_list_sha256.missing",
            "Geometry cache lacks frame_list_sha256",
        )
    if "source_pose_sha256" in payload:
        actual = _scalar(payload["source_pose_sha256"])
        report.require(
            _is_sha256(actual),
            "geometry.source_pose_sha256.format",
            "source_pose_sha256 must contain 64 hexadecimal characters",
        )
        report.require(
            actual == sha256_file(raw_path),
            "geometry.source_pose_sha256",
            "Geometry cache was not derived from the supplied raw pose artifact",
        )
    elif options.strict:
        report.issue(
            "geometry.source_pose_sha256.missing",
            "Geometry cache lacks source_pose_sha256",
        )

    transforms = _se2(motion)
    lookup = {pair: row for row, pair in enumerate(pair_tuples)}
    _validate_se2_algebra(
        pair_tuples, transforms, lookup, trajectory_name, options=options, report=report
    )
    if math.isfinite(scale) and scale > 0:
        _validate_actions_against_raw_pose(
            pairs,
            motion,
            raw=raw,
            scale=scale,
            options=options,
            report=report,
        )
    return {
        "pairs": pairs,
        "pair_tuples": pair_tuples,
        "motion": motion,
        "translation_unit": translation_unit,
        "scale": scale,
    }


def _expected_nwm_pairs(
    num_frames: int,
    *,
    min_offset: int,
    max_offset: int,
    context_size: int,
    len_traj_pred: int,
) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for current in range(context_size - 1, num_frames - len_traj_pred):
        lower = max(0, current + min_offset)
        upper = min(num_frames - 1, current + max_offset)
        rows.extend((current, target) for target in range(lower, upper + 1))
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _validate_actions_against_raw_pose(
    pairs: np.ndarray,
    motion: np.ndarray,
    *,
    raw: Mapping[str, Any],
    scale: float,
    options: ValidationOptions,
    report: _Report,
) -> None:
    """Recompute the declared common-gauge planar camera conversion.

    Every camera is first projected into an SE(2) pose in the first camera's
    x-forward/y-left navigation gauge. Pair actions are then the exact
    ``inv(T_current) @ T_goal`` transform. Camera height, pitch and roll are
    deliberately discarded before constructing any pair.
    """

    if len(pairs) == 0:
        report.metrics["geometry.pose_action_checks"] = 0
        return
    rotations = np.asarray(raw["rotations_c2w"], dtype=np.float64)
    centers = np.asarray(raw["centers"], dtype=np.float64)
    first_from_world = rotations[0].T
    centers_first_camera = (first_from_world @ (centers - centers[0]).T).T
    positions_navigation = (
        np.column_stack(
            (
                centers_first_camera[:, 2],
                -centers_first_camera[:, 0],
            )
        )
        * scale
    )
    orientations_first_camera = first_from_world[None] @ rotations
    forward_first_camera = orientations_first_camera[:, :, 2]
    yaw_navigation = np.arctan2(-forward_first_camera[:, 0], forward_first_camera[:, 2])

    current = pairs[:, 0]
    goal = pairs[:, 1]
    displacement = positions_navigation[goal] - positions_navigation[current]
    cosine = np.cos(yaw_navigation[current])
    sine = np.sin(yaw_navigation[current])
    expected = np.column_stack(
        (
            displacement[:, 0] * cosine + displacement[:, 1] * sine,
            -displacement[:, 0] * sine + displacement[:, 1] * cosine,
            _wrap_angle(yaw_navigation[goal] - yaw_navigation[current]),
        )
    )
    translation_error = np.linalg.norm(expected[:, :2] - motion[:, :2], axis=1)
    yaw_error = np.degrees(np.abs(_wrap_angle(expected[:, 2] - motion[:, 2])))
    report.metrics["geometry.pose_action_checks"] = len(pairs)
    report.threshold_max(
        "geometry.pose_action_translation_max",
        float(np.max(translation_error)),
        options.thresholds.max_pose_action_translation,
    )
    report.threshold_max(
        "geometry.pose_action_yaw_deg_max",
        float(np.max(yaw_error)),
        options.thresholds.max_pose_action_yaw_deg,
    )


def _validate_se2_algebra(
    pairs: Sequence[tuple[int, int]],
    transforms: np.ndarray,
    lookup: Mapping[tuple[int, int], int],
    trajectory_name: str,
    *,
    options: ValidationOptions,
    report: _Report,
) -> None:
    identity_translation: list[float] = []
    identity_yaw: list[float] = []
    inverse_translation: list[float] = []
    inverse_yaw: list[float] = []
    for row, (current, goal) in enumerate(pairs):
        if current == goal:
            translation, yaw = _se2_error(np.eye(3), transforms[row])
            identity_translation.append(translation)
            identity_yaw.append(yaw)
        reverse = lookup.get((goal, current))
        if reverse is not None and row <= reverse:
            translation, yaw = _se2_error(
                np.eye(3), transforms[row] @ transforms[reverse]
            )
            inverse_translation.append(translation)
            inverse_yaw.append(yaw)

    _record_algebra_metric(
        "geometry.identity_translation_max",
        identity_translation,
        options.thresholds.max_identity_translation,
        options,
        report,
    )
    _record_algebra_metric(
        "geometry.identity_yaw_deg_max",
        identity_yaw,
        options.thresholds.max_identity_yaw_deg,
        options,
        report,
    )
    report.metrics["geometry.identity_checks"] = len(identity_translation)
    _record_algebra_metric(
        "geometry.inverse_translation_max",
        inverse_translation,
        options.thresholds.max_inverse_translation,
        options,
        report,
    )
    _record_algebra_metric(
        "geometry.inverse_yaw_deg_max",
        inverse_yaw,
        options.thresholds.max_inverse_yaw_deg,
        options,
        report,
    )
    report.metrics["geometry.inverse_checks"] = len(inverse_translation)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for current, goal in pairs:
        adjacency[current].append(goal)
    candidates = [pair for pair in pairs if adjacency.get(pair[1])]
    seed = int.from_bytes(hashlib.sha256(trajectory_name.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    composition_translation: list[float] = []
    composition_yaw: list[float] = []
    attempts = 0
    max_attempts = max(options.max_composition_checks * 5, 100)
    while (
        candidates
        and len(composition_translation) < options.max_composition_checks
        and attempts < max_attempts
    ):
        current, middle = candidates[attempts % len(candidates)]
        goal = rng.choice(adjacency[middle])
        direct = lookup.get((current, goal))
        attempts += 1
        if direct is None:
            continue
        first = lookup[(current, middle)]
        second = lookup[(middle, goal)]
        translation, yaw = _se2_error(
            transforms[first] @ transforms[second], transforms[direct]
        )
        composition_translation.append(translation)
        composition_yaw.append(yaw)
    _record_algebra_metric(
        "geometry.composition_translation_max",
        composition_translation,
        options.thresholds.max_composition_translation,
        options,
        report,
    )
    _record_algebra_metric(
        "geometry.composition_yaw_deg_max",
        composition_yaw,
        options.thresholds.max_composition_yaw_deg,
        options,
        report,
    )
    report.metrics["geometry.composition_checks"] = len(composition_translation)


def _record_algebra_metric(
    name: str,
    values: Sequence[float],
    threshold: float | None,
    options: ValidationOptions,
    report: _Report,
) -> None:
    if values:
        report.threshold_max(name, max(values), threshold)
    else:
        report.metrics[name] = None
        report.issue(
            f"{name}.skipped",
            f"No eligible pairs for {name}",
            error=options.require_algebra_coverage,
        )


def _umeyama(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) mapping source points onto target points."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Umeyama inputs must both have shape [N,3]")
    if len(source) < 3:
        raise ValueError(
            "At least three points are required for validation-only Sim(3)"
        )
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.sum(source_centered * source_centered) / len(source))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Predicted camera centers are degenerate")
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    scale = float(np.sum(singular * sign) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid Sim(3) scale {scale}")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _circular_mean(angles: np.ndarray) -> float:
    return float(math.atan2(np.sin(angles).mean(), np.cos(angles).mean()))


def _direction_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    valid = (predicted_norm > 1e-8) & (target_norm > 1e-8)
    if not np.any(valid):
        return {"count": 0, "cosine_mean": float("nan"), "angle_deg_mean": float("nan")}
    cosine = np.sum(predicted[valid] * target[valid], axis=1) / (
        predicted_norm[valid] * target_norm[valid]
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    return {
        "count": int(np.sum(valid)),
        "cosine_mean": float(np.mean(cosine)),
        "angle_deg_mean": float(np.mean(np.degrees(np.arccos(cosine)))),
    }


def _load_gt(path: str) -> tuple[np.ndarray, np.ndarray]:
    # The caller opts into reading the trusted local TartanDrive pickle.
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, Mapping)
        or "position" not in payload
        or "yaw" not in payload
    ):
        raise ValueError("traj_data.pkl must contain position and yaw")
    positions = np.asarray(payload["position"], dtype=np.float64)
    yaw = np.asarray(payload["yaw"], dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError(f"GT position must have shape [N,2|3], got {positions.shape}")
    if (
        len(yaw) != len(positions)
        or not np.isfinite(positions).all()
        or not np.isfinite(yaw).all()
    ):
        raise ValueError("GT position/yaw shape or finiteness check failed")
    return positions, yaw


def _validate_gt(
    gt_path: str,
    *,
    raw: Mapping[str, Any],
    geometry: Mapping[str, Any],
    options: ValidationOptions,
    report: _Report,
) -> None:
    report.metrics["gt.only_for_validation"] = True
    try:
        positions, gt_yaw = _load_gt(gt_path)
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        report.issue("gt.load", f"Cannot load GT validation data: {exc}")
        return
    frame_ids = np.asarray(raw["frame_ids"], dtype=np.int64)
    if not report.require(
        len(positions) == len(frame_ids),
        "gt.num_frames",
        f"GT/raw frame count mismatch: {len(positions)} != {len(frame_ids)}",
    ):
        return
    target = (
        positions
        if positions.shape[1] == 3
        else np.column_stack((positions, np.zeros(len(positions))))
    )
    source = np.asarray(raw["centers"], dtype=np.float64)
    try:
        scale, rotation, translation = _umeyama(source, target)
    except ValueError as exc:
        report.issue("gt.sim3.skipped", str(exc), error=False)
        return
    aligned = scale * (rotation @ source.T).T + translation
    ate = np.linalg.norm(aligned - target, axis=1)
    report.metrics["gt.sim3_validation_only"] = True
    report.metrics["gt.sim3_scale"] = scale
    report.threshold_max(
        "gt.ate_rmse",
        float(np.sqrt(np.mean(np.square(ate)))),
        options.thresholds.max_gt_ate_rmse,
    )
    report.metrics["gt.ate_median"] = float(np.median(ate))

    delta = options.gt_rpe_delta
    if delta < 1 or delta >= len(frame_ids):
        report.issue("gt.rpe.skipped", f"Invalid RPE delta {delta}", error=False)
    else:
        pred_delta = aligned[delta:] - aligned[:-delta]
        gt_delta = target[delta:] - target[:-delta]
        translation_error = np.linalg.norm(pred_delta - gt_delta, axis=1)
        report.threshold_max(
            "gt.rpe_translation_rmse",
            float(np.sqrt(np.mean(np.square(translation_error)))),
            options.thresholds.max_gt_rpe_translation_rmse,
        )
        direction = _direction_metrics(pred_delta[:, :2], gt_delta[:, :2])
        report.metrics["gt.translation_direction_count"] = direction["count"]
        report.metrics["gt.translation_direction_cosine_mean"] = _finite_float(
            direction["cosine_mean"]
        )
        report.metrics["gt.translation_direction_angle_deg_mean"] = _finite_float(
            direction["angle_deg_mean"]
        )

        aligned_camera_rotations = rotation[None] @ np.asarray(raw["rotations_c2w"])
        forward = aligned_camera_rotations[:, :, 2]  # OpenCV camera forward is +z.
        camera_yaw = np.arctan2(forward[:, 1], forward[:, 0])
        yaw_offset = _circular_mean(_wrap_angle(gt_yaw - camera_yaw))
        predicted_yaw = _wrap_angle(camera_yaw + yaw_offset)
        pred_dyaw = _wrap_angle(predicted_yaw[delta:] - predicted_yaw[:-delta])
        gt_dyaw = _wrap_angle(gt_yaw[delta:] - gt_yaw[:-delta])
        yaw_error_deg = np.degrees(np.abs(_wrap_angle(pred_dyaw - gt_dyaw)))
        report.metrics["gt.camera_to_body_yaw_offset_rad_validation_only"] = yaw_offset
        report.threshold_max(
            "gt.rpe_yaw_mae_deg",
            float(np.mean(yaw_error_deg)),
            options.thresholds.max_gt_rpe_yaw_mae_deg,
        )
        pred_relative = (
            np.swapaxes(aligned_camera_rotations[:-delta], -1, -2)
            @ aligned_camera_rotations[delta:]
        )
        pred_rotation_deg = _rotation_angle_deg(pred_relative)
        gt_rotation_deg = np.degrees(np.abs(gt_dyaw))
        rotation_error = np.abs(pred_rotation_deg - gt_rotation_deg)
        report.threshold_max(
            "gt.rpe_rotation_mae_deg",
            float(np.mean(rotation_error)),
            options.thresholds.max_gt_rpe_rotation_mae_deg,
        )

    _validate_actions_against_gt(
        positions[:, :2], gt_yaw, geometry=geometry, options=options, report=report
    )


def _validate_actions_against_gt(
    positions: np.ndarray,
    yaw: np.ndarray,
    *,
    geometry: Mapping[str, Any],
    options: ValidationOptions,
    report: _Report,
) -> None:
    pairs = np.asarray(geometry["pairs"], dtype=np.int64)
    predicted = np.asarray(geometry["motion"], dtype=np.float64)
    if len(pairs) == 0:
        report.issue("gt.action.skipped", "Geometry cache has no actions", error=False)
        return
    current = pairs[:, 0]
    goal = pairs[:, 1]
    displacement = positions[goal] - positions[current]
    cosine = np.cos(yaw[current])
    sine = np.sin(yaw[current])
    gt_xy = np.column_stack(
        (
            displacement[:, 0] * cosine + displacement[:, 1] * sine,
            -displacement[:, 0] * sine + displacement[:, 1] * cosine,
        )
    )
    if geometry["translation_unit"] == "waypoint_spacing_units":
        spacing = options.waypoint_spacing
        if spacing is not None and spacing > 0:
            gt_xy = gt_xy / spacing
            report.metrics["gt.action_scale_policy_validation_only"] = (
                "known_waypoint_spacing"
            )
            report.metrics["gt.action_waypoint_spacing"] = spacing
        else:
            adjacent = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
            adjacent = adjacent[np.isfinite(adjacent) & (adjacent > 1e-8)]
            if len(adjacent) == 0:
                report.issue(
                    "gt.action.spacing",
                    "Cannot derive validation-only GT scale from a stationary trajectory",
                    error=False,
                )
                return
            gt_spacing = float(np.median(adjacent))
            gt_xy = gt_xy / gt_spacing
            report.metrics["gt.action_scale_policy_validation_only"] = (
                "per_trajectory_median_adjacent_step_to_waypoint_unit"
            )
            report.metrics["gt.action_gt_median_adjacent_step"] = gt_spacing
    gt_dyaw = _wrap_angle(yaw[goal] - yaw[current])
    translation_error = np.linalg.norm(predicted[:, :2] - gt_xy, axis=1)
    yaw_error_deg = np.degrees(np.abs(_wrap_angle(predicted[:, 2] - gt_dyaw)))
    report.threshold_max(
        "gt.action_translation_rmse",
        float(np.sqrt(np.mean(np.square(translation_error)))),
        options.thresholds.max_action_translation_rmse,
    )
    report.metrics["gt.action_translation_mae"] = float(np.mean(translation_error))
    report.threshold_max(
        "gt.action_yaw_mae_deg",
        float(np.mean(yaw_error_deg)),
        options.thresholds.max_action_yaw_mae_deg,
    )
    direction = _direction_metrics(predicted[:, :2], gt_xy)
    report.metrics["gt.action_direction_count"] = direction["count"]
    if direction["count"]:
        report.threshold_min(
            "gt.action_direction_cosine_mean",
            direction["cosine_mean"],
            options.thresholds.min_action_direction_cosine,
        )
        report.metrics["gt.action_direction_angle_deg_mean"] = direction[
            "angle_deg_mean"
        ]
    meaningful_yaw = (np.abs(gt_dyaw) > math.radians(0.1)) & (
        np.abs(predicted[:, 2]) > math.radians(0.1)
    )
    report.metrics["gt.action_yaw_sign_count"] = int(np.sum(meaningful_yaw))
    if np.any(meaningful_yaw):
        agreement = float(
            np.mean(
                np.sign(predicted[meaningful_yaw, 2])
                == np.sign(gt_dyaw[meaningful_yaw])
            )
        )
        report.threshold_min(
            "gt.action_yaw_sign_agreement",
            agreement,
            options.thresholds.min_action_yaw_sign_agreement,
        )
    else:
        report.metrics["gt.action_yaw_sign_agreement"] = None
        report.issue(
            "gt.action_yaw_sign.skipped",
            "No frame pair has meaningful yaw in both geometry and GT",
            error=False,
        )


def _input_fingerprint(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return {"path": os.path.realpath(path), "missing": True}
    return {
        "path": os.path.realpath(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_frames_fingerprint(trajectory_dir: str | None) -> dict[str, Any] | None:
    """Cheap resume invalidation; full image content hashes are checked on reload."""

    if trajectory_dir is None:
        return None
    try:
        entries = []
        with os.scandir(trajectory_dir) as iterator:
            for entry in iterator:
                stem, extension = os.path.splitext(entry.name)
                if (
                    extension.lower() != ".jpg"
                    or not stem.isdigit()
                    or not entry.is_file()
                ):
                    continue
                stat = entry.stat()
                entries.append((int(stem), stat.st_size, stat.st_mtime_ns))
        entries.sort()
    except OSError:
        return {"path": os.path.realpath(trajectory_dir), "missing": True}
    encoded = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return {
        "path": os.path.realpath(trajectory_dir),
        "num_frames": len(entries),
        "stat_list_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_trajectory(
    raw_pose_path: str | os.PathLike[str],
    geometry_path: str | os.PathLike[str],
    *,
    dataset_name: str,
    trajectory_name: str,
    trajectory_dir: str | os.PathLike[str] | None = None,
    gt_path: str | os.PathLike[str] | None = None,
    options: ValidationOptions | None = None,
) -> dict[str, Any]:
    """Validate one raw-pose/geometry pair without mutating either artifact."""

    options = options or ValidationOptions()
    raw_pose_path = os.path.realpath(os.fspath(raw_pose_path))
    geometry_path = os.path.realpath(os.fspath(geometry_path))
    trajectory_dir_str = (
        None if trajectory_dir is None else os.path.realpath(os.fspath(trajectory_dir))
    )
    gt_path_str = None if gt_path is None else os.path.realpath(os.fspath(gt_path))
    report = _Report()
    result: dict[str, Any] = {
        "dataset_name": dataset_name,
        "trajectory_name": trajectory_name,
        "inputs": {
            "raw_pose": _input_fingerprint(raw_pose_path),
            "geometry": _input_fingerprint(geometry_path),
            "gt": _input_fingerprint(gt_path_str),
            "source_frames": _source_frames_fingerprint(trajectory_dir_str),
        },
    }
    try:
        raw_payload = _load_mapping(raw_pose_path)
        geometry_payload = _load_mapping(geometry_path)
        raw = _validate_raw_pose(
            raw_payload,
            dataset_name=dataset_name,
            trajectory_name=trajectory_name,
            trajectory_dir=trajectory_dir_str,
            options=options,
            report=report,
        )
        if raw is not None:
            geometry = _validate_geometry(
                geometry_payload,
                raw=raw,
                raw_path=raw_pose_path,
                dataset_name=dataset_name,
                trajectory_name=trajectory_name,
                options=options,
                report=report,
            )
            if geometry is not None and gt_path_str is not None:
                _validate_gt(
                    gt_path_str,
                    raw=raw,
                    geometry=geometry,
                    options=options,
                    report=report,
                )
    except Exception as exc:  # noqa: BLE001 - preserve the rest of a long NAS audit.
        report.issue("validation.exception", f"{type(exc).__name__}: {exc}")

    errors = sum(issue["severity"] == "error" for issue in report.issues)
    result.update(
        {
            "status": "pass" if errors == 0 else "fail",
            "error_count": errors,
            "warning_count": len(report.issues) - errors,
            "metrics": report.metrics,
            "issues": report.issues,
        }
    )
    return result


def _safe_artifact_path(root: str, pattern: str, dataset: str, trajectory: str) -> str:
    root = os.path.realpath(os.path.expanduser(root))
    relative = pattern.format(dataset_name=dataset, trajectory_name=trajectory)
    path = os.path.realpath(os.path.join(root, relative))
    if path != root and not path.startswith(root + os.sep):
        raise ValueError(f"Unsafe artifact path produced for {trajectory!r}")
    return path


def _json_config_fingerprint(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.realpath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".validation-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize_trajectories(
    trajectories: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(
        record.get("status", "unknown") for record in trajectories.values()
    )
    issue_counts = Counter(
        issue.get("code", "unknown")
        for record in trajectories.values()
        for issue in record.get("issues", [])
        if issue.get("severity") == "error"
    )
    numeric: dict[str, list[float]] = defaultdict(list)
    for record in trajectories.values():
        for name, value in record.get("metrics", {}).items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                numeric[name].append(float(value))
    aggregates = {
        name: {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
        for name, values in sorted(numeric.items())
    }
    failed = status_counts.get("fail", 0)
    return {
        "status": "pass" if failed == 0 else "fail",
        "total": len(trajectories),
        "passed": status_counts.get("pass", 0),
        "failed": failed,
        "status_counts": dict(sorted(status_counts.items())),
        "error_code_counts": dict(sorted(issue_counts.items())),
        "metric_aggregates": aggregates,
    }


def validate_dataset(
    trajectory_names: Iterable[str],
    *,
    raw_pose_root: str,
    geometry_root: str,
    dataset_name: str,
    output_path: str,
    data_root: str | None = None,
    raw_pattern: str = "{dataset_name}/{trajectory_name}.pt",
    geometry_pattern: str = "{dataset_name}/{trajectory_name}.pt",
    use_gt: bool = True,
    resume: bool = True,
    flush_every: int = 10,
    options: ValidationOptions | None = None,
) -> dict[str, Any]:
    """Validate a dataset and atomically checkpoint an incremental JSON report."""

    options = options or ValidationOptions()
    names = list(
        dict.fromkeys(name.strip() for name in trajectory_names if name.strip())
    )
    config_document = {
        "validator_version": VALIDATOR_VERSION,
        "dataset_name": dataset_name,
        "raw_pose_root": os.path.realpath(raw_pose_root),
        "geometry_root": os.path.realpath(geometry_root),
        "data_root": None if data_root is None else os.path.realpath(data_root),
        "raw_pattern": raw_pattern,
        "geometry_pattern": geometry_pattern,
        "use_gt": use_gt,
        "options": asdict(options),
        "trajectory_names_sha256": hashlib.sha256(
            "\n".join(names).encode()
        ).hexdigest(),
    }
    config_fingerprint = _json_config_fingerprint(config_document)
    previous: dict[str, Any] = {}
    if resume and os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as handle:
                candidate = json.load(handle)
            if candidate.get("config_fingerprint") == config_fingerprint:
                previous = dict(candidate.get("trajectories", {}))
        except (OSError, ValueError, TypeError):
            previous = {}

    trajectories: dict[str, Any] = {}
    reused = 0
    flush_every = max(1, int(flush_every))

    def current_document(*, complete: bool) -> dict[str, Any]:
        summary = summarize_trajectories(trajectories)
        summary["reused"] = reused
        summary["remaining"] = len(names) - len(trajectories)
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "complete": complete,
            "config": config_document,
            "config_fingerprint": config_fingerprint,
            "summary": summary,
            "trajectories": trajectories,
        }

    for index, trajectory in enumerate(names, start=1):
        raw_path = _safe_artifact_path(
            raw_pose_root, raw_pattern, dataset_name, trajectory
        )
        geometry_path = _safe_artifact_path(
            geometry_root, geometry_pattern, dataset_name, trajectory
        )
        trajectory_dir = (
            None if data_root is None else os.path.join(data_root, trajectory)
        )
        gt_path = (
            os.path.join(trajectory_dir, "traj_data.pkl")
            if use_gt and trajectory_dir is not None
            else None
        )
        input_fingerprints = {
            "raw_pose": _input_fingerprint(raw_path),
            "geometry": _input_fingerprint(geometry_path),
            "gt": _input_fingerprint(gt_path),
            "source_frames": _source_frames_fingerprint(trajectory_dir),
        }
        old = previous.get(trajectory)
        if isinstance(old, Mapping) and old.get("inputs") == input_fingerprints:
            trajectories[trajectory] = old
            reused += 1
        else:
            trajectories[trajectory] = validate_trajectory(
                raw_path,
                geometry_path,
                dataset_name=dataset_name,
                trajectory_name=trajectory,
                trajectory_dir=trajectory_dir,
                gt_path=gt_path,
                options=options,
            )
        if index % flush_every == 0:
            _atomic_json(output_path, current_document(complete=False))
    document = current_document(complete=True)
    _atomic_json(output_path, document)
    return document


__all__ = [
    "ValidationOptions",
    "ValidationThresholds",
    "frame_list_records",
    "frame_list_sha256",
    "sha256_file",
    "summarize_trajectories",
    "validate_dataset",
    "validate_trajectory",
]
