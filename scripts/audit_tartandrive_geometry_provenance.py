#!/usr/bin/env python3
"""Read-only protocol/provenance audit for the formal TartanDrive cache.

This intentionally checks a single, immutable formal protocol.  It is not a
general artifact validator: changing any pinned model, code, input, inference,
or pair-generation setting makes the audit fail and produces an atomic JSON
receipt explaining why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_action.tartandrive_vggt_omega import (
    build_nwm_frame_pairs,
    safe_torch_load,
    sha256_file,
)

AUDIT_SCHEMA_VERSION = 1
AUDITOR_VERSION = "1.0"
EXPECTED_DATASET = "tartan_drive"
EXPECTED_TRAJECTORY_COUNT = 1251
EXPECTED_MODEL_ID = "facebook/VGGT-Omega-1B-512"
EXPECTED_CHECKPOINT_SHA256 = (
    "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
)
EXPECTED_CODE_REVISION = "282ec70363edeff59424bf43731658092fba3d37"
EXPECTED_MODEL_REVISION = "ba9db085d6b7349b738fa2e37d198bb4dd077954"
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "e194815bd80b13d960b12d80fbf927457c16c0243bd896518c54b89932cbc0e6"
)
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "2b89f69d011baec55da36f155e416d42bc9edca735799c503e4f84d1915a925b"
)

# This is the fingerprint observed in the formal artifacts and independently
# reproduced from the exact canonical configuration below.  Descriptor overlap
# is 64 even though effective overlap is zero for whole-trajectory inference.
EXPECTED_EXTRACTION_FINGERPRINT = (
    "b424ae19050608d03fdaa9ca26541f12080d8e9d06cdc0a6f97edea9e05b5476"
)
EXPECTED_EXTRACTION_CONFIGURATION: dict[str, Any] = {
    "preprocessing": {
        "implementation": "vggt_omega.utils.load_fn.load_and_preprocess_images",
        "resolution": 384,
        "resize_mode": "max_size",
        "patch_size": 16,
    },
    "inference": {
        "path": "fast",
        "effective_dtype": "bfloat16",
        "allow_tf32": True,
        "strict_checkpoint_load": True,
        "depth_head_executed": False,
        "aggregator_cached_layer_policy": "final_layer_only",
    },
    "windowing": {
        "window_size": 0,
        "overlap": 64,
        "partition_policy": "contiguous_trailing_overlap_v1",
        "alignment_policy": "single_identity_or_sequential_overlap_sim3_v1",
        "pose_selection": "first_prediction_wins_overlap",
        "allow_degenerate_scale": False,
        "scale_fallback": "error",
    },
    "determinism": {"seed": 0},
}

EXPECTED_RAW_PREPROCESSING = {
    "implementation": "vggt_omega.utils.load_fn.load_and_preprocess_images",
    "resize_mode": "max_size",
    "image_resolution": 384,
    "patch_size": 16,
}
EXPECTED_RAW_INFERENCE = {
    "path": "fast",
    "effective_dtype": "bfloat16",
    "strict_checkpoint_load": True,
    "depth_head_executed": False,
    "aggregator_cached_layer_policy": "final_layer_only",
}
EXPECTED_WINDOW_ALIGNMENT = {
    "policy": "single_window_identity",
    "overlap": 0,
    "min_overlap": 3,
    "pose_selection": "first_prediction_wins_overlap",
    "scale_fallback": "error",
}
EXPECTED_PAIR_GENERATION = {
    "min_offset": -64,
    "max_offset": 64,
    "context_size": 4,
    "len_traj_pred": 64,
    "coverage": "exact_BaseDataset_training_index_domain",
}
EXPECTED_GEOMETRY_POLICY_CONFIGURATION = {
    "alignment_policy": "tartandrive_forward_camera",
    "degenerate_scale_policy": "empty_only",
    "nonzero_epsilon": 1e-6,
    "camera_to_navigation": None,
    "meters_per_model_unit": None,
    "translation_unit": "waypoint_spacing_units",
    "waypoint_spacing_meters": 0.72,
    "min_offset": -64,
    "max_offset": 64,
    "context_size": 4,
    "len_traj_pred": 64,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _display(value: Any, limit: int = 300) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _equal(actual: Any, expected: Any) -> bool:
    try:
        result = actual == expected
        return isinstance(result, bool) and result
    except (RuntimeError, TypeError, ValueError):
        return False


def _issue(issues: list[dict[str, str]], code: str, message: str) -> None:
    issues.append({"code": code, "severity": "error", "message": message})


def _require_equal(
    issues: list[dict[str, str]],
    code: str,
    actual: Any,
    expected: Any,
) -> bool:
    if _equal(actual, expected):
        return True
    _issue(
        issues,
        code,
        f"expected {_display(expected)}, got {_display(actual)}",
    )
    return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _mapping(
    payload: Mapping[str, Any],
    key: str,
    issues: list[dict[str, str]],
    code: str,
) -> Mapping[str, Any] | None:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return value
    _issue(issues, code, f"{key} must be a mapping, got {type(value).__name__}")
    return None


def _tensor(
    payload: Mapping[str, Any],
    key: str,
    issues: list[dict[str, str]],
    *,
    dtype: torch.dtype,
    ndim: int,
    trailing_shape: Sequence[int] = (),
) -> torch.Tensor | None:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        _issue(
            issues,
            f"tensor.{key}.type",
            f"{key} must be a torch.Tensor, got {type(value).__name__}",
        )
        return None
    if value.dtype != dtype:
        _issue(
            issues,
            f"tensor.{key}.dtype",
            f"{key} must have dtype {dtype}, got {value.dtype}",
        )
    if value.ndim != ndim or (
        trailing_shape
        and tuple(value.shape[-len(trailing_shape) :]) != tuple(trailing_shape)
    ):
        _issue(
            issues,
            f"tensor.{key}.shape",
            f"{key} has invalid shape {tuple(value.shape)}",
        )
        return None
    return value


def _check_descriptor(
    descriptor: Any,
    *,
    expected_configuration: Mapping[str, Any],
    expected_fingerprint: str,
    code_prefix: str,
    issues: list[dict[str, str]],
    expected_schema_version: int | None = 1,
) -> None:
    if not isinstance(descriptor, Mapping):
        _issue(issues, f"{code_prefix}.type", "descriptor must be a mapping")
        return
    if expected_schema_version is not None:
        _require_equal(
            issues,
            f"{code_prefix}.schema_version",
            descriptor.get("schema_version"),
            expected_schema_version,
        )
    configuration = descriptor.get("configuration")
    if not isinstance(configuration, Mapping):
        _issue(
            issues,
            f"{code_prefix}.configuration.type",
            "descriptor configuration must be a mapping",
        )
        return
    try:
        recomputed = _canonical_sha256(dict(configuration))
    except (TypeError, ValueError) as error:
        _issue(
            issues,
            f"{code_prefix}.configuration.canonical_json",
            f"descriptor configuration is not canonical-JSON compatible: {error}",
        )
        return
    _require_equal(
        issues,
        f"{code_prefix}.fingerprint.self_consistency",
        descriptor.get("fingerprint"),
        recomputed,
    )
    _require_equal(
        issues,
        f"{code_prefix}.fingerprint.formal_protocol",
        descriptor.get("fingerprint"),
        expected_fingerprint,
    )
    _require_equal(
        issues,
        f"{code_prefix}.configuration.formal_protocol",
        dict(configuration),
        dict(expected_configuration),
    )


def _audit_raw(
    path: Path,
    trajectory_name: str,
    dataset_name: str,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        result["sha256"] = sha256_file(path)
    except OSError as error:
        _issue(issues, "raw.sha256", f"cannot hash {path}: {error}")
        return result
    try:
        payload = safe_torch_load(path)
    except Exception as error:  # noqa: BLE001 - corrupt artifacts must be receipted.
        _issue(
            issues,
            "raw.load",
            f"cannot safely load {path}: {type(error).__name__}: {error}",
        )
        return result

    fixed = {
        "schema_version": 1,
        "artifact_type": "vggt_omega_camera_poses",
        "dataset_name": dataset_name,
        "trajectory_name": trajectory_name,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "code_revision": EXPECTED_CODE_REVISION,
        "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
        "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "pose_convention": "world_to_camera_opencv",
        "complete": True,
    }
    for key, expected in fixed.items():
        _require_equal(issues, f"raw.{key}", payload.get(key), expected)
    frame_list_sha256 = payload.get("frame_list_sha256")
    if not _is_sha256(frame_list_sha256):
        _issue(
            issues,
            "raw.frame_list_sha256",
            "frame_list_sha256 must contain 64 hexadecimal characters",
        )
    else:
        result["frame_list_sha256"] = frame_list_sha256

    _check_descriptor(
        payload.get("extraction_descriptor"),
        expected_configuration=EXPECTED_EXTRACTION_CONFIGURATION,
        expected_fingerprint=EXPECTED_EXTRACTION_FINGERPRINT,
        code_prefix="raw.extraction_descriptor",
        issues=issues,
    )
    _require_equal(
        issues,
        "raw.preprocessing",
        payload.get("preprocessing"),
        EXPECTED_RAW_PREPROCESSING,
    )
    _require_equal(
        issues,
        "raw.inference",
        payload.get("inference"),
        EXPECTED_RAW_INFERENCE,
    )
    _require_equal(
        issues,
        "raw.window_alignment",
        payload.get("window_alignment"),
        EXPECTED_WINDOW_ALIGNMENT,
    )

    frame_ids = _tensor(payload, "frame_ids", issues, dtype=torch.int64, ndim=1)
    num_frames: int | None = None
    if frame_ids is not None:
        num_frames = int(frame_ids.numel())
        result["num_frames"] = num_frames
        if not torch.equal(frame_ids.cpu(), torch.arange(num_frames)):
            _issue(
                issues,
                "raw.frame_ids.domain",
                "frame_ids must be contiguous from zero",
            )
    extrinsics = _tensor(
        payload,
        "extrinsics_w2c",
        issues,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3, 4),
    )
    intrinsics = _tensor(
        payload,
        "intrinsics",
        issues,
        dtype=torch.float32,
        ndim=3,
        trailing_shape=(3, 3),
    )
    for key, tensor in (("extrinsics_w2c", extrinsics), ("intrinsics", intrinsics)):
        if tensor is not None and not bool(torch.isfinite(tensor).all()):
            _issue(issues, f"raw.{key}.finite", f"{key} contains non-finite values")
    if num_frames is not None:
        if extrinsics is not None and extrinsics.shape[0] != num_frames:
            _issue(
                issues, "raw.extrinsics_w2c.length", "extrinsics/frame count mismatch"
            )
        if intrinsics is not None and intrinsics.shape[0] != num_frames:
            _issue(issues, "raw.intrinsics.length", "intrinsics/frame count mismatch")

    image_size = _tensor(payload, "image_size_hw", issues, dtype=torch.int64, ndim=1)
    if image_size is not None:
        if tuple(image_size.shape) != (2,):
            _issue(
                issues, "raw.image_size_hw.shape", "image_size_hw must have shape [2]"
            )
        elif bool((image_size <= 0).any()) or int(image_size.max()) != 384:
            _issue(
                issues,
                "raw.image_size_hw.resolution",
                f"max_size preprocessing must produce positive max dimension 384; got {image_size.tolist()}",
            )

    windows = payload.get("windows")
    if not isinstance(windows, list) or len(windows) != 1:
        _issue(
            issues,
            "raw.windows.whole_trajectory",
            "formal whole-trajectory inference must contain exactly one window",
        )
    elif num_frames is not None and isinstance(windows[0], Mapping):
        window = windows[0]
        _require_equal(issues, "raw.window.start_index", window.get("start_index"), 0)
        _require_equal(
            issues,
            "raw.window.end_index_exclusive",
            window.get("end_index_exclusive"),
            num_frames,
        )
        window_ids = window.get("frame_ids")
        if not isinstance(window_ids, torch.Tensor) or not torch.equal(
            window_ids.cpu(), torch.arange(num_frames)
        ):
            _issue(
                issues,
                "raw.window.frame_ids",
                "the sole window must cover every frame exactly once",
            )
    elif isinstance(windows, list) and windows:
        _issue(issues, "raw.window.type", "window record must be a mapping")
    return result


def _audit_geometry(
    path: Path,
    trajectory_name: str,
    dataset_name: str,
    raw: Mapping[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        result["sha256"] = sha256_file(path)
    except OSError as error:
        _issue(issues, "geometry.sha256", f"cannot hash {path}: {error}")
        return result
    try:
        payload = safe_torch_load(path)
    except Exception as error:  # noqa: BLE001 - corrupt artifacts must be receipted.
        _issue(
            issues,
            "geometry.load",
            f"cannot safely load {path}: {type(error).__name__}: {error}",
        )
        return result

    fixed = {
        "schema_version": 1,
        "motion_type": "geometry",
        "dataset_name": dataset_name,
        "trajectory_name": trajectory_name,
        "pair_direction": "current_to_goal",
        "normalization": "raw",
        "coordinate_frame": "current_navigation_frame",
        "translation_unit": "waypoint_spacing_units",
        "yaw_unit": "radians",
        "components": ["delta_x", "delta_y", "delta_yaw"],
        "source_artifact_type": "vggt_omega_camera_poses",
        "model_id": EXPECTED_MODEL_ID,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "alignment_policy": "tartandrive_forward_camera",
        "ground_truth_usage": "none",
        "pair_generation": EXPECTED_PAIR_GENERATION,
        "complete": True,
    }
    for key, expected in fixed.items():
        _require_equal(issues, f"geometry.{key}", payload.get(key), expected)

    raw_sha = raw.get("sha256")
    if raw_sha is None:
        _issue(
            issues,
            "geometry.source_pose_sha256.unverifiable",
            "raw artifact could not be hashed",
        )
    else:
        _require_equal(
            issues,
            "geometry.source_pose_sha256",
            payload.get("source_pose_sha256"),
            raw_sha,
        )
    raw_frame_list = raw.get("frame_list_sha256")
    if raw_frame_list is not None:
        _require_equal(
            issues,
            "geometry.frame_list_sha256",
            payload.get("frame_list_sha256"),
            raw_frame_list,
        )
    elif not _is_sha256(payload.get("frame_list_sha256")):
        _issue(
            issues,
            "geometry.frame_list_sha256.format",
            "frame_list_sha256 must contain 64 hexadecimal characters",
        )

    policy_fingerprint = _canonical_sha256(EXPECTED_GEOMETRY_POLICY_CONFIGURATION)
    _check_descriptor(
        payload.get("policy_descriptor"),
        expected_configuration=EXPECTED_GEOMETRY_POLICY_CONFIGURATION,
        expected_fingerprint=policy_fingerprint,
        code_prefix="geometry.policy_descriptor",
        issues=issues,
        expected_schema_version=None,
    )

    pairs = _tensor(
        payload,
        "frame_pairs",
        issues,
        dtype=torch.int64,
        ndim=2,
        trailing_shape=(2,),
    )
    motion = _tensor(
        payload,
        "motion",
        issues,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    if pairs is not None:
        result["num_pairs"] = int(pairs.shape[0])
    if pairs is not None and motion is not None:
        if motion.shape[0] != pairs.shape[0]:
            _issue(
                issues, "geometry.motion.length", "motion/frame_pairs length mismatch"
            )
        if not bool(torch.isfinite(motion).all()):
            _issue(
                issues, "geometry.motion.finite", "motion contains non-finite values"
            )
        elif motion.numel() and bool((motion[:, 2].abs() > math.pi + 1e-6).any()):
            _issue(
                issues, "geometry.motion.yaw_range", "delta_yaw lies outside [-pi, pi]"
            )

    num_frames = raw.get("num_frames")
    if pairs is not None and isinstance(num_frames, int):
        expected_pairs = torch.from_numpy(build_nwm_frame_pairs(num_frames))
        if not torch.equal(pairs.cpu(), expected_pairs):
            _issue(
                issues,
                "geometry.frame_pairs.exact_domain",
                "frame_pairs do not exactly match the NWM training index domain",
            )
    scale = payload.get("scale")
    try:
        scale_ok = (
            not isinstance(scale, bool)
            and math.isfinite(float(scale))
            and float(scale) > 0
        )
    except (TypeError, ValueError):
        scale_ok = False
    if not scale_ok:
        _issue(
            issues,
            "geometry.scale",
            f"scale must be positive and finite, got {_display(scale)}",
        )
    if payload.get("scale_status") not in {"estimated", "unused_no_nwm_pairs"}:
        _issue(
            issues,
            "geometry.scale_status",
            f"unexpected scale_status {_display(payload.get('scale_status'))}",
        )
    return result


def _audit_trajectory(
    trajectory_name: str,
    raw_path: Path | None,
    geometry_path: Path | None,
    dataset_name: str,
    expected: bool,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    record: dict[str, Any] = {
        "trajectory_name": trajectory_name,
        "expected": expected,
    }
    raw: dict[str, Any] = {}
    if raw_path is None:
        _issue(issues, "file.raw.missing", "raw pose artifact is missing")
    else:
        raw = _audit_raw(raw_path, trajectory_name, dataset_name, issues)
        record["raw"] = raw
    if geometry_path is None:
        _issue(issues, "file.geometry.missing", "geometry motion artifact is missing")
    else:
        record["geometry"] = _audit_geometry(
            geometry_path,
            trajectory_name,
            dataset_name,
            raw,
            issues,
        )
    record["issues"] = issues
    record["status"] = "pass" if not issues else "fail"
    return record


def _read_expected_trajectories(
    split_files: Sequence[str], issues: list[dict[str, str]]
) -> list[str]:
    names: list[str] = []
    for filename in split_files:
        path = Path(filename)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            _issue(issues, "split.read", f"cannot read {path}: {error}")
            continue
        for line_number, raw_name in enumerate(lines, start=1):
            name = raw_name.strip()
            if not name:
                continue
            if Path(name).name != name or name in {".", ".."}:
                _issue(
                    issues,
                    "split.unsafe_trajectory_name",
                    f"unsafe trajectory name at {path}:{line_number}: {name!r}",
                )
                continue
            names.append(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        _issue(
            issues,
            "split.duplicate_trajectory",
            f"trajectory names occur more than once: {duplicates[:20]}",
        )
    return sorted(set(names))


def _dataset_directory(root: str, dataset_name: str) -> Path:
    path = Path(root).expanduser().resolve()
    return path if path.name == dataset_name else path / dataset_name


def _scan_artifacts(
    directory: Path, kind: str, issues: list[dict[str, str]]
) -> dict[str, Path]:
    if not directory.is_dir():
        _issue(
            issues,
            f"directory.{kind}.missing",
            f"directory does not exist: {directory}",
        )
        return {}
    paths = sorted(path for path in directory.glob("*.pt") if path.is_file())
    return {path.stem: path for path in paths}


def _set_issue(issues: list[dict[str, str]], code: str, names: set[str]) -> None:
    if names:
        ordered = sorted(names)
        _issue(
            issues,
            code,
            f"{len(ordered)} trajectories: {ordered[:20]}"
            + (" ..." if len(ordered) > 20 else ""),
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError(f"audit receipt must use a .json path: {path}")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def audit_dataset(
    *,
    raw_pose_root: str,
    geometry_root: str,
    split_files: Sequence[str],
    output: str,
    dataset_name: str = EXPECTED_DATASET,
    expected_count: int = EXPECTED_TRAJECTORY_COUNT,
    workers: int = 16,
) -> dict[str, Any]:
    global_issues: list[dict[str, str]] = []
    if dataset_name != EXPECTED_DATASET:
        _issue(
            global_issues,
            "protocol.dataset_name",
            f"formal protocol requires {EXPECTED_DATASET!r}, got {dataset_name!r}",
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    expected_names = _read_expected_trajectories(split_files, global_issues)
    if len(expected_names) != expected_count:
        _issue(
            global_issues,
            "set.expected_count",
            f"expected {expected_count} split trajectories, got {len(expected_names)}",
        )

    raw_directory = _dataset_directory(raw_pose_root, dataset_name)
    geometry_directory = _dataset_directory(geometry_root, dataset_name)
    raw_paths = _scan_artifacts(raw_directory, "raw", global_issues)
    geometry_paths = _scan_artifacts(geometry_directory, "geometry", global_issues)
    expected_set = set(expected_names)
    raw_set = set(raw_paths)
    geometry_set = set(geometry_paths)
    _set_issue(global_issues, "set.raw_missing", expected_set - raw_set)
    _set_issue(global_issues, "set.raw_extra", raw_set - expected_set)
    _set_issue(global_issues, "set.geometry_missing", expected_set - geometry_set)
    _set_issue(global_issues, "set.geometry_extra", geometry_set - expected_set)
    _set_issue(global_issues, "set.raw_without_geometry", raw_set - geometry_set)
    _set_issue(global_issues, "set.geometry_without_raw", geometry_set - raw_set)
    if len(raw_paths) != expected_count:
        _issue(
            global_issues,
            "count.raw_files",
            f"expected {expected_count} raw .pt files, got {len(raw_paths)}",
        )
    if len(geometry_paths) != expected_count:
        _issue(
            global_issues,
            "count.geometry_files",
            f"expected {expected_count} geometry .pt files, got {len(geometry_paths)}",
        )

    recomputed_fingerprint = _canonical_sha256(EXPECTED_EXTRACTION_CONFIGURATION)
    if recomputed_fingerprint != EXPECTED_EXTRACTION_FINGERPRINT:
        _issue(
            global_issues,
            "protocol.extraction_fingerprint.internal_consistency",
            "auditor's pinned extraction configuration and fingerprint disagree",
        )

    all_names = sorted(expected_set | raw_set | geometry_set)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _audit_trajectory,
                name,
                raw_paths.get(name),
                geometry_paths.get(name),
                dataset_name,
                name in expected_set,
            ): name
            for name in all_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001 - preserve an audit receipt.
                records.append(
                    {
                        "trajectory_name": name,
                        "expected": name in expected_set,
                        "status": "fail",
                        "issues": [
                            {
                                "code": "audit.unexpected_exception",
                                "severity": "error",
                                "message": f"{type(error).__name__}: {error}",
                            }
                        ],
                    }
                )
    records.sort(key=lambda record: record["trajectory_name"])
    failed = sum(record["status"] != "pass" for record in records)
    frames = sum(int(record.get("raw", {}).get("num_frames", 0)) for record in records)
    pairs = sum(
        int(record.get("geometry", {}).get("num_pairs", 0)) for record in records
    )
    status = "pass" if not global_issues and failed == 0 else "fail"
    artifact_index = [
        {
            "trajectory_name": record["trajectory_name"],
            "raw_sha256": record.get("raw", {}).get("sha256"),
            "geometry_sha256": record.get("geometry", {}).get("sha256"),
        }
        for record in records
    ]
    receipt: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_type": "tartandrive_geometry_provenance_audit",
        "auditor_version": AUDITOR_VERSION,
        "complete": True,
        "status": status,
        "read_only_audit": True,
        "inputs": {
            "dataset_name": dataset_name,
            "raw_pose_directory": str(raw_directory),
            "geometry_directory": str(geometry_directory),
            "split_files": [str(Path(path).resolve()) for path in split_files],
        },
        "protocol": {
            "model_id": EXPECTED_MODEL_ID,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "omega_code_revision": EXPECTED_CODE_REVISION,
            "model_revision": EXPECTED_MODEL_REVISION,
            "input_manifest_sha256": EXPECTED_INPUT_MANIFEST_SHA256,
            "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "extraction_descriptor": {
                "fingerprint": EXPECTED_EXTRACTION_FINGERPRINT,
                "recomputed_fingerprint": recomputed_fingerprint,
                "configuration": EXPECTED_EXTRACTION_CONFIGURATION,
            },
            "geometry_policy_descriptor": {
                "fingerprint": _canonical_sha256(
                    EXPECTED_GEOMETRY_POLICY_CONFIGURATION
                ),
                "configuration": EXPECTED_GEOMETRY_POLICY_CONFIGURATION,
            },
            "pair_generation": EXPECTED_PAIR_GENERATION,
        },
        "counts": {
            "expected_trajectories": len(expected_names),
            "required_trajectories": expected_count,
            "raw_files": len(raw_paths),
            "geometry_files": len(geometry_paths),
            "audited_trajectories": len(records),
            "passed_trajectories": len(records) - failed,
            "failed_trajectories": failed,
            "frames": frames,
            "frame_pairs": pairs,
        },
        "trajectory_set_sha256": _canonical_sha256(expected_names),
        "artifact_index_sha256": _canonical_sha256(artifact_index),
        "global_issues": global_issues,
        "trajectories": records,
    }
    _atomic_json(Path(output), receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of the pinned formal VGGT-Omega TartanDrive raw-pose "
            "and geometry-action artifacts."
        )
    )
    parser.add_argument("--raw-pose-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--split-files", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Atomic JSON audit receipt")
    parser.add_argument("--dataset-name", default=EXPECTED_DATASET)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TRAJECTORY_COUNT)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_count < 0:
        _parser().error("--expected-count must be non-negative")
    receipt = audit_dataset(
        raw_pose_root=args.raw_pose_root,
        geometry_root=args.geometry_root,
        split_files=args.split_files,
        output=args.output,
        dataset_name=args.dataset_name,
        expected_count=args.expected_count,
        workers=args.workers,
    )
    print(
        json.dumps(
            {"status": receipt["status"], **receipt["counts"]},
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
