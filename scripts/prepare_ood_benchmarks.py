#!/usr/bin/env python3
"""Prepare and audit the fixed 4-second out-of-domain NWM benchmark.

The four benchmark datasets are ``planetary_rover`` (Mars and Moon combined),
``unitree_go2``, ``tum_rgbd`` and ``uzh_fpv``.  Every output trajectory uses
the repository's native layout::

    <data-root>/<dataset>/<trajectory>/{0.jpg, ..., traj_data.pkl}

``traj_data.pkl`` contains measured/interpolated metric positions and measured
orientations reduced to yaw.  Images are always real source frames: temporal
resampling selects the nearest acquired frame and never interpolates pixels.

The fixed prediction indexing contract is four context frames, sixteen future
frames, and 500 deterministic windows per dataset. Unitree, TUM, and UZH are
resampled to 4 Hz, so this is a physical 4-second horizon. Planetary Rover has
no fixed-rate physical timestamps and uses the same 16-waypoint spatial
horizon instead. A 100-window ``navigation_eval.pkl`` is also emitted so the
same real trajectories can be loaded by ``planning_eval.py``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

import numpy as np
from PIL import Image


FPS = 4
CONTEXT_SIZE = 4
HORIZON_FRAMES = 16
SAMPLE_COUNT = 500
NAVIGATION_SAMPLE_COUNT = 100
QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm/data")
DEFAULT_REFERENCE_ROOT = DEFAULT_DATA_ROOT / "go_stanford"
DEFAULT_UNITREE_SOURCE = Path(
    "/file_system/nas/algorithm/dujun.nie/datasets/unitree-go2-data/nwm_4hz"
)
DEFAULT_TUM_SOURCE = Path(
    "/file_system/nas/algorithm/dujun.nie/datasets/nwm_ood_sources/tum_rgbd"
)
DEFAULT_UZH_SOURCE = Path(
    "/file_system/nas/algorithm/dujun.nie/datasets/nwm_ood_sources/uzh_fpv"
)

TUM_ARCHIVE_SEQUENCES = ("xyz", "floor", "desk", "desk2", "room")
TUM_URL = (
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
    "rgbd_dataset_freiburg1_{sequence}.tgz"
)
TUM_HF_REPOSITORY = "voviktyl/TUM_RGBD-SLAM"
TUM_HF_REVISION = "76818471ae555dd6cd4e100b7cedcccda660448d"
TUM_HF_SEQUENCES = (
    ("freiburg1_desk", "rgbd_dataset_freiburg1_desk"),
    ("freiburg1_desk2", "rgbd_dataset_freiburg1_desk2"),
    ("freiburg1_room", "rgbd_dataset_freiburg1_room"),
    ("freiburg2_xyz", "rgbd_dataset_freiburg2_xyz"),
    (
        "freiburg3_long_office_household",
        "rgbd_dataset_freiburg3_long_office_household",
    ),
)
TUM_HF_BASE_URL = (
    f"https://huggingface.co/datasets/{TUM_HF_REPOSITORY}/resolve/"
    f"{TUM_HF_REVISION}"
)
UZH_REVISION = "97c0a1132b22f8908925d057e1aa1c12148f056e"
# The pinned HF derivative was built from a lexicographically sorted source
# file list.  Keep that exact order: episode duration and a direct source-frame
# pixel match pin episode 0 to indoor_45_12 (not indoor_45_2).
UZH_SEQUENCES = (
    "indoor_45_12",
    "indoor_45_13",
    "indoor_45_14",
    "indoor_45_2",
    "indoor_45_4",
    "indoor_45_9",
    "indoor_forward_10",
    "indoor_forward_3",
    "indoor_forward_5",
    "indoor_forward_6",
    "indoor_forward_7",
    "indoor_forward_9",
)
UZH_GT_URL = (
    "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/v3/"
    "{sequence}_snapdragon_with_gt.zip"
)


@dataclass(frozen=True)
class Anchor:
    trajectory: str
    current: int
    displacement_m: float


class UZHSequenceTooShort(ValueError):
    """Raised when the public video and official GT overlap cannot form a window."""

    def __init__(self, sequence: str, valid_frames: int) -> None:
        self.sequence = sequence
        self.valid_frames = valid_frames
        super().__init__(
            f"UZH sequence too short after GT bracketing: {sequence} "
            f"({valid_frames} < {CONTEXT_SIZE + HORIZON_FRAMES} frames)"
        )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    atomic_write(path, payload.encode("utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_pickle(path: Path, payload: Any) -> None:
    atomic_write(path, pickle.dumps(payload, protocol=4))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def wrap_radians(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2 * np.pi) - np.pi


def quantile_dict(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot summarize an empty distribution")
    return {
        str(level): float(value)
        for level, value in zip(QUANTILES, np.quantile(array, QUANTILES))
    }


def nearest_indices(
    timestamps: Sequence[float], grid: Sequence[float], max_error: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique real frames nearest to an explicit time grid."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Source timestamps must be strictly increasing")
    right = np.clip(np.searchsorted(timestamps, grid), 0, len(timestamps) - 1)
    left = np.maximum(0, right - 1)
    use_left = np.abs(timestamps[left] - grid) <= np.abs(timestamps[right] - grid)
    selected = np.where(use_left, left, right)
    errors = np.abs(timestamps[selected] - grid)
    if np.any(np.diff(selected) <= 0):
        raise ValueError("Temporal grid selected a source image more than once")
    if len(errors) and float(np.max(errors)) > max_error:
        raise ValueError(
            f"Source cadence is too sparse: max error {float(np.max(errors)):.6f}s "
            f"> {max_error:.6f}s"
        )
    return selected.astype(np.int64), errors


def tum_bracketed_grid_indices(
    rgb_timestamps: Sequence[float], pose_timestamps: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the 4 Hz real RGB frames whose timestamps can be pose-interpolated."""

    rgb_timestamps = np.asarray(rgb_timestamps, dtype=np.float64)
    pose_timestamps = np.asarray(pose_timestamps, dtype=np.float64)
    if not len(rgb_timestamps) or np.any(np.diff(rgb_timestamps) <= 0):
        raise ValueError("TUM RGB timestamps must be strictly increasing")
    if len(pose_timestamps) < 2 or np.any(np.diff(pose_timestamps) <= 0):
        raise ValueError("TUM pose timestamps must be strictly increasing")
    bracketed_rgb_indices = np.flatnonzero(
        (rgb_timestamps >= pose_timestamps[0])
        & (rgb_timestamps <= pose_timestamps[-1])
    )
    if not len(bracketed_rgb_indices):
        raise ValueError("TUM RGB frames and poses do not overlap")
    bracketed_rgb_timestamps = rgb_timestamps[bracketed_rgb_indices]
    grid = bracketed_rgb_timestamps[0] + np.arange(
        int(
            np.floor(
                (bracketed_rgb_timestamps[-1] - bracketed_rgb_timestamps[0])
                * FPS
            )
        )
        + 1
    ) / FPS
    selected_in_bracket, errors = nearest_indices(
        bracketed_rgb_timestamps, grid, max_error=0.5 / FPS
    )
    return bracketed_rgb_indices[selected_in_bracket], grid, errors


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-12) or not np.isfinite(norms).all():
        raise ValueError("Invalid quaternion")
    return quaternions / norms


def slerp_quaternions(
    left: np.ndarray, right: np.ndarray, fraction: np.ndarray
) -> np.ndarray:
    """Vectorized shortest-path SLERP for XYZW quaternions."""

    left = normalize_quaternions(left)
    right = normalize_quaternions(right)
    fraction = np.asarray(fraction, dtype=np.float64).reshape(-1, 1)
    dots = np.sum(left * right, axis=1, keepdims=True)
    right = np.where(dots < 0, -right, right)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    near = dots > 0.9995
    theta = np.arccos(dots)
    denominator = np.sin(theta)
    denominator = np.where(near, 1.0, denominator)
    result = (
        np.sin((1.0 - fraction) * theta) / denominator * left
        + np.sin(fraction * theta) / denominator * right
    )
    linear = (1.0 - fraction) * left + fraction * right
    result = np.where(near, linear, result)
    return normalize_quaternions(result)


def interpolate_poses(
    pose_times: Sequence[float],
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    target_times: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_times = np.asarray(pose_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    quaternions_xyzw = normalize_quaternions(quaternions_xyzw)
    if np.any(np.diff(pose_times) <= 0):
        raise ValueError("Pose timestamps must be strictly increasing")
    if len(positions) != len(pose_times) or len(quaternions_xyzw) != len(pose_times):
        raise ValueError("Pose array lengths differ")
    if len(target_times) and (
        target_times[0] < pose_times[0] or target_times[-1] > pose_times[-1]
    ):
        raise ValueError("Target image time is not bracketed by real poses")
    right = np.searchsorted(pose_times, target_times, side="left")
    right = np.clip(right, 1, len(pose_times) - 1)
    left = right - 1
    duration = pose_times[right] - pose_times[left]
    fraction = (target_times - pose_times[left]) / duration
    output_position = (
        positions[left] * (1.0 - fraction[:, None])
        + positions[right] * fraction[:, None]
    )
    output_quaternion = slerp_quaternions(
        quaternions_xyzw[left], quaternions_xyzw[right], fraction
    )
    return output_position, output_quaternion, duration


def quaternion_yaw_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternions(quaternion)
    x, y, z, w = quaternion.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quaternion_camera_forward_yaw(quaternion: np.ndarray) -> np.ndarray:
    """TUM optical +Z, rather than image-right +X, is navigation forward."""
    x, y, z, w = normalize_quaternions(quaternion).T
    forward_x, forward_y = 2 * (x * z + w * y), 2 * (y * z - w * x)
    if np.any(np.hypot(forward_x, forward_y) < 1e-6):
        raise ValueError("Vertical optical axis has no planar heading")
    return np.arctan2(forward_y, forward_x)


def jpeg_bytes(image: Image.Image, *, quality: int = 95) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(
        output, format="JPEG", quality=quality, optimize=True, subsampling=0
    )
    return output.getvalue()


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with (path / "traj_data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    position = np.asarray(payload["position"], dtype=np.float64)
    yaw = np.asarray(payload["yaw"], dtype=np.float64).reshape(-1)
    if position.ndim != 2 or position.shape[1] < 2 or len(position) != len(yaw):
        raise ValueError(f"Invalid trajectory arrays in {path}: {position.shape}/{yaw.shape}")
    if not np.isfinite(position).all() or not np.isfinite(yaw).all():
        raise ValueError(f"Non-finite trajectory in {path}")
    return position, yaw


def reference_displacements(reference_root: Path, reference_split: Path) -> np.ndarray:
    with reference_split.open("rb") as stream:
        entries = pickle.load(stream)
    cache: dict[str, np.ndarray] = {}
    distances = []
    for entry in entries:
        name, current = str(entry[0]), int(entry[1])
        if name not in cache:
            cache[name] = load_trajectory(reference_root / name)[0]
        position = cache[name]
        distances.append(
            float(np.linalg.norm(position[current + HORIZON_FRAMES, :2] - position[current, :2]))
        )
    if len(distances) != SAMPLE_COUNT:
        raise ValueError(
            f"Go Stanford reference must contain {SAMPLE_COUNT} samples, got {len(distances)}"
        )
    return np.asarray(distances, dtype=np.float64)


def candidate_anchors(output: Path, names: Sequence[str]) -> list[Anchor]:
    candidates: list[Anchor] = []
    for name in sorted(names):
        position, _ = load_trajectory(output / name)
        for current in range(CONTEXT_SIZE - 1, len(position) - HORIZON_FRAMES):
            distance = float(
                np.linalg.norm(
                    position[current + HORIZON_FRAMES, :2] - position[current, :2]
                )
            )
            candidates.append(Anchor(name, current, distance))
    return candidates


def _remove_nearest_targets(targets: list[float], values: Sequence[float]) -> list[float]:
    remaining = list(targets)
    for value in sorted(values):
        index = min(
            range(len(remaining)),
            key=lambda item: (abs(remaining[item] - value), item),
        )
        remaining.pop(index)
    return remaining


def match_distance_distribution(
    candidates: Sequence[Anchor],
    target_distances: Sequence[float],
    count: int,
    forced: Sequence[Anchor] = (),
) -> list[Anchor]:
    """Find a deterministic order-preserving minimum-cost 1-D match.

    Candidate and reference distances are sorted before dynamic programming.
    The resulting subset minimizes absolute displacement error without
    replacement.  ``forced`` is used only to retain explicitly requested scene
    coverage (the Moon portion of the combined planetary dataset).
    """

    forced = list(dict.fromkeys(forced))
    if len(forced) > count:
        raise ValueError("More forced anchors than requested samples")
    forced_set = set(forced)
    available = sorted(
        (item for item in candidates if item not in forced_set),
        key=lambda item: (item.displacement_m, item.trajectory, item.current),
    )
    targets = sorted(float(value) for value in target_distances)
    targets = _remove_nearest_targets(
        targets, [item.displacement_m for item in forced]
    )
    required = count - len(forced)
    if len(available) < required or len(targets) < required:
        raise ValueError(
            f"Insufficient anchors/targets: {len(available)}/{len(targets)} for {required}"
        )
    if len(targets) != required:
        quantile = (np.arange(required, dtype=np.float64) + 0.5) / required
        targets = np.quantile(np.asarray(targets), quantile).tolist()
    n = len(available)
    back = np.full((required, n), -1, dtype=np.int32)
    candidate_values = np.asarray(
        [item.displacement_m for item in available], dtype=np.float64
    )
    previous = np.abs(candidate_values - targets[0])
    previous[n - required + 1 :] = np.inf
    for target_index in range(1, required):
        prefix_value = np.empty(n, dtype=np.float64)
        prefix_arg = np.empty(n, dtype=np.int32)
        best_value = np.inf
        best_arg = -1
        for candidate_index, value in enumerate(previous):
            if value < best_value:
                best_value = float(value)
                best_arg = candidate_index
            prefix_value[candidate_index] = best_value
            prefix_arg[candidate_index] = best_arg
        current = np.full(n, np.inf, dtype=np.float64)
        first = target_index
        last = n - (required - target_index)
        indices = np.arange(first, last + 1)
        predecessor = indices - 1
        current[indices] = (
            np.abs(candidate_values[indices] - targets[target_index])
            + prefix_value[predecessor]
        )
        back[target_index, indices] = prefix_arg[predecessor]
        previous = current
    end = int(np.argmin(previous))
    if not np.isfinite(previous[end]):
        raise RuntimeError("Distance matching did not find a finite solution")
    selected_indices = [end]
    for target_index in range(required - 1, 0, -1):
        end = int(back[target_index, end])
        if end < 0:
            raise RuntimeError("Broken dynamic-programming backpointer")
        selected_indices.append(end)
    selected_indices.reverse()
    selected = [available[index] for index in selected_indices] + list(forced)
    return sorted(
        selected,
        key=lambda item: (item.displacement_m, item.trajectory, item.current),
    )


def evenly_spaced(items: Sequence[Any], count: int) -> list[Any]:
    if len(items) < count:
        raise ValueError(f"Need {count} items, only {len(items)} are available")
    indices = np.floor(np.linspace(0, len(items), count, endpoint=False)).astype(int)
    return [items[int(index)] for index in indices]


def infer_spacing(output: Path, names: Sequence[str]) -> float:
    steps = []
    for name in names:
        position, _ = load_trajectory(output / name)
        steps.extend(np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1).tolist())
    moving = np.asarray([value for value in steps if value > 1e-8], dtype=np.float64)
    if not len(moving):
        raise ValueError("Dataset has no measured planar translation")
    return float(np.mean(moving))


def trajectory_names(output: Path) -> list[str]:
    return sorted(
        child.name
        for child in output.iterdir()
        if child.is_dir() and (child / "traj_data.pkl").is_file()
    )


def validate_images(output: Path, names: Sequence[str]) -> tuple[int, list[float]]:
    frame_count = 0
    steps: list[float] = []
    for name in names:
        position, _ = load_trajectory(output / name)
        expected = [f"{index}.jpg" for index in range(len(position))]
        actual = sorted(
            (path.name for path in (output / name).glob("*.jpg")),
            key=lambda value: int(Path(value).stem),
        )
        if actual != expected:
            raise ValueError(f"Non-contiguous images in {output / name}")
        for index in {0, len(position) // 2, len(position) - 1}:
            with Image.open(output / name / f"{index}.jpg") as image:
                image.verify()
        frame_count += len(position)
        steps.extend(np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1).tolist())
    return frame_count, steps


def prune_stale_indexed_images(trajectory: Path, frame_count: int) -> list[str]:
    """Remove only numeric JPEGs left beyond a regenerated trajectory's end."""

    removed = []
    for path in trajectory.glob("*.jpg"):
        if path.stem.isdigit() and int(path.stem) >= frame_count:
            path.unlink()
            removed.append(path.name)
    return sorted(removed, key=lambda name: int(Path(name).stem))


def finalize_dataset(
    dataset: str,
    output: Path,
    split_root: Path,
    reference_root: Path,
    reference_split: Path,
    *,
    spacing: float | None,
    provenance: dict[str, Any],
    cadence: str = "fixed_4hz",
    force_prefix: str | None = None,
    force_count: int = 0,
) -> dict[str, Any]:
    names = trajectory_names(output)
    if not names:
        raise ValueError(f"No trajectories found under {output}")
    frames, all_steps = validate_images(output, names)
    candidates = candidate_anchors(output, names)
    if len(candidates) < SAMPLE_COUNT:
        raise ValueError(
            f"{dataset} has only {len(candidates)} valid 4-second windows; "
            f"{SAMPLE_COUNT} are required"
        )
    reference = reference_displacements(reference_root, reference_split)
    forced: list[Anchor] = []
    if force_prefix and force_count:
        matching = sorted(
            (item for item in candidates if item.trajectory.startswith(force_prefix)),
            key=lambda item: (item.trajectory, item.current),
        )
        forced = evenly_spaced(matching, min(force_count, len(matching)))
    selected = match_distance_distribution(
        candidates, reference, SAMPLE_COUNT, forced=forced
    )
    selected_entries = [
        (
            item.trajectory,
            item.current,
            -min(item.current, 64),
            HORIZON_FRAMES,
        )
        for item in selected
    ]
    navigation_source = sorted(selected, key=lambda item: (item.trajectory, item.current))
    navigation = [
        (item.trajectory, item.current, 8, 8)
        for item in evenly_spaced(navigation_source, NAVIGATION_SAMPLE_COUNT)
    ]
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(f"{name}\n" for name in names)
    for filename in ("traj_names.txt", "rollout_traj_names.txt", "all_traj_names.txt"):
        atomic_text(split_root / filename, manifest_text)
    atomic_pickle(split_root / "time.pkl", selected_entries)
    atomic_pickle(split_root / "navigation_eval.pkl", navigation)

    if spacing is None:
        spacing = infer_spacing(output, names)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Invalid waypoint spacing: {spacing}")
    selected_distances = [item.displacement_m for item in selected]
    reference_distances = reference.tolist()
    report_path = output / "dataset_report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "schema_version": 1,
            "dataset": dataset,
            "created_or_finalized_utc": utc_now(),
            "trajectories": len(names),
            "frames": frames,
            "candidate_windows_context4_horizon16": len(candidates),
            "prediction_samples": SAMPLE_COUNT,
            "navigation_samples": NAVIGATION_SAMPLE_COUNT,
            "input_fps": FPS,
            "context_frames": CONTEXT_SIZE,
            "future_frames": HORIZON_FRAMES,
            "horizon_seconds": HORIZON_FRAMES / FPS,
            "image_resampling": "nearest real acquired frame; pixels never interpolated",
            "trajectory_cadence": cadence,
            "metric_waypoint_spacing": spacing,
            "spacing_policy": "mean nonzero measured planar displacement at output cadence"
            if provenance.get("spacing_policy") is None
            else provenance["spacing_policy"],
            "all_planar_step_m": quantile_dict(all_steps),
            "go_stanford_reference_4s_displacement_m": quantile_dict(
                reference_distances
            ),
            "selected_4s_displacement_m": quantile_dict(selected_distances),
            "selected_mean_absolute_reference_quantile_error_m": float(
                np.mean(
                    np.abs(
                        np.sort(np.asarray(selected_distances))
                        - np.sort(np.asarray(reference_distances))
                    )
                )
            ),
            "forced_scene_coverage": {
                "trajectory_prefix": force_prefix,
                "requested": force_count,
                "selected": len(forced),
            }
            if force_prefix
            else None,
            "provenance": provenance,
        }
    )
    atomic_json(
        output / "dataset_config.json",
        {
            "dataset": dataset,
            "data_folder": str(output.resolve()),
            "test_split": str(split_root.resolve()),
            "metric_waypoint_spacing": spacing,
            "input_fps": FPS,
            "context_frames": CONTEXT_SIZE,
            "future_frames": HORIZON_FRAMES,
            "prediction_sample_count": SAMPLE_COUNT,
            "navigation_sample_count": NAVIGATION_SAMPLE_COUNT,
        },
    )
    # Hash only after every pinned split file has reached its final contents.
    report["splits"] = {
        filename: {
            "path": str((split_root / filename).resolve()),
            "sha256": sha256_file(split_root / filename),
        }
        for filename in (
            "traj_names.txt",
            "rollout_traj_names.txt",
            "time.pkl",
            "navigation_eval.pkl",
        )
    }
    report["selected_entries_sha256"] = sha256_bytes(
        json.dumps(selected_entries, separators=(",", ":")).encode("utf-8")
    )
    atomic_json(report_path, report)
    atomic_json(output / "ood_protocol_success.json", report)
    return report


def hardlink_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing target differs in size: {target}")
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def prepare_unitree(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output.resolve()
    report = json.loads((source / "conversion_report.json").read_text(encoding="utf-8"))
    source_root = source / "go2"
    names = sorted(child.name for child in source_root.iterdir() if child.is_dir())
    operations: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    for name in names:
        for path in sorted((source_root / name).iterdir()):
            if path.is_file():
                operation = hardlink_or_copy(path, output / name / path.name)
                operations[operation] += 1
    atomic_json(
        output / "source_manifest.json",
        {
            "source": str(source),
            "source_conversion_report_sha256": sha256_file(
                source / "conversion_report.json"
            ),
            "episodes": report["episodes"],
            "segments": len(report["segments"]),
            "materialization": operations,
        },
    )
    return finalize_dataset(
        "unitree_go2",
        output,
        args.split_root.resolve(),
        args.reference_root.resolve(),
        args.reference_split.resolve(),
        spacing=float(report["metric_waypoint_spacing"]),
        provenance={
            "source": str(source),
            "source_kind": "seven locally recorded Unitree Go2 videos",
            "real_pose": True,
            "pose_binding": "same aligned source row as each selected real frame",
            "source_conversion_report_sha256": sha256_file(
                source / "conversion_report.json"
            ),
            "spacing_policy": report["spacing_policy"],
        },
    )


def parse_tum_text(
    payload: bytes, columns: int, *, collapse_duplicate_timestamps: bool = False
) -> np.ndarray:
    rows = []
    for raw in payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < columns:
            raise ValueError(f"Malformed TUM row: {line}")
        rows.append([float(value) for value in fields[:columns]])
    array = np.asarray(rows, dtype=np.float64)
    if not len(array) or np.any(np.diff(array[:, 0]) < 0):
        raise ValueError("Invalid TUM timestamps")
    if np.any(np.diff(array[:, 0]) == 0):
        if not collapse_duplicate_timestamps:
            raise ValueError("Invalid TUM timestamps")
        collapsed = []
        for timestamp in np.unique(array[:, 0]):
            group = array[array[:, 0] == timestamp]
            row = group.mean(axis=0)
            row[0] = timestamp
            if columns >= 8:
                quaternions = group[:, 4:8].copy()
                quaternions[(quaternions @ quaternions[0]) < 0] *= -1
                quaternion = quaternions.mean(axis=0)
                norm = np.linalg.norm(quaternion)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("Invalid duplicate TUM quaternion")
                row[4:8] = quaternion / norm
            collapsed.append(row)
        array = np.asarray(collapsed, dtype=np.float64)
    if np.any(np.diff(array[:, 0]) <= 0):
        raise ValueError("Invalid TUM timestamps after duplicate collapse")
    return array


def download_cached(url: str, target: Path, *, attempts: int = 8) -> dict[str, Any]:
    """Download one pinned source file atomically, retrying transient failures."""

    if target.is_file() and target.stat().st_size > 0:
        return {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "cached": True,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download-{os.getpid()}")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "CompACT-OOD-benchmark/1.0"}
            )
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
                "wb"
            ) as stream:
                expected_header = response.headers.get("Content-Length")
                expected = int(expected_header) if expected_header else None
                written = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    written += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if expected is not None and written != expected:
                raise IOError(f"Short download for {url}: {written} != {expected}")
            if written <= 0:
                raise IOError(f"Empty download for {url}")
            os.replace(temporary, target)
            return {
                "path": str(target),
                "bytes": written,
                "sha256": sha256_file(target),
                "cached": False,
            }
        except Exception as error:  # network errors vary by Python/OpenSSL version
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_error


def tum_hf_url(relative_path: str) -> str:
    return f"{TUM_HF_BASE_URL}/{relative_path}?download=true"


def prepare_tum_hf_sequence(
    source: Path,
    sequence: str,
    repository_directory: str,
    output: Path,
    *,
    download_workers: int,
) -> dict[str, Any]:
    """Materialize only the real RGB frames needed by the 4 Hz protocol."""

    cache_root = source / "hf_mirror" / repository_directory
    text_sources: dict[str, dict[str, Any]] = {}
    for basename in ("rgb.txt", "groundtruth.txt"):
        relative = f"{repository_directory}/{basename}"
        text_sources[basename] = download_cached(
            tum_hf_url(relative), cache_root / basename
        )

    rgb_payload = (cache_root / "rgb.txt").read_bytes()
    groundtruth_payload = (cache_root / "groundtruth.txt").read_bytes()
    rgb_timestamps: list[float] = []
    rgb_paths: list[str] = []
    for raw in rgb_payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative_path = line.split(maxsplit=1)
        rgb_timestamps.append(float(timestamp))
        rgb_paths.append(relative_path)
    rgb_timestamps_array = np.asarray(rgb_timestamps, dtype=np.float64)
    if not len(rgb_timestamps_array) or np.any(np.diff(rgb_timestamps_array) <= 0):
        raise ValueError(f"Invalid TUM RGB timestamps for {sequence}")
    raw_groundtruth_rows = sum(
        1
        for raw in groundtruth_payload.decode("utf-8").splitlines()
        if raw.strip() and not raw.strip().startswith("#")
    )
    groundtruth = parse_tum_text(
        groundtruth_payload, 8, collapse_duplicate_timestamps=True
    )
    selected, grid, errors = tum_bracketed_grid_indices(
        rgb_timestamps_array, groundtruth[:, 0]
    )
    selected_times = rgb_timestamps_array[selected]
    positions, quaternions, brackets = interpolate_poses(
        groundtruth[:, 0],
        groundtruth[:, 1:4],
        groundtruth[:, 4:8],
        selected_times,
    )
    yaw = quaternion_camera_forward_yaw(quaternions)

    requested: dict[int, tuple[str, Path]] = {}
    for source_index in selected.tolist():
        relative_image = rgb_paths[source_index]
        requested[source_index] = (
            f"{repository_directory}/{relative_image}",
            cache_root / relative_image,
        )
    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {
            executor.submit(download_cached, tum_hf_url(relative), cache_path): source_index
            for source_index, (relative, cache_path) in requested.items()
        }
        for count, future in enumerate(as_completed(futures), 1):
            source_index = futures[future]
            completed[source_index] = future.result()
            if count % 100 == 0 or count == len(futures):
                print(
                    f"{sequence}: source RGB {count}/{len(futures)}",
                    flush=True,
                )

    name = f"tum_{sequence}"
    trajectory = output / name
    trajectory.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_index, source_index_value in enumerate(selected):
        source_index = int(source_index_value)
        relative_image = rgb_paths[source_index]
        source_path = cache_root / relative_image
        source_bytes = source_path.read_bytes()
        with Image.open(io.BytesIO(source_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(source_bytes)) as image:
            source_size = image.size
            encoded = jpeg_bytes(image)
        atomic_write(trajectory / f"{output_index}.jpg", encoded)
        rows.append(
            {
                "frame": output_index,
                "grid_timestamp": float(grid[output_index]),
                "image_timestamp": float(selected_times[output_index]),
                "nearest_image_error_s": float(errors[output_index]),
                "pose_bracket_s": float(brackets[output_index]),
                "position_xyz": positions[output_index].tolist(),
                "quaternion_xyzw": quaternions[output_index].tolist(),
                "yaw": float(yaw[output_index]),
                "source_relative_path": f"{repository_directory}/{relative_image}",
                "source_url": tum_hf_url(
                    f"{repository_directory}/{relative_image}"
                ),
                "source_image_sha256": completed[source_index]["sha256"],
                "source_size": list(source_size),
                "processed_jpeg_sha256": sha256_bytes(encoded),
            }
        )
    atomic_pickle(
        trajectory / "traj_data.pkl", {"position": positions[:, :2], "yaw": yaw}
    )
    write_jsonl(trajectory / "frame_metadata.jsonl", rows)
    return {
        "trajectory": name,
        "sequence": sequence,
        "repository_directory": repository_directory,
        "frames": len(selected),
        "duration_s": float(selected_times[-1] - selected_times[0]),
        "max_nearest_image_error_s": float(np.max(errors)),
        "max_pose_bracket_s": float(np.max(brackets)),
        "duplicate_groundtruth_rows_collapsed": raw_groundtruth_rows
        - len(groundtruth),
        "duplicate_groundtruth_policy": "mean metric position and hemisphere-aligned normalized mean quaternion at identical published timestamps",
        "metadata_sources": text_sources,
    }


def tum_archive_metadata(archive_path: Path) -> tuple[bytes, bytes, dict[str, bytes]]:
    text: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            basename = Path(member.name).name
            if basename in {"rgb.txt", "groundtruth.txt"} and member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read {member.name}")
                text[basename] = stream.read()
    missing = {"rgb.txt", "groundtruth.txt"}.difference(text)
    if missing:
        raise ValueError(f"{archive_path} is missing {sorted(missing)}")
    rgb_rows: dict[str, bytes] = {}
    for raw in text["rgb.txt"].decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative_path = line.split(maxsplit=1)
        rgb_rows[relative_path] = timestamp.encode("ascii")
    return text["rgb.txt"], text["groundtruth.txt"], rgb_rows


def prepare_tum_sequence(
    archive_path: Path, sequence: str, output: Path
) -> dict[str, Any]:
    rgb_payload, groundtruth_payload, _ = tum_archive_metadata(archive_path)
    rgb_timestamps = []
    rgb_paths = []
    for raw in rgb_payload.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative_path = line.split(maxsplit=1)
        rgb_timestamps.append(float(timestamp))
        rgb_paths.append(relative_path)
    rgb_timestamps_array = np.asarray(rgb_timestamps, dtype=np.float64)
    raw_groundtruth_rows = sum(
        1
        for raw in groundtruth_payload.decode("utf-8").splitlines()
        if raw.strip() and not raw.strip().startswith("#")
    )
    groundtruth = parse_tum_text(
        groundtruth_payload, 8, collapse_duplicate_timestamps=True
    )
    selected, grid, errors = tum_bracketed_grid_indices(
        rgb_timestamps_array, groundtruth[:, 0]
    )
    selected_times = rgb_timestamps_array[selected]
    positions, quaternions, brackets = interpolate_poses(
        groundtruth[:, 0],
        groundtruth[:, 1:4],
        groundtruth[:, 4:8],
        selected_times,
    )
    yaw = quaternion_camera_forward_yaw(quaternions)
    name = f"tum_freiburg1_{sequence}"
    trajectory = output / name
    trajectory.mkdir(parents=True, exist_ok=True)
    wanted = {rgb_paths[int(source_index)]: output_index for output_index, source_index in enumerate(selected)}
    emitted: set[int] = set()
    image_metadata: dict[int, dict[str, Any]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            relative = "/".join(Path(member.name).parts[1:])
            if relative not in wanted:
                continue
            source_stream = archive.extractfile(member)
            if source_stream is None:
                raise ValueError(f"Cannot read {member.name}")
            source_bytes = source_stream.read()
            with Image.open(io.BytesIO(source_bytes)) as image:
                encoded = jpeg_bytes(image)
                source_size = image.size
            output_index = wanted[relative]
            atomic_write(trajectory / f"{output_index}.jpg", encoded)
            emitted.add(output_index)
            image_metadata[output_index] = {
                "source_relative_path": relative,
                "source_image_sha256": sha256_bytes(source_bytes),
                "source_size": list(source_size),
                "processed_jpeg_sha256": sha256_bytes(encoded),
            }
    if emitted != set(range(len(selected))):
        missing = sorted(set(range(len(selected))).difference(emitted))
        raise ValueError(f"TUM archive image extraction incomplete: {missing[:10]}")
    atomic_pickle(
        trajectory / "traj_data.pkl",
        {"position": positions[:, :2], "yaw": yaw},
    )
    rows = []
    for output_index, source_index in enumerate(selected):
        rows.append(
            {
                "frame": output_index,
                "grid_timestamp": float(grid[output_index]),
                "image_timestamp": float(selected_times[output_index]),
                "nearest_image_error_s": float(errors[output_index]),
                "pose_bracket_s": float(brackets[output_index]),
                "position_xyz": positions[output_index].tolist(),
                "quaternion_xyzw": quaternions[output_index].tolist(),
                "yaw": float(yaw[output_index]),
                **image_metadata[output_index],
            }
        )
    write_jsonl(trajectory / "frame_metadata.jsonl", rows)
    return {
        "trajectory": name,
        "sequence": sequence,
        "frames": len(selected),
        "duration_s": float(selected_times[-1] - selected_times[0]),
        "max_nearest_image_error_s": float(np.max(errors)),
        "max_pose_bracket_s": float(np.max(brackets)),
        "duplicate_groundtruth_rows_collapsed": raw_groundtruth_rows
        - len(groundtruth),
        "duplicate_groundtruth_policy": "mean metric position and hemisphere-aligned normalized mean quaternion at identical published timestamps",
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "source_url": TUM_URL.format(sequence=sequence),
    }


def prepare_tum(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output.resolve()
    records = []
    if args.source_mode == "archives":
        for index, sequence in enumerate(TUM_ARCHIVE_SEQUENCES, 1):
            archive = source / f"rgbd_dataset_freiburg1_{sequence}.tgz"
            if not archive.is_file():
                raise FileNotFoundError(archive)
            print(
                f"TUM {index}/{len(TUM_ARCHIVE_SEQUENCES)}: {sequence}", flush=True
            )
            records.append(prepare_tum_sequence(archive, sequence, output))
        source_manifest = {
            "mode": "official_archives",
            "sequences": records,
        }
    else:
        for index, (sequence, repository_directory) in enumerate(
            TUM_HF_SEQUENCES, 1
        ):
            print(f"TUM {index}/{len(TUM_HF_SEQUENCES)}: {sequence}", flush=True)
            records.append(
                prepare_tum_hf_sequence(
                    source,
                    sequence,
                    repository_directory,
                    output,
                    download_workers=args.download_workers,
                )
            )
        source_manifest = {
            "mode": "pinned_huggingface_mirror_selective_rgb",
            "repository": TUM_HF_REPOSITORY,
            "revision": TUM_HF_REVISION,
            "base_url": TUM_HF_BASE_URL,
            "sequences": records,
        }
    atomic_json(output / "source_manifest.json", source_manifest)
    return finalize_dataset(
        "tum_rgbd",
        output,
        args.split_root.resolve(),
        args.reference_root.resolve(),
        args.reference_split.resolve(),
        spacing=None,
        provenance={
            "official_dataset": "TUM RGB-D",
            "official_page": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
            "source_manifest": source_manifest,
            "sequences": records,
            "real_pose": True,
            "pose_policy": "linear metric-position interpolation and shortest-path quaternion SLERP at each selected real RGB timestamp",
            "planar_policy": "NWM traj_data.pkl uses measured world XY and quaternion yaw; measured Z is retained per frame in frame_metadata.jsonl",
        },
    )


class HTTPRangeReader(io.RawIOBase):
    """Small seekable HTTP reader used to extract metadata from large ZIPs."""

    def __init__(self, url: str):
        super().__init__()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url, method="HEAD")
        with self.opener.open(request, timeout=120) as response:
            self.url = response.geturl()
            length = response.headers.get("Content-Length")
            if length is None:
                raise ValueError(f"Remote ZIP has no Content-Length: {url}")
            self.length = int(length)
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Invalid seek mode: {whence}")
        if position < 0:
            raise ValueError("Negative seek")
        self.position = min(position, self.length)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        if size == 0:
            return b""
        end = min(self.length - 1, self.position + size - 1)
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        with self.opener.open(request, timeout=180) as response:
            if getattr(response, "status", None) != 206:
                raise RuntimeError(
                    f"Server ignored byte range for {self.url}: {response.status}"
                )
            payload = response.read()
        expected = end - self.position + 1
        if len(payload) != expected:
            raise IOError(f"Short HTTP range: {len(payload)} != {expected}")
        self.position = end + 1
        return payload


def fetch_remote_zip_members(
    url: str, output: Path, basenames: Sequence[str]
) -> dict[str, dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    existing = {
        name: output / name for name in basenames if (output / name).is_file()
    }
    if len(existing) == len(basenames):
        return {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in existing.items()
        }
    reader = HTTPRangeReader(url)
    with zipfile.ZipFile(reader) as archive:
        by_basename: dict[str, list[str]] = {name: [] for name in basenames}
        for member in archive.namelist():
            basename = Path(member).name
            if basename in by_basename:
                by_basename[basename].append(member)
        ambiguous = {
            name: members
            for name, members in by_basename.items()
            if len(members) != 1
        }
        if ambiguous:
            raise ValueError(f"Missing or ambiguous remote ZIP members: {ambiguous}")
        for name, members in by_basename.items():
            atomic_write(output / name, archive.read(members[0]))
    return {
        name: {"path": str(output / name), "sha256": sha256_file(output / name)}
        for name in basenames
    }


def parse_uzh_images(path: Path) -> tuple[np.ndarray, list[str]]:
    times = []
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 3:
            _, timestamp, name = fields
        elif len(fields) == 2:
            # Retain compatibility with older exports that omit the image id.
            timestamp, name = fields
        else:
            raise ValueError(f"Malformed UZH image row: {line}")
        times.append(float(timestamp))
        names.append(name)
    array = np.asarray(times, dtype=np.float64)
    if not len(array) or np.any(np.diff(array) <= 0):
        raise ValueError(f"Invalid UZH image timestamps: {path}")
    return array, names


def parse_uzh_groundtruth(path: Path) -> np.ndarray:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.replace(",", " ").split()
        if len(fields) < 8:
            raise ValueError(f"Malformed UZH ground-truth row: {line}")
        rows.append([float(value) for value in fields[:8]])
    array = np.asarray(rows, dtype=np.float64)
    if not len(array) or np.any(np.diff(array[:, 0]) <= 0):
        raise ValueError(f"Invalid UZH ground truth: {path}")
    return array


def align_uzh_episode(
    table: Any,
    sequence: str,
    episode_index: int,
    pose_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = table[table["episode_index"] == episode_index].sort_values("frame_index")
    if not len(rows) or rows["frame_index"].tolist() != list(range(len(rows))):
        raise ValueError(f"Invalid HF episode {episode_index}")
    official_times, official_names = parse_uzh_images(pose_root / "left_images.txt")
    groundtruth = parse_uzh_groundtruth(pose_root / "groundtruth.txt")
    source_times = rows["original_timestamp_s"].to_numpy(dtype=np.float64)
    grid = np.arange(int(np.floor(source_times[-1] * FPS)) + 1) / FPS
    selected, grid_errors = nearest_indices(source_times, grid, max_error=0.5 / FPS)
    # The public derivative preserves capture times but can contain a frame that
    # is absent from left_images.txt (and vice versa) around camera dropouts.
    # Map only the already selected 4 Hz real frames; their 250 ms separation
    # guarantees a one-to-one association even when the native ~27 Hz lists have
    # different dropout patterns.
    # The derivative defines original_timestamp_s relative to the first GT
    # sample, not relative to the first camera frame.  This is independently
    # pinned by episode durations and a direct decoded-frame match against the
    # official ZIP (episode 0 / indoor_45_12: corr=0.99888 at GT time zero).
    expected_official_times = groundtruth[0, 0] + source_times[selected]
    official_indices, official_errors = nearest_indices(
        official_times, expected_official_times, max_error=0.5 / FPS
    )
    selected_official_times = official_times[official_indices]
    valid = (selected_official_times >= groundtruth[0, 0]) & (
        selected_official_times <= groundtruth[-1, 0]
    )
    selected = selected[valid]
    grid = grid[valid]
    grid_errors = grid_errors[valid]
    official_indices = official_indices[valid]
    official_errors = official_errors[valid]
    selected_official_times = selected_official_times[valid]
    if len(selected) < CONTEXT_SIZE + HORIZON_FRAMES:
        raise UZHSequenceTooShort(sequence, len(selected))
    positions, quaternions, brackets = interpolate_poses(
        groundtruth[:, 0],
        groundtruth[:, 1:4],
        groundtruth[:, 4:8],
        selected_official_times,
    )
    yaw = quaternion_yaw_xyzw(quaternions)
    selected_rows = rows.iloc[selected]
    metadata = []
    for frame, (_, row) in enumerate(selected_rows.iterrows()):
        metadata.append(
            {
                "frame": frame,
                "episode_index": episode_index,
                "global_video_frame": int(row["index"]),
                "hf_frame_index": int(row["frame_index"]),
                "grid_time_s": float(grid[frame]),
                "hf_original_timestamp_s": float(row["original_timestamp_s"]),
                "official_image_timestamp_s": float(selected_official_times[frame]),
                "official_image_name": official_names[int(official_indices[frame])],
                "nearest_grid_error_s": float(grid_errors[frame]),
                "hf_official_timestamp_error_s": float(official_errors[frame]),
                "pose_bracket_s": float(brackets[frame]),
                "position_xyz": positions[frame].tolist(),
                "quaternion_xyzw": quaternions[frame].tolist(),
                "yaw": float(yaw[frame]),
                "derived_control_roll_pitch_yawrate_throttle": np.asarray(
                    row["action"], dtype=np.float64
                ).tolist(),
            }
        )
    record = {
        "trajectory": f"uzh_fpv_{sequence}",
        "sequence": sequence,
        "episode_index": episode_index,
        "frames": len(metadata),
        "max_grid_error_s": float(np.max(grid_errors)),
        "max_hf_official_timestamp_error_s": float(np.max(official_errors)),
        "max_pose_bracket_s": float(np.max(brackets)),
        "hf_episode_zero_official_timestamp_s": float(groundtruth[0, 0]),
        "positions": positions,
        "yaw": yaw,
    }
    return record, metadata


def decode_uzh_video(
    video: Path,
    targets: dict[int, tuple[Path, int]],
    expected_frames: int,
) -> dict[int, str]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")
    frame_bytes = 128 * 128 * 3
    hashes: dict[int, str] = {}
    for index in range(expected_frames):
        payload = process.stdout.read(frame_bytes)
        if len(payload) != frame_bytes:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            process.kill()
            raise RuntimeError(
                f"ffmpeg stopped at frame {index}/{expected_frames}: {stderr[-2000:]}"
            )
        if index in targets:
            trajectory, output_index = targets[index]
            image = Image.frombytes("RGB", (128, 128), payload)
            encoded = jpeg_bytes(image)
            atomic_write(trajectory / f"{output_index}.jpg", encoded)
            hashes[index] = sha256_bytes(encoded)
    trailing = process.stdout.read(1)
    return_code = process.wait()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if return_code != 0 or trailing:
        raise RuntimeError(
            f"ffmpeg decode failed or frame count changed: rc={return_code}, "
            f"extra={bool(trailing)}, stderr={stderr[-2000:]}"
        )
    if set(hashes) != set(targets):
        raise RuntimeError("Not every selected UZH frame was decoded")
    return hashes


def prepare_uzh(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd

    source = args.source.resolve()
    output = args.output.resolve()
    table_path = source / "data/chunk-000/file-000.parquet"
    episodes_path = source / "meta/episodes/chunk-000/file-000.parquet"
    video_path = source / "videos/observation.images.front/chunk-000/file-000.mp4"
    table = pd.read_parquet(table_path)
    episode_table = pd.read_parquet(episodes_path)
    if len(table) != 24242 or int(episode_table["length"].sum()) != len(table):
        raise ValueError("Pinned UZH derivative frame count changed")
    records = []
    excluded_sequences = []
    metadata_by_name: dict[str, list[dict[str, Any]]] = {}
    pose_sources = {}
    for sequence_index, sequence in enumerate(UZH_SEQUENCES):
        episode_index = 2 * sequence_index  # left camera; right-camera duplicate excluded
        pose_root = source / "official_pose" / sequence
        url = UZH_GT_URL.format(sequence=sequence)
        print(
            f"UZH {sequence_index + 1}/{len(UZH_SEQUENCES)}: {sequence}",
            flush=True,
        )
        pose_sources[sequence] = {
            "url": url,
            "members": fetch_remote_zip_members(
                url, pose_root, ("groundtruth.txt", "left_images.txt")
            ),
        }
        try:
            record, metadata = align_uzh_episode(
                table, sequence, episode_index, pose_root
            )
        except UZHSequenceTooShort as error:
            exclusion = {
                "sequence": sequence,
                "episode_index": episode_index,
                "valid_4hz_frames": error.valid_frames,
                "required_frames": CONTEXT_SIZE + HORIZON_FRAMES,
                "reason": "public derivative and official GT overlap is too short for one 4-second evaluation window",
            }
            excluded_sequences.append(exclusion)
            print(f"UZH excluded: {exclusion}", flush=True)
            continue
        name = record["trajectory"]
        trajectory = output / name
        trajectory.mkdir(parents=True, exist_ok=True)
        record["pruned_stale_generated_images"] = prune_stale_indexed_images(
            trajectory, len(metadata)
        )
        atomic_pickle(
            trajectory / "traj_data.pkl",
            {
                "position": record.pop("positions")[:, :2],
                "yaw": record.pop("yaw"),
            },
        )
        records.append(record)
        metadata_by_name[name] = metadata
    targets: dict[int, tuple[Path, int]] = {}
    for record in records:
        name = record["trajectory"]
        for row in metadata_by_name[name]:
            global_index = row["global_video_frame"]
            if global_index in targets:
                raise ValueError(f"Duplicate global UZH frame {global_index}")
            targets[global_index] = (output / name, row["frame"])
    image_hashes = decode_uzh_video(video_path, targets, len(table))
    for name, metadata in metadata_by_name.items():
        for row in metadata:
            row["processed_jpeg_sha256"] = image_hashes[row["global_video_frame"]]
        write_jsonl(output / name / "frame_metadata.jsonl", metadata)
    source_manifest = {
        "hf_dataset": "vdmaas98/uzh-fpv-drone-racing-128",
        "hf_revision": UZH_REVISION,
        "hf_files": {
            str(path.relative_to(source)): sha256_file(path)
            for path in (table_path, episodes_path, video_path)
        },
        "official_pose_sources": pose_sources,
        "sequences": records,
        "excluded_sequences": excluded_sequences,
    }
    atomic_json(output / "source_manifest.json", source_manifest)
    return finalize_dataset(
        "uzh_fpv",
        output,
        args.split_root.resolve(),
        args.reference_root.resolve(),
        args.reference_split.resolve(),
        spacing=None,
        provenance={
            **source_manifest,
            "official_dataset_page": "https://fpv.ifi.uzh.ch/datasets/",
            "license": "CC BY-NC-SA 3.0 (non-commercial)",
            "real_pose": True,
            "pose_policy": "official public UZH-FPV continuous-time ground truth interpolated at exact official left-camera timestamps",
            "episode_alignment_policy": "pinned HF lexicographic source order; original_timestamp_s is relative to the first official GT timestamp; verified by episode durations and direct official/HF frame pixel matching",
            "camera_policy": "left camera only; right-camera duplicate episodes excluded",
            "control_caveat": "HF 4D controls are derived from real GT attitude/IMU, not recorded pilot sticks; NWM conditioning uses official metric pose deltas, not those derived controls",
            "planar_policy": "NWM traj_data.pkl uses official world XY and quaternion yaw; official Z is retained per frame in frame_metadata.jsonl",
        },
    )


def finalize_existing(args: argparse.Namespace) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "source_kind": args.source_kind,
        "real_pose": True,
    }
    spacing = args.spacing
    if args.source_report:
        source_report = args.source_report.resolve()
        payload = json.loads(source_report.read_text(encoding="utf-8"))
        provenance.update(
            {
                "source_report": str(source_report),
                "source_report_sha256": sha256_file(source_report),
            }
        )
        if spacing is None and payload.get("metric_waypoint_spacing") is not None:
            spacing = float(payload["metric_waypoint_spacing"])
    return finalize_dataset(
        args.dataset,
        args.output.resolve(),
        args.split_root.resolve(),
        args.reference_root.resolve(),
        args.reference_split.resolve(),
        spacing=spacing,
        provenance=provenance,
        cadence=args.cadence,
        force_prefix=args.force_prefix,
        force_count=args.force_count,
    )


def validate_prepared(dataset: str, output: Path, split_root: Path) -> dict[str, Any]:
    report = json.loads((output / "dataset_report.json").read_text(encoding="utf-8"))
    with (split_root / "time.pkl").open("rb") as stream:
        time_entries = pickle.load(stream)
    with (split_root / "navigation_eval.pkl").open("rb") as stream:
        navigation_entries = pickle.load(stream)
    if len(time_entries) != SAMPLE_COUNT or len(set(time_entries)) != SAMPLE_COUNT:
        raise ValueError(f"{dataset}: invalid prediction split")
    if (
        len(navigation_entries) != NAVIGATION_SAMPLE_COUNT
        or len(set(navigation_entries)) != NAVIGATION_SAMPLE_COUNT
    ):
        raise ValueError(f"{dataset}: invalid navigation split")
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, current, lower, upper in time_entries:
        if name not in cache:
            cache[name] = load_trajectory(output / name)
        position, _ = cache[name]
        if current < CONTEXT_SIZE - 1 or current + HORIZON_FRAMES >= len(position):
            raise ValueError(f"{dataset}: invalid prediction anchor {(name, current)}")
        if current + lower < 0 or current + upper >= len(position):
            raise ValueError(f"{dataset}: invalid bounds {(name, current, lower, upper)}")
        for frame in range(current - CONTEXT_SIZE + 1, current + HORIZON_FRAMES + 1):
            if not (output / name / f"{frame}.jpg").is_file():
                raise FileNotFoundError(output / name / f"{frame}.jpg")
    for name, current, lower, upper in navigation_entries:
        if lower != 8 or upper != 8:
            raise ValueError(f"{dataset}: navigation goal is not fixed at eight steps")
        if name not in cache:
            cache[name] = load_trajectory(output / name)
        if current + 8 >= len(cache[name][0]):
            raise ValueError(f"{dataset}: invalid navigation anchor")
    for filename, metadata in report["splits"].items():
        path = Path(metadata["path"])
        if path != split_root / filename or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"{dataset}: split hash changed for {filename}")
    return {
        "dataset": dataset,
        "status": "passed",
        "prediction_samples": len(time_entries),
        "navigation_samples": len(navigation_entries),
        "trajectories_touched": len(cache),
        "split_sha256": report["splits"]["time.pkl"]["sha256"],
    }


def common_parser(subparser: argparse.ArgumentParser, default_output: str) -> None:
    subparser.add_argument(
        "--output", type=Path, default=DEFAULT_DATA_ROOT / default_output
    )
    subparser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT / "data_splits" / default_output / "test",
    )
    subparser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    subparser.add_argument(
        "--reference-split",
        type=Path,
        default=PROJECT_ROOT / "data_splits/go_stanford/test/time.pkl",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    unitree = subparsers.add_parser("unitree")
    unitree.add_argument("--source", type=Path, default=DEFAULT_UNITREE_SOURCE)
    common_parser(unitree, "unitree_go2")

    tum = subparsers.add_parser("tum")
    tum.add_argument("--source", type=Path, default=DEFAULT_TUM_SOURCE)
    tum.add_argument(
        "--source-mode",
        choices=("huggingface", "archives"),
        default="huggingface",
        help="Use the pinned selective RGB mirror by default; archives preserves the legacy full-tar path.",
    )
    tum.add_argument("--download-workers", type=int, default=16)
    common_parser(tum, "tum_rgbd")

    uzh = subparsers.add_parser("uzh")
    uzh.add_argument("--source", type=Path, default=DEFAULT_UZH_SOURCE)
    common_parser(uzh, "uzh_fpv")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--dataset", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--split-root", type=Path, required=True)
    finalize.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    finalize.add_argument(
        "--reference-split",
        type=Path,
        default=PROJECT_ROOT / "data_splits/go_stanford/test/time.pkl",
    )
    finalize.add_argument("--spacing", type=float)
    finalize.add_argument("--source-report", type=Path)
    finalize.add_argument("--source-kind", default="existing real-pose dataset")
    finalize.add_argument("--cadence", default="fixed_4hz")
    finalize.add_argument("--force-prefix")
    finalize.add_argument("--force-count", type=int, default=0)

    validate = subparsers.add_parser("validate")
    validate.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT
    )
    validate.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    validate.add_argument(
        "--datasets",
        nargs="+",
        default=["planetary_rover", "unitree_go2", "tum_rgbd", "uzh_fpv"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "unitree":
        result = prepare_unitree(args)
    elif args.command == "tum":
        result = prepare_tum(args)
    elif args.command == "uzh":
        result = prepare_uzh(args)
    elif args.command == "finalize":
        result = finalize_existing(args)
    elif args.command == "validate":
        result = {
            dataset: validate_prepared(
                dataset,
                args.data_root.resolve() / dataset,
                args.project_root.resolve() / "data_splits" / dataset / "test",
            )
            for dataset in args.datasets
        }
    else:  # pragma: no cover - argparse enforces choices
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
