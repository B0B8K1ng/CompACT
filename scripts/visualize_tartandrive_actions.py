#!/usr/bin/env python3
"""Render TartanDrive RGB videos with GT and geometry actions side by side.

This is a post-extraction, read-only diagnostic.  GT is loaded only for the
visual comparison and never modifies the geometry cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

GT_COLOR = (80, 220, 80)  # BGR green
GEOMETRY_COLOR = (40, 165, 255)  # BGR orange
TEXT_COLOR = (235, 235, 235)
MUTED_COLOR = (165, 175, 185)
BACKGROUND = (22, 25, 30)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def compute_gt_action(
    positions: np.ndarray,
    yaw: np.ndarray,
    current: int,
    goal: int,
    *,
    waypoint_spacing: float,
) -> np.ndarray:
    """Match the validator/BaseDataset current-navigation-frame convention."""

    displacement = positions[goal, :2] - positions[current, :2]
    cosine = math.cos(float(yaw[current]))
    sine = math.sin(float(yaw[current]))
    action = np.asarray(
        [
            displacement[0] * cosine + displacement[1] * sine,
            -displacement[0] * sine + displacement[1] * cosine,
            wrap_angle(float(yaw[goal] - yaw[current])),
        ],
        dtype=np.float64,
    )
    action[:2] /= waypoint_spacing
    return action


def _quantile_indices(length: int, count: int) -> list[int]:
    if count < 1 or length < count:
        raise ValueError(f"Cannot select {count} unique rows from {length}")
    chosen: list[int] = []
    for index in range(count):
        quantile = (2 * index + 1) / (2 * count)
        candidate = round((length - 1) * quantile)
        if candidate in chosen:
            candidate = next(
                value
                for radius in range(1, length)
                for value in (candidate - radius, candidate + radius)
                if 0 <= value < length and value not in chosen
            )
        chosen.append(candidate)
    return chosen


def select_stratified_trajectories(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_split: int,
    min_frames: int,
) -> list[dict[str, Any]]:
    """Select fixed length quantiles independently from train and test."""

    selected: list[dict[str, Any]] = []
    for split in ("train", "test"):
        eligible = sorted(
            (
                dict(row)
                for row in rows
                if row.get("split") == split
                and int(row.get("num_frames", 0)) >= min_frames
            ),
            key=lambda row: (int(row["num_frames"]), str(row["trajectory_name"])),
        )
        indices = _quantile_indices(len(eligible), per_split)
        for quantile_index, row_index in enumerate(indices):
            row = eligible[row_index]
            row["selection_quantile"] = (2 * quantile_index + 1) / (
                2 * per_split
            )
            selected.append(row)
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
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
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping) or not {"position", "yaw"} <= payload.keys():
        raise ValueError(f"Invalid TartanDrive GT payload: {path}")
    positions = np.asarray(payload["position"], dtype=np.float64)
    yaw = np.asarray(payload["yaw"], dtype=np.float64).reshape(-1)
    if (
        positions.ndim != 2
        or positions.shape[1] not in (2, 3)
        or len(positions) != len(yaw)
        or not np.isfinite(positions).all()
        or not np.isfinite(yaw).all()
    ):
        raise ValueError(f"Invalid position/yaw arrays in {path}")
    return positions, yaw


def _load_geometry(path: Path) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("complete") is not True
        or payload.get("motion_type") != "geometry"
        or payload.get("pair_direction") != "current_to_goal"
        or payload.get("coordinate_frame") != "current_navigation_frame"
        or payload.get("translation_unit") != "waypoint_spacing_units"
    ):
        raise ValueError(f"Geometry cache has incompatible metadata: {path}")
    pairs = torch.as_tensor(payload["frame_pairs"], dtype=torch.int64).numpy()
    motion = torch.as_tensor(payload["motion"], dtype=torch.float32).numpy()
    if pairs.shape != (len(motion), 2) or motion.shape[1:] != (3,):
        raise ValueError(f"Geometry cache has invalid tensor shapes: {path}")
    if not np.isfinite(motion).all():
        raise ValueError(f"Geometry cache contains non-finite actions: {path}")
    lookup = {
        (int(pair[0]), int(pair[1])): value.astype(np.float64, copy=False)
        for pair, value in zip(pairs, motion, strict=True)
    }
    return lookup, dict(payload)


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = TEXT_COLOR,
    scale: float = 0.52,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _project_action(
    action: np.ndarray,
    *,
    origin: tuple[int, int],
    pixels_per_unit: float,
    max_radius: float,
) -> tuple[int, int]:
    xy = np.asarray(action[:2], dtype=np.float64)
    norm = float(np.linalg.norm(xy))
    if norm > max_radius:
        xy *= max_radius / norm
    # Navigation +x is screen-up and +y (left) is screen-left.
    return (
        round(origin[0] - xy[1] * pixels_per_unit),
        round(origin[1] - xy[0] * pixels_per_unit),
    )


def _draw_action(
    canvas: np.ndarray,
    action: np.ndarray,
    *,
    origin: tuple[int, int],
    pixels_per_unit: float,
    max_radius: float,
    color: tuple[int, int, int],
) -> None:
    endpoint = _project_action(
        action,
        origin=origin,
        pixels_per_unit=pixels_per_unit,
        max_radius=max_radius,
    )
    cv2.arrowedLine(canvas, origin, endpoint, color, 4, cv2.LINE_AA, tipLength=0.12)
    heading_length = 0.18 * max_radius
    heading_xy = np.asarray(
        [math.cos(float(action[2])), math.sin(float(action[2]))]
    ) * heading_length
    heading_end = (
        round(endpoint[0] - heading_xy[1] * pixels_per_unit),
        round(endpoint[1] - heading_xy[0] * pixels_per_unit),
    )
    cv2.arrowedLine(
        canvas, endpoint, heading_end, color, 2, cv2.LINE_AA, tipLength=0.22
    )
    cv2.circle(canvas, endpoint, 5, color, -1, cv2.LINE_AA)


def _direction_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first[:2]))
    second_norm = float(np.linalg.norm(second[:2]))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return None
    return float(
        np.clip(np.dot(first[:2], second[:2]) / (first_norm * second_norm), -1, 1)
    )


def _render_frame(
    image: np.ndarray,
    *,
    trajectory_name: str,
    split: str,
    current: int,
    goal: int,
    num_frames: int,
    gt_action: np.ndarray,
    geometry_action: np.ndarray,
    plot_radius: float,
    sequence_index: int,
    sequence_length: int,
) -> np.ndarray:
    canvas = np.full((720, 1280, 3), BACKGROUND, dtype=np.uint8)
    canvas[:, :960] = cv2.resize(image, (960, 720), interpolation=cv2.INTER_CUBIC)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (960, 88), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    _put_text(canvas, trajectory_name, (24, 34), scale=0.72, thickness=2)
    _put_text(
        canvas,
        f"split={split}  frame {current}->{goal} / {num_frames - 1}  k={goal-current}",
        (24, 68),
        color=(210, 220, 230),
        scale=0.56,
    )

    panel_x = 960
    _put_text(canvas, "Action comparison", (panel_x + 18, 34), scale=0.66, thickness=2)
    cv2.line(canvas, (panel_x + 18, 54), (panel_x + 302, 54), (75, 82, 90), 1)
    cv2.line(canvas, (panel_x + 22, 78), (panel_x + 58, 78), GT_COLOR, 5)
    _put_text(canvas, "GT action", (panel_x + 70, 84), color=GT_COLOR)
    cv2.line(canvas, (panel_x + 175, 78), (panel_x + 211, 78), GEOMETRY_COLOR, 5)
    _put_text(canvas, "Geometry", (panel_x + 220, 84), color=GEOMETRY_COLOR)

    origin = (panel_x + 160, 252)
    plot_half_size = 126
    pixels_per_unit = plot_half_size / plot_radius
    cv2.rectangle(
        canvas,
        (origin[0] - plot_half_size, origin[1] - plot_half_size),
        (origin[0] + plot_half_size, origin[1] + plot_half_size),
        (60, 66, 74),
        1,
    )
    cv2.arrowedLine(
        canvas,
        (origin[0], origin[1] + plot_half_size),
        (origin[0], origin[1] - plot_half_size),
        (105, 112, 122),
        1,
        cv2.LINE_AA,
        tipLength=0.06,
    )
    cv2.arrowedLine(
        canvas,
        (origin[0] + plot_half_size, origin[1]),
        (origin[0] - plot_half_size, origin[1]),
        (105, 112, 122),
        1,
        cv2.LINE_AA,
        tipLength=0.06,
    )
    _put_text(canvas, "+x forward", (origin[0] + 8, origin[1] - plot_half_size + 16), color=MUTED_COLOR, scale=0.40)
    _put_text(canvas, "+y left", (origin[0] - plot_half_size + 5, origin[1] - 8), color=MUTED_COLOR, scale=0.40)
    _put_text(canvas, f"radius={plot_radius:.2f}", (panel_x + 34, 402), color=MUTED_COLOR, scale=0.42)
    _draw_action(
        canvas,
        gt_action,
        origin=origin,
        pixels_per_unit=pixels_per_unit,
        max_radius=plot_radius,
        color=GT_COLOR,
    )
    _draw_action(
        canvas,
        geometry_action,
        origin=origin,
        pixels_per_unit=pixels_per_unit,
        max_radius=plot_radius,
        color=GEOMETRY_COLOR,
    )

    rows = (
        ("", "dx", "dy", "dyaw"),
        (
            "GT",
            f"{gt_action[0]:+.3f}",
            f"{gt_action[1]:+.3f}",
            f"{math.degrees(float(gt_action[2])):+.2f}d",
        ),
        (
            "Geom",
            f"{geometry_action[0]:+.3f}",
            f"{geometry_action[1]:+.3f}",
            f"{math.degrees(float(geometry_action[2])):+.2f}d",
        ),
    )
    y = 446
    for row_index, row in enumerate(rows):
        color = MUTED_COLOR if row_index == 0 else (GT_COLOR if row_index == 1 else GEOMETRY_COLOR)
        for column, (text, x) in enumerate(zip(row, (980, 1055, 1132, 1200), strict=True)):
            _put_text(canvas, text, (x, y), color=color, scale=0.43 if column else 0.46)
        y += 29

    translation_error = float(np.linalg.norm(geometry_action[:2] - gt_action[:2]))
    yaw_error = abs(float(wrap_angle(geometry_action[2] - gt_action[2])))
    cosine = _direction_cosine(gt_action, geometry_action)
    _put_text(canvas, f"translation error: {translation_error:.3f}", (panel_x + 24, 558), scale=0.46)
    _put_text(canvas, f"yaw error: {math.degrees(yaw_error):.2f} deg", (panel_x + 24, 586), scale=0.46)
    _put_text(
        canvas,
        f"direction cosine: {'n/a' if cosine is None else f'{cosine:.3f}'}",
        (panel_x + 24, 614),
        scale=0.46,
    )
    _put_text(canvas, "GT used for visualization only", (panel_x + 24, 651), color=MUTED_COLOR, scale=0.42)

    progress_left, progress_right = panel_x + 24, panel_x + 296
    cv2.rectangle(canvas, (progress_left, 680), (progress_right, 692), (67, 72, 80), -1)
    fraction = (sequence_index + 1) / max(sequence_length, 1)
    cv2.rectangle(
        canvas,
        (progress_left, 680),
        (progress_left + round((progress_right - progress_left) * fraction), 692),
        (90, 175, 235),
        -1,
    )
    _put_text(canvas, f"{sequence_index + 1}/{sequence_length}", (panel_x + 130, 713), color=MUTED_COLOR, scale=0.42)
    return canvas


def _ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def _encode_video(
    frames: Sequence[np.ndarray],
    output_path: Path,
    *,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.mp4")
    if temporary.exists():
        temporary.unlink()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        "1280x720",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "4",
        "-y",
        str(temporary),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape != (720, 1280, 3) or frame.dtype != np.uint8:
                raise ValueError(f"Unexpected rendered frame: {frame.shape}/{frame.dtype}")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        if temporary.exists():
            temporary.unlink()
        raise
    if return_code != 0:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr.strip()}")
    os.replace(temporary, output_path)
    probe = _ffprobe(output_path)
    if int(probe["width"]) != 1280 or int(probe["height"]) != 720:
        raise RuntimeError(f"Encoded video has wrong dimensions: {probe}")
    return probe


def _trajectory_actions(
    *,
    positions: np.ndarray,
    yaw: np.ndarray,
    geometry_lookup: Mapping[tuple[int, int], np.ndarray],
    action_offset: int,
    waypoint_spacing: float,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    currents = sorted(
        current
        for current, goal in geometry_lookup
        if goal == current + action_offset
    )
    if not currents:
        raise ValueError(f"No geometry pairs with action offset {action_offset}")
    gt = np.stack(
        [
            compute_gt_action(
                positions,
                yaw,
                current,
                current + action_offset,
                waypoint_spacing=waypoint_spacing,
            )
            for current in currents
        ]
    )
    geometry = np.stack(
        [geometry_lookup[(current, current + action_offset)] for current in currents]
    )
    return currents, gt, geometry


def _summary_metrics(gt: np.ndarray, geometry: np.ndarray) -> dict[str, Any]:
    translation_error = np.linalg.norm(geometry[:, :2] - gt[:, :2], axis=1)
    yaw_error_deg = np.degrees(np.abs(wrap_angle(geometry[:, 2] - gt[:, 2])))
    cosines = [
        value
        for first, second in zip(gt, geometry, strict=True)
        if (value := _direction_cosine(first, second)) is not None
    ]
    meaningful = (np.abs(gt[:, 2]) > math.radians(0.1)) & (
        np.abs(geometry[:, 2]) > math.radians(0.1)
    )
    return {
        "frames": len(gt),
        "translation_rmse": float(np.sqrt(np.mean(np.square(translation_error)))),
        "translation_mae": float(np.mean(translation_error)),
        "yaw_mae_deg": float(np.mean(yaw_error_deg)),
        "direction_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "yaw_sign_agreement": (
            float(np.mean(np.sign(gt[meaningful, 2]) == np.sign(geometry[meaningful, 2])))
            if np.any(meaningful)
            else None
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ten TartanDrive videos with GT and geometry actions."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/file_system/nas/algorithm/dujun.nie/nwm/data/tartan"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path(
            "/file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/"
            "vggt_omega_tartan/geometry_motion"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/"
            "vggt_omega_tartan/manifests/tartan_frames.json"
        ),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path(
            "/file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/"
            "vggt_omega_tartan/validation/report.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/file_system/nas/algorithm/dujun.nie/nwm/geometry_actions/"
            "vggt_omega_tartan/visualizations/action_comparison_k8_10"
        ),
    )
    parser.add_argument("--per-split", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--action-offset", type=int, default=8)
    parser.add_argument("--waypoint-spacing", type=float, default=0.72)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trajectories", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.action_offset < 1 or args.waypoint_spacing <= 0 or args.fps <= 0:
        raise ValueError("action-offset, waypoint-spacing and fps must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["trajectories"]
    by_name = {str(row["trajectory_name"]): dict(row) for row in rows}
    if args.trajectories:
        selected = [by_name[name] for name in args.trajectories]
    else:
        selected = select_stratified_trajectories(
            rows,
            per_split=args.per_split,
            min_frames=args.min_frames,
        )
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    validation_rows = validation["trajectories"]

    args.output_root.mkdir(parents=True, exist_ok=True)
    renderer_snapshot = args.output_root / "provenance" / Path(__file__).name
    _atomic_bytes(Path(__file__).read_bytes(), renderer_snapshot)
    receipt_rows: list[dict[str, Any]] = []
    for order, row in enumerate(selected, start=1):
        name = str(row["trajectory_name"])
        split = str(row["split"])
        trajectory_dir = args.data_root / name
        gt_path = trajectory_dir / "traj_data.pkl"
        geometry_path = args.geometry_root / "tartan_drive" / f"{name}.pt"
        positions, yaw = _load_gt(gt_path)
        geometry_lookup, geometry_payload = _load_geometry(geometry_path)
        if len(positions) != int(row["num_frames"]):
            raise ValueError(f"Manifest/GT frame mismatch for {name}")
        currents, gt_actions, geometry_actions = _trajectory_actions(
            positions=positions,
            yaw=yaw,
            geometry_lookup=geometry_lookup,
            action_offset=args.action_offset,
            waypoint_spacing=args.waypoint_spacing,
        )
        norms = np.concatenate(
            (
                np.linalg.norm(gt_actions[:, :2], axis=1),
                np.linalg.norm(geometry_actions[:, :2], axis=1),
            )
        )
        plot_radius = max(1.0, float(np.percentile(norms, 99)) * 1.15)
        frames: list[np.ndarray] = []
        for sequence_index, current in enumerate(currents):
            image_path = trajectory_dir / f"{current}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot decode image: {image_path}")
            frames.append(
                _render_frame(
                    image,
                    trajectory_name=name,
                    split=split,
                    current=current,
                    goal=current + args.action_offset,
                    num_frames=len(positions),
                    gt_action=gt_actions[sequence_index],
                    geometry_action=geometry_actions[sequence_index],
                    plot_radius=plot_radius,
                    sequence_index=sequence_index,
                    sequence_length=len(currents),
                )
            )
        output_path = args.output_root / f"{order:02d}_{split}_{name}_k{args.action_offset}.mp4"
        probe = _encode_video(
            frames,
            output_path,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
        metrics = _summary_metrics(gt_actions, geometry_actions)
        record = {
            "order": order,
            "trajectory_name": name,
            "split": split,
            "selection_quantile": row.get("selection_quantile"),
            "trajectory_frames": len(positions),
            "first_current_frame": currents[0],
            "last_current_frame": currents[-1],
            "action_offset": args.action_offset,
            "rendered_frames": len(frames),
            "plot_radius_waypoint_units": plot_radius,
            "video": {
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": _sha256_file(output_path),
                "ffprobe": probe,
            },
            "geometry_cache": {
                "path": str(geometry_path),
                "sha256": _sha256_file(geometry_path),
                "source_pose_sha256": geometry_payload["source_pose_sha256"],
            },
            "gt": {"path": str(gt_path), "sha256": _sha256_file(gt_path)},
            "visualized_k_metrics": metrics,
            "all_pair_validation_metrics": validation_rows[name]["metrics"],
        }
        receipt_rows.append(record)
        print(json.dumps({"status": "complete", **record}, ensure_ascii=False))

    receipt = {
        "schema_version": 1,
        "artifact_type": "tartandrive_gt_geometry_action_videos",
        "complete": True,
        "selection_policy": (
            "explicit" if args.trajectories else "train_test_equal_length_quantiles"
        ),
        "gt_usage": "visualization_only",
        "coordinate_frame": "current_navigation_frame",
        "translation_unit": "waypoint_spacing_units",
        "yaw_unit": "radians",
        "components": ["delta_x", "delta_y", "delta_yaw"],
        "action_offset": args.action_offset,
        "fps": args.fps,
        "waypoint_spacing": args.waypoint_spacing,
        "render_config": {
            "resolution": [1280, 720],
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "crf": args.crf,
            "preset": args.preset,
            "per_split": args.per_split,
            "min_frames": args.min_frames,
            "opencv_version": cv2.__version__,
            "ffmpeg_version": subprocess.check_output(
                ["ffmpeg", "-version"], text=True
            ).splitlines()[0],
        },
        "renderer_source": {
            "path": str(renderer_snapshot),
            "sha256": _sha256_file(renderer_snapshot),
        },
        "manifest": {"path": str(args.manifest), "sha256": _sha256_file(args.manifest)},
        "validation_report": {
            "path": str(args.validation_report),
            "sha256": _sha256_file(args.validation_report),
        },
        "videos": receipt_rows,
    }
    _atomic_json(receipt, args.output_root / "visualization_receipt.json")
    print(
        json.dumps(
            {
                "status": "complete",
                "videos": len(receipt_rows),
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
