"""CPU tests for reproducible VGGT/TartanDrive geometry validation."""

from __future__ import annotations

import pickle

import numpy as np
import torch

from geometry_action.tartandrive_vggt_omega import build_geometry_payload
from geometry_action.validation import (
    ValidationOptions,
    frame_list_records,
    frame_list_sha256,
    sha256_file,
    validate_dataset,
    validate_trajectory,
)


def _c2w_rotation() -> np.ndarray:
    # OpenCV +z forward -> navigation/world +x; +x right -> world -y.
    return np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )


def _extrinsics(centers: np.ndarray) -> torch.Tensor:
    rotation_w2c = _c2w_rotation().T
    result = []
    for center in centers:
        translation = -rotation_w2c @ center
        result.append(np.column_stack((rotation_w2c, translation)))
    return torch.tensor(np.stack(result), dtype=torch.float32)


def _extrinsics_from_c2w(
    rotations_c2w: np.ndarray, centers: np.ndarray
) -> torch.Tensor:
    result = []
    for rotation_c2w, center in zip(rotations_c2w, centers, strict=True):
        rotation_w2c = rotation_c2w.T
        result.append(np.column_stack((rotation_w2c, -rotation_w2c @ center)))
    return torch.tensor(np.stack(result), dtype=torch.float32)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _make_artifacts(
    tmp_path, *, corrupt_motion: bool = False, corrupt_rotation: bool = False
):
    dataset = "tartan_drive"
    trajectory = "traj_001"
    trajectory_dir = tmp_path / "data" / trajectory
    raw_dir = tmp_path / "raw" / dataset
    geometry_dir = tmp_path / "geometry" / dataset
    trajectory_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)

    num_frames = 5
    for frame in range(num_frames):
        (trajectory_dir / f"{frame}.jpg").write_bytes(f"jpeg-{frame}".encode())
    positions = np.column_stack((np.arange(num_frames), np.zeros(num_frames))).astype(
        np.float64
    )
    yaw = np.zeros(num_frames, dtype=np.float64)
    with open(trajectory_dir / "traj_data.pkl", "wb") as handle:
        pickle.dump({"position": positions, "yaw": yaw}, handle)

    centers = np.column_stack(
        (np.arange(num_frames), np.zeros(num_frames), np.zeros(num_frames))
    ).astype(np.float32)
    extrinsics = _extrinsics(centers)
    if corrupt_rotation:
        extrinsics[0, 0, 0] += 0.2
    intrinsics = torch.eye(3).repeat(num_frames, 1, 1)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    frame_ids = torch.arange(num_frames, dtype=torch.int64)
    records = frame_list_records(trajectory_dir, trajectory, frame_ids.tolist())
    frames_hash = frame_list_sha256(records)

    # Two overlapping windows exercise alignment residual recomputation.
    windows = []
    for index, (start, end, overlap) in enumerate(((0, 4, []), (1, 5, [1, 2, 3]))):
        windows.append(
            {
                "window_index": index,
                "start_index": start,
                "end_index_exclusive": end,
                "frame_ids": frame_ids[start:end],
                "extrinsics_w2c_local": extrinsics[start:end].clone(),
                "intrinsics_local": intrinsics[start:end].clone(),
                "image_size_hw": torch.tensor([240, 320], dtype=torch.int64),
                "alignment_to_global": {
                    "scale": 1.0,
                    "rotation": torch.eye(3),
                    "translation": torch.zeros(3),
                },
                "overlap_frame_ids": torch.tensor(overlap, dtype=torch.int64),
                "overlap_center_rmse": 0.0,
                "overlap_rotation_rmse_deg": 0.0,
            }
        )
    raw_payload = {
        "schema_version": 1,
        "artifact_type": "vggt_omega_camera_poses",
        "dataset_name": dataset,
        "trajectory_name": trajectory,
        "model_id": "facebook/VGGT-Omega-1B-512",
        "model_revision": "revision",
        "checkpoint_sha256": "a" * 64,
        "code_revision": "commit",
        "pose_convention": "world_to_camera_opencv",
        "frame_ids": frame_ids,
        "frame_list_sha256": frames_hash,
        "extrinsics_w2c": extrinsics,
        "intrinsics": intrinsics,
        "image_size_hw": torch.tensor([240, 320], dtype=torch.int64),
        "preprocessing": {"max_size": 512},
        "windows": windows,
        "window_alignment": {
            "policy": "sequential_overlap_sim3",
            "overlap": 3,
            "min_overlap": 3,
            "pose_selection": "first_prediction_wins_overlap",
            "scale_fallback": "one_when_degenerate",
        },
        "complete": True,
    }
    raw_path = raw_dir / f"{trajectory}.pt"
    torch.save(raw_payload, raw_path)

    pairs = torch.tensor(
        [
            (current, goal)
            for current in range(num_frames)
            for goal in range(num_frames)
        ],
        dtype=torch.int64,
    )
    motion = torch.tensor(
        [[float(goal - current), 0.0, 0.0] for current, goal in pairs.tolist()],
        dtype=torch.float32,
    )
    if corrupt_motion:
        motion[:, 0].mul_(-1)
    geometry_payload = {
        "schema_version": 1,
        "motion_type": "geometry",
        "dataset_name": dataset,
        "trajectory_name": trajectory,
        "pair_direction": "current_to_goal",
        "normalization": "raw",
        "coordinate_frame": "current_navigation_frame",
        "translation_unit": "waypoint_spacing_units",
        "yaw_unit": "radians",
        "components": ["delta_x", "delta_y", "delta_yaw"],
        "frame_pairs": pairs,
        "motion": motion,
        "source_pose_sha256": sha256_file(raw_path),
        "source_artifact_type": "vggt_omega_camera_poses",
        "frame_list_sha256": frames_hash,
        "alignment_policy": "tartandrive_forward_camera",
        "pair_generation": {
            "min_offset": -64,
            "max_offset": 64,
            "context_size": 1,
            "len_traj_pred": 0,
            "coverage": "exact_BaseDataset_training_index_domain",
        },
        "scale_policy": "per_trajectory_median_adjacent_step_to_waypoint_unit",
        "scale": 1.0,
        "scale_status": "estimated",
        "camera_to_navigation_policy": (
            "opencv_camera_poses_projected_to_first_frame_navigation_se2"
        ),
        "camera_to_navigation_axes": {
            "absolute_x": "first_camera_z",
            "absolute_y": "negative_first_camera_x",
            "absolute_heading": (
                "atan2(-forward_x_in_first_camera,forward_z_in_first_camera)"
            ),
            "relative_action": "inverse_se2_current_times_se2_goal",
        },
        "scale_diagnostics": {"measurement_frame": "first_frame_navigation_se2"},
        "ground_truth_usage": "none",
        "complete": True,
    }
    geometry_path = geometry_dir / f"{trajectory}.pt"
    torch.save(geometry_payload, geometry_path)
    return {
        "dataset": dataset,
        "trajectory": trajectory,
        "trajectory_dir": trajectory_dir,
        "raw_root": tmp_path / "raw",
        "geometry_root": tmp_path / "geometry",
        "raw_path": raw_path,
        "geometry_path": geometry_path,
        "gt_path": trajectory_dir / "traj_data.pkl",
    }


def _validate(paths):
    return validate_trajectory(
        paths["raw_path"],
        paths["geometry_path"],
        dataset_name=paths["dataset"],
        trajectory_name=paths["trajectory"],
        trajectory_dir=paths["trajectory_dir"],
        gt_path=paths["gt_path"],
        options=ValidationOptions(),
    )


def test_valid_artifacts_pass_all_structural_and_gt_checks(tmp_path) -> None:
    paths = _make_artifacts(tmp_path)
    before_raw = paths["raw_path"].read_bytes()
    before_geometry = paths["geometry_path"].read_bytes()

    result = _validate(paths)

    assert result["status"] == "pass", result["issues"]
    assert result["metrics"]["raw.overlap_frames_checked"] == 3
    assert result["metrics"]["geometry.pose_action_checks"] == 25
    assert result["metrics"]["geometry.composition_translation_max"] == 0.0
    assert result["metrics"]["gt.only_for_validation"] is True
    assert result["metrics"]["gt.ate_rmse"] < 1e-9
    assert result["metrics"]["gt.action_direction_cosine_mean"] == 1.0
    # Validation-only Sim(3)/scale must never contaminate either proxy artifact.
    assert paths["raw_path"].read_bytes() == before_raw
    assert paths["geometry_path"].read_bytes() == before_geometry


def test_wrong_action_direction_fails_pose_and_gt_checks(tmp_path) -> None:
    result = _validate(_make_artifacts(tmp_path, corrupt_motion=True))
    codes = {
        issue["code"] for issue in result["issues"] if issue["severity"] == "error"
    }

    assert result["status"] == "fail"
    assert "threshold.geometry.pose_action_translation_max" in codes
    # GT is report-only unless the caller explicitly opts into a GT threshold.
    assert "threshold.gt.action_direction_cosine_mean" not in codes
    assert result["metrics"]["gt.action_direction_cosine_mean"] == -1.0


def test_non_so3_camera_pose_fails(tmp_path) -> None:
    result = _validate(_make_artifacts(tmp_path, corrupt_rotation=True))
    codes = {
        issue["code"] for issue in result["issues"] if issue["severity"] == "error"
    }

    assert result["status"] == "fail"
    assert "threshold.raw.so3_orthogonality_max" in codes


def test_validator_accepts_payload_from_the_official_extraction_builder(
    tmp_path,
) -> None:
    paths = _make_artifacts(tmp_path)
    raw_payload = torch.load(paths["raw_path"], map_location="cpu", weights_only=True)

    # Deliberately include pitch and roll. Both sides must first project every
    # absolute pose to the common first-frame SE(2) gauge; projecting each 3-D
    # pair independently would not preserve inverse/composition here.
    count = len(raw_payload["frame_ids"])
    first_camera_positions = np.column_stack(
        (
            -0.15 * np.square(np.arange(count)),
            0.1 * np.sin(np.arange(count)),
            np.arange(count, dtype=np.float64),
        )
    )
    base = _c2w_rotation().astype(np.float64)
    centers_world = (base @ first_camera_positions.T).T
    rotations_c2w = np.stack(
        [
            base
            @ _rotation_y(0.12 * frame)
            @ _rotation_x(0.06 * (-1) ** frame)
            @ _rotation_z(0.04 * frame)
            for frame in range(count)
        ]
    )
    varied_extrinsics = _extrinsics_from_c2w(rotations_c2w, centers_world)
    raw_payload["extrinsics_w2c"] = varied_extrinsics
    for window in raw_payload["windows"]:
        start = window["start_index"]
        end = window["end_index_exclusive"]
        window["extrinsics_w2c_local"] = varied_extrinsics[start:end].clone()
    torch.save(raw_payload, paths["raw_path"])

    geometry_payload = build_geometry_payload(
        raw_payload,
        source_pose_sha256=sha256_file(paths["raw_path"]),
        min_offset=-64,
        max_offset=64,
        context_size=1,
        len_traj_pred=0,
    )
    torch.save(geometry_payload, paths["geometry_path"])

    result = _validate(paths)

    assert result["status"] == "pass", result["issues"]
    assert result["metrics"]["geometry.pose_action_checks"] == 25
    assert result["metrics"]["geometry.inverse_translation_max"] < 1e-6
    assert result["metrics"]["geometry.composition_translation_max"] < 1e-6


def test_dataset_report_is_atomic_json_and_resumable(tmp_path) -> None:
    paths = _make_artifacts(tmp_path)
    output = tmp_path / "validation" / "report.json"
    kwargs = {
        "raw_pose_root": str(paths["raw_root"]),
        "geometry_root": str(paths["geometry_root"]),
        "dataset_name": paths["dataset"],
        "output_path": str(output),
        "data_root": str(tmp_path / "data"),
        "flush_every": 1,
        "options": ValidationOptions(),
    }

    first = validate_dataset([paths["trajectory"]], **kwargs)
    second = validate_dataset([paths["trajectory"]], **kwargs)

    assert first["complete"] is True
    assert first["summary"]["status"] == "pass"
    assert second["summary"]["reused"] == 1
    assert second["trajectories"][paths["trajectory"]]["status"] == "pass"
