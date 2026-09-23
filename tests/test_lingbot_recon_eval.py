from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_lingbot_recon_eval.py"
SPEC = importlib.util.spec_from_file_location("run_lingbot_recon_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_split_preserves_fixed_sample_ids(tmp_path: Path) -> None:
    split = tmp_path / "time.pkl"
    rows = [("a", 4, -4, 64), ("b", 9, -9, 64)]
    split.write_bytes(pickle.dumps(rows))
    samples = MODULE.load_split(split, expected_count=2)
    assert [sample.sample_id for sample in samples] == [0, 1]
    assert samples[1].target_frame == 25
    assert MODULE.sample_seed(42, samples[1].sample_id) == 43


def test_load_split_rejects_insufficient_future_horizon(tmp_path: Path) -> None:
    split = tmp_path / "time.pkl"
    split.write_bytes(pickle.dumps([("a", 4, -4, 15)]))
    with pytest.raises(ValueError, match="4-second target"):
        MODULE.load_split(split, expected_count=1)


def test_interpolation_unwraps_yaw_and_keeps_endpoints() -> None:
    position = np.array([[0.0, 0.0], [4.0, 2.0]])
    yaw = np.deg2rad(np.array([179.0, -179.0]))
    xy, angles = MODULE.interpolate_planar_trajectory(position, yaw, output_frames=5)
    np.testing.assert_allclose(xy[[0, -1]], position, atol=1e-12)
    assert np.rad2deg(angles[-1] - angles[0]) == pytest.approx(2.0)


def test_planar_pose_uses_opencv_camera_axes() -> None:
    pose = MODULE.planar_to_opencv_c2w(
        np.array([[2.0, 3.0]]), np.array([0.0])
    )[0]
    np.testing.assert_allclose(pose[:3, 0], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(pose[:3, 1], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(pose[:3, 2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(pose[:3, 3], [2.0, 3.0, 0.0])
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3))
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)


def test_action_arrays_match_confirmed_protocol() -> None:
    trajectory = {
        "position": np.stack((np.arange(17), np.zeros(17)), axis=1),
        "yaw": np.zeros(17),
    }
    poses, intrinsics = MODULE.build_action_arrays(trajectory)
    assert poses.shape == (65, 4, 4)
    assert poses.dtype == np.float32
    assert intrinsics.shape == (65, 4)
    np.testing.assert_allclose(intrinsics[0], [416.0, 320.0, 416.0, 240.0])
    np.testing.assert_allclose(poses[[0, -1], 0, 3], [0.0, 16.0])
