"""CPU-only contracts for the fixed four-second OOD benchmark builder."""

import math
import pickle

import numpy as np

from scripts.prepare_ood_benchmarks import (
    Anchor,
    UZH_SEQUENCES,
    candidate_anchors,
    evenly_spaced,
    interpolate_poses,
    match_distance_distribution,
    nearest_indices,
    parse_uzh_images,
    parse_tum_text,
    prune_stale_indexed_images,
    quaternion_yaw_xyzw,
    tum_bracketed_grid_indices,
)


def _yaw_quaternion(angle):
    return np.asarray([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])


def test_nearest_indices_selects_unique_real_frames_and_reports_error():
    timestamps = np.asarray([0.00, 0.09, 0.21, 0.31, 0.42, 0.52])
    grid = np.asarray([0.00, 0.10, 0.20, 0.30, 0.40, 0.50])

    selected, errors = nearest_indices(timestamps, grid, max_error=0.03)

    np.testing.assert_array_equal(selected, np.arange(6))
    np.testing.assert_allclose(errors, [0.00, 0.01, 0.01, 0.01, 0.02, 0.02])


def test_tum_duplicate_published_timestamps_are_pose_averaged_deterministically():
    payload = (
        b"1.0 0 0 0 0 0 0 1\n"
        b"1.0 2 0 0 0 0 0 -1\n"
        b"2.0 3 0 0 0 0 0 1\n"
    )

    rows = parse_tum_text(payload, 8, collapse_duplicate_timestamps=True)

    np.testing.assert_allclose(rows[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(rows[0, 1:4], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rows[0, 4:8], [0.0, 0.0, 0.0, 1.0])


def test_tum_grid_never_selects_rgb_outside_real_pose_bracket():
    rgb = np.asarray([0.95, 1.05, 1.30, 1.55, 1.80, 2.05])
    poses = np.asarray([1.0, 1.5, 2.0])

    selected, grid, errors = tum_bracketed_grid_indices(rgb, poses)

    np.testing.assert_array_equal(selected, [1, 2, 3, 4])
    np.testing.assert_allclose(grid, [1.05, 1.30, 1.55, 1.80])
    np.testing.assert_allclose(errors, 0.0)
    assert np.all(rgb[selected] >= poses[0])
    assert np.all(rgb[selected] <= poses[-1])


def test_uzh_image_manifest_uses_timestamp_column_not_numeric_image_id(tmp_path):
    manifest = tmp_path / "left_images.txt"
    manifest.write_text(
        "# id timestamp image_name\n"
        "0 1398.526170378000 img/image_0_0.png\n"
        "1 1398.559342690000 img/image_0_1.png\n",
        encoding="utf-8",
    )

    timestamps, names = parse_uzh_images(manifest)

    np.testing.assert_allclose(timestamps, [1398.526170378, 1398.55934269])
    assert names == ["img/image_0_0.png", "img/image_0_1.png"]


def test_uzh_hf_episode_order_is_pinned_lexicographic_source_order():
    assert UZH_SEQUENCES == tuple(sorted(UZH_SEQUENCES))
    assert UZH_SEQUENCES[:4] == (
        "indoor_45_12",
        "indoor_45_13",
        "indoor_45_14",
        "indoor_45_2",
    )


def test_regeneration_prunes_only_numeric_jpegs_beyond_new_length(tmp_path):
    for name in ("0.jpg", "3.jpg", "4.jpg", "notes.jpg", "4.png"):
        (tmp_path / name).write_bytes(b"test")

    removed = prune_stale_indexed_images(tmp_path, 4)

    assert removed == ["4.jpg"]
    assert {path.name for path in tmp_path.iterdir()} == {
        "0.jpg",
        "3.jpg",
        "notes.jpg",
        "4.png",
    }


def test_pose_interpolation_uses_metric_position_and_shortest_quaternion_path():
    left = math.radians(170.0)
    right = math.radians(-170.0)
    positions, quaternions, brackets = interpolate_poses(
        [0.0, 2.0],
        np.asarray([[0.0, 0.0, 0.0], [4.0, 2.0, 0.0]]),
        np.stack([_yaw_quaternion(left), _yaw_quaternion(right)]),
        [1.0],
    )

    np.testing.assert_allclose(positions, [[2.0, 1.0, 0.0]], atol=1e-12)
    assert math.isclose(abs(float(quaternion_yaw_xyzw(quaternions)[0])), math.pi)
    np.testing.assert_allclose(brackets, [2.0])


def test_fixed_split_has_exactly_500_deterministic_real_anchors(tmp_path):
    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    # len - horizon(16) - context_begin(3) = 500 valid anchors exactly.
    frame_count = 519
    position = np.column_stack(
        [np.arange(frame_count, dtype=np.float64) * 0.1, np.zeros(frame_count)]
    )
    with (trajectory / "traj_data.pkl").open("wb") as stream:
        pickle.dump(
            {"position": position, "yaw": np.zeros(frame_count)},
            stream,
            protocol=4,
        )

    candidates = candidate_anchors(tmp_path, ["trajectory"])
    target = np.asarray([item.displacement_m for item in candidates])
    selected_once = match_distance_distribution(candidates, target, 500)
    selected_twice = match_distance_distribution(candidates, target, 500)

    assert len(candidates) == len(selected_once) == 500
    assert len(set(selected_once)) == 500
    assert selected_once == selected_twice
    assert selected_once[0].current >= 3
    assert selected_once[-1].current + 16 < frame_count


def test_forced_coverage_and_navigation_subsample_remain_unique():
    candidates = [Anchor(f"scene_{index % 5}", index, index / 100.0) for index in range(800)]
    forced = candidates[::40][:10]
    targets = np.linspace(0.0, 7.99, 500)

    selected = match_distance_distribution(candidates, targets, 500, forced=forced)
    navigation = evenly_spaced(
        sorted(selected, key=lambda item: (item.trajectory, item.current)), 100
    )

    assert set(forced).issubset(selected)
    assert len(selected) == len(set(selected)) == 500
    assert len(navigation) == len(set(navigation)) == 100
