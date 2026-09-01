import math

import numpy as np

from scripts.visualize_tartandrive_actions import (
    _direction_cosine,
    _project_action,
    compute_gt_action,
    select_stratified_trajectories,
    wrap_angle,
)


def test_compute_gt_action_uses_current_navigation_frame_and_spacing() -> None:
    positions = np.asarray([[0.0, 0.0], [0.0, 1.44]])
    yaw = np.asarray([math.pi / 2, math.pi])

    action = compute_gt_action(
        positions, yaw, 0, 1, waypoint_spacing=0.72
    )

    np.testing.assert_allclose(action, [2.0, 0.0, math.pi / 2], atol=1e-12)


def test_stratified_selection_is_split_balanced_and_deterministic() -> None:
    rows = [
        {"trajectory_name": f"{split}_{length}", "split": split, "num_frames": length}
        for split in ("train", "test")
        for length in range(100, 200, 10)
    ]

    selected = select_stratified_trajectories(rows, per_split=2, min_frames=100)

    assert [row["trajectory_name"] for row in selected] == [
        "train_120",
        "train_170",
        "test_120",
        "test_170",
    ]
    assert [row["selection_quantile"] for row in selected] == [0.25, 0.75, 0.25, 0.75]


def test_projection_and_wrapped_direction_metrics() -> None:
    origin = (100, 100)
    assert _project_action(
        np.asarray([2.0, 1.0, 0.0]),
        origin=origin,
        pixels_per_unit=10.0,
        max_radius=10.0,
    ) == (90, 80)
    assert _direction_cosine(
        np.asarray([1.0, 0.0, 0.0]), np.asarray([2.0, 0.0, 0.0])
    ) == 1.0
    assert math.isclose(float(wrap_angle(3 * math.pi)), -math.pi)
