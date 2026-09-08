import copy
import math
from pathlib import Path

import numpy as np
import torch

from geometry_action.tartandrive_vggt_omega import (
    VGGTOmegaCameraExtractor,
    atomic_torch_save,
    build_geometry_payload,
    build_input_manifest,
    build_nwm_frame_pairs,
    build_raw_pose_payload,
    c2w_to_w2c,
    deterministic_shard,
    geometry_actions_tartandrive_forward_camera,
    geometry_payload_matches,
    geometry_policy_descriptor,
    raw_extraction_descriptor,
    raw_pose_matches,
    stitch_pose_windows,
    tartandrive_image_only_scale,
)


def _straight_w2c(num_frames: int, step: float = 2.0) -> np.ndarray:
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    c2w[:, 2, 3] = np.arange(num_frames) * step
    return c2w_to_w2c(c2w)


def test_opencv_to_navigation_axes_scale_and_reverse() -> None:
    extrinsics = _straight_w2c(68)
    scale, diagnostics = tartandrive_image_only_scale(extrinsics)
    assert scale == 0.5
    assert diagnostics["num_nonzero_steps"] == 67
    pairs = np.asarray([[0, 1], [0, 2], [2, 0], [3, 3]])
    actual = geometry_actions_tartandrive_forward_camera(
        extrinsics, pairs, scale=scale
    )
    np.testing.assert_allclose(
        actual,
        [[1, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 0, 0]],
        atol=1e-6,
    )


def test_left_and_right_yaw_sign() -> None:
    extrinsics = np.repeat(np.eye(4)[None, :3, :], 3, axis=0)
    left_relative = np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    right_relative = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    # R_relative = R_cw_current @ R_cw_target.T.
    extrinsics[1, :3, :3] = left_relative.T
    extrinsics[2, :3, :3] = right_relative.T
    actions = geometry_actions_tartandrive_forward_camera(
        extrinsics, np.asarray([[0, 1], [0, 2]]), scale=1.0
    )
    np.testing.assert_allclose(
        actions[:, 2], [math.pi / 2, -math.pi / 2], atol=1e-6
    )


def test_pitch_roll_are_dropped_before_se2_inverse_and_composition() -> None:
    def rotation_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rotation_y(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rotation_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    c2w = np.repeat(np.eye(4)[None], 3, axis=0)
    c2w[0, :3, :3] = rotation_z(0.2) @ rotation_x(0.1)
    c2w[1, :3, :3] = rotation_y(0.4) @ rotation_x(-0.3)
    c2w[2, :3, :3] = rotation_z(-0.25) @ rotation_y(0.8)
    c2w[:, :3, 3] = [[1, 3, -2], [2, 5, 0], [4, 4, 3]]
    extrinsics = c2w_to_w2c(c2w)
    actions = geometry_actions_tartandrive_forward_camera(
        extrinsics, np.asarray([[0, 1], [1, 0], [1, 2], [0, 2]]), scale=0.7
    )

    def se2(action: np.ndarray) -> np.ndarray:
        c, s = math.cos(float(action[2])), math.sin(float(action[2]))
        return np.asarray(
            [[c, -s, action[0]], [s, c, action[1]], [0, 0, 1]],
            dtype=np.float64,
        )

    np.testing.assert_allclose(se2(actions[0]) @ se2(actions[1]), np.eye(3), atol=1e-6)
    np.testing.assert_allclose(
        se2(actions[0]) @ se2(actions[2]), se2(actions[3]), atol=1e-6
    )


def test_pair_domain_exactly_matches_nwm_training_index() -> None:
    assert build_nwm_frame_pairs(67).shape == (0, 2)
    pairs = build_nwm_frame_pairs(68)
    assert pairs.shape == (68, 2)
    np.testing.assert_array_equal(pairs[:, 0], np.full(68, 3))
    assert set(pairs[:, 1].tolist()) == set(range(68))

    longer = build_nwm_frame_pairs(70)
    assert set(longer[:, 0].tolist()) == {3, 4, 5}
    for current in (3, 4, 5):
        targets = longer[longer[:, 0] == current, 1]
        assert targets[0] == 0
        assert targets[-1] == min(69, current + 64)


def test_lpt_sharding_is_complete_disjoint_and_cost_balanced() -> None:
    items = [
        {"trajectory_name": str(index), "num_frames": frames}
        for index, frames in enumerate([100, 99, 10, 9, 8, 7])
    ]
    shards = [deterministic_shard(items, rank, 2) for rank in range(2)]
    names = [{item["trajectory_name"] for item in shard} for shard in shards]
    assert names[0].isdisjoint(names[1])
    assert names[0] | names[1] == {str(index) for index in range(len(items))}
    costs = [sum(item["num_frames"] ** 2 for item in shard) for shard in shards]
    assert abs(costs[0] - costs[1]) < 500
    assert shards == [deterministic_shard(items, rank, 2) for rank in range(2)]


def test_parallel_manifest_is_identical_to_single_worker(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    split_root = tmp_path / "splits"
    for name, count, split in (("b", 3, "train"), ("a", 2, "test")):
        trajectory = data_root / name
        trajectory.mkdir(parents=True)
        for frame_id in range(count):
            (trajectory / f"{frame_id}.jpg").write_bytes(
                f"{name}-{frame_id}".encode()
            )
        directory = split_root / split
        directory.mkdir(parents=True)
        (directory / "traj_names.txt").write_text(name + "\n", encoding="utf-8")
    one = build_input_manifest(
        data_root=data_root,
        split_root=split_root,
        splits=["train", "test"],
        workers=1,
    )
    four = build_input_manifest(
        data_root=data_root,
        split_root=split_root,
        splits=["train", "test"],
        workers=4,
    )
    assert one == four
    assert [item["trajectory_name"] for item in one["trajectories"]] == ["a", "b"]


def test_window_sim3_stitch_recovers_global_camera_poses() -> None:
    global_c2w = np.repeat(np.eye(4)[None], 9, axis=0)
    global_c2w[:, 0, 3] = np.arange(9)
    for index, angle in enumerate(np.arange(9) * 0.03):
        global_c2w[index, :3, :3] = [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]

    alignment_rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    alignment_scale = 2.5
    alignment_translation = np.asarray([5.0, -3.0, 2.0])
    local = global_c2w[3:].copy()
    local[:, :3, :3] = np.einsum(
        "ij,njk->nik", alignment_rotation.T, global_c2w[3:, :3, :3]
    )
    local[:, :3, 3] = (
        np.einsum(
            "ij,nj->ni",
            alignment_rotation.T,
            global_c2w[3:, :3, 3] - alignment_translation,
        )
        / alignment_scale
    )
    intrinsics = np.repeat(np.eye(3)[None], 6, axis=0)
    predictions = [
        {
            "start_index": 0,
            "end_index_exclusive": 6,
            "extrinsics_w2c": c2w_to_w2c(global_c2w[:6]),
            "intrinsics": intrinsics,
            "image_size_hw": (384, 512),
        },
        {
            "start_index": 3,
            "end_index_exclusive": 9,
            "extrinsics_w2c": c2w_to_w2c(local),
            "intrinsics": intrinsics,
            "image_size_hw": (384, 512),
        },
    ]
    stitched, _, windows, metadata = stitch_pose_windows(
        predictions, num_frames=9, requested_overlap=3
    )
    np.testing.assert_allclose(stitched, c2w_to_w2c(global_c2w), atol=1e-5)
    assert metadata["policy"] == "sequential_overlap_sim3"
    assert windows[1]["overlap_center_rmse"] < 1e-7
    assert windows[1]["overlap_rotation_rmse_deg"] < 1e-6


def test_window_sim3_stitch_treats_nonpositive_scale_as_degenerate() -> None:
    global_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 9, axis=0)
    global_c2w[:, 0, 3] = np.arange(9, dtype=np.float64)
    local_c2w = global_c2w[3:].copy()
    local_c2w[:, 0, 3] *= -1.0
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None], 6, axis=0)
    predictions = [
        {
            "start_index": 0,
            "end_index_exclusive": 6,
            "extrinsics_w2c": c2w_to_w2c(global_c2w[:6]),
            "intrinsics": intrinsics,
            "image_size_hw": (384, 512),
        },
        {
            "start_index": 3,
            "end_index_exclusive": 9,
            "extrinsics_w2c": c2w_to_w2c(local_c2w),
            "intrinsics": intrinsics,
            "image_size_hw": (384, 512),
        },
    ]

    stitched, _, windows, _ = stitch_pose_windows(
        predictions,
        num_frames=9,
        requested_overlap=3,
        allow_degenerate_scale=True,
    )

    assert np.isfinite(stitched).all()
    assert windows[1]["alignment_to_global"]["scale"] == 1.0
    assert windows[1]["alignment_to_global"]["degenerate_scale"] is True


def test_schema_and_resume_fingerprints(tmp_path: Path) -> None:
    extrinsics = _straight_w2c(68)
    extraction = raw_extraction_descriptor(
        resolution=512,
        resize_mode="max_size",
        window_size=0,
        overlap=32,
        inference_path="fast",
        dtype="bfloat16",
        allow_tf32=True,
        allow_degenerate_window_scale=False,
        seed=0,
    )
    raw = build_raw_pose_payload(
        dataset_name="tartan_drive",
        trajectory_name="trajectory",
        frame_list_sha256="f" * 64,
        extrinsics_w2c=extrinsics,
        intrinsics=np.repeat(np.eye(3)[None], 68, axis=0),
        image_size_hw=(384, 512),
        windows=[],
        window_alignment={
            "policy": "single_window_identity",
            "overlap": 0,
            "min_overlap": 3,
            "pose_selection": "first_prediction_wins_overlap",
            "scale_fallback": "error",
        },
        checkpoint_sha256="c" * 64,
        code_revision="revision",
        model_revision="model-revision",
        resize_mode="max_size",
        resolution=512,
        inference_path="fast",
        dtype="bfloat16",
        input_manifest_sha256="i" * 64,
        checkpoint_manifest_sha256="m" * 64,
        extraction_descriptor=extraction,
    )
    raw_path = tmp_path / "raw.pt"
    atomic_torch_save(raw, raw_path)
    assert raw_pose_matches(
        raw_path,
        dataset_name="tartan_drive",
        trajectory_name="trajectory",
        frame_list_sha256="f" * 64,
        checkpoint_sha256="c" * 64,
        code_revision="revision",
        checkpoint_manifest_sha256="m" * 64,
        model_revision="model-revision",
        input_manifest_sha256="i" * 64,
        extraction_fingerprint=extraction["fingerprint"],
    )

    policy = geometry_policy_descriptor(
        alignment_policy="tartandrive_forward_camera",
        degenerate_scale_policy="empty_only",
        nonzero_epsilon=1e-6,
        camera_to_navigation=None,
        meters_per_model_unit=None,
        translation_unit="waypoint_spacing_units",
        waypoint_spacing_meters=None,
        min_offset=-64,
        max_offset=64,
        context_size=4,
        len_traj_pred=64,
    )
    geometry = build_geometry_payload(raw, source_pose_sha256="s" * 64)
    motion_path = tmp_path / "motion.pt"
    atomic_torch_save(geometry, motion_path)
    assert geometry_payload_matches(
        motion_path,
        dataset_name="tartan_drive",
        trajectory_name="trajectory",
        frame_list_sha256="f" * 64,
        checkpoint_sha256="c" * 64,
        alignment_policy="tartandrive_forward_camera",
        policy_fingerprint=policy["fingerprint"],
    )
    assert not geometry_payload_matches(
        motion_path,
        dataset_name="tartan_drive",
        trajectory_name="trajectory",
        frame_list_sha256="f" * 64,
        checkpoint_sha256="c" * 64,
        alignment_policy="tartandrive_forward_camera",
        policy_fingerprint="0" * 64,
    )


def test_raw_resume_rejects_every_result_affecting_configuration_change(
    tmp_path: Path,
) -> None:
    base = {
        "resolution": 512,
        "resize_mode": "max_size",
        "window_size": 0,
        "overlap": 32,
        "inference_path": "fast",
        "dtype": "bfloat16",
        "allow_tf32": True,
        "allow_degenerate_window_scale": False,
        "seed": 0,
    }
    descriptor = raw_extraction_descriptor(**base)
    raw = build_raw_pose_payload(
        dataset_name="tartan_drive",
        trajectory_name="trajectory",
        frame_list_sha256="f" * 64,
        extrinsics_w2c=_straight_w2c(68),
        intrinsics=np.repeat(np.eye(3)[None], 68, axis=0),
        image_size_hw=(384, 512),
        windows=[],
        window_alignment={
            "policy": "single_window_identity",
            "overlap": 0,
            "min_overlap": 3,
            "pose_selection": "first_prediction_wins_overlap",
            "scale_fallback": "error",
        },
        checkpoint_sha256="c" * 64,
        code_revision="revision",
        model_revision="model-revision",
        resize_mode="max_size",
        resolution=512,
        inference_path="fast",
        dtype="bfloat16",
        input_manifest_sha256="i" * 64,
        checkpoint_manifest_sha256="m" * 64,
        extraction_descriptor=descriptor,
    )
    path = tmp_path / "raw.pt"
    atomic_torch_save(raw, path)

    def matches(fingerprint: str = descriptor["fingerprint"], **metadata) -> bool:
        expected = {
            "model_revision": "model-revision",
            "input_manifest_sha256": "i" * 64,
            **metadata,
        }
        return raw_pose_matches(
            path,
            dataset_name="tartan_drive",
            trajectory_name="trajectory",
            frame_list_sha256="f" * 64,
            checkpoint_sha256="c" * 64,
            code_revision="revision",
            checkpoint_manifest_sha256="m" * 64,
            extraction_fingerprint=fingerprint,
            **expected,
        )

    assert matches()
    changes = (
        {"resolution": 384},
        {"resize_mode": "balanced"},
        {"window_size": 64},
        {"overlap": 16},
        {"inference_path": "full"},
        {"dtype": "float16"},
        {"allow_tf32": False},
        {"allow_degenerate_window_scale": True},
        {"seed": 17},
    )
    for change in changes:
        changed = raw_extraction_descriptor(**{**base, **change})
        assert not matches(changed["fingerprint"]), change
    assert not matches(model_revision="other-revision")
    assert not matches(input_manifest_sha256="0" * 64)

    legacy = dict(raw)
    legacy.pop("extraction_descriptor")
    atomic_torch_save(legacy, path)
    assert not matches()

    tampered = copy.deepcopy(raw)
    tampered["extraction_descriptor"]["configuration"]["inference"][
        "allow_tf32"
    ] = False
    atomic_torch_save(tampered, path)
    assert not matches()

    inconsistent = copy.deepcopy(raw)
    inconsistent["preprocessing"]["image_resolution"] = 384
    atomic_torch_save(inconsistent, path)
    assert not matches()


def test_fast_path_caches_only_final_aggregator_layer() -> None:
    snapshots = []

    class Aggregator:
        def __init__(self) -> None:
            self.depth = 24
            self.cached_layer_indices = {4, 11, 17, 23}

    class Model:
        def __init__(self) -> None:
            self.aggregator = Aggregator()
            self.dense_head = object()

        def __call__(self, images):
            snapshots.append(
                (set(self.aggregator.cached_layer_indices), self.dense_head)
            )
            return {"images": images}

    extractor = object.__new__(VGGTOmegaCameraExtractor)
    extractor.model = Model()
    extractor.retain_dense_head = True
    extractor._full_cached_layer_indices = {4, 11, 17, 23}
    extractor._camera_cached_layer_indices = {23}
    images = torch.zeros(1)
    extractor._forward(images, full=False)
    assert snapshots[-1] == ({23}, None)
    assert extractor.model.aggregator.cached_layer_indices == {4, 11, 17, 23}
    assert extractor.model.dense_head is not None
