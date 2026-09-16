from __future__ import annotations

import hashlib

import numpy as np

from navanywhere_recipe import build_sampling_recipe_from_inventory
from precompute_navanywhere_nav1_latent_actions import _matching_pair_rows
from scripts.build_navanywhere_v2_split import choose_validation_trajectories


def _record(source: str, trajectory: str, observations: int) -> dict[str, object]:
    frames = observations + 3
    return {
        "source_id": source,
        "trajectory_id": trajectory,
        "frame_count": frames,
        "observation_count": observations,
        "first_frame_index": 0,
        "last_frame_index": frames - 1,
        "frame_indices_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
    }


def test_v2_holdout_is_smallest_adequate_trajectory_per_source() -> None:
    inventory = sorted(
        [
            _record("a", "too_short", 100),
            _record("a", "small", 256),
            _record("a", "large", 900),
            _record("b", "only_short", 20),
            _record("b", "largest_short", 40),
        ],
        key=lambda item: (item["source_id"], item["trajectory_id"]),
    )
    selected = choose_validation_trajectories(
        inventory,
        seed=7,
        count_per_source=1,
        min_observations=256,
    )
    assert [(item["source_id"], item["trajectory_id"]) for item in selected] == [
        ("a", "small"),
        ("b", "largest_short"),
    ]


def test_recipe_from_inventory_keeps_uniform_source_policy() -> None:
    inventory = sorted(
        [_record("a", "a0", 8), _record("b", "b0", 12)],
        key=lambda item: (item["source_id"], item["trajectory_id"]),
    )
    recipe = build_sampling_recipe_from_inventory(
        inventory,
        seed=11,
        context_size=4,
        goals_per_obs=4,
        samples_per_epoch=64,
    )
    assert recipe["totals"] == {
        "sources": 2,
        "trajectories": 2,
        "observations": 20,
        "samples_per_epoch": 64,
    }
    assert recipe["observation_sampling"]["source_weighting"] == "uniform"


def test_pair_reuse_matches_only_exact_sorted_rows() -> None:
    requested = np.asarray([[3, 2], [3, 3], [4, 4], [7, 8]], dtype=np.int64)
    available = np.asarray([[2, 2], [3, 2], [4, 4], [7, 7]], dtype=np.int64)
    requested_rows, available_rows = _matching_pair_rows(requested, available)
    assert requested_rows.tolist() == [0, 2]
    assert available_rows.tolist() == [1, 2]
