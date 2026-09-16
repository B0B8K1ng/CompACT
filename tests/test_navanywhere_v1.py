from __future__ import annotations

import hashlib

from scripts.build_navanywhere_v1_split import (
    DEFAULT_SOURCES,
    choose_validation_trajectories,
)


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


def test_v1_source_contract_has_only_the_original_thirteen_sources() -> None:
    assert len(DEFAULT_SOURCES) == 13
    assert "ego4d" not in DEFAULT_SOURCES
    assert "The_Great_Outdoors" not in DEFAULT_SOURCES


def test_v1_holdout_is_smallest_adequate_trajectory_per_source() -> None:
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
