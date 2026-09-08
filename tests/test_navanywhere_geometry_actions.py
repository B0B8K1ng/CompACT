from __future__ import annotations

import numpy as np

from precompute_navanywhere_geometry_actions import (
    _position_pairs,
    _window_cost,
    build_navanywhere_frame_pairs,
)


def test_frame_pairs_cover_every_strict_local_lookup() -> None:
    indices = np.arange(10, 16, dtype=np.int64)
    pairs = build_navanywhere_frame_pairs(
        indices, context_size=4, max_abs_frame_offset=2
    )
    assert pairs.tolist() == [
        [13, 11], [13, 12], [13, 13], [13, 14], [13, 15],
        [14, 12], [14, 13], [14, 14], [14, 15],
        [15, 13], [15, 14], [15, 15],
    ]
    assert _position_pairs(indices, pairs).tolist() == [
        [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
        [4, 2], [4, 3], [4, 4], [4, 5],
        [5, 3], [5, 4], [5, 5],
    ]


def test_frame_pairs_respect_gaps_and_short_trajectories() -> None:
    assert build_navanywhere_frame_pairs([1, 2], context_size=4).shape == (0, 2)
    pairs = build_navanywhere_frame_pairs(
        [10, 11, 13, 14], context_size=2, max_abs_frame_offset=2
    )
    assert pairs.tolist() == [
        [11, 10], [11, 11], [11, 13],
        [13, 11], [13, 13], [13, 14],
        [14, 13], [14, 14],
    ]


def test_window_cost_accounts_for_overlapping_windows() -> None:
    assert _window_cost(10, 128, 32) == 16**2
    assert _window_cost(128, 128, 32) == 128**2
    assert _window_cost(129, 128, 32) == 128**2 + 33**2
