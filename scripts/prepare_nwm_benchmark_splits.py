#!/usr/bin/env python3
"""Create the missing fixed rollout/navigation splits for the unified benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm/data")
ROLLOUT_DATASETS = ("planetary_rover", "unitree_go2", "tum_rgbd", "uzh_fpv")
HURON_DATASET = "sacson"
CONTEXT_FRAMES = 4
DIRECT_FUTURE_FRAMES = 16
DIRECT_SAMPLE_COUNT = 500
ROLLOUT_FUTURE_FRAMES = 64
ROLLOUT_SAMPLE_COUNT = 150
NAVIGATION_HORIZON = 8
NAVIGATION_SAMPLE_COUNT = 100


@dataclass(frozen=True)
class Window:
    trajectory: str
    current: int
    displacement: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_pickle(path: Path, payload: Any) -> None:
    atomic_bytes(path, pickle.dumps(payload, protocol=4))


def load_positions(data_root: Path, dataset: str, trajectory: str) -> np.ndarray:
    path = data_root / dataset / trajectory / "traj_data.pkl"
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    positions = np.asarray(payload["position"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"Invalid positions in {path}: {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError(f"Non-finite positions in {path}")
    return positions


def split_trajectories(
    split_root: Path,
    filenames: tuple[str, ...] = ("rollout_traj_names.txt", "traj_names.txt"),
) -> list[str]:
    for filename in filenames:
        path = split_root / filename
        if path.is_file():
            names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if names:
                return names
    raise FileNotFoundError(f"No trajectory manifest under {split_root}")


def candidate_windows(
    data_root: Path,
    dataset: str,
    split_root: Path,
    horizon: int,
    *,
    manifest_names: tuple[str, ...] = ("rollout_traj_names.txt", "traj_names.txt"),
    skip_missing: bool = False,
) -> list[Window]:
    result = []
    for trajectory in sorted(split_trajectories(split_root, manifest_names)):
        trajectory_path = data_root / dataset / trajectory / "traj_data.pkl"
        if skip_missing and not trajectory_path.is_file():
            continue
        positions = load_positions(data_root, dataset, trajectory)
        for current in range(CONTEXT_FRAMES - 1, len(positions) - horizon):
            displacement = float(
                np.linalg.norm(positions[current + horizon, :2] - positions[current, :2])
            )
            result.append(Window(trajectory, current, displacement))
    return result


def representative_windows(candidates: Sequence[Window], count: int) -> list[Window]:
    """Select deterministic trajectory coverage plus movement-distance quantiles."""
    forced = coverage_windows(candidates, count)
    forced_set = set(forced)
    available = sorted(
        (item for item in candidates if item not in forced_set),
        key=lambda item: (item.displacement, item.trajectory, item.current),
    )
    required = count - len(forced)
    if required < 0 or len(available) < required:
        raise ValueError(f"Cannot select {count} windows from {len(candidates)} candidates")
    if required:
        # Pick the centre of equally sized rank buckets. This preserves the full
        # local movement-distance distribution without any random state.
        indices = np.floor(
            (np.arange(required, dtype=np.float64) + 0.5)
            * len(available)
            / required
        ).astype(np.int64)
        selected = forced + [available[int(index)] for index in indices]
    else:
        selected = forced
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("Representative split selection is not unique and complete")
    return sorted(selected, key=lambda item: (item.trajectory, item.current))


def build_huron_split(
    data_root: Path,
    horizon: int,
    count: int,
    manifest_names: tuple[str, ...],
    navigation: bool = False,
) -> list[tuple[Any, ...]]:
    split_root = PROJECT_ROOT / "data_splits" / HURON_DATASET / "test"
    candidates = candidate_windows(
        data_root,
        HURON_DATASET,
        split_root,
        horizon,
        manifest_names=manifest_names,
        skip_missing=True,
    )
    selected = representative_windows(candidates, count)
    if navigation:
        return [
            (item.trajectory, item.current, horizon, horizon) for item in selected
        ]
    return [
        (item.trajectory, item.current, -min(item.current, horizon), horizon)
        for item in selected
    ]


def remove_nearest_targets(targets: list[float], selected: Sequence[Window]) -> list[float]:
    remaining = list(targets)
    for window in sorted(selected, key=lambda item: item.displacement):
        index = min(
            range(len(remaining)),
            key=lambda item: (abs(remaining[item] - window.displacement), item),
        )
        remaining.pop(index)
    return remaining


def coverage_windows(candidates: Sequence[Window], limit: int) -> list[Window]:
    by_trajectory: dict[str, list[Window]] = {}
    for window in candidates:
        by_trajectory.setdefault(window.trajectory, []).append(window)
    names = sorted(by_trajectory)
    if len(names) > limit:
        indices = np.floor(np.linspace(0, len(names), limit, endpoint=False)).astype(int)
        names = [names[int(index)] for index in indices]
    return [
        sorted(by_trajectory[name], key=lambda item: item.current)[
            len(by_trajectory[name]) // 2
        ]
        for name in names
    ]


def match_distribution(
    candidates: Sequence[Window], targets: Sequence[float], count: int
) -> list[Window]:
    forced = coverage_windows(candidates, count)
    forced_set = set(forced)
    available = sorted(
        (item for item in candidates if item not in forced_set),
        key=lambda item: (item.displacement, item.trajectory, item.current),
    )
    remaining_targets = remove_nearest_targets(
        sorted(float(value) for value in targets), forced
    )
    required = count - len(forced)
    if required == 0:
        return sorted(forced, key=lambda item: (item.trajectory, item.current))
    if len(available) < required or len(remaining_targets) != required:
        raise ValueError(
            f"Cannot select {count} windows from {len(candidates)} candidates"
        )

    values = np.asarray([item.displacement for item in available], dtype=np.float64)
    previous = np.abs(values - remaining_targets[0])
    previous[len(available) - required + 1 :] = np.inf
    back = np.full((required, len(available)), -1, dtype=np.int32)
    for target_index in range(1, required):
        prefix_values = np.minimum.accumulate(previous)
        prefix_args = np.empty(len(previous), dtype=np.int32)
        best_index = 0
        for index in range(len(previous)):
            if previous[index] < previous[best_index]:
                best_index = index
            prefix_args[index] = best_index
        current = np.full(len(available), np.inf, dtype=np.float64)
        first = target_index
        last = len(available) - (required - target_index)
        indices = np.arange(first, last + 1)
        predecessors = indices - 1
        current[indices] = (
            np.abs(values[indices] - remaining_targets[target_index])
            + prefix_values[predecessors]
        )
        back[target_index, indices] = prefix_args[predecessors]
        previous = current
    end = int(np.argmin(previous))
    selected_indices = [end]
    for target_index in range(required - 1, 0, -1):
        end = int(back[target_index, end])
        if end < 0:
            raise RuntimeError("Broken split-selection backpointer")
        selected_indices.append(end)
    selected_indices.reverse()
    selected = forced + [available[index] for index in selected_indices]
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("Split selection is not unique and complete")
    return sorted(selected, key=lambda item: (item.trajectory, item.current))


def go_stanford_rollout_targets(data_root: Path) -> list[float]:
    path = PROJECT_ROOT / "data_splits/go_stanford/test/rollout.pkl"
    with path.open("rb") as stream:
        entries = pickle.load(stream)
    targets = []
    for trajectory, current, *_ in entries:
        positions = load_positions(data_root, "go_stanford", str(trajectory))
        current = int(current)
        targets.append(
            float(
                np.linalg.norm(
                    positions[current + ROLLOUT_FUTURE_FRAMES, :2]
                    - positions[current, :2]
                )
            )
        )
    if len(targets) != ROLLOUT_SAMPLE_COUNT:
        raise ValueError(f"Go Stanford rollout split has {len(targets)} entries")
    return targets


def build_rollout_split(data_root: Path, dataset: str) -> list[tuple[Any, ...]]:
    split_root = PROJECT_ROOT / "data_splits" / dataset / "test"
    candidates = candidate_windows(
        data_root, dataset, split_root, ROLLOUT_FUTURE_FRAMES
    )
    selected = match_distribution(
        candidates, go_stanford_rollout_targets(data_root), ROLLOUT_SAMPLE_COUNT
    )
    return [
        (
            item.trajectory,
            item.current,
            -min(item.current, 64),
            ROLLOUT_FUTURE_FRAMES,
        )
        for item in selected
    ]


def build_go_stanford_navigation_split(
    data_root: Path,
) -> list[tuple[Any, ...]]:
    split_root = PROJECT_ROOT / "data_splits/go_stanford/test"
    candidates = candidate_windows(
        data_root, "go_stanford", split_root, NAVIGATION_HORIZON
    )
    selected = coverage_windows(candidates, NAVIGATION_SAMPLE_COUNT)
    if len(selected) != NAVIGATION_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {NAVIGATION_SAMPLE_COUNT} Go Stanford navigation windows"
        )
    return [
        (item.trajectory, item.current, NAVIGATION_HORIZON, NAVIGATION_HORIZON)
        for item in sorted(selected, key=lambda item: (item.trajectory, item.current))
    ]


def write_or_check(path: Path, payload: list[tuple[Any, ...]], check: bool) -> dict[str, Any]:
    serialized = pickle.dumps(payload, protocol=4)
    if check:
        if not path.is_file() or path.read_bytes() != serialized:
            raise RuntimeError(f"Split is missing or stale: {path}")
    else:
        atomic_pickle(path, payload)
    return {
        "path": str(path.resolve()),
        "sample_count": len(payload),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--repair-huron",
        action="store_true",
        help=(
            "Build fixed HuRoN splits only from locally available SACSoN "
            "trajectories, replacing historical entries whose source chunks "
            "are absent from the public processed copy."
        ),
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    outputs: dict[str, Any] = {"rollout": {}, "navigation": {}}
    if args.repair_huron:
        huron_root = PROJECT_ROOT / "data_splits/sacson/test"
        outputs["direct"] = {
            "huron": write_or_check(
                huron_root / "time.pkl",
                build_huron_split(
                    data_root,
                    DIRECT_FUTURE_FRAMES,
                    DIRECT_SAMPLE_COUNT,
                    ("traj_names.txt",),
                ),
                args.check,
            )
        }
        outputs["rollout"]["huron"] = write_or_check(
            huron_root / "rollout.pkl",
            build_huron_split(
                data_root,
                ROLLOUT_FUTURE_FRAMES,
                ROLLOUT_SAMPLE_COUNT,
                ("rollout_traj_names.txt", "traj_names.txt"),
            ),
            args.check,
        )
        outputs["navigation"]["huron"] = write_or_check(
            huron_root / "navigation_eval.pkl",
            build_huron_split(
                data_root,
                NAVIGATION_HORIZON,
                NAVIGATION_SAMPLE_COUNT,
                ("rollout_traj_names.txt", "traj_names.txt"),
                navigation=True,
            ),
            args.check,
        )
        print(json.dumps(outputs, indent=2, sort_keys=True))
        return
    for dataset in ROLLOUT_DATASETS:
        path = PROJECT_ROOT / "data_splits" / dataset / "test/rollout.pkl"
        outputs["rollout"][dataset] = write_or_check(
            path, build_rollout_split(data_root, dataset), args.check
        )
    navigation_path = (
        PROJECT_ROOT / "data_splits/go_stanford/test/navigation_eval.pkl"
    )
    outputs["navigation"]["go_stanford"] = write_or_check(
        navigation_path,
        build_go_stanford_navigation_split(data_root),
        args.check,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
