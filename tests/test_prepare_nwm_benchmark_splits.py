import pickle
from pathlib import Path

import numpy as np

from scripts import prepare_nwm_benchmark_splits as splits


def write_trajectory(root: Path, name: str, length: int) -> None:
    path = root / splits.HURON_DATASET / name
    path.mkdir(parents=True)
    positions = np.stack(
        [np.arange(length, dtype=np.float64), np.zeros(length, dtype=np.float64)],
        axis=1,
    )
    with (path / "traj_data.pkl").open("wb") as stream:
        pickle.dump({"position": positions}, stream)


def test_representative_windows_are_unique_and_cover_trajectories() -> None:
    candidates = [
        splits.Window(trajectory, current, float(current))
        for trajectory in ("a", "b", "c")
        for current in range(10)
    ]

    selected = splits.representative_windows(candidates, 9)

    assert len(selected) == len(set(selected)) == 9
    assert {item.trajectory for item in selected} == {"a", "b", "c"}


def test_huron_builder_skips_missing_public_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    split_root = project_root / "data_splits/sacson/test"
    split_root.mkdir(parents=True)
    (split_root / "traj_names.txt").write_text("available_a\nmissing\navailable_b\n")
    data_root = tmp_path / "data"
    write_trajectory(data_root, "available_a", 30)
    write_trajectory(data_root, "available_b", 35)
    monkeypatch.setattr(splits, "PROJECT_ROOT", project_root)

    entries = splits.build_huron_split(
        data_root,
        horizon=4,
        count=8,
        manifest_names=("traj_names.txt",),
    )

    assert len(entries) == len(set(entries)) == 8
    assert {entry[0] for entry in entries} == {"available_a", "available_b"}
    assert all(entry[2] == -min(entry[1], 4) and entry[3] == 4 for entry in entries)
