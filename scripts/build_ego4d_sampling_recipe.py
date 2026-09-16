#!/usr/bin/env python3
"""Build an Ego4D NavAnywhere recipe from its completed release inventory.

The release inventory already audited every JPEG and records contiguous frame
ranges. Reusing it avoids a second 13-million-file directory walk before latent
action extraction; extraction still checks the live inventory trajectory by
trajectory before writing each cache file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navanywhere_recipe import (  # noqa: E402
    DEFAULT_GOAL_STRATA,
    GOAL_SAMPLING,
    OBSERVATION_SAMPLING,
    RECIPE_FORMAT,
    RECIPE_SCHEMA_VERSION,
    canonical_json,
    frame_indices_sha256,
    load_sampling_recipe,
    write_sampling_recipe,
)


DEFAULT_DATA_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere")
DEFAULT_INVENTORY = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/"
    "modelscope_staging/audits/ego4d-20260908/inventory-v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ego4d")
    parser.add_argument("--source-id", default="ego4d")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--goals-per-obs", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_recipe(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root.expanduser().resolve()
    inventory_path = args.inventory.expanduser().resolve()
    release = json.loads(inventory_path.read_text(encoding="utf-8"))
    datasets = release.get("datasets")
    if not isinstance(datasets, list):
        raise TypeError("release inventory has no datasets list")
    matches = [item for item in datasets if item.get("name") == args.dataset_name]
    if len(matches) != 1:
        raise ValueError(
            f"expected one inventory dataset {args.dataset_name!r}, found {len(matches)}"
        )
    dataset = matches[0]
    if dataset.get("anomalies") or int(dataset.get("anomaly_count", -1)) != 0:
        raise RuntimeError("Ego4D release inventory contains anomalies")
    raw_trajectories = dataset.get("trajectories")
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise ValueError("Ego4D release inventory has no trajectories")
    if len(raw_trajectories) != int(dataset.get("trajectory_count", -1)):
        raise ValueError("Ego4D release trajectory count mismatch")

    source_root = (data_root / args.source_id).resolve()
    if not source_root.is_dir() or data_root not in source_root.parents:
        raise FileNotFoundError(f"Ego4D source root is missing or unsafe: {source_root}")
    expected_names = {str(item.get("name")) for item in raw_trajectories}
    actual_names = {
        entry.name for entry in source_root.iterdir() if entry.is_dir()
    }
    if actual_names != expected_names:
        raise RuntimeError(
            "live Ego4D trajectory inventory differs from the completed release: "
            f"missing={len(expected_names - actual_names)}, "
            f"extra={len(actual_names - expected_names)}"
        )

    context_size = int(args.context_size)
    trajectories: list[dict[str, object]] = []
    total_frames = 0
    for item in sorted(raw_trajectories, key=lambda value: str(value["name"])):
        name = str(item["name"])
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe trajectory name: {name!r}")
        frame_count = int(item["frame_count"])
        if frame_count != int(item["file_count"]) or frame_count < context_size:
            raise ValueError(f"invalid frame inventory for {name}")
        if item.get("first_frame") != "0.jpg" or item.get("last_frame") != f"{frame_count - 1}.jpg":
            raise ValueError(f"non-contiguous frame inventory for {name}")
        trajectory_root = source_root / name
        if not (trajectory_root / "0.jpg").is_file() or not (
            trajectory_root / f"{frame_count - 1}.jpg"
        ).is_file():
            raise FileNotFoundError(f"boundary frame is missing for {name}")
        trajectories.append(
            {
                "source_id": str(args.source_id),
                "trajectory_id": name,
                "frame_count": frame_count,
                "observation_count": frame_count - context_size + 1,
                "first_frame_index": 0,
                "last_frame_index": frame_count - 1,
                "frame_indices_sha256": frame_indices_sha256(
                    np.arange(frame_count, dtype=np.int64)
                ),
            }
        )
        total_frames += frame_count

    if total_frames != int(dataset.get("frame_count", -1)):
        raise ValueError("Ego4D release frame total mismatch")
    observations = sum(int(item["observation_count"]) for item in trajectories)
    samples_per_epoch = int(args.samples_per_epoch) or observations
    recipe: dict[str, object] = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "format": RECIPE_FORMAT,
        "seed": int(args.seed),
        "context_size": context_size,
        "goals_per_obs": int(args.goals_per_obs),
        "frame_offset_range": [-64, 64],
        "samples_per_epoch": samples_per_epoch,
        "observation_sampling": {
            "strategy": OBSERVATION_SAMPLING,
            "source_weighting": "uniform",
            "trajectory_weighting_within_source": "uniform",
            "observation_cycle": "coprime_affine_permutation",
        },
        "goal_sampling": {
            "strategy": GOAL_SAMPLING,
            "strata": [list(bounds) for bounds in DEFAULT_GOAL_STRATA],
            "without_replacement_when_possible": True,
        },
        "inventory_sha256": hashlib.sha256(canonical_json(trajectories)).hexdigest(),
        "totals": {
            "sources": 1,
            "trajectories": len(trajectories),
            "observations": observations,
            "samples_per_epoch": samples_per_epoch,
        },
        "sources": {
            str(args.source_id): {
                "trajectories": len(trajectories),
                "observations": observations,
            }
        },
        "trajectories": trajectories,
    }
    return recipe


def main() -> None:
    args = parse_args()
    recipe = build_recipe(args)
    output = write_sampling_recipe(args.output, recipe, overwrite=args.overwrite)
    validated, resolved, digest = load_sampling_recipe(output)
    print(
        json.dumps(
            {
                "recipe": resolved,
                "sha256": digest,
                "inventory": str(args.inventory.expanduser().resolve()),
                "totals": validated["totals"],
                "sources": validated["sources"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
