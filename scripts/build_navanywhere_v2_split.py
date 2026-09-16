#!/usr/bin/env python3
"""Freeze the trajectory-disjoint 15-source NavAnywhere v2 train/val split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navanywhere_recipe import (  # noqa: E402
    build_sampling_recipe_from_inventory,
    canonical_json,
    load_sampling_recipe,
    sha256_file,
    write_sampling_recipe,
)


DEFAULT_SOURCES = (
    "BotanicGarden",
    "CASIA-Nav",
    "CityWalker",
    "DL3DV-10K",
    "EgoWalk",
    "KrishnaCam",
    "LAVN",
    "ROVER",
    "RealEstate10K",
    "SANPO",
    "The_Great_Outdoors",
    "Walking_Tours",
    "ego4d",
    "i2Nav-Robot",
    "uB-VisioGeoloc",
)
SPLIT_FORMAT = "navanywhere_trajectory_split"
SPLIT_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-recipe",
        action="append",
        required=True,
        help="Validated recipe contributing disjoint sources; repeatable.",
    )
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--split-output", required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--goals-per-obs", type=int, default=4)
    parser.add_argument("--val-trajectories-per-source", type=int, default=1)
    parser.add_argument("--min-val-observations", type=int, default=256)
    parser.add_argument(
        "--val-samples-per-source",
        type=int,
        default=128,
        help="Logical validation slots per source; 128 gives 1,920 total slots.",
    )
    parser.add_argument(
        "--expected-source",
        action="append",
        default=None,
        help="Expected source ID; defaults to the canonical 15-source set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _tie_break(seed: int, source: str, trajectory: str) -> str:
    return hashlib.sha256(
        f"navanywhere-v2-val\0{seed}\0{source}\0{trajectory}".encode("utf-8")
    ).hexdigest()


def choose_validation_trajectories(
    inventory: list[dict[str, Any]],
    *,
    seed: int,
    count_per_source: int,
    min_observations: int,
) -> list[dict[str, Any]]:
    """Choose a small useful holdout while minimizing removed train data."""

    if count_per_source < 1 or min_observations < 1:
        raise ValueError("validation split counts must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in inventory:
        grouped[str(item["source_id"])].append(item)
    selected: list[dict[str, Any]] = []
    for source in sorted(grouped):
        candidates = grouped[source]
        if len(candidates) <= count_per_source:
            raise ValueError(
                f"Source {source!r} needs more than {count_per_source} trajectories "
                "for a disjoint train/validation split"
            )
        adequate = [
            item
            for item in candidates
            if int(item["observation_count"]) >= min_observations
        ]
        pool = adequate or candidates
        # Smallest adequate trajectories minimize train-data removal. If no
        # trajectory reaches the requested floor, use the largest available.
        if adequate:
            key = lambda item: (
                int(item["observation_count"]),
                _tie_break(seed, source, str(item["trajectory_id"])),
            )
        else:
            key = lambda item: (
                -int(item["observation_count"]),
                _tie_break(seed, source, str(item["trajectory_id"])),
            )
        selected.extend(sorted(pool, key=key)[:count_per_source])
    return sorted(selected, key=lambda item: (item["source_id"], item["trajectory_id"]))


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Split report already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    expected_sources = set(args.expected_source or DEFAULT_SOURCES)
    if len(expected_sources) != 15:
        raise ValueError(f"NavAnywhere v2 requires exactly 15 sources, got {len(expected_sources)}")

    input_records = []
    inventory: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for raw_path in args.inventory_recipe:
        recipe, path, digest = load_sampling_recipe(
            raw_path,
            seed=args.seed,
            context_size=args.context_size,
            goals_per_obs=args.goals_per_obs,
            frame_offset_range=(-64, 64),
        )
        input_records.append(
            {
                "path": path,
                "sha256": digest,
                "inventory_sha256": recipe["inventory_sha256"],
                "sources": sorted(recipe["sources"]),
            }
        )
        for raw_item in recipe["trajectories"]:
            item = dict(raw_item)
            identity = (str(item["source_id"]), str(item["trajectory_id"]))
            if identity in identities:
                raise ValueError(f"Duplicate trajectory across inventory recipes: {identity}")
            identities.add(identity)
            inventory.append(item)
    inventory.sort(key=lambda item: (item["source_id"], item["trajectory_id"]))
    actual_sources = {str(item["source_id"]) for item in inventory}
    if actual_sources != expected_sources:
        raise ValueError(
            "15-source inventory mismatch: "
            f"missing={sorted(expected_sources - actual_sources)}, "
            f"unexpected={sorted(actual_sources - expected_sources)}"
        )

    validation = choose_validation_trajectories(
        inventory,
        seed=args.seed,
        count_per_source=args.val_trajectories_per_source,
        min_observations=args.min_val_observations,
    )
    validation_ids = {
        (str(item["source_id"]), str(item["trajectory_id"])) for item in validation
    }
    training = [
        item
        for item in inventory
        if (str(item["source_id"]), str(item["trajectory_id"])) not in validation_ids
    ]
    train_recipe = build_sampling_recipe_from_inventory(
        training,
        seed=args.seed,
        context_size=args.context_size,
        goals_per_obs=args.goals_per_obs,
    )
    val_recipe = build_sampling_recipe_from_inventory(
        validation,
        seed=args.seed,
        context_size=args.context_size,
        goals_per_obs=args.goals_per_obs,
        samples_per_epoch=len(expected_sources) * args.val_samples_per_source,
    )
    if set(train_recipe["sources"]) != expected_sources or set(val_recipe["sources"]) != expected_sources:
        raise RuntimeError("Both split recipes must retain all 15 sources")

    train_path = write_sampling_recipe(
        args.train_output, train_recipe, overwrite=bool(args.overwrite)
    )
    val_path = write_sampling_recipe(
        args.val_output, val_recipe, overwrite=bool(args.overwrite)
    )
    per_source: dict[str, Any] = {}
    for source in sorted(expected_sources):
        source_train = [item for item in training if item["source_id"] == source]
        source_val = [item for item in validation if item["source_id"] == source]
        per_source[source] = {
            "sampling_weight": 1.0 / len(expected_sources),
            "train_trajectories": len(source_train),
            "train_observations": sum(int(item["observation_count"]) for item in source_train),
            "validation": [
                {
                    "trajectory_id": item["trajectory_id"],
                    "frame_count": int(item["frame_count"]),
                    "observation_count": int(item["observation_count"]),
                    "frame_indices_sha256": item["frame_indices_sha256"],
                }
                for item in source_val
            ],
        }
    report = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "format": SPLIT_FORMAT,
        "seed": int(args.seed),
        "policy": {
            "unit": "trajectory",
            "validation_trajectories_per_source": int(args.val_trajectories_per_source),
            "selection": "smallest_trajectory_with_at_least_min_val_observations_v1",
            "min_val_observations": int(args.min_val_observations),
            "val_samples_per_source": int(args.val_samples_per_source),
            "train_validation_overlap": 0,
        },
        "inputs": input_records,
        "source_count": len(expected_sources),
        "source_sampling_weight": 1.0 / len(expected_sources),
        "train_recipe": {"path": train_path, "sha256": sha256_file(train_path)},
        "val_recipe": {"path": val_path, "sha256": sha256_file(val_path)},
        "totals": {
            "all_trajectories": len(inventory),
            "train_trajectories": len(training),
            "val_trajectories": len(validation),
            "all_observations": sum(int(item["observation_count"]) for item in inventory),
            "train_observations": int(train_recipe["totals"]["observations"]),
            "val_observations": int(val_recipe["totals"]["observations"]),
            "train_samples_per_epoch": int(train_recipe["samples_per_epoch"]),
            "val_samples_per_epoch": int(val_recipe["samples_per_epoch"]),
        },
        "sources": per_source,
    }
    report["split_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    _atomic_json(Path(args.split_output), report, overwrite=bool(args.overwrite))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
