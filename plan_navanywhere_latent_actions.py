#!/usr/bin/env python3
"""Plan-only accounting for NavAnywhere Stage-1 latent-action pairs.

This command performs no image I/O and no model inference.  It replays the
Stage-1 DistributedSampler contract and the exact recipe goal sampler, then
reports both requested and unique local frame pairs.  Completed TimePT/GeoPT
logs can be supplied as independent reference traces; the command fails closed
if their launch contract or per-epoch proxy counts do not match the replay.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from navanywhere_recipe import (
    affine_observation_slot,
    canonical_json,
    coprime_stride,
    load_sampling_recipe,
    sample_recipe_goal_offsets,
)


SCHEMA_VERSION = 1
FORMAT_NAME = "navanywhere_latent_action_training_pair_plan"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DEFAULT_REFERENCE_LOGS = (
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_stage1/"
    "train_timept_20260902_130035.log",
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/logs/navanywhere_stage1/"
    "train_geopt_20260905_020847.log",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def atomic_binary_dump(value: bytes | bytearray | memoryview, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-recipe", required=True)
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument(
        "--pair-bitmap-output",
        help=(
            "Optional packed pair-selection bitmap. Defaults to OUTPUT with "
            "the .pairs.bin suffix when --output is set."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-steps", type=int, default=200_000)
    parser.add_argument(
        "--usage",
        choices=("training", "validation"),
        default="training",
        help=(
            "Training replays shuffled epochs and advances the recipe epoch; "
            "validation replays the unshuffled evaluation loader at recipe epoch 0."
        ),
    )
    parser.add_argument("--max-abs-frame-offset", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="CPU processes used only for plan accounting; zero runs serially.",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--reference-log",
        action="append",
        default=None,
        help="Completed TimePT/GeoPT launch log used for fail-closed validation.",
    )
    parser.add_argument(
        "--use-default-reference-logs",
        action="store_true",
        help="Validate against this workspace's completed TimePT and GeoPT logs.",
    )
    parser.add_argument(
        "--expected-recipe-sha256",
        help="Optional expected lowercase recipe SHA-256.",
    )
    return parser.parse_args()


@dataclass(frozen=True, slots=True)
class PlanTrajectory:
    ordinal: int
    source_id: str
    trajectory_id: str
    first_frame_index: int
    frame_count: int
    observation_count: int
    observation_base: int
    stride: int
    shift: int


class RecipePlanIndex:
    """Metadata-only form of the recipe indexing used by NavAnywhereDataset."""

    def __init__(self, recipe: Mapping[str, Any], local_limit: int) -> None:
        self.recipe = dict(recipe)
        self.seed = int(recipe["seed"])
        self.context_size = int(recipe["context_size"])
        self.goals_per_obs = int(recipe["goals_per_obs"])
        self.dataset_length = int(recipe["samples_per_epoch"])
        self.min_frame_offset, self.max_frame_offset = map(
            int, recipe["frame_offset_range"]
        )
        self.local_limit = int(local_limit)
        self.pair_width = 2 * self.local_limit + 1
        if self.local_limit < 0:
            raise ValueError("max_abs_frame_offset must be non-negative")

        source_records: dict[str, list[PlanTrajectory]] = {}
        observation_base = 0
        trajectories: list[PlanTrajectory] = []
        for ordinal, item in enumerate(recipe["trajectories"]):
            source = str(item["source_id"])
            trajectory = str(item["trajectory_id"])
            first = int(item["first_frame_index"])
            last = int(item["last_frame_index"])
            frame_count = int(item["frame_count"])
            observation_count = int(item["observation_count"])
            if last - first + 1 != frame_count:
                raise ValueError(
                    "Plan-only replay requires the contiguous frame inventory "
                    "used by the completed Stage-1 runs: "
                    f"{source}/{trajectory}"
                )
            expected_observations = frame_count - self.context_size + 1
            if observation_count != expected_observations:
                raise ValueError(
                    "Recipe observation count does not match the contiguous "
                    f"Stage-1 contract for {source}/{trajectory}"
                )
            stride, shift = coprime_stride(
                observation_count, f"{source}/{trajectory}", self.seed
            )
            record = PlanTrajectory(
                ordinal=ordinal,
                source_id=source,
                trajectory_id=trajectory,
                first_frame_index=first,
                frame_count=frame_count,
                observation_count=observation_count,
                observation_base=observation_base,
                stride=stride,
                shift=shift,
            )
            trajectories.append(record)
            source_records.setdefault(source, []).append(record)
            observation_base += observation_count
        self.trajectories = tuple(trajectories)
        self.sources = tuple(
            tuple(source_records[source]) for source in sorted(source_records)
        )
        self.observation_count = observation_base
        if not self.sources:
            raise ValueError("Sampling recipe contains no sources")

    def locate(self, epoch: int, dataset_index: int) -> tuple[PlanTrajectory, int]:
        if dataset_index < 0 or dataset_index >= self.dataset_length:
            raise IndexError(dataset_index)
        source_count = len(self.sources)
        source_slot = int(dataset_index) % source_count
        source_draw = int(dataset_index) // source_count
        trajectories = self.sources[source_slot]
        trajectory_count = len(trajectories)
        trajectory_slot = source_draw % trajectory_count
        trajectory_draw = source_draw // trajectory_count
        record = trajectories[trajectory_slot]
        source_draws_per_epoch = (
            (self.dataset_length - 1 - source_slot) // source_count + 1
        )
        trajectory_draws_per_epoch = (
            (source_draws_per_epoch - 1 - trajectory_slot) // trajectory_count + 1
            if trajectory_slot < source_draws_per_epoch
            else 0
        )
        observation_slot = affine_observation_slot(
            epoch=epoch,
            trajectory_draw=trajectory_draw,
            trajectory_draws_per_epoch=trajectory_draws_per_epoch,
            observation_count=record.observation_count,
            stride=record.stride,
            shift=record.shift,
        )
        return record, observation_slot

    def sample_offsets(
        self, epoch: int, dataset_index: int, record: PlanTrajectory, observation_slot: int
    ) -> np.ndarray:
        current_position = self.context_size - 1 + int(observation_slot)
        low = max(self.min_frame_offset, -current_position)
        high = min(
            self.max_frame_offset, record.frame_count - 1 - current_position
        )
        available = np.arange(low, high + 1, dtype=np.int64)
        return sample_recipe_goal_offsets(
            available,
            seed=self.seed,
            epoch=epoch,
            dataset_index=dataset_index,
            goals_per_obs=self.goals_per_obs,
            strata=self.recipe["goal_sampling"]["strata"],
        )

    def pair_key(
        self, record: PlanTrajectory, observation_slot: int, offset: int
    ) -> int:
        return (
            (record.observation_base + int(observation_slot)) * self.pair_width
            + int(offset)
            + self.local_limit
        )

    def full_local_pair_count(self) -> int:
        total = 0
        for record in self.trajectories:
            positions = np.arange(
                self.context_size - 1, record.frame_count, dtype=np.int64
            )
            total += int(
                np.sum(
                    1
                    + np.minimum(self.local_limit, positions)
                    + np.minimum(
                        self.local_limit, record.frame_count - 1 - positions
                    ),
                    dtype=np.int64,
                )
            )
        return total


class _LengthOnlyDataset:
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length


def distributed_epoch_indices(
    dataset_length: int,
    *,
    seed: int,
    epoch: int,
    world_size: int,
    batch_size: int,
    steps: int,
    shuffle: bool = True,
) -> np.ndarray:
    """Return the global multiset consumed by synchronous DDP for an epoch.

    A one-replica ``DistributedSampler`` supplies the exact seeded permutation.
    The multi-replica sampler is a stride partition of that same permutation;
    consuming an equal number of full batches on every rank is therefore its
    prefix after standard sampler padding.  Tests compare this reconstruction
    directly with one real sampler instance per rank.
    """

    if min(dataset_length, world_size, batch_size) < 1 or steps < 0:
        raise ValueError("Invalid dataset, distributed, batch, or step setting")
    base_sampler = DistributedSampler(
        _LengthOnlyDataset(dataset_length),
        num_replicas=1,
        rank=0,
        shuffle=bool(shuffle),
        seed=int(seed),
        drop_last=False,
    )
    base_sampler.set_epoch(int(epoch))
    permutation = np.fromiter(
        iter(base_sampler), dtype=np.int64, count=len(base_sampler)
    )
    samples_per_rank = (dataset_length + world_size - 1) // world_size
    total_size = samples_per_rank * world_size
    if total_size > dataset_length:
        padding = total_size - dataset_length
        if padding <= dataset_length:
            permutation = np.concatenate((permutation, permutation[:padding]))
        else:
            repeats = (padding + dataset_length - 1) // dataset_length
            permutation = np.concatenate(
                (permutation, np.tile(permutation, repeats)[:padding])
            )
    requested = int(steps) * int(world_size) * int(batch_size)
    if requested > (samples_per_rank // batch_size) * batch_size * world_size:
        raise ValueError("Requested steps exceed full DataLoader batches in the epoch")
    return permutation[:requested]


def _chunks(values: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


_WORKER_PLAN_INDEX: RecipePlanIndex | None = None


def _process_chunk(task: tuple[int, np.ndarray]) -> dict[str, Any]:
    epoch, indices = task
    plan = _WORKER_PLAN_INDEX
    if plan is None:
        raise RuntimeError("Plan worker was not initialized")
    pair_keys: list[int] = []
    touched: set[int] = set()
    offset_counts = np.zeros(plan.pair_width, dtype=np.int64)
    absolute_offset_sum = 0
    for raw_index in indices:
        dataset_index = int(raw_index)
        record, observation_slot = plan.locate(epoch, dataset_index)
        touched.add(record.ordinal)
        offsets = plan.sample_offsets(
            epoch, dataset_index, record, observation_slot
        )
        for raw_offset in offsets:
            offset = int(raw_offset)
            if abs(offset) <= plan.local_limit:
                pair_keys.append(plan.pair_key(record, observation_slot, offset))
                offset_counts[offset + plan.local_limit] += 1
                absolute_offset_sum += abs(offset)
    return {
        "pair_keys": np.asarray(pair_keys, dtype=np.int64),
        "touched": np.fromiter(touched, dtype=np.int64, count=len(touched)),
        "offset_counts": offset_counts,
        "absolute_offset_sum": absolute_offset_sum,
    }


def _parse_field(text: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Reference log has no {label!r} field")
    return match.group(1)


def parse_reference_log(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Reference Stage-1 log does not exist: {resolved}")
    text = ANSI_ESCAPE.sub("", resolved.read_text(encoding="utf-8", errors="replace"))
    gpu_contract = _parse_field(text, "GPUs/processes")
    process_match = re.search(r"/\s*(\d+)\s*$", gpu_contract)
    if process_match is None:
        raise ValueError(f"Malformed GPUs/processes field in {resolved}")
    completed = re.findall(
        r"Two-stage training complete: stage=proxy_pretrain .*? total=(\d+)", text
    )
    if not completed:
        raise ValueError(f"Reference Stage-1 run did not complete: {resolved}")
    epoch_metrics: dict[int, dict[str, float]] = {}
    for epoch_text, payload_text in re.findall(
        r"Epoch\s+(\d+)\s+proxy metrics:\s*(\{[^\n]+\})", text
    ):
        payload = ast.literal_eval(payload_text)
        epoch = int(epoch_text)
        selected = {
            "total_samples": float(payload["proxy/total_samples"]),
            "eligible_samples": float(payload["proxy/eligible_samples"]),
            "mean_abs_offset_used": float(payload["proxy/mean_abs_offset_used"]),
        }
        if epoch in epoch_metrics and epoch_metrics[epoch] != selected:
            raise ValueError(f"Conflicting epoch {epoch} metrics in {resolved}")
        epoch_metrics[epoch] = selected
    return {
        "path": str(resolved),
        "mode": _parse_field(text, "mode"),
        "recipe_sha256": _parse_field(text, "recipe sha256"),
        "world_size": int(process_match.group(1)),
        "batch_size_per_rank": int(_parse_field(text, "batch/GPU")),
        "max_train_steps": int(_parse_field(text, "max steps")),
        "completed_train_steps": int(completed[-1]),
        "epoch_metrics": epoch_metrics,
    }


def validate_reference_contracts(
    references: Sequence[Mapping[str, Any]],
    *,
    recipe_sha256: str,
    seed: int,
    world_size: int,
    batch_size: int,
    max_train_steps: int,
) -> None:
    if not references:
        return
    modes = {str(reference["mode"]) for reference in references}
    if modes != {"timept", "geopt"}:
        raise ValueError(
            "Reference validation requires exactly completed timept and geopt "
            f"contracts; got {sorted(modes)}"
        )
    expected = {
        "recipe_sha256": recipe_sha256,
        "world_size": int(world_size),
        "batch_size_per_rank": int(batch_size),
        "max_train_steps": int(max_train_steps),
        "completed_train_steps": int(max_train_steps),
    }
    for reference in references:
        for key, value in expected.items():
            if reference[key] != value:
                raise ValueError(
                    f"Reference {reference['mode']} {key}={reference[key]!r}, "
                    f"plan expects {value!r}"
                )
    # The seed is carried by and validated against the immutable recipe.
    if seed < 0:
        raise ValueError("Training seed must be non-negative")


def _validate_epoch_references(
    epoch_record: Mapping[str, Any], references: Sequence[Mapping[str, Any]]
) -> None:
    epoch = int(epoch_record["epoch"])
    if not bool(epoch_record["complete_epoch"]):
        return
    local_draws = int(epoch_record["local_pair_draws"])
    goal_draws = int(epoch_record["goal_draws"])
    absolute_offset_sum = int(epoch_record["absolute_local_offset_sum"])
    reduction_error_bound = float(
        epoch_record["float32_reduction_abs_sum_error_bound"]
    )
    for reference in references:
        metrics = reference["epoch_metrics"].get(epoch)
        if metrics is None:
            raise ValueError(
                f"Reference {reference['mode']} has no completed epoch {epoch} metrics"
            )
        if int(metrics["eligible_samples"]) != local_draws:
            raise ValueError(
                f"Reference {reference['mode']} epoch {epoch} eligible count "
                f"{int(metrics['eligible_samples'])} != replay {local_draws}"
            )
        if int(metrics["total_samples"]) != goal_draws:
            raise ValueError(
                f"Reference {reference['mode']} epoch {epoch} total count "
                f"{int(metrics['total_samples'])} != replay {goal_draws}"
            )
        # TimePT has no used proxy and reports zero for this field. GeoPT uses
        # every eligible proxy, so its mean provides an additional value-level
        # check beyond counts. ProxyMetrics converts each rank's exact Python
        # counters to float32 before NCCL all_reduce; the reduction order is not
        # specified and can differ from a single float32 cast of the global
        # integer sum. Recover the logged numerator and admit only the standard
        # worst-case rounding envelope for `world_size` float32 inputs.
        if reference["mode"] == "geopt":
            reference_abs_sum = (
                float(metrics["mean_abs_offset_used"])
                * float(metrics["eligible_samples"])
            )
            if abs(reference_abs_sum - absolute_offset_sum) > reduction_error_bound:
                raise ValueError(
                    f"Reference geopt epoch {epoch} absolute local-offset sum "
                    f"{reference_abs_sum} differs from exact replay "
                    f"{absolute_offset_sum} beyond float32 all-reduce bound "
                    f"{reduction_error_bound}"
                )


def build_plan_report(
    recipe: Mapping[str, Any],
    *,
    recipe_path: str,
    recipe_sha256: str,
    seed: int,
    world_size: int,
    batch_size: int,
    max_train_steps: int,
    local_limit: int,
    workers: int,
    chunk_size: int,
    usage: str = "training",
    pair_bitmap_output: Path | None = None,
    references: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if int(seed) != int(recipe["seed"]):
        raise ValueError(
            f"Plan seed={seed} does not match recipe seed={recipe['seed']}"
        )
    if usage not in {"training", "validation"}:
        raise ValueError(f"Unknown pair-plan usage: {usage!r}")
    if references and usage != "training":
        raise ValueError("Reference training logs cannot validate a validation plan")
    if min(world_size, batch_size, max_train_steps, chunk_size) < 1 or workers < 0:
        raise ValueError("Invalid plan execution or training contract")
    plan = RecipePlanIndex(recipe, local_limit)
    samples_per_rank = (plan.dataset_length + world_size - 1) // world_size
    steps_per_epoch = samples_per_rank // batch_size
    if steps_per_epoch < 1:
        raise ValueError("Training DataLoader has no full distributed batch")

    validate_reference_contracts(
        references,
        recipe_sha256=recipe_sha256,
        seed=seed,
        world_size=world_size,
        batch_size=batch_size,
        max_train_steps=max_train_steps,
    )

    seen_pairs = np.zeros(plan.observation_count * plan.pair_width, dtype=np.bool_)
    touched_trajectories = np.zeros(len(plan.trajectories), dtype=np.bool_)
    per_epoch: list[dict[str, Any]] = []
    total_local_draws = 0
    total_goal_draws = 0
    total_abs_offset = 0
    total_offset_counts = np.zeros(plan.pair_width, dtype=np.int64)
    remaining_steps = max_train_steps
    epoch = 0
    started = time.perf_counter()

    global _WORKER_PLAN_INDEX
    _WORKER_PLAN_INDEX = plan
    executor: ProcessPoolExecutor | None = None
    if workers:
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("fork")
        )
    try:
        while remaining_steps:
            epoch_steps = min(remaining_steps, steps_per_epoch)
            recipe_epoch = epoch if usage == "training" else 0
            indices = distributed_epoch_indices(
                plan.dataset_length,
                seed=seed,
                epoch=epoch,
                world_size=world_size,
                batch_size=batch_size,
                steps=epoch_steps,
                shuffle=usage == "training",
            )
            tasks = ((recipe_epoch, chunk) for chunk in _chunks(indices, chunk_size))
            results: Iterable[dict[str, Any]]
            if executor is None:
                results = map(_process_chunk, tasks)
            else:
                results = executor.map(_process_chunk, tasks, chunksize=1)
            epoch_local_draws = 0
            epoch_abs_offset = 0
            epoch_offset_counts = np.zeros(plan.pair_width, dtype=np.int64)
            for result in results:
                keys = result["pair_keys"]
                seen_pairs[keys] = True
                touched_trajectories[result["touched"]] = True
                epoch_local_draws += int(len(keys))
                epoch_abs_offset += int(result["absolute_offset_sum"])
                epoch_offset_counts += result["offset_counts"]
            epoch_goal_draws = int(len(indices)) * plan.goals_per_obs
            training_float32_mean_abs = float(np.float32(epoch_abs_offset)) / float(
                np.float32(epoch_local_draws)
            )
            # One ULP per rank covers both initial float32 materialization and
            # any sequential/tree all-reduce association. This is deliberately
            # expressed in numerator units so the integer replay remains the
            # source of truth.
            reduction_error_bound = float(
                world_size * np.spacing(np.float32(epoch_abs_offset))
            )
            record = {
                "epoch": epoch,
                "steps": epoch_steps,
                "complete_epoch": epoch_steps == steps_per_epoch,
                "sample_draws": int(len(indices)),
                "goal_draws": epoch_goal_draws,
                "local_pair_draws": epoch_local_draws,
                "local_pair_rate": epoch_local_draws / epoch_goal_draws,
                "absolute_local_offset_sum": epoch_abs_offset,
                "mean_abs_local_offset": epoch_abs_offset / epoch_local_draws,
                "training_float32_mean_abs_local_offset": (
                    training_float32_mean_abs
                ),
                "float32_reduction_abs_sum_error_bound": reduction_error_bound,
                "local_offset_counts": {
                    str(offset): int(epoch_offset_counts[offset + local_limit])
                    for offset in range(-local_limit, local_limit + 1)
                },
            }
            _validate_epoch_references(record, references)
            per_epoch.append(record)
            total_local_draws += epoch_local_draws
            total_goal_draws += epoch_goal_draws
            total_abs_offset += epoch_abs_offset
            total_offset_counts += epoch_offset_counts
            remaining_steps -= epoch_steps
            print(
                f"[{utc_now()}] epoch={epoch} steps={epoch_steps} "
                f"local_draws={epoch_local_draws} "
                f"unique_so_far={int(np.count_nonzero(seen_pairs))} "
                f"reference={'matched' if references and record['complete_epoch'] else 'n/a'}",
                flush=True,
            )
            epoch += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        _WORKER_PLAN_INDEX = None

    unique_pairs = int(np.count_nonzero(seen_pairs))
    full_pairs = plan.full_local_pair_count()
    pair_bitmap = np.packbits(seen_pairs, bitorder="little")
    bitmap_sha256 = hashlib.sha256(pair_bitmap.tobytes()).hexdigest()
    bitmap_descriptor: dict[str, Any] | None = None
    if pair_bitmap_output is not None:
        resolved_bitmap = pair_bitmap_output.expanduser().resolve()
        atomic_binary_dump(pair_bitmap.tobytes(), resolved_bitmap)
        bitmap_descriptor = {
            "path": str(resolved_bitmap),
            "sha256": bitmap_sha256,
            "encoding": "numpy.packbits",
            "bitorder": "little",
            "pair_key_count": int(seen_pairs.size),
            "packed_byte_count": int(pair_bitmap.nbytes),
        }
    contract = {
        "usage": usage,
        "seed": int(seed),
        "world_size": int(world_size),
        "batch_size_per_rank": int(batch_size),
        "max_train_steps": int(max_train_steps),
        "dataset_length": plan.dataset_length,
        "steps_per_epoch": steps_per_epoch,
        "sampler": {
            "class": "torch.utils.data.DistributedSampler",
            "shuffle": usage == "training",
            "drop_last": False,
            "dataloader_drop_last": True,
            "set_epoch": True,
            "torch_version": torch.__version__,
        },
        "goal_sampling": recipe["goal_sampling"]["strategy"],
        "observation_sampling": recipe["observation_sampling"]["strategy"],
        "numpy_version": np.__version__,
        "max_abs_frame_offset": int(local_limit),
    }
    pair_identity = {
        "sampling_recipe_sha256": recipe_sha256,
        "inventory_sha256": recipe["inventory_sha256"],
        "training_contract_sha256": fingerprint(contract),
        "pair_bitmap_sha256": bitmap_sha256,
        "pair_key_domain": (
            "recipe_trajectory_order/observation_slot/"
            f"signed_offset[-{local_limit},{local_limit}]"
        ),
        "pair_key_count": int(seen_pairs.size),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "complete": True,
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "sampling_recipe": {
            "path": recipe_path,
            "sha256": recipe_sha256,
            "inventory_sha256": recipe["inventory_sha256"],
        },
        "training_contract": contract,
        "training_contract_sha256": pair_identity["training_contract_sha256"],
        "pair_identity": pair_identity,
        "pair_bitmap": bitmap_descriptor,
        "plan_sha256": fingerprint(pair_identity),
        "reference_validation": {
            "status": "matched" if references else "not_requested",
            "logs": [
                {
                    key: value
                    for key, value in reference.items()
                    if key != "epoch_metrics"
                }
                for reference in references
            ],
            "matched_complete_epochs": [
                item["epoch"] for item in per_epoch if item["complete_epoch"]
            ]
            if references
            else [],
        },
        "totals": {
            "planned_batches": int(max_train_steps),
            "train_steps": int(max_train_steps),
            "epochs_touched": len(per_epoch),
            "sample_draws": total_goal_draws // plan.goals_per_obs,
            "goal_draws": total_goal_draws,
            "local_pair_draws": total_local_draws,
            "local_pair_rate": total_local_draws / total_goal_draws,
            "mean_abs_local_offset": total_abs_offset / total_local_draws,
            "unique_local_pairs": unique_pairs,
            "duplicate_local_pair_draws": total_local_draws - unique_pairs,
            "full_local_pair_domain": full_pairs,
            "unique_fraction_of_full_domain": unique_pairs / full_pairs,
            "pair_reduction_vs_full_domain": 1.0 - unique_pairs / full_pairs,
            "trajectories_touched": int(np.count_nonzero(touched_trajectories)),
            "trajectories_total": len(plan.trajectories),
        },
        "local_offset_counts": {
            str(offset): int(total_offset_counts[offset + local_limit])
            for offset in range(-local_limit, local_limit + 1)
        },
        "epochs": per_epoch,
    }
    return report


def _print_summary(report: Mapping[str, Any]) -> None:
    totals = report["totals"]
    print("NavAnywhere latent-action plan-only summary", flush=True)
    print(f"  plan SHA-256:             {report['plan_sha256']}", flush=True)
    print(
        f"  reference validation:    {report['reference_validation']['status']}",
        flush=True,
    )
    print(f"  training steps:           {totals['train_steps']:,}", flush=True)
    print(f"  sampled observations:     {totals['sample_draws']:,}", flush=True)
    print(f"  sampled goals:            {totals['goal_draws']:,}", flush=True)
    print(f"  local pair draws:         {totals['local_pair_draws']:,}", flush=True)
    print(f"  unique local pairs:       {totals['unique_local_pairs']:,}", flush=True)
    print(f"  full local pair domain:   {totals['full_local_pair_domain']:,}", flush=True)
    print(
        "  extraction pair savings: "
        f"{totals['pair_reduction_vs_full_domain']:.2%}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    recipe, recipe_path, recipe_sha = load_sampling_recipe(args.sampling_recipe)
    if args.expected_recipe_sha256 is not None and (
        recipe_sha != str(args.expected_recipe_sha256).lower()
    ):
        raise ValueError(
            f"Recipe SHA-256 {recipe_sha} != expected {args.expected_recipe_sha256}"
        )
    reference_paths = list(args.reference_log or ())
    if args.use_default_reference_logs:
        reference_paths.extend(DEFAULT_REFERENCE_LOGS)
    if len(reference_paths) != len(set(reference_paths)):
        raise ValueError("Duplicate --reference-log paths")
    references = [parse_reference_log(path) for path in reference_paths]
    seed = int(recipe["seed"] if args.seed is None else args.seed)
    output = Path(args.output).expanduser().resolve() if args.output else None
    pair_bitmap_output = (
        Path(args.pair_bitmap_output).expanduser().resolve()
        if args.pair_bitmap_output
        else output.with_suffix(".pairs.bin")
        if output is not None
        else None
    )
    report = build_plan_report(
        recipe,
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha,
        seed=seed,
        world_size=int(args.world_size),
        batch_size=int(args.batch_size),
        max_train_steps=int(args.max_train_steps),
        local_limit=int(args.max_abs_frame_offset),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        usage=str(args.usage),
        pair_bitmap_output=pair_bitmap_output,
        references=references,
    )
    if output is not None:
        atomic_json_dump(report, output)
        print(f"  report:                    {output}", flush=True)
    _print_summary(report)


if __name__ == "__main__":
    main()
