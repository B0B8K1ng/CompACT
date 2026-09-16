from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DistributedSampler

from navanywhere_recipe import build_sampling_recipe, sha256_file, write_sampling_recipe
from plan_navanywhere_latent_actions import (
    RecipePlanIndex,
    _LengthOnlyDataset,
    _validate_epoch_references,
    build_plan_report,
    distributed_epoch_indices,
)
from two_stage_data import NavAnywhereDataset


def test_global_epoch_indices_match_real_distributed_samplers() -> None:
    dataset = _LengthOnlyDataset(37)
    world_size = 3
    batch_size = 4
    steps = 2
    actual_ranks = []
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=19,
        )
        sampler.set_epoch(2)
        actual_ranks.append(list(sampler)[: steps * batch_size])
    expected_interleaved = np.asarray(actual_ranks, dtype=np.int64).T.reshape(-1)
    replayed = distributed_epoch_indices(
        len(dataset),
        seed=19,
        epoch=2,
        world_size=world_size,
        batch_size=batch_size,
        steps=steps,
    )
    assert np.array_equal(replayed, expected_interleaved)


def test_validation_indices_match_unshuffled_distributed_sampler() -> None:
    dataset = _LengthOnlyDataset(41)
    actual_ranks = []
    for rank in range(3):
        sampler = DistributedSampler(
            dataset,
            num_replicas=3,
            rank=rank,
            shuffle=False,
            seed=19,
        )
        actual_ranks.append(list(sampler)[:4])
    expected_interleaved = np.asarray(actual_ranks, dtype=np.int64).T.reshape(-1)
    replayed = distributed_epoch_indices(
        len(dataset),
        seed=19,
        epoch=0,
        world_size=3,
        batch_size=4,
        steps=1,
        shuffle=False,
    )
    assert np.array_equal(replayed, expected_interleaved)


def test_geopt_reference_validation_allows_only_float32_reduction_error() -> None:
    local_draws = 8_934_207
    absolute_offset_sum = 35_536_926
    training_mean = float(np.float32(absolute_offset_sum)) / float(
        np.float32(local_draws)
    )
    exact_mean = absolute_offset_sum / local_draws
    assert training_mean != exact_mean
    epoch_record = {
        "epoch": 0,
        "complete_epoch": True,
        "local_pair_draws": local_draws,
        "goal_draws": 16_529_408,
        "absolute_local_offset_sum": absolute_offset_sum,
        "mean_abs_local_offset": exact_mean,
        "training_float32_mean_abs_local_offset": training_mean,
        "float32_reduction_abs_sum_error_bound": 32.0,
    }

    def reference(mean: float):
        return [
            {
                "mode": "geopt",
                "epoch_metrics": {
                    0: {
                        "eligible_samples": float(local_draws),
                        "total_samples": 16_529_408.0,
                        "mean_abs_offset_used": mean,
                    }
                },
            }
        ]

    _validate_epoch_references(epoch_record, reference(training_mean))
    with pytest.raises(ValueError, match="beyond float32 all-reduce bound"):
        _validate_epoch_references(
            epoch_record,
            reference(training_mean + 64.0 / local_draws),
        )


def _dataset_and_recipe(tmp_path: Path):
    root = tmp_path / "NavAnywhere"
    trajectory = root / "source" / "trajectory"
    trajectory.mkdir(parents=True)
    for frame_index in range(21):
        Image.new("RGB", (2, 2), color=(frame_index,) * 3).save(
            trajectory / f"{frame_index:06d}.jpg"
        )
    recipe = build_sampling_recipe(
        root,
        seed=23,
        context_size=4,
        goals_per_obs=4,
    )
    recipe_path = Path(write_sampling_recipe(tmp_path / "recipe.json", recipe))
    dataset = NavAnywhereDataset(
        root,
        source_ids=["source"],
        sampling_recipe=recipe_path,
        context_size=4,
        goals_per_obs=4,
        min_frame_offset=-64,
        max_frame_offset=64,
        action_mode="none",
        seed=23,
    )
    return dataset, recipe, recipe_path


def test_metadata_plan_reuses_dataset_observation_and_goal_sampling(
    tmp_path: Path,
) -> None:
    dataset, recipe, _ = _dataset_and_recipe(tmp_path)
    plan = RecipePlanIndex(recipe, local_limit=8)
    for epoch in (0, 1, 4):
        dataset.set_epoch(epoch)
        for dataset_index in range(len(dataset)):
            dataset_record, current_position = dataset._locate(dataset_index)
            dataset_offsets = dataset._sample_offsets(
                dataset_record, current_position, dataset_index
            )
            plan_record, observation_slot = plan.locate(epoch, dataset_index)
            plan_offsets = plan.sample_offsets(
                epoch, dataset_index, plan_record, observation_slot
            )
            assert plan_record.source_id == dataset_record.source_id
            assert plan_record.trajectory_id == dataset_record.trajectory_id
            assert observation_slot + dataset.context_size - 1 == current_position
            assert np.array_equal(plan_offsets, dataset_offsets)


def test_plan_report_unique_pairs_match_direct_dataset_replay(tmp_path: Path) -> None:
    dataset, recipe, recipe_path = _dataset_and_recipe(tmp_path)
    recipe_digest = sha256_file(recipe_path)
    steps = 7
    selected = distributed_epoch_indices(
        len(dataset),
        seed=23,
        epoch=0,
        world_size=1,
        batch_size=2,
        steps=steps,
    )
    expected_pairs = set()
    local_draws = 0
    for raw_index in selected:
        index = int(raw_index)
        record, current_position = dataset._locate(index)
        current_frame = int(record.frame_indices[current_position])
        for offset in dataset._sample_offsets(record, current_position, index):
            if abs(int(offset)) <= 8:
                local_draws += 1
                expected_pairs.add(
                    (
                        record.source_id,
                        record.trajectory_id,
                        current_frame,
                        current_frame + int(offset),
                    )
                )
    report = build_plan_report(
        recipe,
        recipe_path=str(recipe_path),
        recipe_sha256=recipe_digest,
        seed=23,
        world_size=1,
        batch_size=2,
        max_train_steps=steps,
        local_limit=8,
        workers=0,
        chunk_size=3,
    )
    assert report["totals"]["local_pair_draws"] == local_draws
    assert report["totals"]["unique_local_pairs"] == len(expected_pairs)
