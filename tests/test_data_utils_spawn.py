"""Regression tests for CUDA-safe, exactly replayable DataLoader workers."""

from __future__ import annotations

import random
import unittest
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import data_utils
from two_stage_checkpoint import build_data_resume_fingerprint
from two_stage_training import _set_loader_epoch_seed


class _WorkerRandomDataset(Dataset):
    """Exercise every RNG seeded by PyTorch's DataLoader worker loop."""

    def __init__(self, length: int = 18) -> None:
        self.length = int(length)
        self.epoch = 0

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> torch.Tensor:
        worker = torch.utils.data.get_worker_info()
        return torch.tensor(
            [
                int(index),
                self.epoch,
                -1 if worker is None else worker.id,
                random.randrange(2**31),
                int(np.random.randint(0, 2**31)),
                int(torch.randint(0, 2**31, ()).item()),
            ],
            dtype=torch.int64,
        )


def _config(*, num_workers: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 1701,
            "training": {"batch_size": 2, "num_workers": num_workers},
            "motion_condition": {"enabled": False},
            "training_stage": "real_finetune",
            "action_mode": "real",
            "finetune": {"scheme": "reset"},
            "dataset": {"test_fixture": True},
            "dataset_selection": {"finetune": ["test_fixture"]},
            "proxy": {},
        }
    )


def _loader(*, num_workers: int = 2):
    dataset = _WorkerRandomDataset()
    config = _config(num_workers=num_workers)
    with mock.patch.object(data_utils.dist, "get_world_size", return_value=1):
        loader, sampler = data_utils.create_dataloader(
            dataset, config, rank=0, is_train=True
        )
    return config, dataset, loader, sampler


def _prepare_epoch(config, dataset, loader, sampler, epoch: int) -> None:
    sampler.set_epoch(epoch)
    dataset.set_epoch(epoch)
    _set_loader_epoch_seed(loader, int(config.seed), rank=0, epoch=epoch)


class SpawnDataLoaderTests(unittest.TestCase):
    def test_workers_use_spawn_without_changing_zero_worker_loaders(self) -> None:
        _, _, loader, _ = _loader(num_workers=2)
        self.assertIsNotNone(loader.multiprocessing_context)
        self.assertEqual(
            loader.multiprocessing_context.get_start_method(), "spawn"
        )
        self.assertFalse(loader.persistent_workers)
        self.assertTrue(loader.pin_memory)

        _, _, inline_loader, _ = _loader(num_workers=0)
        self.assertIsNone(inline_loader.multiprocessing_context)
        self.assertFalse(inline_loader.persistent_workers)

    def test_spawn_workers_replay_an_in_epoch_checkpoint_cursor_exactly(
        self,
    ) -> None:
        config, dataset, loader, sampler = _loader()
        epoch = 4
        _prepare_epoch(config, dataset, loader, sampler, epoch)
        uninterrupted = [batch.clone() for batch in loader]

        # Recreate the complete loader as a resumed process would, then replay
        # the saved cursor from the beginning of the same sampler epoch.
        resumed_config, resumed_dataset, resumed_loader, resumed_sampler = _loader()
        _prepare_epoch(
            resumed_config,
            resumed_dataset,
            resumed_loader,
            resumed_sampler,
            epoch,
        )
        cursor = 5
        resumed_iterator = iter(resumed_loader)
        for _ in range(cursor):
            next(resumed_iterator)
        resumed_remainder = [batch.clone() for batch in resumed_iterator]

        self.assertEqual(len(uninterrupted), len(resumed_remainder) + cursor)
        for expected, actual in zip(uninterrupted[cursor:], resumed_remainder):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_spawn_keeps_existing_checkpoint_fingerprint_compatible(self) -> None:
        config, _, loader, _ = _loader()
        fingerprint = build_data_resume_fingerprint(
            config,
            "warmup",
            dataset_length=18,
            loader_length=len(loader),
        )

        # The process start mechanism does not change sampler indices or seeded
        # worker streams, so old exact-resume fingerprints remain compatible.
        self.assertEqual(
            fingerprint["payload"]["loader_contract"],
            {
                "sampler": "DistributedSampler",
                "shuffle": True,
                "drop_last": True,
                "persistent_workers": False,
            },
        )
        self.assertEqual(
            loader.multiprocessing_context.get_start_method(), "spawn"
        )


if __name__ == "__main__":
    unittest.main()
