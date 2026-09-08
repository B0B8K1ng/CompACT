from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from precompute_navanywhere_nav1_latent_actions import (
    EXPECTED_CHECKPOINT_CLASS,
    atomic_torch_save,
    build_navanywhere_frame_pairs,
    build_planned_navanywhere_frame_pairs,
    encode_pair_batches,
    position_pairs,
    safe_path,
    validate_checkpoint_metadata,
)
from two_stage_data import OfflineProxyStore


def test_frame_pairs_use_real_ids_and_cover_strict_local_domain() -> None:
    indices = np.asarray([10, 11, 13, 14], dtype=np.int64)
    pairs = build_navanywhere_frame_pairs(
        indices, context_size=2, max_abs_frame_offset=2
    )
    assert pairs.tolist() == [
        [11, 10],
        [11, 11],
        [11, 13],
        [13, 11],
        [13, 13],
        [13, 14],
        [14, 13],
        [14, 14],
    ]
    assert position_pairs(indices, pairs).tolist() == [
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 2],
        [3, 3],
    ]


def test_frame_pair_edge_cases() -> None:
    assert build_navanywhere_frame_pairs([5, 7], context_size=3).shape == (0, 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_navanywhere_frame_pairs([5, 5, 6], context_size=2)
    with pytest.raises(ValueError, match="absent frame"):
        position_pairs(
            np.asarray([10, 12], dtype=np.int64),
            np.asarray([[10, 11]], dtype=np.int64),
        )


def test_planned_frame_pairs_decode_packed_global_pair_keys() -> None:
    indices = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    observation_base = 1
    pair_width = 5
    bits = np.zeros((observation_base + 4) * pair_width, dtype=np.bool_)
    for observation_slot, offset in ((0, -1), (0, 0), (2, 1), (3, -2)):
        bits[(observation_base + observation_slot) * pair_width + offset + 2] = True
    packed = np.packbits(bits, bitorder="little")
    pairs = build_planned_navanywhere_frame_pairs(
        indices,
        observation_base=observation_base,
        pair_bitmap=packed,
        context_size=2,
        max_abs_frame_offset=2,
    )
    assert pairs.tolist() == [[11, 10], [11, 11], [13, 14], [14, 12]]


def test_safe_path_rejects_identity_traversal(tmp_path: Path) -> None:
    assert safe_path(tmp_path, "source", "trajectory", ".pt") == (
        tmp_path / "source" / "trajectory.pt"
    ).resolve()
    with pytest.raises(ValueError, match="Unsafe identity"):
        safe_path(tmp_path, "../outside", "trajectory", ".pt")


class _FakeAdapter:
    device = torch.device("cpu")
    checkpoint_metadata = {"hparams": {"lam_latent_dim": 32}}

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        values = videos[:, 0, 0, 0, 0] + 10 * videos[:, 1, 0, 0, 0]
        return values[:, None, None, None].expand(-1, 1, 1, 32).contiguous()


def test_encode_pair_batches_preserves_current_to_goal_direction() -> None:
    frames = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
    motion = encode_pair_batches(
        _FakeAdapter(),
        frames,
        np.asarray([[2, 0], [1, 3]], dtype=np.int64),
        batch_size=1,
        precision="32",
    )
    assert motion.dtype == torch.float32
    assert motion.shape == (2, 32)
    assert torch.equal(motion[:, 0], torch.tensor([2.0, 31.0]))


def test_encode_pair_batches_compact_frame_bank_matches_full_bank() -> None:
    frames = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
    pairs = np.asarray([[2, 0], [2, 1], [3, 5], [4, 3]], dtype=np.int64)
    full = encode_pair_batches(
        _FakeAdapter(), frames, pairs, batch_size=2, precision="32"
    )
    compact = encode_pair_batches(
        _FakeAdapter(),
        frames,
        pairs,
        batch_size=2,
        precision="32",
        gpu_frame_bank_limit_bytes=0,
        pair_chunk_size=2,
    )
    assert torch.equal(compact, full)


def test_checkpoint_contract_rejects_non_pixel_action_model() -> None:
    metadata = {
        "class_path": EXPECTED_CHECKPOINT_CLASS,
        "global_step": 100000,
        "epoch": 3,
        "hparams": {"lam_latent_dim": 32, "lam_patch_size": 16},
        "datamodule_hparams": {
            "image_height": 240,
            "image_width": 320,
            "max_frame_offset": 8,
        },
    }
    validate_checkpoint_metadata(
        metadata, image_height=240, image_width=320, max_abs_frame_offset=8
    )
    metadata["class_path"] = "lam.model.LAM"
    with pytest.raises(ValueError, match="checkpoint class"):
        validate_checkpoint_metadata(
            metadata, image_height=240, image_width=320, max_abs_frame_offset=8
        )


def test_written_payload_is_offline_proxy_store_compatible(tmp_path: Path) -> None:
    path = tmp_path / "source" / "trajectory.pt"
    motion = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    atomic_torch_save(
        {
            "proxy_type": "latent",
            "source_id": "source",
            "trajectory_id": "trajectory",
            "frame_pairs": torch.tensor([[13, 11], [13, 14]], dtype=torch.int64),
            "motion": motion,
        },
        path,
    )
    store = OfflineProxyStore(
        root=tmp_path,
        proxy_type="latent",
        dim=32,
        strict_loading=True,
        max_abs_frame_offset=8,
    )
    result = store.lookup("source", "trajectory", 13, 14)
    assert result.valid
    assert torch.equal(result.proxy_action, motion[1])
