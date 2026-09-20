from __future__ import annotations

from pathlib import Path
import stat

import numpy as np
import pytest
import torch

from precompute_navanywhere_nav1_latent_actions import (
    EXPECTED_CHECKPOINT_CLASS,
    SUPPORTED_CHECKPOINT_CLASSES,
    atomic_json_dump,
    atomic_torch_save,
    build_navanywhere_frame_pairs,
    build_planned_navanywhere_frame_pairs,
    encode_dino_pair_batches,
    encode_pair_batches,
    encode_pair_batches_streaming,
    position_pairs,
    safe_path,
    validate_checkpoint_metadata,
)
from two_stage_data import OfflineProxyStore


def test_atomic_cache_outputs_are_readable_by_shared_group(tmp_path: Path) -> None:
    tensor_path = tmp_path / 'trajectory.pt'
    marker_path = tmp_path / 'metadata.json'
    atomic_torch_save({'motion': torch.zeros(1, 32)}, tensor_path)
    atomic_json_dump({'complete': True}, marker_path)
    for path in (tensor_path, marker_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert torch.load(tensor_path, weights_only=True)['motion'].shape == (1, 32)


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


def test_cache_resume_accepts_changed_extraction_batch_fingerprint(
    tmp_path: Path,
) -> None:
    import precompute_navanywhere_nav1_latent_actions as pre

    indices = np.arange(5, dtype=np.int64)
    pairs = pre.build_navanywhere_frame_pairs(
        indices, context_size=2, max_abs_frame_offset=1
    )
    task = {
        "source_id": "source",
        "trajectory_id": "trajectory",
        "frame_indices_sha256": "indices",
    }
    state = {
        "sampling_recipe_sha256": "recipe",
        "checkpoint": {"sha256": "checkpoint", "latent_dim": 32},
        "extraction": {"fingerprint": "batch-128"},
        "policy": {
            "fingerprint": "policy",
            "configuration": {"context_size": 2, "max_abs_frame_offset": 1},
        },
        "training_pair_plan": None,
        "legacy_full_policy_fingerprint": "legacy",
    }
    payload = {
        "schema_version": pre.SCHEMA_VERSION,
        "format": pre.FORMAT_NAME,
        "proxy_type": "latent",
        "source_id": "source",
        "trajectory_id": "trajectory",
        "frame_indices": torch.from_numpy(indices.copy()),
        "frame_pairs": torch.from_numpy(pairs.copy()),
        "motion": torch.zeros(len(pairs), 32),
        "metadata": {
            "sampling_recipe_sha256": "recipe",
            "frame_indices_sha256": "indices",
            "source_fingerprint": "source",
            "checkpoint_sha256": "checkpoint",
            "extraction_fingerprint": "batch-64",
            "policy_fingerprint": "policy",
        },
        "complete": True,
    }
    path = tmp_path / "trajectory.pt"
    pre.atomic_torch_save(payload, path)

    record = pre._cache_record(
        path,
        state,
        task,
        indices,
        "source",
        None,
        payload=payload,
    )

    assert record["pair_count"] == len(pairs)


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


def test_streaming_pair_batches_match_full_bank_and_preserve_order() -> None:
    frames = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1)
    records = [(index, f"{index}.jpg") for index in range(len(frames))]
    pairs = np.asarray(
        [[8, 1], [3, 8], [1, 2], [4, 3], [8, 0], [2, 5], [0, 7]],
        dtype=np.int64,
    )

    def load_frame(path, *, image_height, image_width):
        assert (image_height, image_width) == (1, 1)
        return frames[int(Path(path).stem)]

    expected = encode_pair_batches(
        _FakeAdapter(), frames, pairs, batch_size=2, precision="32"
    )
    actual, substitutions = encode_pair_batches_streaming(
        _FakeAdapter(),
        records,
        pairs,
        load_frame,
        batch_size=2,
        precision="32",
        image_height=1,
        image_width=1,
        loader_threads=2,
        pair_chunk_size=4,
    )
    assert substitutions == []
    assert torch.equal(actual, expected)


def test_dino_feature_once_encodes_each_referenced_frame_once() -> None:
    frame_values = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
    records = [(index, f"{index}.jpg") for index in range(len(frame_values))]
    pairs = np.asarray(
        [[2, 0], [2, 1], [3, 2], [2, 3], [3, 0], [2, 0]], dtype=np.int64
    )

    class FakeFeatureLAM:
        mu_record = None

        def encode(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            values = features[:, 0, 0, 0] + 10 * features[:, 1, 0, 0]
            z_mu = values[:, None].expand(-1, 32).contiguous()
            self.mu_record = z_mu
            return {"z_mu": z_mu}

    class FakeDINOModel:
        def __init__(self) -> None:
            self.lam = FakeFeatureLAM()
            self.seen: list[int] = []

        def extract_dino_features(self, videos: torch.Tensor) -> torch.Tensor:
            values = videos[:, :, 0, 0, 0]
            self.seen.extend(int(value) for value in values.reshape(-1).tolist())
            return values[:, :, None, None]

    class FakeDINOAdapter:
        device = torch.device("cpu")
        checkpoint_metadata = {"hparams": {"lam_latent_dim": 32}}

        def __init__(self) -> None:
            self.model = FakeDINOModel()

    def load_frame(path, *, image_height, image_width):
        assert (image_height, image_width) == (1, 1)
        return frame_values[int(Path(path).stem)]

    adapter = FakeDINOAdapter()
    actual, substitutions = encode_dino_pair_batches(
        adapter,
        records,
        pairs,
        load_frame,
        precision="32",
        image_height=1,
        image_width=1,
        loader_threads=2,
        frame_batch_size=2,
        lam_batch_size=3,
    )
    expected = encode_pair_batches(
        _FakeAdapter(), frame_values, pairs, batch_size=3, precision="32"
    )
    assert substitutions == []
    assert torch.equal(actual, expected)
    assert adapter.model.seen == [0, 1, 2, 3]
    assert adapter.model.lam.mu_record is None


def test_resumable_chunks_preserve_batch_membership_and_pair_order() -> None:
    class BatchSensitiveAdapter(_FakeAdapter):
        def encode(self, videos):
            return super().encode(videos) + videos[:, 0].sum()

    frames = torch.arange(9, dtype=torch.float32).reshape(9, 1, 1, 1)
    pairs = np.asarray([[8, 1], [3, 8], [1, 2], [4, 3], [8, 0], [2, 5], [0, 7]])
    full = encode_pair_batches(BatchSensitiveAdapter(), frames, pairs, batch_size=2, precision='32')
    chunks = []
    for start in range(0, len(pairs), 4):
        needed, inverse = np.unique(pairs[start:start+4], return_inverse=True)
        chunks.append(encode_pair_batches(BatchSensitiveAdapter(), frames[needed], inverse.reshape(-1, 2),
                                          batch_size=2, precision='32'))
    assert torch.equal(torch.cat(chunks), full)


def test_saved_chunk_rejects_corruption() -> None:
    from scripts.repair_navanywhere_latent_chunks import chunk_is_valid
    pairs = np.asarray([[1, 0], [2, 1]], dtype=np.int64)
    payload = dict(complete=True, start=0, end=2, metadata={'fingerprint': 'test'},
                   frame_pairs=torch.from_numpy(pairs.copy()), motion=torch.ones(2, 32))
    assert chunk_is_valid(payload, 0, 2, payload['metadata'], pairs)
    assert not chunk_is_valid(payload, 0, 2, {'fingerprint': 'changed'}, pairs)
    payload['motion'][0, 0] = float('nan')
    assert not chunk_is_valid(payload, 0, 2, payload['metadata'], pairs)
    payload['motion'] = torch.ones(2, 32)
    payload['frame_pairs'][0, 1] = 9
    assert not chunk_is_valid(payload, 0, 2, payload['metadata'], pairs)


def test_complete_saved_chunk_assembles_training_compatible_shard(tmp_path, monkeypatch) -> None:
    import json
    import sys
    import precompute_navanywhere_nav1_latent_actions as pre
    from scripts import repair_navanywhere_latent_chunks as repair
    indices = np.arange(6, dtype=np.int64)
    pairs = pre.build_navanywhere_frame_pairs(indices, context_size=4, max_abs_frame_offset=8)
    bits = np.zeros(3*17, dtype=np.uint8)
    for current, goal in pairs:
        bits[(current-3)*17+goal-current+8] = 1
    bitmap_path = tmp_path/'pairs.bin'
    bitmap_path.write_bytes(np.packbits(bits, bitorder='little').tobytes())
    task = dict(source_id='source', trajectory_id='trajectory', observation_base=0)
    state = dict(data_root=str(tmp_path), sampling_recipe_sha256='recipe',
                 checkpoint=dict(sha256='checkpoint', latent_dim=32),
                 policy=dict(fingerprint='policy', configuration=dict(context_size=4, max_abs_frame_offset=8)),
                 extraction=dict(fingerprint='extract', configuration=dict(batch_size=64)),
                 training_pair_plan=dict(pair_bitmap_path=str(bitmap_path)),
                 legacy_full_policy_fingerprint='legacy')
    task['frame_indices_sha256'] = 'indices'
    source_fp = 'source'
    metadata = pre._expected_metadata(state, task, source_fp)
    state_path, tasks_path = tmp_path/'state.json', tmp_path/'tasks.json'
    state_path.write_text(json.dumps(state))
    tasks_path.write_text(json.dumps([task]))
    chunk_root, staging = tmp_path/'chunks', tmp_path/'staging'
    motion = torch.arange(len(pairs)*32, dtype=torch.float32).reshape(-1, 32)
    pre.atomic_torch_save(dict(complete=True, start=0, end=len(pairs), metadata=metadata,
        motion=motion, frame_pairs=torch.from_numpy(pairs), invalid_frame_substitutions=[]),
        chunk_root/'source'/'trajectory'/f'{0:09d}_{len(pairs):09d}.pt')
    monkeypatch.setattr(pre, '_scan_task', lambda *args: ([], indices, source_fp))
    monkeypatch.setattr(repair, 'run_workers', lambda *args: pytest.fail('Completed chunks must not initialize GPUs'))
    monkeypatch.setattr(sys, 'argv', ['repair', '--state', str(state_path), '--tasks', str(tasks_path),
        '--staging-root', str(staging), '--chunk-root', str(chunk_root), '--devices', '0,1,2,3,4,5,6,7'])
    repair.main()
    store = OfflineProxyStore(root=staging, proxy_type='latent', dim=32, strict_loading=True, max_abs_frame_offset=8)
    result = store.lookup('source', 'trajectory', int(pairs[-1, 0]), int(pairs[-1, 1]))
    assert result.valid
    assert torch.equal(result.proxy_action, motion[-1])


def test_checkpoint_contract_accepts_pixel_models_and_rejects_other_models() -> None:
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
    for checkpoint_class in (
        "lam.navigation_variants.PixelLAM",
        "lam.navigation_variants.DINOFeatureLAM",
    ):
        metadata["class_path"] = checkpoint_class
        assert metadata["class_path"] in SUPPORTED_CHECKPOINT_CLASSES
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
