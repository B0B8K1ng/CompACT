import copy

import pytest
import torch

import precompute_navanywhere_vae_latents as module


def fixture_cache(tmp_path):
    old = {
        "output_root": str(tmp_path / "old"),
        "sampling_recipe_sha256": "old-recipe",
        "vae": {"fingerprint": "vae"},
        "transform": {"fingerprint": "transform", "image_size": 8},
        "encoding": {"fingerprint": "old-device", "compute_dtype": "bfloat16",
                     "storage_dtype": "bfloat16", "vae_batch_size": 128},
    }
    task = {"source_id": "source", "trajectory_id": "trajectory"}
    frames = [(0, "0.jpg"), (1, "1.jpg")]
    source = tmp_path / "old/source/trajectory.pt"
    source.parent.mkdir(parents=True)
    payload = {
        "schema_version": module.SCHEMA_VERSION, "format": module.FORMAT_NAME,
        "dataset_name": "source", "trajectory_name": "trajectory",
        "frame_indices": torch.tensor([0, 1]),
        "posterior_mean": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4, 1, 1),
        "posterior_logvar": torch.zeros(2, 4, 1, 1, dtype=torch.bfloat16),
        "metadata": module._file_metadata(old, "source-fingerprint"),
    }
    torch.save(payload, source)
    state = copy.deepcopy(old)
    state.update(output_root=str(tmp_path / "new"), sampling_recipe_sha256="new-recipe")
    state["encoding"]["fingerprint"] = "new-device"
    state["reuse"] = {**old, "root": old["output_root"], "metadata_sha256": "old-metadata"}
    return state, task, frames, source, payload


def test_reuse_preserves_tensors_original_file_and_provenance(tmp_path, monkeypatch):
    state, task, frames, source, payload = fixture_cache(tmp_path)
    before = source.read_bytes()
    monkeypatch.setattr(module, "load_vae", lambda *args: pytest.fail("VAE must not load"))
    record = module._reuse_task(state, task, frames, "source-fingerprint")
    result = module._load_cache(tmp_path / "new/source/trajectory.pt")
    for key in ("posterior_mean", "posterior_logvar", "frame_indices"):
        assert torch.equal(result[key], payload[key])
    assert source.read_bytes() == before
    assert result["metadata"]["sampling_recipe_sha256"] == "new-recipe"
    assert result["metadata"]["reused_from"]["metadata"] == payload["metadata"]
    assert record["frame_count"] == 2


def test_reuse_rejects_changed_source(tmp_path):
    state, task, frames, _, _ = fixture_cache(tmp_path)
    with pytest.raises(ValueError, match="source_fingerprint"):
        module._reuse_task(state, task, frames, "changed-source")
    assert not (tmp_path / "new/source/trajectory.pt").exists()


def test_reuse_rejects_changed_frame_inventory(tmp_path):
    state, task, _, _, _ = fixture_cache(tmp_path)
    with pytest.raises(ValueError, match="frame indices"):
        module._reuse_task(state, task, [(0, "0.jpg"), (2, "2.jpg")], "source-fingerprint")


def test_reuse_allows_device_change_but_rejects_numeric_change(tmp_path):
    state, _, _, _, _ = fixture_cache(tmp_path)
    module._validate_reuse_descriptors(state, state["reuse"])
    state["encoding"]["vae_batch_size"] = 256
    with pytest.raises(ValueError, match="vae_batch_size"):
        module._validate_reuse_descriptors(state, state["reuse"])
