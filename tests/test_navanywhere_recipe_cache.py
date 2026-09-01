"""Integrity tests for the frozen NavAnywhere recipe/cache contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from navanywhere_latent_cache import validate_navanywhere_latent_cache
from navanywhere_recipe import (
    build_sampling_recipe,
    canonical_json,
    sha256_file,
    validate_sampling_recipe,
    write_sampling_recipe,
)
import two_stage_training


def _fingerprinted(**values):
    values["fingerprint"] = hashlib.sha256(canonical_json(values)).hexdigest()
    return values


def _write_completed_cache(tmp_path: Path):
    data_root = tmp_path / "NavAnywhere"
    trajectory = data_root / "source_a" / "trajectory_a"
    trajectory.mkdir(parents=True)
    for frame_index in range(6):
        Image.new("RGB", (4, 4), color=(frame_index,) * 3).save(
            trajectory / f"{frame_index:06d}.jpg"
        )
    recipe = build_sampling_recipe(
        data_root,
        seed=23,
        context_size=1,
        goals_per_obs=4,
    )
    recipe_path = Path(
        write_sampling_recipe(tmp_path / "recipe.json", recipe)
    )
    recipe_sha = sha256_file(recipe_path)
    record = recipe["trajectories"][0]

    cache_root = tmp_path / "latents"
    source_root = cache_root / "source_a"
    source_root.mkdir(parents=True)
    latent_path = source_root / "trajectory_a.pt"
    torch.save({"fixture": True}, latent_path)
    manifest_record = {
        "source_id": "source_a",
        "trajectory_id": "trajectory_a",
        "frame_count": 6,
        "frame_indices_sha256": record["frame_indices_sha256"],
        "posterior_shape": [6, 4, 28, 28],
        "file_size_bytes": latent_path.stat().st_size,
    }
    manifest_path = source_root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(manifest_record, sort_keys=True) + "\n", encoding="utf-8"
    )

    vae = _fingerprinted(
        identifier="stabilityai/sd-vae-ft-ema",
        scaling_factor=0.18215,
        latent_channels=4,
    )
    transform = _fingerprinted(
        image_size=224,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    encoding = _fingerprinted(
        compute_dtype="bfloat16",
        storage_dtype="bfloat16",
        vae_batch_size=128,
        vae_fingerprint=vae["fingerprint"],
        transform_fingerprint=transform["fingerprint"],
    )
    metadata = {
        "schema_version": 1,
        "format": "sd_vae_posterior_stats",
        "dataset_kind": "navanywhere",
        "complete": True,
        "status": "complete",
        "sampling_recipe": {
            "sha256": recipe_sha,
            "inventory_sha256": recipe["inventory_sha256"],
        },
        "vae": vae,
        "transform": transform,
        "encoding": encoding,
        "storage": {
            "dtype": "bfloat16",
            "file_pattern": "{source_id}/{trajectory_id}.pt",
        },
        "sources": {
            "source_a": {"manifest_sha256": sha256_file(manifest_path)}
        },
    }
    metadata_path = cache_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    success_path = cache_root / "_SUCCESS.json"
    success_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "sd_vae_posterior_stats",
                "complete": True,
                "metadata_sha256": sha256_file(metadata_path),
                "encoding_fingerprint": encoding["fingerprint"],
                "sampling_recipe_sha256": recipe_sha,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "seed": 23,
            # Training batch is intentionally unrelated to precompute batch.
            "training": {"batch_size": 3},
            "dataset": {
                "context_size": 1,
                "image_size": 224,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "datasets": {"navanywhere": {"goals_per_obs": 4}},
                "precomputed_latents": {
                    "root": str(cache_root),
                    "vae_identifier": "stabilityai/sd-vae-ft-ema",
                    "scaling_factor": 0.18215,
                    "vae_encode_batch_size": 128,
                },
            },
        }
    )
    return config, recipe_path, success_path


def test_completed_cache_is_bound_to_recipe_but_not_training_batch(
    tmp_path: Path,
) -> None:
    config, recipe_path, _ = _write_completed_cache(tmp_path)
    result = validate_navanywhere_latent_cache(
        config,
        recipe_path=str(recipe_path),
        launch_dir=str(tmp_path),
    )
    assert result["recipe_sha256"] == sha256_file(recipe_path)
    assert set(result["manifest_records"]["source_a"]) == {"trajectory_a"}


def test_existing_recipe_rejects_a_different_requested_epoch_size(
    tmp_path: Path,
) -> None:
    _, recipe_path, _ = _write_completed_cache(tmp_path)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    try:
        validate_sampling_recipe(
            recipe, samples_per_epoch=int(recipe["samples_per_epoch"]) + 1
        )
    except ValueError as exc:
        assert "samples_per_epoch" in str(exc)
    else:
        raise AssertionError("recipe accepted a changed epoch size")


def test_cache_rejects_corrupt_completion_fingerprint(tmp_path: Path) -> None:
    config, recipe_path, success_path = _write_completed_cache(tmp_path)
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["metadata_sha256"] = "0" * 64
    success_path.write_text(json.dumps(success), encoding="utf-8")
    try:
        validate_navanywhere_latent_cache(
            config,
            recipe_path=str(recipe_path),
            launch_dir=str(tmp_path),
        )
    except ValueError as exc:
        assert "success.metadata_sha256" in str(exc)
    else:
        raise AssertionError("corrupt cache completion fingerprint was accepted")


def test_cached_posterior_mode_does_not_load_vae_weights(monkeypatch) -> None:
    config = OmegaConf.create(
        {
            "dataset": {
                "precomputed_latents": {
                    "enabled": True,
                    "scaling_factor": 0.18215,
                }
            }
        }
    )
    monkeypatch.setattr(
        two_stage_training,
        "setup_tokenizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("VAE weights must not load in cached mode")
        ),
    )

    class _Log:
        def info(self, *_args, **_kwargs):
            pass

    tokenizer = two_stage_training._setup_training_tokenizer(
        config, torch.device("cpu"), _Log()
    )
    assert tokenizer.scaling_factor == 0.18215
    try:
        tokenizer.encode(torch.zeros(1))
    except RuntimeError as exc:
        assert "Online VAE encoding is disabled" in str(exc)
    else:
        raise AssertionError("cached-only tokenizer unexpectedly encoded pixels")
