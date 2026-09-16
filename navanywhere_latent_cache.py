"""Strict completed-cache validation for NavAnywhere SD-VAE posteriors."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from typing import Any

import torch.distributed as dist

from navanywhere_recipe import load_sampling_recipe


SCHEMA_VERSION = 1
FORMAT_NAME = "sd_vae_posterior_stats"


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read NavAnywhere cache JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"NavAnywhere cache JSON must be an object: {path}")
    return value


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Mapping[str, Any], field: str) -> None:
    descriptor = dict(value)
    actual = descriptor.pop("fingerprint", None)
    expected = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual != expected:
        raise ValueError(f"NavAnywhere latent {field}.fingerprint is corrupt")


def _equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"NavAnywhere latent metadata mismatch for {field}: "
            f"{actual!r} != {expected!r}"
        )


def _resolve_path(value: Any, launch_dir: str) -> str:
    path = os.path.expanduser(os.fspath(value))
    if not os.path.isabs(path):
        path = os.path.join(launch_dir, path)
    return os.path.realpath(path)


def _safe_identity_component(value: str, field: str) -> str:
    value = str(value)
    if not value or value in {".", ".."} or os.path.basename(value) != value:
        raise ValueError(f"Unsafe NavAnywhere latent {field}: {value!r}")
    return value


def _validate_local(
    config: Any,
    *,
    recipe_path: str,
    launch_dir: str,
) -> dict[str, Any]:
    latent_config = config.dataset.precomputed_latents
    root = _resolve_path(latent_config.root, launch_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"NavAnywhere precomputed latent root does not exist: {root}"
        )
    metadata_path = os.path.join(root, "metadata.json")
    success_path = os.path.join(root, "_SUCCESS.json")
    if not os.path.isfile(metadata_path) or not os.path.isfile(success_path):
        raise FileNotFoundError(
            "NavAnywhere latent cache is incomplete; metadata.json and "
            f"_SUCCESS.json are both required under {root}"
        )
    metadata = _read_json(metadata_path)
    success = _read_json(success_path)
    for name, document in (("metadata", metadata), ("success", success)):
        _equal(document.get("schema_version"), SCHEMA_VERSION, f"{name}.schema_version")
        _equal(document.get("format"), FORMAT_NAME, f"{name}.format")
    _equal(metadata.get("dataset_kind"), "navanywhere", "metadata.dataset_kind")
    _equal(metadata.get("complete"), True, "metadata.complete")
    _equal(metadata.get("status"), "complete", "metadata.status")
    _equal(success.get("complete"), True, "success.complete")
    _equal(
        success.get("metadata_sha256"),
        _sha256_file(metadata_path),
        "success.metadata_sha256",
    )

    recipe, resolved_recipe, recipe_sha = load_sampling_recipe(
        recipe_path,
        seed=int(config.seed),
        context_size=int(config.dataset.context_size),
        goals_per_obs=int(config.dataset.datasets.navanywhere.goals_per_obs),
        frame_offset_range=(-64, 64),
    )
    recipe_meta = metadata.get("sampling_recipe")
    if not isinstance(recipe_meta, Mapping):
        raise TypeError("metadata.sampling_recipe must be an object")
    cache_recipe_sha = str(recipe_meta.get("sha256", ""))
    exact_recipe = cache_recipe_sha == recipe_sha
    allow_recipe_subset = bool(latent_config.get("allow_recipe_subset", False))
    if not exact_recipe and not allow_recipe_subset:
        _equal(cache_recipe_sha, recipe_sha, "sampling_recipe.sha256")
    if exact_recipe:
        _equal(
            recipe_meta.get("inventory_sha256"),
            recipe["inventory_sha256"],
            "sampling_recipe.inventory_sha256",
        )

    vae = metadata.get("vae")
    transform = metadata.get("transform")
    encoding = metadata.get("encoding")
    storage = metadata.get("storage")
    sources = metadata.get("sources")
    for name, section in (
        ("vae", vae),
        ("transform", transform),
        ("encoding", encoding),
        ("storage", storage),
        ("sources", sources),
    ):
        if not isinstance(section, Mapping):
            raise TypeError(f"metadata.{name} must be an object")
    _fingerprint(vae, "vae")
    _fingerprint(transform, "transform")
    _fingerprint(encoding, "encoding")
    _equal(vae.get("identifier"), str(latent_config.vae_identifier), "vae.identifier")
    _equal(
        float(vae.get("scaling_factor")),
        float(latent_config.scaling_factor),
        "vae.scaling_factor",
    )
    _equal(vae.get("latent_channels"), 4, "vae.latent_channels")
    _equal(transform.get("image_size"), int(config.dataset.image_size), "transform.image_size")
    _equal(list(transform.get("mean", [])), list(config.dataset.mean), "transform.mean")
    _equal(list(transform.get("std", [])), list(config.dataset.std), "transform.std")
    _equal(encoding.get("compute_dtype"), "bfloat16", "encoding.compute_dtype")
    _equal(encoding.get("storage_dtype"), "bfloat16", "encoding.storage_dtype")
    _equal(
        int(encoding.get("vae_batch_size")),
        int(latent_config.vae_encode_batch_size),
        "encoding.vae_batch_size",
    )
    _equal(
        encoding.get("vae_fingerprint"), vae["fingerprint"], "encoding.vae_fingerprint"
    )
    _equal(
        encoding.get("transform_fingerprint"),
        transform["fingerprint"],
        "encoding.transform_fingerprint",
    )
    _equal(storage.get("dtype"), "bfloat16", "storage.dtype")
    _equal(storage.get("file_pattern"), "{source_id}/{trajectory_id}.pt", "storage.file_pattern")

    records: dict[str, dict[str, dict[str, Any]]] = {}
    source_roots: dict[str, str] = {}
    source_file_sizes: dict[str, dict[str, int]] = {}
    expected_identities = {
        (str(item["source_id"]), str(item["trajectory_id"])): item
        for item in recipe["trajectories"]
    }
    for source_id in sorted({key[0] for key in expected_identities}):
        _safe_identity_component(source_id, "source_id")
        source_meta = sources.get(source_id)
        if not isinstance(source_meta, Mapping):
            raise KeyError(f"metadata.sources has no entry for {source_id!r}")
        # Resolve each source directory once. Calling realpath for every one of
        # ~68K trajectory files causes repeated NAS metadata walks through all
        # parent components and can add tens of minutes to launch validation.
        source_root = os.path.realpath(os.path.join(root, source_id))
        if os.path.commonpath((root, source_root)) != root or source_root == root:
            raise ValueError(f"Unsafe NavAnywhere latent source path: {source_root}")
        source_roots[source_id] = source_root
        manifest_path = os.path.join(source_root, "manifest.jsonl")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"NavAnywhere latent manifest is missing: {manifest_path}")
        _equal(
            source_meta.get("manifest_sha256"),
            _sha256_file(manifest_path),
            f"sources.{source_id}.manifest_sha256",
        )
        source_records: dict[str, dict[str, Any]] = {}
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Invalid record at {manifest_path}:{line_number}")
                trajectory_id = str(value.get("trajectory_id", ""))
                if not trajectory_id or trajectory_id in source_records:
                    raise ValueError(f"Duplicate/empty trajectory at {manifest_path}:{line_number}")
                source_records[trajectory_id] = value
        records[source_id] = source_records
        expected_source_trajectories = {
            trajectory
            for source, trajectory in expected_identities
            if source == source_id
        }
        file_sizes: dict[str, int] = {}
        if len(expected_source_trajectories) * 2 < len(source_records):
            # A tiny validation subset should not enumerate a source directory
            # containing tens of thousands of cache shards.
            for trajectory_id in expected_source_trajectories:
                file_path = os.path.join(source_root, f"{trajectory_id}.pt")
                try:
                    file_stat = os.stat(file_path, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ValueError(
                        f"NavAnywhere latent cache is not a regular file: {file_path}"
                    )
                file_sizes[trajectory_id] = file_stat.st_size
        else:
            # Readdir once for a nearly complete training subset; independent
            # path lookups for all 68K files are much slower on the shared NAS.
            with os.scandir(source_root) as entries:
                for entry in entries:
                    if not entry.name.endswith(".pt"):
                        continue
                    trajectory_id = entry.name[:-3]
                    if trajectory_id not in expected_source_trajectories:
                        continue
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise ValueError(
                            f"NavAnywhere latent cache is not a regular file: {entry.path}"
                        )
                    file_sizes[trajectory_id] = entry.stat(
                        follow_symlinks=False
                    ).st_size
        source_file_sizes[source_id] = file_sizes

    actual_identities = {
        (source, trajectory)
        for source, source_records in records.items()
        for trajectory in source_records
    }
    missing_identities = set(expected_identities) - actual_identities
    unexpected_identities = actual_identities - set(expected_identities)
    if missing_identities or (unexpected_identities and not allow_recipe_subset):
        raise ValueError(
            "NavAnywhere latent manifests do not cover the sampling recipe under "
            f"the configured policy: missing={sorted(missing_identities)[:5]}, "
            f"unexpected={sorted(unexpected_identities)[:5]}, "
            f"allow_recipe_subset={allow_recipe_subset}"
        )
    for identity, recipe_record in expected_identities.items():
        _safe_identity_component(identity[1], "trajectory_id")
        record = records[identity[0]][identity[1]]
        _equal(record.get("source_id"), identity[0], f"manifest.{identity}.source_id")
        _equal(record.get("trajectory_id"), identity[1], f"manifest.{identity}.trajectory_id")
        _equal(
            int(record.get("frame_count", -1)),
            int(recipe_record["frame_count"]),
            f"manifest.{identity}.frame_count",
        )
        _equal(
            record.get("frame_indices_sha256"),
            recipe_record["frame_indices_sha256"],
            f"manifest.{identity}.frame_indices_sha256",
        )
        file_path = os.path.join(
            source_roots[identity[0]], f"{identity[1]}.pt"
        )
        file_size = source_file_sizes[identity[0]].get(identity[1])
        if file_size is None:
            raise FileNotFoundError(f"NavAnywhere latent file is missing: {file_path}")
        if int(record.get("file_size_bytes", -1)) != file_size:
            raise ValueError(f"NavAnywhere latent file size changed: {file_path}")

    expected_file_metadata = {
        "vae_fingerprint": vae["fingerprint"],
        "transform_fingerprint": transform["fingerprint"],
        "encoding_fingerprint": encoding["fingerprint"],
        # Per-trajectory files remain bound to the immutable recipe that
        # created the cache. A requested train/validation subset has a new
        # recipe hash even though every selected frame posterior is identical.
        "sampling_recipe_sha256": cache_recipe_sha,
        "storage_dtype": "bfloat16",
    }
    _equal(
        success.get("encoding_fingerprint"),
        encoding["fingerprint"],
        "success.encoding_fingerprint",
    )
    _equal(
        success.get("sampling_recipe_sha256"),
        cache_recipe_sha,
        "success.sampling_recipe_sha256",
    )
    return {
        "root": root,
        "metadata": metadata,
        "manifest_records": records,
        "expected_file_metadata": expected_file_metadata,
        "recipe": recipe,
        "recipe_path": resolved_recipe,
        "recipe_sha256": recipe_sha,
        "cache_recipe_sha256": cache_recipe_sha,
        "recipe_relation": "exact" if exact_recipe else "cache_superset",
    }


def validate_navanywhere_latent_cache(
    config: Any,
    *,
    recipe_path: str,
    launch_dir: str,
) -> dict[str, Any]:
    """Validate once on rank zero and broadcast the immutable manifest."""
    if not dist.is_available() or not dist.is_initialized():
        return _validate_local(
            config, recipe_path=recipe_path, launch_dir=launch_dir
        )
    message: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            message[0] = {
                "ok": True,
                "result": _validate_local(
                    config, recipe_path=recipe_path, launch_dir=launch_dir
                ),
            }
        except Exception as exc:
            message[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    dist.broadcast_object_list(message, src=0)
    if not message[0]["ok"]:
        raise RuntimeError(
            "NavAnywhere precomputed latent validation failed: "
            + message[0]["error"]
        )
    return message[0]["result"]


__all__ = ["validate_navanywhere_latent_cache"]
