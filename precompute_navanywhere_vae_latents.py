#!/usr/bin/env python3
"""Distributed SD-VAE posterior precompute for a NavAnywhere recipe."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from misc import get_transform
from navanywhere_recipe import (
    frame_indices_sha256,
    load_sampling_recipe,
    scan_trajectory_frames,
)


IMAGE_READ_ATTEMPTS = 3
IMAGE_READ_RETRY_DELAY_SECONDS = 0.25
from precompute_vae_latents import (
    DEFAULT_SCALING_FACTOR,
    FORMAT_NAME,
    SCHEMA_VERSION,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    encode_fixed_batch,
    load_dataset_config,
    load_image_tensor,
    load_vae,
    local_hardware_descriptor,
    make_encoding_descriptor,
    make_software_descriptor,
    make_transform_descriptor,
    make_vae_descriptor,
    sha256_bytes,
    sha256_file,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Precompute one BF16 SD-VAE posterior cache per trajectory in a "
            "frozen NavAnywhere sampling recipe."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sampling-recipe", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reuse-root", help="Completed VAE cache whose matching trajectories are imported without encoding")
    parser.add_argument(
        "--dataset-config", default=str(repo / "conf" / "dataset" / "navanywhere.yaml")
    )
    parser.add_argument("--vae-model-path", required=True)
    parser.add_argument("--vae-identifier", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--vae-batch-size", type=int, default=128)
    parser.add_argument("--loader-threads", type=int, default=8)
    parser.add_argument("--compute-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--storage-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--log-every-trajectories", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(rank: int, message: str) -> None:
    print(f"[{utc_now()}][rank {rank}] {message}", flush=True)


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=12))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NavAnywhere VAE precomputation")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _safe_path(root: Path, source_id: str, trajectory_id: str, suffix: str) -> Path:
    base = (root / source_id).resolve()
    path = (base / f"{trajectory_id}{suffix}").resolve()
    if path == base or base not in path.parents:
        raise ValueError(
            f"Unsafe NavAnywhere identity {source_id!r}/{trajectory_id!r}"
        )
    return path


def _source_fingerprint(frames: list[tuple[int, str]], trajectory_root: Path) -> str:
    records = []
    for frame_index, frame_path in frames:
        path = Path(frame_path)
        stat = path.stat()
        records.append(
            [
                int(frame_index),
                path.relative_to(trajectory_root).as_posix(),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            ]
        )
    return sha256_bytes(canonical_json(records))


def _assign(tasks: list[dict[str, Any]], world_size: int) -> None:
    loads = [0] * world_size
    for task in sorted(
        tasks,
        key=lambda item: (
            -int(item["frame_count"]),
            item["source_id"],
            item["trajectory_id"],
        ),
    ):
        target = min(range(world_size), key=lambda rank: (loads[rank], rank))
        task["assigned_rank"] = target
        loads[target] += int(task["frame_count"])


def _file_metadata(
    state: dict[str, Any], source_fingerprint: str
) -> dict[str, Any]:
    return {
        "vae_fingerprint": state["vae"]["fingerprint"],
        "transform_fingerprint": state["transform"]["fingerprint"],
        "encoding_fingerprint": state["encoding"]["fingerprint"],
        "sampling_recipe_sha256": state["sampling_recipe_sha256"],
        "storage_dtype": "bfloat16",
        "source_fingerprint": source_fingerprint,
    }


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("payload is not a mapping")
    return value


def _cache_record(
    state: dict[str, Any],
    task: dict[str, Any],
    frames: list[tuple[int, str]],
    source_fingerprint: str,
) -> dict[str, Any]:
    path = _safe_path(
        Path(state["output_root"]), task["source_id"], task["trajectory_id"], ".pt"
    )
    payload = _load_cache(path)
    expected_metadata = _file_metadata(state, source_fingerprint)
    required = {
        "schema_version",
        "format",
        "dataset_name",
        "trajectory_name",
        "frame_indices",
        "posterior_mean",
        "posterior_logvar",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"missing keys {sorted(missing)}")
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    if payload["format"] != FORMAT_NAME:
        raise ValueError("format mismatch")
    if payload["dataset_name"] != task["source_id"]:
        raise ValueError("source mismatch")
    if payload["trajectory_name"] != task["trajectory_id"]:
        raise ValueError("trajectory mismatch")
    indices = torch.as_tensor(payload["frame_indices"])
    expected_indices = torch.tensor([item[0] for item in frames], dtype=torch.int64)
    if indices.dtype != torch.int64 or not torch.equal(indices, expected_indices):
        raise ValueError("frame indices mismatch")
    mean = torch.as_tensor(payload["posterior_mean"])
    logvar = torch.as_tensor(payload["posterior_logvar"])
    if mean.dtype != torch.bfloat16 or logvar.dtype != torch.bfloat16:
        raise TypeError("posterior dtype is not bfloat16")
    if mean.ndim != 4 or mean.shape != logvar.shape:
        raise ValueError("posterior shapes differ or are not NCHW")
    expected_shape = (
        len(frames),
        4,
        int(state["transform"]["image_size"]) // 8,
        int(state["transform"]["image_size"]) // 8,
    )
    if tuple(mean.shape) != expected_shape:
        raise ValueError(f"posterior shape {tuple(mean.shape)} != {expected_shape}")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("file metadata is not a mapping")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"file metadata.{key} mismatch")
    substitutions = metadata.get("invalid_frame_substitutions", [])
    if not isinstance(substitutions, list):
        raise TypeError("file metadata.invalid_frame_substitutions is not a list")
    if not torch.isfinite(mean).all() or not torch.isfinite(logvar).all():
        raise ValueError("non-finite posterior statistics")
    record = {
        "source_id": task["source_id"],
        "trajectory_id": task["trajectory_id"],
        "frame_count": len(frames),
        "frame_indices_sha256": frame_indices_sha256(expected_indices.numpy()),
        "posterior_shape": list(mean.shape),
        "file_size_bytes": path.stat().st_size,
        "source_fingerprint": source_fingerprint,
    }
    if substitutions:
        record["invalid_frame_substitutions"] = substitutions
    return record


def _scan_task(state: dict[str, Any], task: dict[str, Any]) -> tuple[list[tuple[int, str]], str]:
    trajectory_root = _safe_path(
        Path(state["data_root"]), task["source_id"], task["trajectory_id"], ""
    )
    frames = scan_trajectory_frames(trajectory_root)
    indices = [item[0] for item in frames]
    if len(frames) != int(task["frame_count"]):
        raise ValueError(f"frame_count changed for {task['source_id']}/{task['trajectory_id']}")
    if frame_indices_sha256(indices) != task["frame_indices_sha256"]:
        raise ValueError(
            f"frame indices changed for {task['source_id']}/{task['trajectory_id']}"
        )
    return frames, _source_fingerprint(frames, trajectory_root)


def _load_frame_with_fallback(
    frames: list[tuple[int, str]], position: int, transform: Any
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    """Load one frame, deterministically replacing an unreadable image.

    The original path is retried to tolerate transient NAS reads.  A genuinely
    unreadable source frame is replaced by the nearest readable frame, preferring
    the previous frame on equal distance.  The caller persists this substitution
    in the trajectory cache metadata.
    """
    candidate_positions = [position]
    for distance in range(1, len(frames)):
        previous = position - distance
        following = position + distance
        if previous >= 0:
            candidate_positions.append(previous)
        if following < len(frames):
            candidate_positions.append(following)

    original_error: OSError | None = None
    attempted: list[str] = []
    for candidate_position in candidate_positions:
        candidate_index, candidate_path = frames[candidate_position]
        attempts = IMAGE_READ_ATTEMPTS if candidate_position == position else 1
        for attempt in range(attempts):
            try:
                image = load_image_tensor(Path(candidate_path), transform)
                if candidate_position == position:
                    return image, None
                original_index, original_path = frames[position]
                return image, {
                    "frame_index": int(original_index),
                    "frame_path": Path(original_path).name,
                    "replacement_frame_index": int(candidate_index),
                    "replacement_frame_path": Path(candidate_path).name,
                    "reason": (
                        f"{type(original_error).__name__}: {original_error}"
                        if original_error is not None
                        else "unreadable image"
                    ),
                }
            except OSError as exc:
                attempted.append(candidate_path)
                if candidate_position == position:
                    original_error = exc
                if attempt + 1 < attempts:
                    time.sleep(IMAGE_READ_RETRY_DELAY_SECONDS * (attempt + 1))

    original_index, original_path = frames[position]
    raise RuntimeError(
        f"No readable replacement for frame {original_index} ({original_path}); "
        f"attempted {len(attempted)} image reads"
    ) from original_error


def _compute_task(
    state: dict[str, Any],
    task: dict[str, Any],
    frames: list[tuple[int, str]],
    source_fingerprint: str,
    *,
    vae: torch.nn.Module,
    transform: Any,
    device: torch.device,
    loader_threads: int,
    rank: int,
) -> dict[str, Any]:
    means: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    substitutions: list[dict[str, Any]] = []
    batch_size = int(state["encoding"]["vae_batch_size"])
    pool = (
        ThreadPoolExecutor(max_workers=loader_threads)
        if loader_threads > 0
        else contextlib.nullcontext()
    )
    with pool as executor:
        for start in range(0, len(frames), batch_size):
            positions = list(range(start, min(start + batch_size, len(frames))))
            if executor is None:
                loaded = [
                    _load_frame_with_fallback(frames, position, transform)
                    for position in positions
                ]
            else:
                loaded = list(
                    executor.map(
                        lambda position: _load_frame_with_fallback(
                            frames, position, transform
                        ),
                        positions,
                    )
                )
            images = [item[0] for item in loaded]
            substitutions.extend(item[1] for item in loaded if item[1] is not None)
            mean, logvar = encode_fixed_batch(
                vae,
                images,
                batch_size,
                device,
                state["encoding"]["compute_dtype"],
                torch.bfloat16,
            )
            means.append(mean)
            logvars.append(logvar)
    mean = torch.cat(means, dim=0).contiguous()
    logvar = torch.cat(logvars, dim=0).contiguous()
    metadata = _file_metadata(state, source_fingerprint)
    if substitutions:
        substitutions.sort(key=lambda item: int(item["frame_index"]))
        metadata["invalid_frame_substitutions"] = substitutions
        for item in substitutions:
            log(
                rank,
                "replaced unreadable frame "
                f"{task['source_id']}/{task['trajectory_id']}/"
                f"{item['frame_path']} with {item['replacement_frame_path']}: "
                f"{item['reason']}",
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "dataset_name": task["source_id"],
        "trajectory_name": task["trajectory_id"],
        "frame_indices": torch.tensor(
            [item[0] for item in frames], dtype=torch.int64
        ),
        "posterior_mean": mean,
        "posterior_logvar": logvar,
        "metadata": metadata,
    }
    output = _safe_path(
        Path(state["output_root"]), task["source_id"], task["trajectory_id"], ".pt"
    )
    atomic_torch_save(output, payload)
    return _cache_record(state, task, frames, source_fingerprint)


def _build_state(
    args: argparse.Namespace,
    world_size: int,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    recipe, recipe_path, recipe_sha = load_sampling_recipe(args.sampling_recipe)
    os.environ["NWM_NAVANYWHERE_ROOT"] = str(data_root)
    config = load_dataset_config(Path(args.dataset_config).resolve(), data_root)
    if int(config["context_size"]) != int(recipe["context_size"]):
        raise ValueError("Dataset config context_size differs from sampling recipe")
    vae = make_vae_descriptor(args.vae_model_path, args.vae_identifier)
    transform = make_transform_descriptor(config)
    software = make_software_descriptor()
    encoding = make_encoding_descriptor(vae, transform, software, hardware, args)
    # This cache is indexed per exact frame, unlike the legacy trajectory
    # training-batch cache. Precompute batching affects numerical provenance,
    # but never constrains the later observation batch size.
    encoding["compatibility_target"] = {
        "cache_granularity": "per_frame_posterior_statistics",
        "training_batch_per_gpu": "independent",
        "training_context_and_goal_count": "independent",
    }
    encoding["fingerprint"] = sha256_bytes(
        canonical_json(
            {key: value for key, value in encoding.items() if key != "fingerprint"}
        )
    )
    tasks = [dict(item) for item in recipe["trajectories"]]
    if args.max_trajectories:
        tasks = tasks[: int(args.max_trajectories)]
    _assign(tasks, world_size)
    state = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "sampling_recipe_path": recipe_path,
        "sampling_recipe_sha256": recipe_sha,
        "recipe": recipe,
        "tasks": tasks,
        "vae": vae,
        "transform": transform,
        "software": software,
        "hardware": {**hardware, "homogeneous_world_size": world_size},
        "encoding": encoding,
    }
    reuse_root = getattr(args, "reuse_root", None)
    if reuse_root:
        root = Path(reuse_root).expanduser().resolve()
        if root == output_root:
            raise ValueError("Reuse root must differ from output root")
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        success = json.loads((root / "_SUCCESS.json").read_text())
        if (metadata.get("complete") is not True or success.get("complete") is not True
                or success.get("metadata_sha256") != sha256_file(metadata_path)):
            raise ValueError("Reuse VAE cache is incomplete or its metadata changed")
        _validate_reuse_descriptors(state, metadata)
        old_recipe, _, old_sha = load_sampling_recipe(metadata["sampling_recipe"]["path"])
        if old_sha != metadata["sampling_recipe"]["sha256"]:
            raise ValueError("Reuse recipe SHA mismatch")
        identities = {(item["source_id"], item["trajectory_id"]) for item in old_recipe["trajectories"]}
        for task in tasks:
            task["reuse"] = (task["source_id"], task["trajectory_id"]) in identities
        state["reuse"] = {
            "root": str(root), "metadata_sha256": sha256_file(metadata_path),
            "sampling_recipe_sha256": old_sha,
            **{key: metadata[key] for key in ("vae", "transform", "encoding", "hardware", "software")},
        }
    return state


def _validate_reuse_descriptors(state: dict[str, Any], metadata: dict[str, Any]) -> None:
    # Device/software provenance may differ; VAE weights, preprocessing, batching
    # and numeric dtypes must agree. Keep original provenance on imported shards.
    for key in ("vae", "transform"):
        if metadata[key]["fingerprint"] != state[key]["fingerprint"]:
            raise ValueError(f"Reuse {key} fingerprint mismatch")
    provenance = {"fingerprint", "hardware_fingerprint", "software_fingerprint"}
    for key in (set(metadata["encoding"]) | set(state["encoding"])) - provenance:
        if metadata["encoding"].get(key) != state["encoding"].get(key):
            raise ValueError(f"Reuse encoding.{key} mismatch")


def _reuse_task(state, task, frames, source_fingerprint):
    reuse = state["reuse"]
    old_state = {
        **state, "output_root": reuse["root"],
        **{key: reuse[key] for key in ("vae", "transform", "encoding", "sampling_recipe_sha256")},
    }
    # Fail on changed/missing old data instead of silently re-encoding it.
    _cache_record(old_state, task, frames, source_fingerprint)
    source = _safe_path(Path(reuse["root"]), task["source_id"], task["trajectory_id"], ".pt")
    payload = _load_cache(source)
    original = payload["metadata"]
    payload["metadata"] = {
        **original, **_file_metadata(state, source_fingerprint),
        "reused_from": {"path": str(source), "metadata": original,
                        "cache_metadata_sha256": reuse["metadata_sha256"]},
    }
    output = _safe_path(Path(state["output_root"]), task["source_id"], task["trajectory_id"], ".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(output, payload)
    return _cache_record(state, task, frames, source_fingerprint)


def _write_completion(
    state: dict[str, Any], all_records: list[dict[str, Any]], partial: bool
) -> None:
    output_root = Path(state["output_root"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(
        all_records, key=lambda item: (item["source_id"], item["trajectory_id"])
    ):
        grouped.setdefault(record["source_id"], []).append(record)
    sources: dict[str, Any] = {}
    for source_id, records in grouped.items():
        manifest = output_root / source_id / "manifest.jsonl"
        payload = b"".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            for record in records
        )
        atomic_write_bytes(manifest, payload)
        sources[source_id] = {
            "trajectories": len(records),
            "frames": sum(int(record["frame_count"]) for record in records),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
        }
    complete = not partial and len(all_records) == len(state["recipe"]["trajectories"])
    substitutions = [
        {
            "source_id": record["source_id"],
            "trajectory_id": record["trajectory_id"],
            **substitution,
        }
        for record in all_records
        for substitution in record.get("invalid_frame_substitutions", [])
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "dataset_kind": "navanywhere",
        "status": "complete" if complete else "partial",
        "complete": complete,
        "updated_at_utc": utc_now(),
        "sampling_recipe": {
            "path": state["sampling_recipe_path"],
            "sha256": state["sampling_recipe_sha256"],
            "inventory_sha256": state["recipe"]["inventory_sha256"],
        },
        "vae": state["vae"],
        "transform": state["transform"],
        "software": state["software"],
        "hardware": state["hardware"],
        "encoding": state["encoding"],
        "reused_cache": state.get("reuse"),
        "storage": {
            "dtype": "bfloat16",
            "layout": "NCHW",
            "tensor_keys": ["frame_indices", "posterior_mean", "posterior_logvar"],
            "file_pattern": "{source_id}/{trajectory_id}.pt",
        },
        "sources": sources,
        "totals": {
            "trajectories": len(all_records),
            "frames": sum(int(record["frame_count"]) for record in all_records),
            "invalid_frame_substitutions": len(substitutions),
        },
        "invalid_frame_substitutions": substitutions,
    }
    metadata_path = output_root / "metadata.json"
    atomic_write_json(metadata_path, metadata)
    success_path = output_root / "_SUCCESS.json"
    if complete:
        atomic_write_json(
            success_path,
            {
                "schema_version": SCHEMA_VERSION,
                "format": FORMAT_NAME,
                "complete": True,
                "completed_at_utc": utc_now(),
                "metadata_sha256": sha256_file(metadata_path),
                "encoding_fingerprint": state["encoding"]["fingerprint"],
                "sampling_recipe_sha256": state["sampling_recipe_sha256"],
                "totals": metadata["totals"],
            },
        )
    else:
        with contextlib.suppress(FileNotFoundError):
            success_path.unlink()


def main() -> None:
    args = parse_args()
    if args.vae_batch_size < 1 or args.loader_threads < 0:
        raise ValueError("Invalid VAE batch size or loader thread count")
    if args.max_trajectories < 0:
        raise ValueError("--max-trajectories cannot be negative")
    rank, world_size, local_rank = distributed_context()
    device = torch.device("cuda", local_rank)
    local_hardware = local_hardware_descriptor(local_rank)
    hardware_all: list[Any] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(hardware_all, local_hardware)
    else:
        hardware_all[0] = local_hardware
    if any(item != hardware_all[0] for item in hardware_all):
        raise RuntimeError("All VAE precompute ranks must use homogeneous GPUs")

    lock_handle = None
    state_message: list[Any] = [None]
    if rank == 0:
        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        lock_handle = (output_root / ".precompute.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            state_message[0] = {
                "ok": True,
                "state": _build_state(args, world_size, hardware_all[0]),
            }
        except Exception as exc:
            state_message[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if world_size > 1:
        dist.broadcast_object_list(state_message, src=0)
    if not state_message[0]["ok"]:
        raise RuntimeError(state_message[0]["error"])
    state = state_message[0]["state"]
    assigned = [task for task in state["tasks"] if int(task["assigned_rank"]) == rank]
    log(
        rank,
        f"assigned trajectories={len(assigned)}, frames="
        f"{sum(int(task['frame_count']) for task in assigned)}, "
        f"reuse_trajectories={sum(bool(task.get('reuse')) for task in assigned)}",
    )
    vae = None
    transform = get_transform(
        state["transform"]["image_size"],
        state["transform"]["mean"],
        state["transform"]["std"],
    )
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task_index, task in enumerate(assigned, 1):
        frames, source_fingerprint = _scan_task(state, task)
        output = _safe_path(
            Path(state["output_root"]), task["source_id"], task["trajectory_id"], ".pt"
        )
        record = None
        if output.is_file() and not args.overwrite:
            try:
                record = _cache_record(state, task, frames, source_fingerprint)
            except Exception as exc:
                log(rank, f"recomputing invalid cache {output}: {type(exc).__name__}: {exc}")
        if record is None:
            if task.get("reuse"):
                record = _reuse_task(state, task, frames, source_fingerprint)
        if record is None:
            if vae is None:
                vae = load_vae(state, device, rank)
            record = _compute_task(
                state,
                task,
                frames,
                source_fingerprint,
                vae=vae,
                transform=transform,
                device=device,
                loader_threads=args.loader_threads,
                rank=rank,
            )
        records.append(record)
        if task_index % max(1, args.log_every_trajectories) == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            frames_done = sum(int(item["frame_count"]) for item in records)
            log(
                rank,
                f"completed {task_index}/{len(assigned)} trajectories, "
                f"{frames_done / elapsed:.1f} frames/s",
            )

    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(records, gathered, dst=0)
    else:
        gathered = [records]
    if rank == 0:
        all_records = [record for rank_records in gathered for record in rank_records]
        _write_completion(
            state,
            all_records,
            partial=bool(args.max_trajectories),
        )
        log(
            rank,
            f"cache {'complete' if not args.max_trajectories else 'partial'}: "
            f"trajectories={len(all_records)}, output={state['output_root']}",
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if lock_handle is not None:
        lock_handle.close()


if __name__ == "__main__":
    main()
