#!/usr/bin/env python3
"""Precompute SD-VAE posterior statistics for CompACT navigation training.

The cache contains one atomic ``.pt`` file per trajectory rather than one file
per frame.  The default encoding path intentionally matches the paper-style
training launch (per-GPU observation batch 16): training flattens 16 samples x
8 images to a VAE batch of 128, uses CUDA BF16 autocast, and samples from the
posterior afterwards.  We therefore encode fixed batches of 128 and preserve
the posterior mean and clamped log-variance as BF16 tensors.  Training can draw
fresh posterior noise on every access without running the VAE encoder again.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import datetime as dt
import fcntl
import hashlib
import inspect
import json
import os
import pickle
import random
import stat as statlib
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import torch
import torch.distributed as dist
import diffusers
import torchvision
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from omegaconf import OmegaConf
from PIL import Image, __version__ as pillow_version

from misc import CenterCropAR, IMAGE_ASPECT_RATIO, get_transform


SCHEMA_VERSION = 1
FORMAT_NAME = "sd_vae_posterior_stats"
DEFAULT_SCALING_FACTOR = 0.18215
DEFAULT_DATA_ROOT = "/file_system/nas/algorithm/dujun.nie/nwm/data"
DEFAULT_OUTPUT_ROOT = (
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/"
    "vae_latents_sd_vae_ft_ema_224"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp.", suffix=f".{os.getpid()}"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o640)
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp.", suffix=f".{os.getpid()}"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fchmod(handle.fileno(), 0o640)
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def torch_load_mmap(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def log(rank: int, message: str) -> None:
    print(f"[{utc_now()}][rank {rank}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Precompute fixed-batch SD-VAE posterior mean/logvar for the train "
            "trajectories in conf/dataset/nwm.yaml."
        )
    )
    parser.add_argument("--data-root", default=os.environ.get("NWM_DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--output-root",
        default=os.environ.get("NWM_LATENT_ROOT", DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--dataset-config", default=str(repo_root / "conf" / "dataset" / "nwm.yaml")
    )
    parser.add_argument(
        "--vae-model-path",
        default=os.environ.get("VAE_MODEL_PATH", "stabilityai/sd-vae-ft-ema"),
    )
    parser.add_argument("--vae-identifier", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=128,
        help="Must match the flattened online VAE batch; paper-style 16x8 is 128.",
    )
    parser.add_argument("--loader-threads", type=int, default=4)
    parser.add_argument("--discovery-threads", type=int, default=32)
    parser.add_argument(
        "--storage-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="BF16 exactly preserves the default online BF16 posterior statistics.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="Limit newly checked/processed trajectories for a smoke test; 0 means all.",
    )
    parser.add_argument("--verify-samples", type=int, default=32)
    parser.add_argument("--verify-atol", type=float, default=0.02)
    parser.add_argument("--verify-rtol", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every-trajectories", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not write cache files; fully validate the existing cache and recompute samples.",
    )
    parser.add_argument(
        "--fail-on-missing-split-trajectories",
        action="store_true",
        help="Fail instead of recording train-split entries absent from the downloaded dataset.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.vae_batch_size < 1:
        raise ValueError("--vae-batch-size must be positive")
    if args.loader_threads < 0:
        raise ValueError("--loader-threads cannot be negative")
    if args.discovery_threads < 1:
        raise ValueError("--discovery-threads must be positive")
    if args.max_trajectories < 0:
        raise ValueError("--max-trajectories cannot be negative")
    if args.verify_samples < 1:
        raise ValueError("--verify-samples must be positive")


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SD-VAE precomputation and verification")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def acquire_root_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    handle = (output_root / ".precompute.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"another precompute process holds {output_root / '.precompute.lock'}"
        ) from exc
    return handle


def safe_trajectory_name(value: str) -> str:
    name = value.strip()
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe trajectory name in split manifest: {value!r}")
    return path.as_posix()


def trajectory_output_path(output_root: Path, dataset_name: str, trajectory_name: str) -> Path:
    dataset_root = (output_root / dataset_name).resolve()
    output = (dataset_root / PurePosixPath(trajectory_name)).with_suffix(".pt").resolve()
    if output != dataset_root and dataset_root not in output.parents:
        raise ValueError(f"trajectory output escapes dataset root: {trajectory_name!r}")
    return output


def load_position_length(traj_data_path: Path) -> int:
    try:
        with traj_data_path.open("rb") as handle:
            traj_data = pickle.load(handle)
        length = len(traj_data["position"])
    except Exception as exc:
        raise RuntimeError(f"cannot read trajectory length from {traj_data_path}: {exc}") from exc
    if length < 1:
        raise RuntimeError(f"trajectory has no positions: {traj_data_path}")
    return int(length)


def make_vae_descriptor(model_path: str, identifier: str) -> dict[str, Any]:
    local_path = Path(model_path).expanduser()
    if not local_path.is_dir():
        raise FileNotFoundError(
            f"VAE must resolve to a local snapshot directory, got: {model_path}"
        )
    local_path = local_path.resolve()
    config_path = local_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"VAE config.json is missing: {config_path}")
    weight_candidates = [
        local_path / "diffusion_pytorch_model.bin",
        local_path / "diffusion_pytorch_model.safetensors",
    ]
    weights_path = next((p for p in weight_candidates if p.is_file()), None)
    if weights_path is None:
        raise FileNotFoundError(f"VAE weights are missing below {local_path}")

    revision = None
    parts = local_path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            revision = parts[index + 1]

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_scaling = float(config.get("scaling_factor", DEFAULT_SCALING_FACTOR))
    if abs(model_scaling - DEFAULT_SCALING_FACTOR) > 1e-9:
        raise RuntimeError(
            f"VAE config scaling_factor={model_scaling} does not match "
            f"CompACT VAEWrapper={DEFAULT_SCALING_FACTOR}"
        )
    descriptor = {
        "identifier": identifier,
        "model_path": str(local_path),
        "snapshot_revision": revision,
        "config_sha256": sha256_file(config_path),
        "weights_filename": weights_path.name,
        "weights_size_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256_file(weights_path),
        "scaling_factor": DEFAULT_SCALING_FACTOR,
        "latent_channels": int(config.get("latent_channels", 4)),
    }
    descriptor["fingerprint"] = sha256_bytes(canonical_json(descriptor))
    return descriptor


def make_transform_descriptor(dataset_config: dict[str, Any]) -> dict[str, Any]:
    image_size = int(dataset_config["image_size"])
    mean = [float(x) for x in dataset_config["mean"]]
    std = [float(x) for x in dataset_config["std"]]
    source = inspect.getsource(CenterCropAR) + "\n" + inspect.getsource(get_transform)
    descriptor = {
        "implementation": "misc.get_transform",
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "center_crop_aspect_ratio": float(IMAGE_ASPECT_RATIO),
        "resize_size": [image_size, image_size],
        "resize_interpolation": "torchvision PIL bilinear default",
        "to_tensor": True,
        "normalize_inplace": True,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
    }
    descriptor["fingerprint"] = sha256_bytes(canonical_json(descriptor))
    return descriptor


def make_encoding_descriptor(
    vae: dict[str, Any],
    transform: dict[str, Any],
    software: dict[str, Any],
    hardware: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    descriptor = {
        "vae_fingerprint": vae["fingerprint"],
        "transform_fingerprint": transform["fingerprint"],
        "compute_dtype": args.compute_dtype,
        "storage_dtype": args.storage_dtype,
        "vae_batch_size": args.vae_batch_size,
        "posterior_fields": ["mean", "logvar"],
        "posterior_logvar_clamp": [-30.0, 20.0],
        "scaling_applied": False,
        "fixed_batch_padding": "repeat_last_frame_for_final_partial_batch",
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
        "compatibility_target": {
            "training_batch_per_gpu": 16,
            "context_images": 4,
            "goal_images": 4,
            "images_per_observation": 8,
            "flattened_vae_batch_per_gpu": 128,
            "bfloat16_autocast": True,
        },
        "software_fingerprint": software["fingerprint"],
        "hardware_fingerprint": hardware["fingerprint"],
    }
    descriptor["fingerprint"] = sha256_bytes(canonical_json(descriptor))
    return descriptor


def make_software_descriptor() -> dict[str, Any]:
    descriptor = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "diffusers": diffusers.__version__,
        "pillow": pillow_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    descriptor["fingerprint"] = sha256_bytes(canonical_json(descriptor))
    return descriptor


def local_hardware_descriptor(local_rank: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(local_rank)
    core = {
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }
    return {**core, "fingerprint": sha256_bytes(canonical_json(core))}


def load_dataset_config(path: Path, data_root: Path) -> dict[str, Any]:
    os.environ["NWM_DATA_ROOT"] = str(data_root)
    config = OmegaConf.load(path)
    # conf/dataset/nwm.yaml inherits pixel normalization from the selected
    # SD-VAE tokenizer config in Hydra.  Compose that small inherited section
    # explicitly so this standalone script uses the exact same transform.
    if config.get("mean") is None or config.get("std") is None:
        sdvae_config_path = Path(__file__).resolve().parent / "conf" / "model" / "tokenizer" / "sdvae.yaml"
        sdvae_config = OmegaConf.load(sdvae_config_path)
        config = OmegaConf.merge(sdvae_config.dataset, config)
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def assign_balanced_ranks(tasks: list[dict[str, Any]], world_size: int) -> None:
    totals = [0 for _ in range(world_size)]
    # Longest-processing-time scheduling balances frames better than list slicing.
    for task in sorted(
        tasks, key=lambda item: (-item["num_frames"], item["dataset_name"], item["trajectory_name"])
    ):
        target = min(range(world_size), key=lambda idx: (totals[idx], idx))
        task["assigned_rank"] = target
        totals[target] += task["num_frames"]


def build_state(
    args: argparse.Namespace, world_size: int, hardware: dict[str, Any]
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.dataset_config).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"NWM data root is not mounted: {data_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"dataset config is missing: {config_path}")

    config = load_dataset_config(config_path, data_root)
    vae = make_vae_descriptor(args.vae_model_path, args.vae_identifier)
    transform = make_transform_descriptor(config)
    software = make_software_descriptor()
    encoding = make_encoding_descriptor(vae, transform, software, hardware, args)
    tasks: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    datasets_meta: dict[str, dict[str, Any]] = {}
    active_dataset_names = [
        str(dataset_name)
        for dataset_name, dataset_cfg in config["datasets"].items()
        if bool(dataset_cfg.get("enabled", True)) and "train" in dataset_cfg
    ]

    for dataset_name, dataset_cfg in config["datasets"].items():
        if not bool(dataset_cfg.get("enabled", True)) or "train" not in dataset_cfg:
            continue
        split_dir = Path(dataset_cfg["train"])
        if not split_dir.is_absolute():
            split_dir = repo_root / split_dir
        split_manifest = split_dir.resolve() / "traj_names.txt"
        if not split_manifest.is_file():
            raise FileNotFoundError(f"train split manifest is missing: {split_manifest}")
        names = [
            safe_trajectory_name(line)
            for line in split_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate trajectories in {split_manifest}")

        configured_data_folder = Path(dataset_cfg["data_folder"]).expanduser().resolve()
        dataset_missing: list[str] = []
        existing_count = 0
        frame_count = 0
        def inspect_trajectory(trajectory_name: str):
            trajectory_dir = configured_data_folder / PurePosixPath(trajectory_name)
            traj_data = trajectory_dir / "traj_data.pkl"
            if not trajectory_dir.is_dir() or not traj_data.is_file():
                return trajectory_name, trajectory_dir, traj_data, None
            num_frames = load_position_length(traj_data)
            return trajectory_name, trajectory_dir, traj_data, num_frames

        with ThreadPoolExecutor(max_workers=args.discovery_threads) as executor:
            inspected = list(executor.map(inspect_trajectory, names))
        for trajectory_name, trajectory_dir, traj_data, num_frames in inspected:
            if num_frames is None:
                dataset_missing.append(trajectory_name)
                continue
            task = {
                "dataset_name": dataset_name,
                "trajectory_name": trajectory_name,
                "trajectory_dir": str(trajectory_dir),
                "traj_data_path": str(traj_data),
                "num_frames": num_frames,
                "output_path": str(
                    trajectory_output_path(output_root, dataset_name, trajectory_name)
                ),
            }
            tasks.append(task)
            existing_count += 1
            frame_count += num_frames

        missing[dataset_name] = dataset_missing
        datasets_meta[dataset_name] = {
            "data_folder": str(configured_data_folder),
            "split": "train",
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": sha256_file(split_manifest),
            "split_entries": len(names),
            "existing_trajectories": existing_count,
            "missing_trajectories": len(dataset_missing),
            "missing_trajectory_names": dataset_missing,
            "expected_frames": frame_count,
        }

    tasks.sort(key=lambda item: (item["dataset_name"], item["trajectory_name"]))
    assign_balanced_ranks(tasks, world_size)
    selected_keys: set[tuple[str, str]]
    if args.max_trajectories:
        # A limited smoke test must exercise every dataset rather than taking a
        # lexicographic RECON-only prefix.  Round-robin is deterministic and
        # guarantees one trajectory per dataset when the budget permits.
        tasks_by_dataset = {
            dataset_name: [
                task for task in tasks if task["dataset_name"] == dataset_name
            ]
            for dataset_name in active_dataset_names
        }
        selected: list[dict[str, Any]] = []
        offset = 0
        limit = min(args.max_trajectories, len(tasks))
        while len(selected) < limit:
            added = False
            for dataset_tasks in tasks_by_dataset.values():
                if offset < len(dataset_tasks) and len(selected) < limit:
                    selected.append(dataset_tasks[offset])
                    added = True
            if not added:
                break
            offset += 1
        selected_keys = {
            (task["dataset_name"], task["trajectory_name"]) for task in selected
        }
    else:
        selected_keys = {
            (task["dataset_name"], task["trajectory_name"]) for task in tasks
        }
    for task in tasks:
        task["selected"] = (task["dataset_name"], task["trajectory_name"]) in selected_keys

    if args.fail_on_missing_split_trajectories and any(missing.values()):
        counts = {name: len(values) for name, values in missing.items() if values}
        raise RuntimeError(f"train split contains missing trajectories: {counts}")

    old_metadata_path = output_root / "metadata.json"
    if old_metadata_path.is_file():
        old = json.loads(old_metadata_path.read_text(encoding="utf-8"))
        old_encoding = old.get("encoding", {}).get("fingerprint")
        if old_encoding and old_encoding != encoding["fingerprint"]:
            raise RuntimeError(
                "output root contains an incompatible cache: "
                f"old encoding={old_encoding}, requested={encoding['fingerprint']}. "
                "Use a different NWM_LATENT_ROOT."
            )

    return {
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "dataset_config_path": str(config_path),
        "dataset_config_sha256": sha256_file(config_path),
        "dataset_config": config,
        "vae": vae,
        "transform": transform,
        "software": software,
        "hardware": hardware,
        "encoding": encoding,
        "tasks": tasks,
        "datasets_meta": datasets_meta,
        "missing": missing,
    }


def broadcast_state(state: dict[str, Any] | None, rank: int) -> dict[str, Any]:
    if not dist.is_initialized():
        assert state is not None
        return state
    values: list[Any] = [state]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], dict):
        raise RuntimeError("rank 0 failed to broadcast precompute state")
    return values[0]


def source_frame_paths(task: dict[str, Any]) -> list[Path]:
    trajectory_dir = Path(task["trajectory_dir"])
    # traj_data['position'] is authoritative.  Do not glob: some NAS directory
    # listings return duplicate entries for two known trajectories.
    paths = [trajectory_dir / f"{index}.jpg" for index in range(task["num_frames"])]
    digest = hashlib.sha256()
    source_entries: list[tuple[str, int, int]] = []
    traj_data_path = Path(task["traj_data_path"])
    traj_stat = traj_data_path.stat()
    source_entries.append(("traj_data.pkl", traj_stat.st_size, traj_stat.st_mtime_ns))
    missing: list[str] = []
    for index, path in enumerate(paths):
        try:
            file_stat = path.stat()
            if not statlib.S_ISREG(file_stat.st_mode):
                missing.append(str(path))
                continue
            source_entries.append((f"{index}.jpg", file_stat.st_size, file_stat.st_mtime_ns))
        except FileNotFoundError:
            missing.append(str(path))
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)}/{len(paths)} expected frames are missing for "
            f"{task['dataset_name']}/{task['trajectory_name']}: {preview}"
        )
    for entry in source_entries:
        digest.update(canonical_json(entry))
        digest.update(b"\n")
    task["source_fingerprint"] = digest.hexdigest()
    return paths


def expected_file_metadata(state: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "vae_fingerprint": state["vae"]["fingerprint"],
        "transform_fingerprint": state["transform"]["fingerprint"],
        "encoding_fingerprint": state["encoding"]["fingerprint"],
        "hardware_fingerprint": state["hardware"]["fingerprint"],
        "scaling_factor": DEFAULT_SCALING_FACTOR,
        "image_size": state["transform"]["image_size"],
        "storage_dtype": state["encoding"]["storage_dtype"],
        "compute_dtype": state["encoding"]["compute_dtype"],
        "vae_batch_size": state["encoding"]["vae_batch_size"],
        "num_frames": task["num_frames"],
        "source_fingerprint": task["source_fingerprint"],
        "source_fingerprint_method": "sha256(filename,size,mtime_ns);traj_data+0..L-1.jpg",
    }


def storage_torch_dtype(state: dict[str, Any]) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[state["encoding"]["storage_dtype"]]


def cache_validation_error(
    state: dict[str, Any], task: dict[str, Any], check_finite: bool
) -> str | None:
    path = Path(task["output_path"])
    if not path.is_file():
        return "file is missing"
    try:
        cached = torch_load_mmap(path)
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
        missing_keys = required.difference(cached)
        if missing_keys:
            return f"missing keys: {sorted(missing_keys)}"
        if cached["schema_version"] != SCHEMA_VERSION or cached["format"] != FORMAT_NAME:
            return "schema/format mismatch"
        if cached["dataset_name"] != task["dataset_name"]:
            return "dataset_name mismatch"
        if cached["trajectory_name"] != task["trajectory_name"]:
            return "trajectory_name mismatch"

        frame_indices = cached["frame_indices"]
        expected_frames = task["num_frames"]
        if not isinstance(frame_indices, torch.Tensor) or frame_indices.dtype != torch.int64:
            return "frame_indices must be an int64 tensor"
        if frame_indices.shape != (expected_frames,):
            return f"frame_indices shape is {tuple(frame_indices.shape)}, expected {(expected_frames,)}"
        if not torch.equal(frame_indices, torch.arange(expected_frames, dtype=torch.int64)):
            return "frame_indices are not the authoritative contiguous position range"

        latent_hw = state["transform"]["image_size"] // 8
        expected_shape = (expected_frames, state["vae"]["latent_channels"], latent_hw, latent_hw)
        expected_dtype = storage_torch_dtype(state)
        for key in ("posterior_mean", "posterior_logvar"):
            tensor = cached[key]
            if not isinstance(tensor, torch.Tensor):
                return f"{key} is not a tensor"
            if tuple(tensor.shape) != expected_shape:
                return f"{key} shape is {tuple(tensor.shape)}, expected {expected_shape}"
            if tensor.dtype != expected_dtype:
                return f"{key} dtype is {tensor.dtype}, expected {expected_dtype}"
            if check_finite:
                for chunk in tensor.split(256, dim=0):
                    if not bool(torch.isfinite(chunk).all()):
                        return f"{key} contains NaN or Inf"

        expected_metadata = expected_file_metadata(state, task)
        for key, value in expected_metadata.items():
            if cached["metadata"].get(key) != value:
                return f"metadata.{key} mismatch"
        return None
    except Exception as exc:
        return f"cannot load/validate: {type(exc).__name__}: {exc}"


def load_image_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def make_autocast_context(device: torch.device, compute_dtype: str):
    if compute_dtype == "float32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def load_vae(state: dict[str, Any], device: torch.device, rank: int) -> AutoencoderKL:
    log(rank, f"loading SD-VAE from {state['vae']['model_path']}")
    use_safetensors = str(state["vae"].get("weights_filename", "")).endswith(
        ".safetensors"
    )
    vae = AutoencoderKL.from_pretrained(
        state["vae"]["model_path"], use_safetensors=use_safetensors
    ).eval()
    vae.requires_grad_(False)
    vae.to(device)
    config_scaling = float(getattr(vae.config, "scaling_factor", DEFAULT_SCALING_FACTOR))
    if abs(config_scaling - DEFAULT_SCALING_FACTOR) > 1e-9:
        raise RuntimeError(
            f"loaded VAE scaling_factor={config_scaling}, expected={DEFAULT_SCALING_FACTOR}"
        )
    return vae


def encode_fixed_batch(
    vae: AutoencoderKL,
    image_tensors: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
    compute_dtype: str,
    storage_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = len(image_tensors)
    if valid < 1 or valid > batch_size:
        raise ValueError(f"invalid VAE batch occupancy: {valid}/{batch_size}")
    images = torch.stack(image_tensors, dim=0)
    if valid < batch_size:
        # Fixed shape is intentional: BF16 convolution kernels can give slightly
        # different posterior statistics for different batch shapes.
        images = torch.cat(
            [images, images[-1:].expand(batch_size - valid, -1, -1, -1)], dim=0
        )
    images = images.to(device, non_blocking=True)
    with torch.inference_mode(), make_autocast_context(device, compute_dtype):
        posterior = vae.encode(images).latent_dist
        mean = posterior.mean[:valid]
        # DiagonalGaussianDistribution already clamps to [-30, 20].  Clamp
        # explicitly as a version-independent guarantee for the cache contract.
        logvar = posterior.logvar[:valid].clamp(-30.0, 20.0)
    mean = mean.to(dtype=storage_dtype, device="cpu")
    logvar = logvar.to(dtype=storage_dtype, device="cpu")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(logvar).all()):
        raise RuntimeError("VAE produced non-finite posterior statistics")
    return mean, logvar


def build_cache_payload(
    state: dict[str, Any], task: dict[str, Any], mean: torch.Tensor, logvar: torch.Tensor
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "dataset_name": task["dataset_name"],
        "trajectory_name": task["trajectory_name"],
        "frame_indices": torch.arange(task["num_frames"], dtype=torch.int64),
        "posterior_mean": mean.contiguous(),
        "posterior_logvar": logvar.contiguous(),
        "metadata": expected_file_metadata(state, task),
    }


def classify_rank_tasks(
    state: dict[str, Any], args: argparse.Namespace, rank: int
) -> tuple[list[tuple[dict[str, Any], list[Path]]], int, int, dict[str, str]]:
    assigned = [
        task
        for task in state["tasks"]
        if task["assigned_rank"] == rank and task["selected"]
    ]
    pending: list[tuple[dict[str, Any], list[Path]]] = []
    skipped = 0
    checked_frames = 0
    source_fingerprints: dict[str, str] = {}
    world_size = int(state["hardware"]["homogeneous_world_size"])
    # NAS metadata lookup dominates this pass. Preserve the requested discovery
    # concurrency globally instead of multiplying it by the distributed ranks.
    source_workers = max(1, min(len(assigned), args.discovery_threads // world_size))
    log(rank, f"source fingerprint scan workers={source_workers}")
    with ThreadPoolExecutor(max_workers=source_workers) as executor:
        scanned = executor.map(source_frame_paths, assigned)
        for index, (task, frame_paths) in enumerate(zip(assigned, scanned), start=1):
            source_fingerprints[f"{task['dataset_name']}\0{task['trajectory_name']}"] = task[
                "source_fingerprint"
            ]
            checked_frames += len(frame_paths)
            error = "overwrite requested" if args.overwrite else cache_validation_error(
                state, task, check_finite=False
            )
            if error is None:
                skipped += 1
            else:
                pending.append((task, frame_paths))
            if index % max(1, args.log_every_trajectories * 4) == 0:
                log(
                    rank,
                    f"classified {index}/{len(assigned)} trajectories "
                    f"(cached={skipped}, to_compute={len(pending)})",
                )
    return pending, skipped, checked_frames, source_fingerprints


def precompute_rank(
    state: dict[str, Any], args: argparse.Namespace, rank: int, device: torch.device
) -> dict[str, Any]:
    pending, skipped, checked_frames, source_fingerprints = classify_rank_tasks(
        state, args, rank
    )
    if not pending:
        log(rank, f"all assigned cache files are valid (skipped={skipped})")
        return {
            "ok": True,
            "rank": rank,
            "computed": 0,
            "skipped": skipped,
            "frames": checked_frames,
            "source_fingerprints": source_fingerprints,
        }

    vae = load_vae(state, device, rank)
    transform = get_transform(
        state["transform"]["image_size"],
        state["transform"]["mean"],
        state["transform"]["std"],
    )
    storage_dtype = storage_torch_dtype(state)
    batch_size = state["encoding"]["vae_batch_size"]
    compute_dtype = state["encoding"]["compute_dtype"]
    pool_context = (
        ThreadPoolExecutor(max_workers=args.loader_threads)
        if args.loader_threads > 0
        else contextlib.nullcontext()
    )

    batch_refs: list[tuple[str, Path]] = []
    tasks_by_key: dict[str, dict[str, Any]] = {}
    means: dict[str, list[torch.Tensor]] = {}
    logvars: dict[str, list[torch.Tensor]] = {}
    frame_counts: dict[str, int] = {}
    computed = 0
    encoded_frames = 0
    started = time.perf_counter()

    def key_for(task: dict[str, Any]) -> str:
        return f"{task['dataset_name']}\0{task['trajectory_name']}"

    def flush_batch(executor) -> None:
        nonlocal batch_refs, computed, encoded_frames
        if not batch_refs:
            return
        paths = [item[1] for item in batch_refs]
        if executor is None:
            images = [load_image_tensor(path, transform) for path in paths]
        else:
            images = list(executor.map(lambda path: load_image_tensor(path, transform), paths))
        try:
            batch_mean, batch_logvar = encode_fixed_batch(
                vae,
                images,
                batch_size,
                device,
                compute_dtype,
                storage_dtype,
            )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"VAE batch {batch_size} OOM on rank {rank}; correctness requires the fixed "
                "online-equivalent batch shape. Use a GPU with enough free memory rather "
                "than silently changing VAE_BATCH_SIZE."
            ) from exc

        start = 0
        while start < len(batch_refs):
            key = batch_refs[start][0]
            end = start + 1
            while end < len(batch_refs) and batch_refs[end][0] == key:
                end += 1
            means[key].append(batch_mean[start:end])
            logvars[key].append(batch_logvar[start:end])
            frame_counts[key] += end - start
            start = end

        encoded_frames += len(batch_refs)
        touched_keys = set(item[0] for item in batch_refs)
        batch_refs = []
        for key in touched_keys:
            task = tasks_by_key[key]
            if frame_counts[key] != task["num_frames"]:
                continue
            trajectory_mean = torch.cat(means.pop(key), dim=0)
            trajectory_logvar = torch.cat(logvars.pop(key), dim=0)
            payload = build_cache_payload(state, task, trajectory_mean, trajectory_logvar)
            atomic_torch_save(Path(task["output_path"]), payload)
            error = cache_validation_error(state, task, check_finite=True)
            if error is not None:
                raise RuntimeError(f"post-write validation failed for {task['output_path']}: {error}")
            del frame_counts[key]
            del tasks_by_key[key]
            computed += 1
            if computed % max(1, args.log_every_trajectories) == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                log(
                    rank,
                    f"saved {computed}/{len(pending)} trajectories, "
                    f"{encoded_frames:,} frames, {encoded_frames / elapsed:.1f} frames/s",
                )

    with pool_context as executor_value:
        executor = executor_value if args.loader_threads > 0 else None
        for task, frame_paths in pending:
            key = key_for(task)
            tasks_by_key[key] = task
            means[key] = []
            logvars[key] = []
            frame_counts[key] = 0
            for path in frame_paths:
                batch_refs.append((key, path))
                if len(batch_refs) == batch_size:
                    flush_batch(executor)
        flush_batch(executor)

    if tasks_by_key or means or logvars or frame_counts:
        raise RuntimeError("internal error: unfinished trajectory accumulators after final VAE batch")
    del vae
    torch.cuda.empty_cache()
    elapsed = max(time.perf_counter() - started, 1e-6)
    log(
        rank,
        f"precompute done: computed={computed}, skipped={skipped}, "
        f"frames={encoded_frames:,}, throughput={encoded_frames / elapsed:.1f} frames/s",
    )
    return {
        "ok": True,
        "rank": rank,
        "computed": computed,
        "skipped": skipped,
        "frames": checked_frames,
        "encoded_frames": encoded_frames,
        "elapsed_seconds": elapsed,
        "source_fingerprints": source_fingerprints,
    }


def collect_valid_records(
    state: dict[str, Any],
    check_sources: bool,
    check_finite: bool,
    rank: int,
    tasks: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in state["datasets_meta"]
    }
    invalid: list[dict[str, str]] = []
    tasks = state["tasks"] if tasks is None else tasks
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        try:
            if check_sources:
                source_frame_paths(task)
            error = cache_validation_error(state, task, check_finite=check_finite)
        except Exception as exc:
            error = f"source validation failed: {type(exc).__name__}: {exc}"
        if error is not None:
            invalid.append(
                {
                    "dataset_name": task["dataset_name"],
                    "trajectory_name": task["trajectory_name"],
                    "error": error,
                }
            )
        else:
            output = Path(task["output_path"])
            frame_count = task["num_frames"]
            records[task["dataset_name"]].append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_name": task["dataset_name"],
                    "trajectory_name": task["trajectory_name"],
                    "relative_path": output.relative_to(Path(state["output_root"])).as_posix(),
                    "num_frames": frame_count,
                    "file_size_bytes": output.stat().st_size,
                    "frame_min": 0,
                    "frame_max": frame_count - 1,
                    "vae_fingerprint": state["vae"]["fingerprint"],
                    "transform_fingerprint": state["transform"]["fingerprint"],
                    "encoding_fingerprint": state["encoding"]["fingerprint"],
                    "source_fingerprint": task["source_fingerprint"],
                }
            )
        if index % 1000 == 0 or index == total:
            log(rank, f"full validation {index}/{total}, invalid={len(invalid)}")
    return records, invalid


def write_manifests(output_root: Path, records: dict[str, list[dict[str, Any]]]) -> None:
    for dataset_name, entries in records.items():
        entries.sort(key=lambda item: item["trajectory_name"])
        payload = b"".join(canonical_json(entry) + b"\n" for entry in entries)
        atomic_write_bytes(output_root / dataset_name / "manifest.jsonl", payload)


def sampled_frame_locations(
    tasks: list[dict[str, Any]], sample_count: int, seed: int
) -> list[tuple[dict[str, Any], int]]:
    def locate(
        scoped_tasks: list[dict[str, Any]], flat_index: int
    ) -> tuple[dict[str, Any], int]:
        cumulative: list[int] = []
        running = 0
        for scoped_task in scoped_tasks:
            running += scoped_task["num_frames"]
            cumulative.append(running)
        task_index = bisect.bisect_right(cumulative, flat_index)
        previous = cumulative[task_index - 1] if task_index else 0
        return scoped_tasks[task_index], flat_index - previous

    total = sum(task["num_frames"] for task in tasks)
    if total == 0:
        raise RuntimeError("cannot verify an empty latent cache")
    count = min(sample_count, total)
    rng = random.Random(seed)
    locations: list[tuple[dict[str, Any], int]] = []
    seen: set[tuple[str, str, int]] = set()

    # Always cover every configured dataset when the sample budget permits it.
    datasets = sorted({task["dataset_name"] for task in tasks})
    if count >= len(datasets):
        for dataset_name in datasets:
            scoped = [task for task in tasks if task["dataset_name"] == dataset_name]
            scoped_total = sum(task["num_frames"] for task in scoped)
            task, frame = locate(scoped, rng.randrange(scoped_total))
            key = (task["dataset_name"], task["trajectory_name"], frame)
            locations.append((task, frame))
            seen.add(key)

    while len(locations) < count:
        task, frame = locate(tasks, rng.randrange(total))
        key = (task["dataset_name"], task["trajectory_name"], frame)
        if key in seen:
            continue
        locations.append((task, frame))
        seen.add(key)
    return locations


def semantic_verify(
    state: dict[str, Any],
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    vae: AutoencoderKL | None = None,
) -> dict[str, Any]:
    locations = sampled_frame_locations(tasks, args.verify_samples, args.seed)
    transform = get_transform(
        state["transform"]["image_size"],
        state["transform"]["mean"],
        state["transform"]["std"],
    )
    if vae is None:
        vae = load_vae(state, device, rank)

    images = [
        load_image_tensor(Path(task["trajectory_dir"]) / f"{frame_index}.jpg", transform)
        for task, frame_index in locations
    ]
    direct_mean, direct_logvar = encode_fixed_batch(
        vae,
        images,
        state["encoding"]["vae_batch_size"],
        device,
        state["encoding"]["compute_dtype"],
        storage_torch_dtype(state),
    )
    cached_objects: dict[str, dict[str, Any]] = {}
    cached_mean: list[torch.Tensor] = []
    cached_logvar: list[torch.Tensor] = []
    for task, frame_index in locations:
        output_path = task["output_path"]
        if output_path not in cached_objects:
            cached_objects[output_path] = torch_load_mmap(Path(output_path))
        cached = cached_objects[output_path]
        cached_mean.append(cached["posterior_mean"][frame_index])
        cached_logvar.append(cached["posterior_logvar"][frame_index])
    cached_mean_tensor = torch.stack(cached_mean)
    cached_logvar_tensor = torch.stack(cached_logvar)

    def metrics(cached: torch.Tensor, direct: torch.Tensor) -> tuple[float, float, bool]:
        delta = (cached.float() - direct.float()).abs()
        return (
            float(delta.max().item()),
            float(delta.mean().item()),
            bool(torch.allclose(cached.float(), direct.float(), atol=args.verify_atol, rtol=args.verify_rtol)),
        )

    mean_max, mean_avg, mean_ok = metrics(cached_mean_tensor, direct_mean)
    logvar_max, logvar_avg, logvar_ok = metrics(cached_logvar_tensor, direct_logvar)
    per_dataset: dict[str, dict[str, float | int]] = {}
    for dataset_name in sorted({task["dataset_name"] for task, _ in locations}):
        indices = [
            index
            for index, (task, _) in enumerate(locations)
            if task["dataset_name"] == dataset_name
        ]
        ds_mean_max, ds_mean_avg, _ = metrics(
            cached_mean_tensor[indices], direct_mean[indices]
        )
        ds_logvar_max, ds_logvar_avg, _ = metrics(
            cached_logvar_tensor[indices], direct_logvar[indices]
        )
        per_dataset[dataset_name] = {
            "samples": len(indices),
            "mean_max_abs_error": ds_mean_max,
            "mean_mean_abs_error": ds_mean_avg,
            "logvar_max_abs_error": ds_logvar_max,
            "logvar_mean_abs_error": ds_logvar_avg,
        }

    def posterior_sample(
        mean: torch.Tensor, logvar: torch.Tensor, sample_seed: int
    ) -> torch.Tensor:
        parameters = torch.cat([mean, logvar], dim=1).to(device)
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        with make_autocast_context(device, state["encoding"]["compute_dtype"]):
            posterior = DiagonalGaussianDistribution(parameters)
            return posterior.sample(generator=generator).mul(DEFAULT_SCALING_FACTOR).cpu()

    sample_seed = args.seed + 1729
    direct_sample = posterior_sample(direct_mean, direct_logvar, sample_seed)
    cached_sample = posterior_sample(cached_mean_tensor, cached_logvar_tensor, sample_seed)
    cached_repeat = posterior_sample(cached_mean_tensor, cached_logvar_tensor, sample_seed)
    cached_different_seed = posterior_sample(
        cached_mean_tensor, cached_logvar_tensor, sample_seed + 1
    )
    sample_delta = (direct_sample.float() - cached_sample.float()).abs()
    sample_ok = bool(
        torch.allclose(
            direct_sample.float(),
            cached_sample.float(),
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
    )
    repeat_equal = bool(torch.equal(cached_sample, cached_repeat))
    different_seed_equal = bool(torch.equal(cached_sample, cached_different_seed))
    for dataset_name, dataset_metrics in per_dataset.items():
        indices = [
            index
            for index, (task, _) in enumerate(locations)
            if task["dataset_name"] == dataset_name
        ]
        dataset_sample_delta = (
            direct_sample[indices].float() - cached_sample[indices].float()
        ).abs()
        dataset_metrics["sample_max_abs_error"] = float(
            dataset_sample_delta.max().item()
        )
        dataset_metrics["sample_mean_abs_error"] = float(
            dataset_sample_delta.mean().item()
        )

    result: dict[str, Any] = {
        "samples": len(locations),
        "mean_max_abs_error": mean_max,
        "mean_mean_abs_error": mean_avg,
        "logvar_max_abs_error": logvar_max,
        "logvar_mean_abs_error": logvar_avg,
        "sample_max_abs_error": float(sample_delta.max().item()),
        "sample_mean_abs_error": float(sample_delta.mean().item()),
        "same_seed_cached_repeat_bitwise_equal": repeat_equal,
        "different_seed_cached_bitwise_equal": different_seed_equal,
        "per_dataset": per_dataset,
    }
    log(rank, f"semantic verification: {json.dumps(result, sort_keys=True)}")
    if not mean_ok or not logvar_ok or not sample_ok or not repeat_equal or different_seed_equal:
        raise RuntimeError(
            "semantic verification or posterior resampling check failed: "
            f"{result}; tolerances atol={args.verify_atol}, rtol={args.verify_rtol}"
        )
    return result


def finalize_cache(
    state: dict[str, Any], args: argparse.Namespace, device: torch.device, rank: int
) -> dict[str, Any]:
    output_root = Path(state["output_root"])
    selected = [task for task in state["tasks"] if task["selected"]]
    if args.max_trajectories:
        records, invalid = collect_valid_records(
            state,
            check_sources=False,
            check_finite=True,
            rank=rank,
            tasks=selected,
        )
        if invalid:
            raise RuntimeError(
                f"{len(invalid)} smoke-scope cache files failed final validation: {invalid[:5]}"
            )
        semantic = semantic_verify(state, selected, args, device, rank)
        partial_report = {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_NAME,
            "status": "smoke_scope_valid",
            "complete": False,
            "updated_at_utc": utc_now(),
            "selected_trajectories": len(selected),
            "selected_frames": sum(task["num_frames"] for task in selected),
            "total_existing_trajectories": len(state["tasks"]),
            "encoding_fingerprint": state["encoding"]["fingerprint"],
            "semantic_verification": semantic,
        }
        atomic_write_json(output_root / "_PARTIAL_VALIDATION.json", partial_report)
        log(
            rank,
            "limited smoke scope validated; canonical manifests, metadata.json, and "
            "_SUCCESS.json were intentionally left untouched",
        )
        return {
            "complete": False,
            "totals": {
                "completed_trajectories": len(selected),
                "completed_frames": sum(task["num_frames"] for task in selected),
                "full_existing_trajectories": len(state["tasks"]),
            },
            "semantic_verification": semantic,
        }

    records, invalid = collect_valid_records(
        state, check_sources=False, check_finite=True, rank=rank
    )
    valid_keys = {
        (record["dataset_name"], record["trajectory_name"])
        for entries in records.values()
        for record in entries
    }
    selected_keys = {(task["dataset_name"], task["trajectory_name"]) for task in selected}
    selected_invalid = [
        item
        for item in invalid
        if (item["dataset_name"], item["trajectory_name"]) in selected_keys
    ]
    if selected_invalid:
        preview = selected_invalid[:5]
        raise RuntimeError(
            f"{len(selected_invalid)} selected cache files failed final validation: {preview}"
        )

    valid_tasks = [
        task
        for task in state["tasks"]
        if (task["dataset_name"], task["trajectory_name"]) in valid_keys
    ]
    semantic = semantic_verify(state, valid_tasks, args, device, rank)
    write_manifests(output_root, records)

    datasets = {}
    for name, base in state["datasets_meta"].items():
        completed_entries = records.get(name, [])
        item = dict(base)
        item["completed_trajectories"] = len(completed_entries)
        item["completed_frames"] = sum(entry["num_frames"] for entry in completed_entries)
        item["complete"] = item["completed_trajectories"] == item["existing_trajectories"]
        item["manifest"] = str(output_root / name / "manifest.jsonl")
        item["manifest_sha256"] = sha256_file(output_root / name / "manifest.jsonl")
        datasets[name] = item

    complete = not invalid and all(item["complete"] for item in datasets.values())
    old_metadata_path = output_root / "metadata.json"
    created_at = utc_now()
    if old_metadata_path.is_file():
        with contextlib.suppress(Exception):
            created_at = json.loads(old_metadata_path.read_text(encoding="utf-8")).get(
                "created_at_utc", created_at
            )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT_NAME,
        "status": "complete" if complete else "partial",
        "complete": complete,
        "created_at_utc": created_at,
        "updated_at_utc": utc_now(),
        "vae": state["vae"],
        "transform": state["transform"],
        "software": state["software"],
        "hardware": state["hardware"],
        "encoding": state["encoding"],
        "storage": {
            "dtype": state["encoding"]["storage_dtype"],
            "layout": "NCHW",
            "tensor_keys": ["frame_indices", "posterior_mean", "posterior_logvar"],
            "file_pattern": "{dataset_name}/{trajectory_name}.pt",
        },
        "dataset_config": {
            "path": state["dataset_config_path"],
            "sha256": state["dataset_config_sha256"],
            "split": "train",
        },
        "datasets": datasets,
        "totals": {
            "split_entries": sum(item["split_entries"] for item in datasets.values()),
            "existing_trajectories": len(state["tasks"]),
            "missing_trajectories": sum(item["missing_trajectories"] for item in datasets.values()),
            "expected_frames": sum(task["num_frames"] for task in state["tasks"]),
            "completed_trajectories": len(valid_tasks),
            "completed_frames": sum(task["num_frames"] for task in valid_tasks),
            "invalid_or_missing_cache_files": len(invalid),
        },
        "semantic_verification": semantic,
        "invalid_cache_files": invalid[:100],
    }
    atomic_write_json(old_metadata_path, metadata)
    metadata_sha256 = sha256_file(old_metadata_path)
    success_path = output_root / "_SUCCESS.json"
    if complete:
        success = {
            "schema_version": SCHEMA_VERSION,
            "format": FORMAT_NAME,
            "complete": True,
            "completed_at_utc": utc_now(),
            "metadata_sha256": metadata_sha256,
            "encoding_fingerprint": state["encoding"]["fingerprint"],
            "totals": metadata["totals"],
        }
        atomic_write_json(success_path, success)
    else:
        with contextlib.suppress(FileNotFoundError):
            success_path.unlink()
        log(
            rank,
            f"partial cache is valid for selected scope but {len(invalid)} existing "
            "trajectories remain invalid/missing; _SUCCESS.json was not created",
        )
    return metadata


def verify_only(
    state: dict[str, Any], args: argparse.Namespace, device: torch.device, rank: int
) -> dict[str, Any]:
    tasks = [task for task in state["tasks"] if task["selected"]]
    selected_keys = {(task["dataset_name"], task["trajectory_name"]) for task in tasks}
    records, invalid = collect_valid_records(
        state,
        check_sources=True,
        check_finite=True,
        rank=rank,
        tasks=tasks if args.max_trajectories else None,
    )
    selected_invalid = [
        item
        for item in invalid
        if (item["dataset_name"], item["trajectory_name"]) in selected_keys
    ]
    if selected_invalid:
        raise RuntimeError(
            f"verify-only found {len(selected_invalid)} invalid selected files: "
            f"{selected_invalid[:5]}"
        )
    valid_keys = {
        (entry["dataset_name"], entry["trajectory_name"])
        for entries in records.values()
        for entry in entries
    }
    valid_tasks = [
        task
        for task in tasks
        if (task["dataset_name"], task["trajectory_name"]) in valid_keys
    ]
    semantic = semantic_verify(state, valid_tasks, args, device, rank)
    full_scope = args.max_trajectories == 0
    if full_scope:
        if invalid:
            raise RuntimeError(f"full cache verification found {len(invalid)} invalid files")
        success_path = Path(state["output_root"]) / "_SUCCESS.json"
        metadata_path = Path(state["output_root"]) / "metadata.json"
        if not success_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("complete cache is missing metadata.json or _SUCCESS.json")
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success.get("metadata_sha256") != sha256_file(metadata_path):
            raise RuntimeError("_SUCCESS.json metadata_sha256 does not match metadata.json")
    return {"verified_trajectories": len(valid_tasks), "semantic": semantic}


def all_gather_results(local_result: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return [local_result]
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_result)
    return gathered  # type: ignore[return-value]


def main() -> int:
    args = parse_args()
    validate_args(args)
    rank, world_size, local_rank = distributed_context()
    device = torch.device("cuda", local_rank)
    local_hardware = local_hardware_descriptor(local_rank)
    if dist.is_initialized():
        hardware_values: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(hardware_values, local_hardware)
    else:
        hardware_values = [local_hardware]
    hardware_fingerprints = {value["fingerprint"] for value in hardware_values}
    if len(hardware_fingerprints) != 1:
        raise RuntimeError(
            f"heterogeneous GPUs are not allowed for reproducible VAE extraction: {hardware_values}"
        )
    hardware = dict(local_hardware)
    hardware["homogeneous_world_size"] = world_size
    # Match the online training process, which leaves both flags at PyTorch's
    # defaults.  Enabling benchmark here could choose a different BF16 kernel
    # and make verification self-consistent while differing from training.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    # train.py sets these two flags at module import time.  The VAE contains a
    # mid-block attention layer, so mirror them explicitly before encoding.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    lock_handle = None
    running_marker: Path | None = None
    try:
        build_result: dict[str, Any] | None = None
        if rank == 0:
            try:
                output_root = Path(args.output_root).expanduser().resolve()
                lock_handle = acquire_root_lock(output_root)
                running_marker = output_root / ".RUNNING.json"
                atomic_write_json(
                    running_marker,
                    {
                        "pid": os.getpid(),
                        "host": os.uname().nodename,
                        "started_at_utc": utc_now(),
                        "world_size": world_size,
                        "command": sys.argv,
                    },
                )
                state = build_state(args, world_size, hardware)
                build_result = {"ok": True, "state": state}
                log(
                    rank,
                    f"discovered {len(state['tasks']):,} existing train trajectories / "
                    f"{sum(t['num_frames'] for t in state['tasks']):,} frames; "
                    f"missing split entries={sum(len(v) for v in state['missing'].values()):,}",
                )
            except Exception:
                build_result = {"ok": False, "error": traceback.format_exc()}
        if dist.is_initialized():
            values: list[Any] = [build_result]
            dist.broadcast_object_list(values, src=0)
            build_result = values[0]
        assert build_result is not None
        if not build_result["ok"]:
            raise RuntimeError(f"precompute setup failed on rank 0:\n{build_result['error']}")
        state = build_result["state"]

        if args.verify_only:
            if world_size != 1:
                raise RuntimeError("--verify-only must be launched as a single process")
            result = verify_only(state, args, device, rank)
            log(rank, f"verification succeeded: {json.dumps(result, sort_keys=True)}")
            return 0

        try:
            local_result = precompute_rank(state, args, rank, device)
        except Exception:
            local_result = {"ok": False, "rank": rank, "error": traceback.format_exc()}
        gathered = all_gather_results(local_result, world_size)
        failures = [item for item in gathered if not item.get("ok")]

        if not failures:
            fingerprints: dict[str, str] = {}
            for worker_result in gathered:
                fingerprints.update(worker_result.get("source_fingerprints", {}))
            for task in state["tasks"]:
                key = f"{task['dataset_name']}\0{task['trajectory_name']}"
                if task["selected"]:
                    if key not in fingerprints:
                        failures.append(
                            {
                                "ok": False,
                                "rank": "merge",
                                "error": f"missing source fingerprint for {key!r}",
                            }
                        )
                        break
                    task["source_fingerprint"] = fingerprints[key]

        final_result: dict[str, Any] | None = None
        if rank == 0:
            if failures:
                final_result = {"ok": False, "error": f"worker failures: {failures}"}
            else:
                try:
                    metadata = finalize_cache(state, args, device, rank)
                    final_result = {
                        "ok": True,
                        "complete": metadata["complete"],
                        "totals": metadata["totals"],
                        "workers": gathered,
                    }
                except Exception:
                    final_result = {"ok": False, "error": traceback.format_exc()}
        if dist.is_initialized():
            values = [final_result]
            dist.broadcast_object_list(values, src=0)
            final_result = values[0]
        assert final_result is not None
        if not final_result["ok"]:
            raise RuntimeError(f"precompute failed:\n{final_result['error']}")
        if rank == 0:
            log(rank, f"precompute and validation succeeded: {json.dumps(final_result, sort_keys=True)}")
        return 0
    finally:
        if rank == 0 and running_marker is not None:
            with contextlib.suppress(FileNotFoundError):
                running_marker.unlink()
        if lock_handle is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
