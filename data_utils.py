from collections import defaultdict
import hashlib
import inspect
import json
import logging
import os
import sys
import diffusers
from omegaconf import DictConfig, OmegaConf
import psutil
import torch
import torch.distributed as dist
import torchvision
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from datasets import TrainingDataset
from misc import CenterCropAR, IMAGE_ASPECT_RATIO, get_transform
from hydra.utils import get_original_cwd
from tabulate import tabulate
import time

from motion_condition import (
    motion_condition_collate,
    resolve_dataset_motion_map,
    validate_motion_types,
)


logger = logging.getLogger(__name__)
LATENT_SCHEMA_VERSION = 1
LATENT_FORMAT = "sd_vae_posterior_stats"
# CUDA/PyTorch can report a few MiB less or more usable memory for the same GPU
# model after a driver, firmware, or ECC configuration change.  That difference
# does not affect the numerical contents of an already extracted posterior
# cache, so keep the hardware-family checks strict while allowing only a small
# capacity-reporting drift at runtime.
RUNTIME_GPU_MEMORY_TOLERANCE_BYTES = 64 * 1024 * 1024


def _original_cwd() -> str:
    try:
        return get_original_cwd()
    except ValueError:
        # Enables standalone cache validation tests before Hydra initializes.
        return os.getcwd()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file {path}: {exc}") from exc


def _require_equal(actual, expected, field: str):
    if actual != expected:
        raise ValueError(
            f"Precomputed latent metadata mismatch for {field}: "
            f"{actual!r} != {expected!r}"
        )


def _validate_descriptor_fingerprint(descriptor: dict, field: str) -> None:
    value = dict(descriptor)
    fingerprint = value.pop("fingerprint", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    _require_equal(
        fingerprint,
        hashlib.sha256(encoded).hexdigest(),
        f"{field}.fingerprint",
    )


def _validate_precomputed_latent_cache_local(config: DictConfig) -> dict:
    """Strictly validate a completed cache before any DataLoader is created."""
    latent_config = config.dataset.precomputed_latents
    _require_equal(
        config.model.tokenizer.get("_target_"),
        "tokenizer_wrapper.VAEWrapper",
        "model.tokenizer._target_",
    )
    if config.get("tokenizer_path", None) is not None:
        raise ValueError(
            "Precomputed SD-VAE latents are incompatible with tokenizer_path; "
            "use the configured tokenizer_wrapper.VAEWrapper directly."
        )
    root = os.path.realpath(os.path.expanduser(str(latent_config.root)))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Precomputed latent root does not exist: {root}")

    metadata_path = os.path.join(root, "metadata.json")
    success_path = os.path.join(root, "_SUCCESS.json")
    if not os.path.isfile(metadata_path) or not os.path.isfile(success_path):
        raise FileNotFoundError(
            "Precomputed latent cache is incomplete: both metadata.json and "
            f"_SUCCESS.json are required under {root}"
        )

    metadata = _read_json(metadata_path)
    success = _read_json(success_path)
    for document_name, document in (("metadata.json", metadata), ("_SUCCESS.json", success)):
        if not isinstance(document, dict):
            raise TypeError(f"{document_name} must contain a JSON object")
        _require_equal(
            document.get("schema_version"), LATENT_SCHEMA_VERSION,
            f"{document_name}.schema_version",
        )
        _require_equal(document.get("format"), LATENT_FORMAT, f"{document_name}.format")

    _require_equal(metadata.get("status"), "complete", "metadata.json.status")
    _require_equal(metadata.get("complete"), True, "metadata.json.complete")
    _require_equal(success.get("complete"), True, "_SUCCESS.json.complete")
    _require_equal(
        success.get("metadata_sha256"),
        _sha256_file(metadata_path),
        "_SUCCESS.json.metadata_sha256",
    )

    vae = metadata.get("vae")
    transform = metadata.get("transform")
    software = metadata.get("software")
    hardware = metadata.get("hardware")
    encoding = metadata.get("encoding")
    storage = metadata.get("storage")
    datasets_metadata = metadata.get("datasets")
    for name, section in (
        ("vae", vae),
        ("transform", transform),
        ("software", software),
        ("hardware", hardware),
        ("encoding", encoding),
        ("storage", storage),
        ("datasets", datasets_metadata),
    ):
        if not isinstance(section, dict):
            raise TypeError(f"metadata.json.{name} must be an object")

    _require_equal(
        vae.get("identifier"),
        str(latent_config.vae_identifier),
        "vae.identifier",
    )
    _require_equal(
        float(vae.get("scaling_factor")),
        float(latent_config.scaling_factor),
        "vae.scaling_factor",
    )
    _require_equal(vae.get("latent_channels"), 4, "vae.latent_channels")
    if not isinstance(vae.get("fingerprint"), str) or not vae["fingerprint"]:
        raise ValueError("metadata.json.vae.fingerprint must be a non-empty string")
    _validate_descriptor_fingerprint(vae, "vae")

    _require_equal(transform.get("image_size"), int(config.dataset.image_size), "transform.image_size")
    _require_equal(list(transform.get("mean", [])), list(config.dataset.mean), "transform.mean")
    _require_equal(list(transform.get("std", [])), list(config.dataset.std), "transform.std")
    _require_equal(
        float(transform.get("center_crop_aspect_ratio")),
        float(IMAGE_ASPECT_RATIO),
        "transform.center_crop_aspect_ratio",
    )
    resize_size = transform.get("resize_size")
    if resize_size not in (
        int(config.dataset.image_size),
        [int(config.dataset.image_size), int(config.dataset.image_size)],
    ):
        raise ValueError(
            f"Precomputed latent transform.resize_size mismatch: {resize_size!r}"
        )
    if not isinstance(transform.get("fingerprint"), str) or not transform["fingerprint"]:
        raise ValueError("metadata.json.transform.fingerprint must be a non-empty string")
    _validate_descriptor_fingerprint(transform, "transform")
    transform_source = inspect.getsource(CenterCropAR) + "\n" + inspect.getsource(get_transform)
    _require_equal(
        hashlib.sha256(transform_source.encode("utf-8")).hexdigest(),
        transform.get("source_sha256"),
        "transform.source_sha256",
    )
    _require_equal(
        transform.get("implementation"), "misc.get_transform", "transform.implementation"
    )
    _require_equal(
        transform.get("resize_interpolation"),
        "torchvision PIL bilinear default",
        "transform.resize_interpolation",
    )
    _require_equal(transform.get("to_tensor"), True, "transform.to_tensor")
    _require_equal(
        transform.get("normalize_inplace"), True, "transform.normalize_inplace"
    )

    _validate_descriptor_fingerprint(software, "software")
    expected_software = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "diffusers": diffusers.__version__,
        "pillow": pillow_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    for key, expected_value in expected_software.items():
        _require_equal(software.get(key), expected_value, f"software.{key}")

    hardware_core = {
        "gpu_name": hardware.get("gpu_name"),
        "compute_capability": hardware.get("compute_capability"),
        "total_memory_bytes": hardware.get("total_memory_bytes"),
    }
    hardware_fingerprint = hashlib.sha256(
        json.dumps(
            hardware_core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _require_equal(
        hardware.get("fingerprint"), hardware_fingerprint, "hardware.fingerprint"
    )
    if not isinstance(hardware.get("homogeneous_world_size"), int) or hardware[
        "homogeneous_world_size"
    ] < 1:
        raise ValueError("metadata.json.hardware.homogeneous_world_size must be positive")

    _require_equal(encoding.get("vae_fingerprint"), vae["fingerprint"], "encoding.vae_fingerprint")
    _require_equal(
        encoding.get("transform_fingerprint"),
        transform["fingerprint"],
        "encoding.transform_fingerprint",
    )
    _require_equal(encoding.get("compute_dtype"), "bfloat16", "encoding.compute_dtype")
    _require_equal(encoding.get("storage_dtype"), "bfloat16", "encoding.storage_dtype")
    _require_equal(
        encoding.get("vae_batch_size"),
        int(latent_config.vae_encode_batch_size),
        "encoding.vae_batch_size",
    )
    _require_equal(encoding.get("posterior_fields"), ["mean", "logvar"], "encoding.posterior_fields")
    _require_equal(
        encoding.get("posterior_logvar_clamp"),
        [-30.0, 20.0],
        "encoding.posterior_logvar_clamp",
    )
    _require_equal(encoding.get("scaling_applied"), False, "encoding.scaling_applied")
    _require_equal(
        encoding.get("fixed_batch_padding"),
        "repeat_last_frame_for_final_partial_batch",
        "encoding.fixed_batch_padding",
    )
    _require_equal(encoding.get("cudnn_benchmark"), False, "encoding.cudnn_benchmark")
    _require_equal(
        encoding.get("cudnn_deterministic"), False, "encoding.cudnn_deterministic"
    )
    _require_equal(
        encoding.get("cuda_matmul_allow_tf32"),
        True,
        "encoding.cuda_matmul_allow_tf32",
    )
    _require_equal(
        encoding.get("cudnn_allow_tf32"), True, "encoding.cudnn_allow_tf32"
    )
    _require_equal(
        encoding.get("software_fingerprint"),
        software["fingerprint"],
        "encoding.software_fingerprint",
    )
    _require_equal(
        encoding.get("hardware_fingerprint"),
        hardware["fingerprint"],
        "encoding.hardware_fingerprint",
    )
    runtime_flags = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    for key, actual_value in runtime_flags.items():
        _require_equal(actual_value, encoding[key], f"runtime.{key}")
    active_dataset_configs = {
        str(name): dataset
        for name, dataset in config.dataset.datasets.items()
        if bool(dataset.get("enabled", True))
    }
    goals_per_observation = {
        int(dataset.get("goals_per_obs", 0))
        for dataset in active_dataset_configs.values()
        if "train" in dataset
    }
    if goals_per_observation != {4}:
        raise ValueError(
            "The validated latent-cache compatibility target requires exactly "
            f"4 goals per training observation; got {sorted(goals_per_observation)}"
        )
    compatibility_target = {
        "training_batch_per_gpu": 16,
        "context_images": int(config.dataset.context_size),
        "goal_images": 4,
        "images_per_observation": int(config.dataset.context_size) + 4,
        "flattened_vae_batch_per_gpu": int(latent_config.vae_encode_batch_size),
        "bfloat16_autocast": True,
    }
    _require_equal(
        encoding.get("compatibility_target"),
        compatibility_target,
        "encoding.compatibility_target",
    )
    _require_equal(
        int(config.training.batch_size),
        compatibility_target["training_batch_per_gpu"],
        "training.batch_size",
    )
    if not isinstance(encoding.get("fingerprint"), str) or not encoding["fingerprint"]:
        raise ValueError("metadata.json.encoding.fingerprint must be a non-empty string")
    _validate_descriptor_fingerprint(encoding, "encoding")

    _require_equal(storage.get("dtype"), "bfloat16", "storage.dtype")
    tensor_keys = storage.get("tensor_keys", [])
    if tensor_keys != ["frame_indices", "posterior_mean", "posterior_logvar"]:
        raise ValueError(
            "metadata.json.storage.tensor_keys must be exactly frame_indices, "
            "posterior_mean, posterior_logvar"
        )
    _require_equal(storage.get("layout"), "NCHW", "storage.layout")
    if not bool(config.bfloat16):
        raise ValueError(
            "This cache contains bfloat16 posteriors from the paper training path, "
            "but bfloat16=0 was requested. Disable precomputed latents or use bfloat16=1."
        )

    # Verify that the configured tokenizer resolves to the same VAE config used
    # for extraction. The launcher supplies a local NAS snapshot, so this check
    # catches accidentally switching checkpoints without hashing the large model
    # weights on every launch.
    model_path = os.path.expanduser(str(config.model.tokenizer.model_path))
    if not os.path.isabs(model_path):
        model_path = os.path.join(_original_cwd(), model_path)
    if os.path.isdir(model_path):
        vae_config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(vae_config_path):
            raise FileNotFoundError(f"Configured VAE has no config.json: {model_path}")
        _require_equal(
            _sha256_file(vae_config_path),
            vae.get("config_sha256"),
            "vae.config_sha256",
        )
        weights_filename = vae.get("weights_filename")
        if not isinstance(weights_filename, str) or not weights_filename:
            raise ValueError("metadata.json.vae.weights_filename is missing")
        weights_path = os.path.join(model_path, weights_filename)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Configured VAE is missing cached weights {weights_path}"
            )
        _require_equal(
            os.path.getsize(weights_path),
            vae.get("weights_size_bytes"),
            "vae.weights_size_bytes",
        )
        _require_equal(
            _sha256_file(weights_path),
            vae.get("weights_sha256"),
            "vae.weights_sha256",
        )
    else:
        raise ValueError(
            "Precomputed latent mode requires model.tokenizer.model_path to be "
            f"the local VAE snapshot used for extraction; got {model_path!r}"
        )

    expected_file_metadata = {
        "vae_fingerprint": vae["fingerprint"],
        "transform_fingerprint": transform["fingerprint"],
        "encoding_fingerprint": encoding["fingerprint"],
        "hardware_fingerprint": hardware["fingerprint"],
        "scaling_factor": float(latent_config.scaling_factor),
        "image_size": int(config.dataset.image_size),
        "storage_dtype": "bfloat16",
        "compute_dtype": "bfloat16",
        "vae_batch_size": int(latent_config.vae_encode_batch_size),
    }

    manifest_records = {}
    for dataset_name, data_config in active_dataset_configs.items():
        if dataset_name not in datasets_metadata:
            raise ValueError(
                f"metadata.json has no entry for configured dataset {dataset_name!r}"
            )
        dataset_metadata = datasets_metadata[dataset_name]
        if not isinstance(dataset_metadata, dict):
            raise TypeError(f"metadata.json.datasets.{dataset_name} must be an object")

        split_path = os.path.join(
            _original_cwd(), str(data_config.train), "traj_names.txt"
        )
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Training split manifest is missing: {split_path}")
        _require_equal(
            _sha256_file(split_path),
            dataset_metadata.get("split_manifest_sha256"),
            f"datasets.{dataset_name}.split_manifest_sha256",
        )
        with open(split_path, "r", encoding="utf-8") as handle:
            split_names = [line.strip() for line in handle if line.strip()]
        if len(split_names) != len(set(split_names)):
            raise ValueError(f"Training split contains duplicate trajectories: {split_path}")
        _require_equal(
            len(split_names),
            dataset_metadata.get("split_entries"),
            f"datasets.{dataset_name}.split_entries",
        )

        manifest_path = os.path.join(root, dataset_name, "manifest.jsonl")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Latent dataset manifest is missing: {manifest_path}")
        _require_equal(
            _sha256_file(manifest_path),
            dataset_metadata.get("manifest_sha256"),
            f"datasets.{dataset_name}.manifest_sha256",
        )
        records = {}
        completed_frames = 0
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {manifest_path}:{line_number}: {exc}"
                    ) from exc
                _require_equal(record.get("schema_version"), 1, f"{manifest_path}:{line_number}.schema_version")
                _require_equal(record.get("dataset_name"), dataset_name, f"{manifest_path}:{line_number}.dataset_name")
                trajectory_name = record.get("trajectory_name")
                if not isinstance(trajectory_name, str) or not trajectory_name:
                    raise ValueError(
                        f"Invalid trajectory_name in {manifest_path}:{line_number}"
                    )
                if trajectory_name in records:
                    raise ValueError(
                        f"Duplicate trajectory {trajectory_name!r} in {manifest_path}"
                    )
                _require_equal(record.get("vae_fingerprint"), vae["fingerprint"], f"{manifest_path}:{line_number}.vae_fingerprint")
                _require_equal(record.get("transform_fingerprint"), transform["fingerprint"], f"{manifest_path}:{line_number}.transform_fingerprint")
                _require_equal(record.get("encoding_fingerprint"), encoding["fingerprint"], f"{manifest_path}:{line_number}.encoding_fingerprint")
                source_fingerprint = record.get("source_fingerprint")
                if not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64:
                    raise ValueError(
                        f"Invalid source_fingerprint in {manifest_path}:{line_number}"
                    )
                num_frames = record.get("num_frames")
                if not isinstance(num_frames, int) or num_frames < 1:
                    raise ValueError(
                        f"Invalid num_frames in {manifest_path}:{line_number}: {num_frames!r}"
                    )
                _require_equal(record.get("frame_min"), 0, f"{manifest_path}:{line_number}.frame_min")
                _require_equal(record.get("frame_max"), num_frames - 1, f"{manifest_path}:{line_number}.frame_max")
                expected_relative_path = os.path.join(dataset_name, f"{trajectory_name}.pt")
                _require_equal(
                    os.path.normpath(str(record.get("relative_path"))),
                    os.path.normpath(expected_relative_path),
                    f"{manifest_path}:{line_number}.relative_path",
                )
                cache_path = os.path.realpath(os.path.join(root, expected_relative_path))
                if not cache_path.startswith(root + os.sep):
                    raise ValueError(f"Unsafe latent relative path: {expected_relative_path}")
                if not os.path.isfile(cache_path):
                    raise FileNotFoundError(f"Manifest references missing latent file: {cache_path}")
                _require_equal(
                    os.path.getsize(cache_path),
                    record.get("file_size_bytes"),
                    f"{manifest_path}:{line_number}.file_size_bytes",
                )
                records[trajectory_name] = record
                completed_frames += num_frames

        split_name_set = set(split_names)
        unexpected = sorted(set(records).difference(split_name_set))
        if unexpected:
            raise ValueError(
                f"Latent manifest {manifest_path} contains {len(unexpected)} "
                f"trajectory names outside the training split; first: {unexpected[:5]}"
            )

        # Missing source trajectories are intentionally excluded from the NWM
        # training index. Every trajectory that still exists in the configured
        # source data must have a cache record; otherwise fail before training.
        source_root = str(data_config.data_folder)
        if not os.path.isabs(source_root):
            source_root = os.path.join(_original_cwd(), source_root)
        source_root = os.path.realpath(source_root)
        _require_equal(
            source_root,
            os.path.realpath(str(dataset_metadata.get("data_folder"))),
            f"datasets.{dataset_name}.data_folder",
        )
        missing_source = []
        missing_cache = []
        for trajectory_name in split_names:
            source_traj = os.path.join(source_root, trajectory_name, "traj_data.pkl")
            if os.path.isfile(source_traj):
                if trajectory_name not in records:
                    missing_cache.append(trajectory_name)
            else:
                missing_source.append(trajectory_name)
        if missing_cache:
            raise FileNotFoundError(
                f"Latent cache for {dataset_name} is incomplete: {len(missing_cache)} "
                f"existing training trajectories are missing; first: {missing_cache[:5]}"
            )
        _require_equal(
            len(records),
            dataset_metadata.get("completed_trajectories"),
            f"datasets.{dataset_name}.completed_trajectories",
        )
        _require_equal(
            len(records),
            dataset_metadata.get("existing_trajectories"),
            f"datasets.{dataset_name}.existing_trajectories",
        )
        _require_equal(
            len(missing_source),
            dataset_metadata.get("missing_trajectories"),
            f"datasets.{dataset_name}.missing_trajectories",
        )
        _require_equal(
            missing_source,
            dataset_metadata.get("missing_trajectory_names"),
            f"datasets.{dataset_name}.missing_trajectory_names",
        )
        _require_equal(
            completed_frames,
            dataset_metadata.get("completed_frames"),
            f"datasets.{dataset_name}.completed_frames",
        )
        manifest_records[dataset_name] = records

    totals = metadata.get("totals")
    if not isinstance(totals, dict):
        raise TypeError("metadata.json.totals must be an object")
    total_completed_trajectories = sum(
        int(item["completed_trajectories"]) for item in datasets_metadata.values()
    )
    total_completed_frames = sum(
        int(item["completed_frames"]) for item in datasets_metadata.values()
    )
    _require_equal(
        totals.get("completed_trajectories"),
        total_completed_trajectories,
        "totals.completed_trajectories",
    )
    _require_equal(
        totals.get("completed_frames"),
        total_completed_frames,
        "totals.completed_frames",
    )
    _require_equal(
        totals.get("invalid_or_missing_cache_files"),
        0,
        "totals.invalid_or_missing_cache_files",
    )
    _require_equal(success.get("encoding_fingerprint"), encoding["fingerprint"], "_SUCCESS.json.encoding_fingerprint")
    _require_equal(success.get("totals"), totals, "_SUCCESS.json.totals")

    return {
        "root": root,
        "expected_file_metadata": expected_file_metadata,
        "manifest_records": manifest_records,
        "metadata": metadata,
    }


def _validate_runtime_latent_hardware(metadata: dict) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for precomputed latent training")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    expected = metadata["hardware"]
    _require_equal(
        properties.name,
        expected.get("gpu_name"),
        "runtime.hardware.gpu_name",
    )
    _require_equal(
        [properties.major, properties.minor],
        expected.get("compute_capability"),
        "runtime.hardware.compute_capability",
    )

    actual_memory = int(properties.total_memory)
    try:
        expected_memory = int(expected["total_memory_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "metadata.json.hardware.total_memory_bytes must be an integer"
        ) from exc
    memory_difference = abs(actual_memory - expected_memory)
    if memory_difference > RUNTIME_GPU_MEMORY_TOLERANCE_BYTES:
        raise ValueError(
            "Precomputed latent metadata mismatch for "
            "runtime.hardware.total_memory_bytes: "
            f"{actual_memory} != {expected_memory} "
            f"(difference {memory_difference} bytes exceeds the allowed "
            f"{RUNTIME_GPU_MEMORY_TOLERANCE_BYTES}-byte same-GPU tolerance)"
        )
    if memory_difference:
        logger.warning(
            "Allowing a small runtime GPU-memory reporting difference for the "
            "same GPU model and compute capability: runtime=%d, cache=%d, "
            "difference=%d bytes",
            actual_memory,
            expected_memory,
            memory_difference,
        )


def validate_precomputed_latent_cache(config: DictConfig) -> dict:
    """Validate on rank zero once, then propagate either result or error."""
    if not dist.is_available() or not dist.is_initialized():
        result = _validate_precomputed_latent_cache_local(config)
        _validate_runtime_latent_hardware(result["metadata"])
        return result

    message = [None]
    if dist.get_rank() == 0:
        try:
            message[0] = {
                "ok": True,
                "result": _validate_precomputed_latent_cache_local(config),
            }
        except Exception as exc:
            message[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(message, src=0)
    if not message[0]["ok"]:
        raise RuntimeError(
            "Precomputed latent cache validation failed: " + message[0]["error"]
        )
    result = message[0]["result"]
    # Every rank validates its own device; rank zero's metadata scan alone
    # cannot detect a heterogeneous multi-GPU training node.
    local_error = None
    try:
        _validate_runtime_latent_hardware(result["metadata"])
    except Exception as exc:
        local_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    hardware_errors = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(hardware_errors, local_error)
    hardware_errors = [error for error in hardware_errors if error is not None]
    if hardware_errors:
        raise RuntimeError(
            "Precomputed latent hardware validation failed: "
            + "; ".join(hardware_errors)
        )
    return result


def prepare_datasets(
    config: DictConfig,
    *,
    finetune_substage: str | None = None,
    include_test: bool = True,
):
    """Create and prepare training and test datasets."""
    train_dataset = []
    test_dataset = []
    train_motion_audit = []

    training_stage = str(config.get("training_stage", "legacy"))
    latent_config = config.dataset.get("precomputed_latents", {})
    use_precomputed_latents = bool(latent_config.get("enabled", False))
    validated_latents = None
    if use_precomputed_latents:
        validated_latents = validate_precomputed_latent_cache(config)
        logger.info(
            "Precomputed SD-VAE posterior mode enabled for training only: "
            f"root={validated_latents['root']}, "
            f"worker_lru_trajectories={int(latent_config.cache_size)}"
        )
    else:
        logger.info(
            "Precomputed latent mode disabled; training images will be decoded "
            "and encoded by the VAE online"
        )

    active_datasets = {
        str(name): dataset_config
        for name, dataset_config in config.dataset.datasets.items()
        if bool(dataset_config.get("enabled", True))
    }
    if not active_datasets:
        raise ValueError("At least one dataset must be enabled")

    if training_stage == "real_finetune":
        expected_training_datasets = {"recon", "scand", "tartan_drive", "sacson"}
        selected_training_datasets = {
            name
            for name, dataset_config in active_datasets.items()
            if "train" in dataset_config
            and not bool(dataset_config.get("evaluation_only", False))
        }
        if selected_training_datasets != expected_training_datasets:
            raise ValueError(
                "real_finetune must train on exactly RECON, SCAND, TartanDrive, "
                "and HuRoN (the repository key is 'sacson'); got "
                f"{sorted(selected_training_datasets)}"
            )
        declared_selection = config.get("dataset_selection", {}).get(
            "finetune", None
        )
        if declared_selection is not None and set(map(str, declared_selection)) != (
            expected_training_datasets
        ):
            raise ValueError(
                "dataset_selection.finetune must be exactly "
                "[recon, scand, tartan_drive, sacson]"
            )
        global_distance = (
            int(config.dataset.distance.min_dist_cat),
            int(config.dataset.distance.max_dist_cat),
        )
        if global_distance != (-64, 64):
            raise ValueError(
                "real_finetune must retain the complete frame-offset range "
                f"[-64, 64], got {list(global_distance)}"
            )
        for name in selected_training_datasets:
            data_config = active_datasets[name]
            effective_distance = (
                int(
                    data_config.get("distance", {}).get(
                        "min_dist_cat", global_distance[0]
                    )
                ),
                int(
                    data_config.get("distance", {}).get(
                        "max_dist_cat", global_distance[1]
                    )
                ),
            )
            if effective_distance != (-64, 64):
                raise ValueError(
                    f"real_finetune dataset {name!r} must sample [-64, 64], "
                    f"got {list(effective_distance)}"
                )

    motion_config = config.get("motion_condition", None)
    use_motion_condition = bool(
        motion_config is not None and motion_config.get("enabled", False)
    )
    train_motion_types = None
    eval_motion_types = None
    dataset_motion_map = None
    if use_motion_condition:
        train_motion_types = validate_motion_types(motion_config, training=True)
        training_dataset_names = [
            str(name)
            for name, dataset_config in active_datasets.items()
            if "train" in dataset_config
            and not bool(dataset_config.get("evaluation_only", False))
        ]
        dataset_motion_map = resolve_dataset_motion_map(
            motion_config, training_dataset_names
        )
        eval_motion_type = str(motion_config.get("eval_type", "real"))
        available_types = validate_motion_types(motion_config)
        if eval_motion_type not in available_types:
            raise ValueError(
                f"motion_condition.eval_type={eval_motion_type!r} is not available"
            )
        eval_motion_types = (eval_motion_type,)
        logger.info(
            f"Motion condition enabled: train_types={train_motion_types}, "
            f"eval_type={eval_motion_type}"
        )
        if dataset_motion_map is not None:
            logger.info(
                "Using fixed dataset motion assignments: "
                f"{dataset_motion_map}"
            )
        if eval_motion_type not in train_motion_types:
            logger.warning(
                f"Evaluation uses the {eval_motion_type} adapter, but it is not in "
                f"train_types={train_motion_types}; that adapter will remain at its "
                "seeded initialization unless a checkpoint initializes it."
            )

    for dataset_name, data_config in active_datasets.items():

        split_types = ["train", "test"] if include_test else ["train"]
        for data_split_type in split_types:
            if data_split_type in data_config:
                if data_split_type == "train" and bool(
                    data_config.get("evaluation_only", False)
                ):
                    logger.info(
                        f"Skipping evaluation-only dataset {dataset_name} for training"
                    )
                    continue
                goals_per_obs = int(data_config["goals_per_obs"])
                if data_split_type == "test":
                    goals_per_obs = 4  # standardize testing

                min_dist_cat = data_config.get("distance", {}).get(
                    "min_dist_cat", config.dataset.distance.min_dist_cat
                )
                max_dist_cat = data_config.get("distance", {}).get(
                    "max_dist_cat", config.dataset.distance.max_dist_cat
                )
                len_traj_pred = data_config.get(
                    "len_traj_pred", config.dataset.len_traj_pred
                )
                if data_split_type == "train":
                    split_motion_types = (
                        (dataset_motion_map[dataset_name],)
                        if dataset_motion_map is not None
                        else train_motion_types
                    )
                else:
                    split_motion_types = eval_motion_types

                dataset = TrainingDataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config.dataset.image_size,
                    min_dist_cat=min_dist_cat,
                    max_dist_cat=max_dist_cat,
                    len_traj_pred=len_traj_pred,
                    context_size=config.dataset.context_size,
                    normalize=config.dataset.normalize,
                    goals_per_obs=goals_per_obs,
                    transform=get_transform(
                        config.dataset.image_size,
                        config.dataset.mean,
                        config.dataset.std,
                    ),
                    action_stats=config.dataset.action_stats,
                    waypoint_spacing=data_config.metric_waypoint_spacing,
                    predefined_index=None,
                    traj_stride=1,
                    precomputed_latent_root=(
                        validated_latents["root"]
                        if data_split_type == "train" and validated_latents
                        else None
                    ),
                    precomputed_latent_cache_size=int(
                        latent_config.get("cache_size", 8)
                    ),
                    precomputed_latent_metadata=(
                        validated_latents["expected_file_metadata"]
                        if data_split_type == "train" and validated_latents
                        else None
                    ),
                    precomputed_latent_records=(
                        validated_latents["manifest_records"][dataset_name]
                        if data_split_type == "train" and validated_latents
                        else None
                    ),
                    motion_condition=motion_config,
                    motion_types=split_motion_types,
                    two_stage_config=(
                        config
                        if data_split_type == "train"
                        and str(config.get("training_stage", "legacy")) != "legacy"
                        else None
                    ),
                    finetune_substage=(
                        finetune_substage if data_split_type == "train" else None
                    ),
                )

                if data_split_type == "train":
                    train_dataset.append(dataset)
                    if use_motion_condition:
                        train_motion_audit.append(
                            (dataset_name, len(dataset), tuple(split_motion_types))
                        )
                else:
                    test_dataset.append(dataset)

                motion_label = (
                    tuple(split_motion_types) if use_motion_condition else "legacy-real"
                )
                print(
                    f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}, "
                    f"input={'precomputed posterior' if dataset.uses_precomputed_latents else 'pixels'}, "
                    f"motion={motion_label}"
                )

    # Combine all datasets from different robots
    if train_motion_audit:
        total_samples = sum(size for _, size, _ in train_motion_audit)
        for dataset_name, size, motion_types in train_motion_audit:
            logger.info(
                f"Training mix: dataset={dataset_name}, motion={motion_types}, "
                f"samples={size}, expected_share={size / total_samples:.2%}"
            )
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset) if test_dataset else None

    return train_dataset, test_dataset


def prepare_proxy_pretrain_dataset(config: DictConfig):
    """Build the NavAnywhere-only image dataset used by stage-1 training."""
    from navanywhere_latent_cache import validate_navanywhere_latent_cache
    from two_stage_data import NavAnywhereDataset

    if str(config.get("training_stage", "legacy")) != "proxy_pretrain":
        raise ValueError(
            "prepare_proxy_pretrain_dataset requires training_stage=proxy_pretrain"
        )
    datasets_config = config.dataset.datasets
    enabled = [
        str(name)
        for name, dataset_config in datasets_config.items()
        if bool(dataset_config.get("enabled", True))
    ]
    if enabled != ["navanywhere"]:
        raise ValueError(
            "Stage-1 training must enable NavAnywhere only; "
            f"enabled datasets are {enabled}"
        )
    data_config = datasets_config.navanywhere
    root = data_config.get("root", data_config.get("data_folder"))
    if root is None:
        raise ValueError("dataset.datasets.navanywhere.root is required")
    root = os.path.expanduser(str(root))
    if not os.path.isabs(root):
        root = os.path.join(_original_cwd(), root)
    source_ids = data_config.get("source_ids", None)
    if source_ids is not None:
        source_ids = list(source_ids)
    manifest = data_config.get("manifest", None)
    if manifest is not None and str(manifest).strip().lower() in {"", "none", "null"}:
        manifest = None
    elif isinstance(manifest, (str, os.PathLike)):
        manifest = os.path.expanduser(os.fspath(manifest))
        if not os.path.isabs(manifest):
            manifest = os.path.join(_original_cwd(), manifest)
    recipe_config = config.dataset.get("sampling_recipe", {})
    sampling_recipe = recipe_config.get("path", None)
    if sampling_recipe is not None and str(sampling_recipe).strip().lower() in {
        "",
        "none",
        "null",
    }:
        sampling_recipe = None
    elif sampling_recipe is not None:
        sampling_recipe = os.path.expanduser(os.fspath(sampling_recipe))
        if not os.path.isabs(sampling_recipe):
            sampling_recipe = os.path.join(_original_cwd(), sampling_recipe)
        sampling_recipe = os.path.realpath(sampling_recipe)
        if not os.path.isfile(sampling_recipe):
            raise FileNotFoundError(
                f"NavAnywhere sampling recipe does not exist: {sampling_recipe}"
            )
        configured_recipe_sha = recipe_config.get("sha256", None)
        actual_recipe_sha = _sha256_file(sampling_recipe)
        if configured_recipe_sha not in (None, "", "null") and str(
            configured_recipe_sha
        ) != actual_recipe_sha:
            raise ValueError(
                "dataset.sampling_recipe.sha256 does not match the recipe file: "
                f"{configured_recipe_sha!r} != {actual_recipe_sha!r}"
            )
        config.dataset.sampling_recipe.sha256 = actual_recipe_sha
    if sampling_recipe is not None and manifest is not None:
        raise ValueError(
            "Use dataset.sampling_recipe.path or NavAnywhere manifest, not both"
        )
    frame_range = data_config.get("frame_offset_range", None)
    min_offset = int(
        data_config.get(
            "min_frame_offset",
            config.dataset.distance.get("min_dist_cat", -64),
        )
    )
    max_offset = int(
        data_config.get(
            "max_frame_offset",
            config.dataset.distance.get("max_dist_cat", 64),
        )
    )
    if frame_range is not None:
        min_offset, max_offset = map(int, frame_range)
    if (min_offset, max_offset) != (-64, 64):
        raise ValueError(
            "Two-stage NavAnywhere pretraining must retain frame offsets [-64, 64]; "
            f"got [{min_offset}, {max_offset}]"
        )
    action_mode = str(config.get("action_mode", "none"))
    raw_proxy_config = config.get("proxy", {})
    proxy_config = (
        OmegaConf.to_container(raw_proxy_config, resolve=True)
        if OmegaConf.is_config(raw_proxy_config)
        else dict(raw_proxy_config)
    )
    if not isinstance(proxy_config, dict):
        raise TypeError("proxy configuration must be a mapping")
    if not bool(proxy_config.get("use_precomputed_only", True)):
        raise ValueError(
            "Stage-1 training only supports offline proxy caches; "
            "proxy.use_precomputed_only must be true"
        )
    proxy_limit = int(proxy_config.get("max_abs_frame_offset", 8))
    if proxy_limit != 8:
        raise ValueError(
            "Stage-1 proxy eligibility is fixed to abs(frame_offset)<=8"
        )
    if action_mode != "none":
        storage_path = proxy_config.get("storage_path")
        if storage_path is None:
            raise ValueError(
                f"action_mode={action_mode!r} requires proxy.storage_path"
            )
        storage_path = os.path.expanduser(str(storage_path))
        if not os.path.isabs(storage_path):
            storage_path = os.path.join(_original_cwd(), storage_path)
        proxy_config["storage_path"] = storage_path
        proxy_type = str(proxy_config.get("type", action_mode))
        if proxy_type != action_mode:
            raise ValueError(
                f"proxy.type={proxy_type!r} must match action_mode={action_mode!r}"
            )
    latent_config = config.dataset.get("precomputed_latents", {})
    validated_latents = None
    if bool(latent_config.get("enabled", False)):
        if sampling_recipe is None:
            raise ValueError(
                "NavAnywhere precomputed latents require "
                "dataset.sampling_recipe.path so cache/frame identity is frozen"
            )
        if latent_config.get("root", None) in (None, "", "null"):
            raise ValueError(
                "dataset.precomputed_latents.root is required when enabled"
            )
        validated_latents = validate_navanywhere_latent_cache(
            config,
            recipe_path=sampling_recipe,
            launch_dir=_original_cwd(),
        )
        logger.info(
            "NavAnywhere precomputed SD-VAE posteriors enabled: root=%s, "
            "recipe_sha256=%s, worker_lru_trajectories=%d",
            validated_latents["root"],
            validated_latents["recipe_sha256"],
            int(latent_config.get("cache_size", 8)),
        )
    dataset = NavAnywhereDataset(
        root=root,
        source_ids=source_ids,
        manifest=manifest,
        sampling_recipe=sampling_recipe,
        transform=get_transform(
            config.dataset.image_size,
            config.dataset.mean,
            config.dataset.std,
        ),
        context_size=int(config.dataset.context_size),
        goals_per_obs=int(data_config.get("goals_per_obs", 1)),
        min_frame_offset=min_offset,
        max_frame_offset=max_offset,
        action_mode=action_mode,
        proxy=proxy_config,
        proxy_dim=(
            None
            if action_mode == "none"
            else int(proxy_config.get("dim"))
        ),
        proxy_max_abs_frame_offset=proxy_limit,
        strict_loading=bool(proxy_config.get("strict_loading", False)),
        seed=int(config.seed),
        precomputed_latent_root=(
            validated_latents["root"] if validated_latents else None
        ),
        precomputed_latent_cache_size=int(latent_config.get("cache_size", 8)),
        precomputed_latent_metadata=(
            validated_latents["expected_file_metadata"]
            if validated_latents
            else None
        ),
        precomputed_latent_records=(
            validated_latents["manifest_records"] if validated_latents else None
        ),
    )
    logger.info(
        "Stage-1 dataset: NavAnywhere only, samples=%d, action_mode=%s, "
        "offset_range=[%d,%d]",
        len(dataset),
        action_mode,
        min_offset,
        max_offset,
    )
    if dataset.recipe_summary is not None:
        logger.info("NavAnywhere sampling recipe: %s", dataset.recipe_summary)
    return dataset


def create_dataloader(dataset, config: DictConfig, rank, is_train=True):
    """Create dataloader for training or testing."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=is_train,
        seed=config.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        collate_fn=(
            motion_condition_collate
            if bool(
                config.get("motion_condition", {}).get("enabled", False)
            )
            else None
        ),
    )

    return loader, sampler


def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res["rss"] += mmap.rss
        res["pss"] += mmap.pss
        res["uss"] += mmap.private_clean + mmap.private_dirty
        res["shared"] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith("/"):
            res["shared_file"] += mmap.shared_clean + mmap.shared_dirty
    return res


def find_dataloader_workers(parent_pid: int = None) -> list[int]:
    """Find PIDs of DataLoader worker processes."""
    if parent_pid is None:
        parent_pid = os.getpid()

    worker_pids = []
    try:
        parent = psutil.Process(parent_pid)
        all_children = parent.children(recursive=True)
        print(f"Parent PID {parent_pid} has {len(all_children)} child processes:")

        for child in all_children:
            try:
                child_name = child.name()
                child_cmdline = " ".join(child.cmdline())
                print(
                    f"  Child PID {child.pid}: name='{child_name}', cmdline='{child_cmdline}'"
                )

                # More flexible detection for DataLoader workers
                if (
                    "python" in child_name.lower()
                    or "dataloader" in child_cmdline.lower()
                    or child.pid != parent_pid
                ):  # Any child process
                    worker_pids.append(child.pid)
                    print("    -> Added as worker")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except psutil.NoSuchProcess:
        print(f"Parent process {parent_pid} not found")

    print(f"Found {len(worker_pids)} worker PIDs: {worker_pids}")
    return worker_pids


class MemoryMonitor:
    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    def analyze_high_memory_workers(self) -> str:
        """Analyze what's in high memory workers."""
        self._refresh()

        result = ["=== High Memory Worker Analysis ==="]
        high_mem_pids = []

        # Find high memory workers (USS > 5GB)
        for pid in self.pids:
            if pid in self.data and self.data[pid]["uss"] > 5 * 1024**3:
                high_mem_pids.append(pid)

        result.append(f"High memory workers: {high_mem_pids}")
        result.append("")

        # Analyze each high memory worker
        for pid in high_mem_pids:
            try:
                proc = psutil.Process(pid)
                result.append(f"--- Worker PID {pid} ---")
                result.append(f"Status: {proc.status()}")
                result.append(f"CPU %: {proc.cpu_percent():.1f}")
                result.append(f"Threads: {proc.num_threads()}")

                # Memory maps analysis
                maps = proc.memory_maps()
                heap_rss = sum(m.rss for m in maps if "[heap]" in m.path)
                anon_rss = sum(m.rss for m in maps if m.path == "")

                result.append(f"Heap memory: {self.format(heap_rss)}")
                result.append(f"Anonymous memory: {self.format(anon_rss)}")
                result.append(f"Total memory maps: {len(maps)}")
                result.append("")

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                result.append(f"PID {pid}: Error - {e}")

        return "\n".join(result)

    @staticmethod
    def format(size: int) -> str:
        for unit in ("", "K", "M", "G"):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)
