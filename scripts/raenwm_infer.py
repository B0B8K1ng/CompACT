#!/usr/bin/env python3
"""Run official RAE-NWM weights on CompACT's pinned NWM evaluation samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_reproducibility import samplewise_randn


EXPECTED_SOURCE_REVISION = "0219ce41c44d515f86719dd763c1efe7c7f72519"
HORIZONS = (1, 2, 4, 8, 16)
INPUT_FPS = 4
DIRECT_SAMPLE_COUNT = 500
ROLLOUT_SAMPLE_COUNT = 150
DATASET_LAYOUTS = {
    "recon": {"data": "recon", "split": "recon", "loader": "recon"},
    "scand": {"data": "scand", "split": "scand", "loader": "scand"},
    "huron": {"data": "sacson", "split": "sacson", "loader": "sacson"},
    "tartan_drive": {
        "data": "tartan",
        "split": "tartan_drive",
        "loader": "recon",
    },
    "go_stanford": {
        "data": "go_stanford",
        "split": "go_stanford",
        "loader": "go_stanford",
    },
    "planetary_rover": {
        "data": "planetary_rover",
        "split": "planetary_rover",
        "loader": "recon",
    },
    "unitree_go2": {
        "data": "unitree_go2",
        "split": "unitree_go2",
        "loader": "recon",
    },
    "tum_rgbd": {"data": "tum_rgbd", "split": "tum_rgbd", "loader": "recon"},
    "uzh_fpv": {"data": "uzh_fpv", "split": "uzh_fpv", "loader": "recon"},
}
WAYPOINT_SPACING = {
    "recon": 0.25,
    "scand": 0.38,
    "huron": 0.255,
    "tartan_drive": 0.72,
    "go_stanford": 0.12,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(source: Path) -> str:
    head = source / ".git/HEAD"
    if not head.is_file():
        raise FileNotFoundError(f"RAE-NWM source is not a Git checkout: {source}")
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def compose_se2(delta: Any) -> Any:
    """Compose body-frame (dx, dy, dtheta) deltas as in official RAE-NWM."""
    import torch

    if delta.ndim != 3 or delta.shape[-1] < 3:
        raise ValueError(f"Expected [B,T,D>=3] deltas, got {tuple(delta.shape)}")
    x = torch.zeros(delta.shape[0], device=delta.device, dtype=delta.dtype)
    y = torch.zeros_like(x)
    theta = torch.zeros_like(x)
    for step in range(delta.shape[1]):
        dx = delta[:, step, 0]
        dy = delta[:, step, 1]
        dtheta = delta[:, step, 2]
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        x = x + cosine * dx - sine * dy
        y = y + sine * dx + cosine * dy
        theta = theta + dtheta
    theta = theta - 2.0 * math.pi * torch.floor((theta + math.pi) / (2.0 * math.pi))
    result = torch.stack((x, y, theta), dim=-1)
    if delta.shape[-1] > 3:
        result = torch.cat((result, delta[..., 3:].sum(dim=1)), dim=-1)
    return result


def save_image(path: Path, image: Any) -> None:
    import numpy as np
    from PIL import Image

    array = (
        image.detach().float().cpu().nan_to_num().clamp(0, 1).permute(1, 2, 0).numpy()
    )
    Image.fromarray((array * 255).astype(np.uint8), mode="RGB").save(path)


def configure_imports(source: Path) -> None:
    source = source.resolve()
    rae_source = source / "RAE/src"
    for path in (str(rae_source), str(source)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    os.chdir(source)


def distributed_context(seed: int) -> tuple[int, int, Any]:
    import numpy as np
    import torch
    import torch.distributed as torch_dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch_dist.init_process_group(backend="nccl", init_method="env://")
    process_seed = seed * world_size + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    return world_size, rank, torch.device("cuda", local_rank)


def build_dataset(
    args: argparse.Namespace, dataset_name: str, official_misc: Any, dataset_cls: Any
) -> Any:
    if dataset_name not in DATASET_LAYOUTS:
        raise KeyError(f"Unknown benchmark dataset: {dataset_name}")
    layout = DATASET_LAYOUTS[dataset_name]
    split_root = args.project_root / "data_splits" / layout["split"] / "test"
    split = split_root / f"{args.eval_type}.pkl"
    if not split.is_file():
        raise FileNotFoundError(split)
    spacing = waypoint_spacing(args, dataset_name)

    # Official RAE-NWM reads its own config/data_config.yaml during construction.
    # Unknown OOD names use a known placeholder only for that lookup; explicit
    # paths and the measured spacing below remain the actual dataset contract.
    dataset = dataset_cls(
        data_folder=str(args.data_root / layout["data"]),
        data_split_folder=str(split_root),
        dataset_name=layout["loader"],
        image_size=224,
        min_dist_cat=-64,
        max_dist_cat=64,
        len_traj_pred=args.future_frames,
        traj_stride=8,
        context_size=4,
        normalize=True,
        transform=official_misc.transform,
        goals_per_obs=4,
        predefined_index=str(split),
        traj_names=(
            "rollout_traj_names.txt"
            if args.eval_type == "rollout"
            and (split_root / "rollout_traj_names.txt").is_file()
            else "traj_names.txt"
        ),
    )
    dataset.dataset_name = dataset_name
    dataset.data_config = {"metric_waypoint_spacing": spacing}
    expected_count = (
        DIRECT_SAMPLE_COUNT if args.eval_type == "time" else ROLLOUT_SAMPLE_COUNT
    )
    if len(dataset) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} samples in {split}, found {len(dataset)}"
        )
    return dataset


def waypoint_spacing(args: argparse.Namespace, dataset_name: str) -> float:
    if dataset_name in WAYPOINT_SPACING:
        return float(WAYPOINT_SPACING[dataset_name])
    layout = DATASET_LAYOUTS[dataset_name]
    config_path = args.data_root / layout["data"] / "dataset_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No waypoint spacing for {dataset_name}; expected {config_path}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    spacing = float(payload["metric_waypoint_spacing"])
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Invalid waypoint spacing for {dataset_name}: {spacing}")
    return spacing


def build_models(args: argparse.Namespace, device: Any) -> tuple[Any, Any, Any]:
    import torch
    from omegaconf import OmegaConf
    from RAE.src.stage1.rae import RAE
    from RAE.src.stage2.transport.transport import (
        ModelType,
        PathType,
        Sampler,
        Transport,
        WeightType,
    )
    from RAE.src.utils.model_utils import instantiate_from_config

    from models import CDiT_models

    config = OmegaConf.load(args.source / "config/raenwm.yaml")
    rae_config = OmegaConf.load(
        args.source / "RAE/configs/stage1/pretrained/DINOv2-B.yaml"
    )["stage_1"]
    rae_config.params.encoder_config_path = str(args.dino_model)
    rae_config.params.encoder_params.dinov2_path = str(args.dino_model)
    rae_config.params.decoder_config_path = str(
        args.source / "RAE/configs/decoder/ViTXL"
    )
    rae_config.params.pretrained_decoder_path = str(args.decoder)
    rae_config.params.normalization_stat_path = str(args.normalization_stats)
    rae: RAE = instantiate_from_config(rae_config).to(device).eval()

    latent_size = int(config.image_size) // 14
    model = CDiT_models[str(config.model)](
        context_size=int(config.context_size),
        input_size=latent_size,
        in_channels=rae.latent_dim,
        learn_sigma=bool(config.get("learn_sigma", False)),
        head_width=int(config.get("head_width", rae.latent_dim)),
        head_depth=int(config.get("head_depth", 2)),
        head_num_heads=int(config.get("head_num_heads", 16)),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "ema" not in checkpoint:
        raise KeyError(f"Checkpoint has no EMA weights: {args.checkpoint}")
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["ema"].items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "RAE-NWM checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval().to(device)
    if args.compile:
        model = torch.compile(model)

    transport_config = config.get("transport", {})
    shift_dimension = int(rae.latent_dim) * latent_size * latent_size
    shift_base = float(transport_config.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(float(shift_dimension) / shift_base)
    if transport_config.get("time_dist_shift") is not None:
        time_dist_shift = float(transport_config.time_dist_shift)
    if bool(transport_config.get("time_dist_shift_disable", False)):
        time_dist_shift = 1.0
    transport = Transport(
        model_type=getattr(
            ModelType, str(transport_config.get("model_type", "velocity")).upper()
        ),
        path_type=getattr(
            PathType, str(transport_config.get("path_type", "linear")).upper()
        ),
        loss_type=getattr(
            WeightType, str(transport_config.get("loss_type", "velocity")).upper()
        ),
        time_dist_type=str(transport_config.get("time_dist_type", "uniform")),
        time_dist_shift=time_dist_shift,
        train_eps=1e-3,
        sample_eps=1e-3,
    )
    return model, Sampler(transport), rae


def sample_latent(
    model: Any,
    sampler: Any,
    rae: Any,
    conditioning: Any,
    action: Any,
    relative_frames: int,
    sample_ids: Any,
    stream: str,
    args: argparse.Namespace,
    device: Any,
) -> Any:
    import torch

    conditioning = conditioning.to(device)
    action = action.to(device)
    batch = conditioning.shape[0]
    latent_size = conditioning.shape[-1]
    noise = samplewise_randn(
        sample_ids,
        (rae.latent_dim, latent_size, latent_size),
        base_seed=args.seed,
        stream=stream,
        device=device,
    )
    relative_time = torch.full((batch,), relative_frames / 128.0, device=device)
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,
        num_steps=args.num_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    )
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return sample_fn(
            noise,
            model,
            y=action,
            x_cond=conditioning,
            rel_t=relative_time,
        )[-1]


def decode_latents(rae: Any, latents: Any, target_size: tuple[int, int]) -> Any:
    import torch

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        decoded = rae.decode(latents).float().nan_to_num().clamp(0, 1)
    if decoded.shape[-2:] != target_size:
        decoded = torch.nn.functional.interpolate(
            decoded,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
    return decoded


def predict_horizon(
    model: Any,
    sampler: Any,
    rae: Any,
    observations: Any,
    action: Any,
    horizon: int,
    sample_ids: Any,
    dataset_name: str,
    args: argparse.Namespace,
    device: Any,
) -> Any:
    import torch

    observations = observations.to(device)
    batch, context = observations.shape[:2]
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pixels = observations.flatten(0, 1) * 0.5 + 0.5
        conditioning = rae.encode(pixels).unflatten(0, (batch, context))[:, :4]
    samples = sample_latent(
        model,
        sampler,
        rae,
        conditioning,
        action,
        horizon * INPUT_FPS,
        sample_ids,
        f"{dataset_name}/time/{horizon}s",
        args,
        device,
    )
    return decode_latents(rae, samples, tuple(observations.shape[-2:]))


def predict_rollout(
    model: Any,
    sampler: Any,
    rae: Any,
    observations: Any,
    deltas: Any,
    sample_ids: Any,
    dataset_name: str,
    rollout_fps: int,
    output: Path,
    args: argparse.Namespace,
    device: Any,
) -> None:
    import torch

    if INPUT_FPS % rollout_fps:
        raise ValueError(f"rollout_fps={rollout_fps} must divide {INPUT_FPS}")
    stride = INPUT_FPS // rollout_fps
    grouped = deltas[:, : args.future_frames].unflatten(1, (-1, stride))
    batch, steps = grouped.shape[:2]
    actions = compose_se2(grouped.flatten(0, 1)).unflatten(0, (batch, steps))
    observations = observations.to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pixels = observations.flatten(0, 1) * 0.5 + 0.5
        current = rae.encode(pixels).unflatten(
            0, (observations.shape[0], observations.shape[1])
        )
    for step in range(steps):
        prediction = sample_latent(
            model,
            sampler,
            rae,
            current[:, -4:],
            actions[:, step],
            stride,
            sample_ids,
            f"{dataset_name}/rollout_{rollout_fps}fps/step={step}",
            args,
            device,
        )
        decoded = decode_latents(rae, prediction, tuple(observations.shape[-2:]))
        for offset, sample_index in enumerate(sample_ids.reshape(-1)):
            sample = output / f"id_{int(sample_index.item())}"
            sample.mkdir(parents=True, exist_ok=True)
            save_image(sample / f"{step}.png", decoded[offset])
        current = torch.cat((current[:, 1:], prediction.unsqueeze(1)), dim=1)


def incomplete_indices(output: Path, indices: Any, force: bool, frames: Any) -> list[int]:
    if force:
        return list(range(len(indices)))
    keep: list[int] = []
    for offset, value in enumerate(indices.view(-1)):
        sample = output / f"id_{int(value.item())}"
        if not all((sample / f"{frame}.png").is_file() for frame in frames):
            keep.append(offset)
    return keep


def write_manifest(args: argparse.Namespace, world_size: int) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    expected_count = (
        DIRECT_SAMPLE_COUNT if args.eval_type == "time" else ROLLOUT_SAMPLE_COUNT
    )
    dataset_manifest = {}
    for name in args.datasets:
        layout = DATASET_LAYOUTS[name]
        split = (
            args.project_root
            / "data_splits"
            / layout["split"]
            / "test"
            / f"{args.eval_type}.pkl"
        )
        dataset_manifest[name] = {
            "sample_count": expected_count,
            "split": str(split.resolve()),
            "split_sha256": sha256_file(split),
            "metric_waypoint_spacing": waypoint_spacing(args, name),
        }
    output = args.output_root / f"{args.eval_type}_inference_manifest.json"
    previous_datasets: dict[str, Any] = {}
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        previous_datasets = previous.get("datasets", {})
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "rae-nwm",
        "source": {
            "repository": "https://github.com/20robo/raenwm",
            "revision": source_revision(args.source),
            "path": str(args.source.resolve()),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "decoder": {
            "path": str(args.decoder.resolve()),
            "sha256": sha256_file(args.decoder),
        },
        "normalization_stats": {
            "path": str(args.normalization_stats.resolve()),
            "sha256": sha256_file(args.normalization_stats),
        },
        "dinov2_model": str(args.dino_model.resolve()),
        "datasets": {**previous_datasets, **dataset_manifest},
        "protocol": {
            "type": (
                "one-shot direct visual prediction"
                if args.eval_type == "time"
                else "autoregressive visual rollout"
            ),
            "horizons_seconds": (
                list(args.horizons) if args.eval_type == "time" else [1, 2, 4, 8, 16]
            ),
            "rollout_fps": (
                list(args.rollout_fps) if args.eval_type == "rollout" else None
            ),
            "future_frames": args.future_frames,
            "input_fps": INPUT_FPS,
            "action_stats": {"min": [-64, -64], "max": [64, 64]},
            "action_composition": "SE(2), matching official RAE-NWM inference",
            "sampling_method": args.sampling_method,
            "sampling_steps": args.num_steps,
            "seed": args.seed,
            "noise_policy": "sha256(base_seed,dataset,evaluation,sample_id,step)",
            "distributed_world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "max_samples": args.max_samples,
            "compile": args.compile,
        },
    }
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--eval-type", choices=("time", "rollout"), default="time")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--rollout-fps", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--future-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sampling-method", default="euler")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit each dataset to its first N split entries (for smoke tests only).",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.horizons = tuple(args.horizons)
    if args.eval_type == "time" and (
        not args.horizons
        or any(horizon <= 0 for horizon in args.horizons)
        or tuple(sorted(set(args.horizons))) != args.horizons
    ):
        raise ValueError("--horizons must be positive, unique, and increasing")
    if args.eval_type == "time" and args.future_frames < max(args.horizons) * INPUT_FPS:
        raise ValueError(
            f"--future-frames={args.future_frames} is shorter than the requested "
            f"{max(args.horizons)}s horizon at {INPUT_FPS} Hz"
        )
    if args.eval_type == "rollout":
        if sorted(set(args.rollout_fps)) != args.rollout_fps:
            raise ValueError("--rollout-fps must be unique and increasing")
        if any(fps <= 0 or INPUT_FPS % fps for fps in args.rollout_fps):
            raise ValueError(f"--rollout-fps values must divide {INPUT_FPS}")
        if args.future_frames != 64:
            raise ValueError("The registered rollout protocol requires 64 future frames")
    args.source = args.source.resolve()
    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    required = (
        args.checkpoint,
        args.decoder,
        args.normalization_stats,
        args.dino_model / "model.safetensors",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RAE-NWM assets: {missing}")
    revision = source_revision(args.source)
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError(
            f"RAE-NWM source revision mismatch: expected {EXPECTED_SOURCE_REVISION}, got {revision}"
        )
    configure_imports(args.source)

    import torch
    import torch.distributed as torch_dist

    import misc as official_misc
    from datasets import EvalDataset

    world_size, rank, device = distributed_context(args.seed)
    model, sampler, rae = build_models(args, device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        dataset = build_dataset(args, dataset_name, official_misc, EvalDataset)
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("--max-samples must be positive")
            dataset = torch.utils.data.Subset(
                dataset, range(min(args.max_samples, len(dataset)))
            )
        distributed_sampler = list(range(rank, len(dataset), world_size))
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=distributed_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        for batch_index, (indices, observations, _, deltas) in enumerate(loader):
            torch.cuda.reset_peak_memory_stats(device)
            generated = 0
            if args.eval_type == "time":
                output = args.output_root / dataset_name / "time"
                output.mkdir(parents=True, exist_ok=True)
                keep = incomplete_indices(output, indices, args.force, args.horizons)
                if keep:
                    select = torch.as_tensor(keep, dtype=torch.long)
                    selected_indices = indices.index_select(0, select)
                    selected_observations = observations.index_select(0, select)
                    selected_deltas = deltas.index_select(0, select)
                    for horizon in args.horizons:
                        action = compose_se2(
                            selected_deltas[:, : horizon * INPUT_FPS]
                        )
                        predictions = predict_horizon(
                            model,
                            sampler,
                            rae,
                            selected_observations,
                            action,
                            horizon,
                            selected_indices,
                            dataset_name,
                            args,
                            device,
                        )
                        for offset, sample_index in enumerate(
                            selected_indices.view(-1)
                        ):
                            sample = output / f"id_{int(sample_index.item())}"
                            sample.mkdir(parents=True, exist_ok=True)
                            save_image(sample / f"{horizon}.png", predictions[offset])
                    generated += len(keep)
            else:
                for rollout_fps in args.rollout_fps:
                    output = (
                        args.output_root
                        / dataset_name
                        / f"rollout_{rollout_fps}fps"
                    )
                    output.mkdir(parents=True, exist_ok=True)
                    frame_count = args.future_frames * rollout_fps // INPUT_FPS
                    keep = incomplete_indices(
                        output, indices, args.force, range(frame_count)
                    )
                    if not keep:
                        continue
                    select = torch.as_tensor(keep, dtype=torch.long)
                    selected_indices = indices.index_select(0, select)
                    predict_rollout(
                        model,
                        sampler,
                        rae,
                        observations.index_select(0, select),
                        deltas.index_select(0, select),
                        selected_indices,
                        dataset_name,
                        rollout_fps,
                        output,
                        args,
                        device,
                    )
                    generated += len(keep)
            print(
                f"[rae-nwm] rank={rank} dataset={dataset_name} "
                f"eval_type={args.eval_type} batch={batch_index + 1}/{len(loader)} "
                f"generated={generated} "
                f"peak_memory_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}",
                flush=True,
            )

    if world_size > 1:
        torch_dist.barrier()
    write_manifest(args, world_size)
    if world_size > 1:
        torch_dist.barrier()
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
