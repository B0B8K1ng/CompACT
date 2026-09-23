#!/usr/bin/env python3
"""Run LingBot-World v2 1.3B on the fixed RECON 4-second split.

This adapter intentionally lives outside the LingBot checkout.  It converts the
RECON planar trajectory into LingBot's OpenCV camera convention, loads one
pipeline for the whole run, and writes only the 4-second prediction expected by
the existing NWM metric script.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


DEFAULT_PROMPT = (
    "A first-person view from a mobile robot moving smoothly through the "
    "environment with a stable forward-facing camera."
)
DEFAULT_SPLIT = Path("data_splits/recon/test/time.pkl")
DEFAULT_DATA_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm/data/recon")
DEFAULT_OUTPUT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/"
    "lingbot_world_v2_1.3b_causal_fast/recon/time"
)
DEFAULT_LINGBOT_REPO = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/external_lingbot_eval/"
    "lingbot-world-v2"
)
DEFAULT_CHECKPOINT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/weights/"
    "lingbot-world-v2-1.3b-causal-fast"
)
DEFAULT_ASSETS = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/weights/"
    "lingbot-world-v2-14b-shared"
)
SOURCE_COMMIT = "1895d300d8ac936401689b26389f51cbd36530eb"
MODEL_REVISION = "7e36a5f919f86cb4255cc9bfc30adb44963fbde1"
ASSETS_REVISION = "5c33dd40b213598c418fd25bff30fdbd23fd38a7"


@dataclass(frozen=True)
class ReconSample:
    sample_id: int
    trajectory: str
    current_frame: int
    min_offset: int
    max_offset: int

    @property
    def target_frame(self) -> int:
        return self.current_frame + 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_split(path: Path, expected_count: int = 500) -> list[ReconSample]:
    with path.open("rb") as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list):
        raise TypeError(f"RECON split must be a list, got {type(rows).__name__}")
    if len(rows) != expected_count:
        raise ValueError(
            f"RECON split count changed: expected {expected_count}, got {len(rows)}"
        )
    samples: list[ReconSample] = []
    for sample_id, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 4:
            raise ValueError(f"invalid split row {sample_id}: {row!r}")
        trajectory, current, minimum, maximum = row
        sample = ReconSample(
            sample_id=sample_id,
            trajectory=str(trajectory),
            current_frame=int(current),
            min_offset=int(minimum),
            max_offset=int(maximum),
        )
        if sample.max_offset < 16:
            raise ValueError(
                f"sample {sample_id} cannot provide a 4-second target: {row!r}"
            )
        samples.append(sample)
    return samples


def select_samples(
    samples: Sequence[ReconSample], start_index: int, limit: int | None
) -> list[ReconSample]:
    if start_index < 0 or start_index >= len(samples):
        raise IndexError(f"start-index must be in [0, {len(samples)}), got {start_index}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    stop = len(samples) if limit is None else min(len(samples), start_index + limit)
    return list(samples[start_index:stop])


def sample_seed(base_seed: int, sample_id: int) -> int:
    if base_seed < 0 or sample_id < 0:
        raise ValueError("base seed and sample id must be non-negative")
    seed = base_seed + sample_id
    if seed >= 2**63:
        raise ValueError("derived seed exceeds torch.Generator's signed 64-bit range")
    return seed


def interpolate_planar_trajectory(
    position: np.ndarray, yaw: np.ndarray, output_frames: int = 65
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if position.ndim != 2 or position.shape[1] < 2:
        raise ValueError(f"position must have shape [N, >=2], got {position.shape}")
    if len(position) != len(yaw) or len(yaw) < 2:
        raise ValueError("position and yaw must have the same length >= 2")
    if output_frames < 2:
        raise ValueError("output_frames must be at least 2")
    if not np.isfinite(position[:, :2]).all() or not np.isfinite(yaw).all():
        raise ValueError("trajectory contains non-finite values")

    source_t = np.linspace(0.0, 1.0, len(yaw), dtype=np.float64)
    target_t = np.linspace(0.0, 1.0, output_frames, dtype=np.float64)
    xy = np.stack(
        [np.interp(target_t, source_t, position[:, axis]) for axis in range(2)],
        axis=1,
    )
    continuous_yaw = np.unwrap(yaw)
    yaw_interp = np.interp(target_t, source_t, continuous_yaw)
    return xy, yaw_interp


def planar_to_opencv_c2w(
    position: np.ndarray, yaw: np.ndarray, camera_height: float = 0.0
) -> np.ndarray:
    """Embed level planar poses as OpenCV camera-to-world matrices.

    OpenCV camera axes are x-right, y-down, z-forward.  RECON yaw defines the
    forward direction in the world XY plane.  A constant camera height cancels
    when LingBot converts the sequence to relative framewise poses.
    """
    position = np.asarray(position, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if position.shape != (len(yaw), 2):
        raise ValueError(
            f"position must have shape ({len(yaw)}, 2), got {position.shape}"
        )
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(yaw), axis=0)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # Columns of c2w rotation: camera right, camera down, camera forward.
    poses[:, :3, 0] = np.stack((sin_yaw, -cos_yaw, np.zeros_like(yaw)), axis=1)
    poses[:, :3, 1] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    poses[:, :3, 2] = np.stack((cos_yaw, sin_yaw, np.zeros_like(yaw)), axis=1)
    poses[:, 0, 3] = position[:, 0]
    poses[:, 1, 3] = position[:, 1]
    poses[:, 2, 3] = float(camera_height)
    return poses


def build_action_arrays(
    trajectory: dict[str, Any], frame_num: int = 65
) -> tuple[np.ndarray, np.ndarray]:
    xy, yaw = interpolate_planar_trajectory(
        np.asarray(trajectory["position"]),
        np.asarray(trajectory["yaw"]),
        output_frames=frame_num,
    )
    poses = planar_to_opencv_c2w(xy, yaw)
    # Virtual 832x480 calibration obtained by mapping a centered, square-pixel
    # 640x480 camera with 90-degree horizontal FOV into LingBot coordinates.
    intrinsics = np.repeat(
        np.array([[416.0, 320.0, 416.0, 240.0]], dtype=np.float32),
        frame_num,
        axis=0,
    )
    return poses, intrinsics


def load_sample_trajectory(data_root: Path, sample: ReconSample) -> dict[str, Any]:
    path = data_root / sample.trajectory / "traj_data.pkl"
    with path.open("rb") as stream:
        trajectory = pickle.load(stream)
    if not isinstance(trajectory, dict) or not {"position", "yaw"} <= trajectory.keys():
        raise ValueError(f"invalid trajectory payload: {path}")
    end = sample.target_frame + 1
    position = np.asarray(trajectory["position"])
    yaw = np.asarray(trajectory["yaw"])
    if end > len(position) or end > len(yaw):
        raise IndexError(
            f"sample {sample.sample_id} needs frames through {sample.target_frame}, "
            f"but {path} contains position={len(position)}, yaw={len(yaw)}"
        )
    return {
        "position": position[sample.current_frame:end],
        "yaw": yaw[sample.current_frame:end],
    }


def validate_inputs(args: argparse.Namespace, selected: Sequence[ReconSample]) -> None:
    required_files = (
        args.lingbot_repo / "wan" / "image2video.py",
        args.lingbot_repo / "SOURCE_COMMIT",
        args.checkpoint_dir / "transformers" / "model.safetensors.index.json",
        args.checkpoint_dir / "WEIGHTS_MANIFEST.json",
        args.assets_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        args.assets_dir / "Wan2.1_VAE.pth",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing LingBot inputs: {missing}")
    source_commit = (args.lingbot_repo / "SOURCE_COMMIT").read_text(
        encoding="utf-8"
    ).strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(
            f"LingBot source commit changed: expected {SOURCE_COMMIT}, got {source_commit}"
        )
    for sample in selected:
        image = args.data_root / sample.trajectory / f"{sample.current_frame}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        load_sample_trajectory(args.data_root, sample)


def valid_prediction(path: Path, output_size: int) -> bool:
    try:
        with Image.open(path) as image:
            return image.mode == "RGB" and image.size == (output_size, output_size)
    except (FileNotFoundError, OSError):
        return False


def save_prediction(video: Any, path: Path, output_size: int) -> tuple[int, int, int]:
    import torch

    if not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"unexpected LingBot output shape: {getattr(video, 'shape', None)}")
    if video.shape[1] != 65:
        raise ValueError(f"LingBot generated {video.shape[1]} frames instead of 65")
    frame = video[:, -1].detach().float().cpu().clamp(-1.0, 1.0)
    frame = frame.add(1.0).mul(127.5).round().to(torch.uint8)
    image = Image.fromarray(frame.permute(1, 2, 0).numpy(), mode="RGB")
    generated_size = image.size
    image = image.resize((output_size, output_size), resample=Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    image.save(temporary, format="PNG")
    temporary.replace(path)
    return video.shape[1], generated_size[0], generated_size[1]


def build_manifest(args: argparse.Namespace, selected: Sequence[ReconSample]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "updated_at": utc_now(),
        "model": {
            "name": "LingBot-World 2.0 1.3B causal-fast",
            "modelscope_checkpoint": str(args.checkpoint_dir.resolve()),
            "huggingface_verified_revision": MODEL_REVISION,
            "shared_assets": str(args.assets_dir.resolve()),
            "shared_assets_huggingface_verified_revision": ASSETS_REVISION,
            "source": str(args.lingbot_repo.resolve()),
            "source_commit": SOURCE_COMMIT,
            "inference_mode": "causal_fast",
            "sampling_steps": 4,
        },
        "protocol": {
            "name": "direct_4s_v1_lingbot_adapter",
            "dataset": "recon",
            "split": str(args.split.resolve()),
            "full_split_count": args.expected_count,
            "selected_sample_ids": [sample.sample_id for sample in selected],
            "input": "single current RGB frame at native 640x480 resolution",
            "target": "current_frame + 16 at 4 Hz (4 seconds)",
            "trajectory": "17 RECON SE(2) poses linearly/Slerp-angle interpolated to 65 poses at 16 Hz",
            "camera_convention": "OpenCV c2w; camera z forward, x right, y down",
            "camera_assumption": "level forward-facing camera; fixed height; body and camera forward aligned",
            "intrinsics_virtual_832x480_fx_fy_cx_cy": [416.0, 320.0, 416.0, 240.0],
            "intrinsics_assumption": "centered principal point, square-pixel native 640x480 camera, 90-degree horizontal FOV",
            "frame_num": args.frame_num,
            "chunk_size": args.chunk_size,
            "fps": 16,
            "max_area": args.max_area,
            "prompt": args.prompt,
            "base_seed": args.base_seed,
            "sample_seed": "base_seed + sample_id",
            "saved_frame": "last generated frame",
            "saved_resolution": [args.output_size, args.output_size],
            "resize": "Pillow bilinear direct resize (no crop)",
            "metrics": ["lpips_alex", "dreamsim", "psnr"],
        },
        "execution": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_id_within_visible_set": args.device_id,
            "pipeline_reused_across_samples": True,
            "t5_prompt_embedding_cache_reused": True,
            "resume_existing_valid_pngs": True,
        },
        "output_dir": str(args.output_dir.resolve()),
        "completed_count": 0,
        "completed_samples": [],
    }


def load_pipeline(args: argparse.Namespace) -> Any:
    sys.path.insert(0, str(args.lingbot_repo.resolve()))
    import torch
    import wan
    from wan.configs import WAN_CONFIGS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LingBot evaluation")
    torch.cuda.set_device(args.device_id)
    cfg = WAN_CONFIGS["i2v-1.3B"]
    return wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=str(args.checkpoint_dir),
        assets_dir=str(args.assets_dir),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
        local_attn_size=-1,
        sink_size=0,
        infer_mode="causal_fast",
    )


def run(args: argparse.Namespace) -> None:
    import torch

    samples = load_split(args.split, expected_count=args.expected_count)
    selected = select_samples(samples, args.start_index, args.limit)
    validate_inputs(args, selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir.parent / "run_manifest.json"
    manifest = build_manifest(args, selected)

    completed: list[dict[str, Any]] = []
    pending: list[ReconSample] = []
    for sample in selected:
        output = args.output_dir / f"id_{sample.sample_id}" / "4.png"
        if valid_prediction(output, args.output_size):
            completed.append(
                {
                    "sample_id": sample.sample_id,
                    "trajectory": sample.trajectory,
                    "current_frame": sample.current_frame,
                    "target_frame": sample.target_frame,
                    "seed": sample_seed(args.base_seed, sample.sample_id),
                    "output": str(output.resolve()),
                    "resumed": True,
                }
            )
        else:
            pending.append(sample)
    manifest["completed_samples"] = completed
    manifest["completed_count"] = len(completed)
    atomic_write_json(manifest_path, manifest)
    print(
        f"selected={len(selected)} completed={len(completed)} pending={len(pending)} "
        f"output={args.output_dir}",
        flush=True,
    )
    if not pending:
        manifest["status"] = "complete"
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        return

    pipe = load_pipeline(args)
    first_generation = True
    try:
        for ordinal, sample in enumerate(pending, start=1):
            seed = sample_seed(args.base_seed, sample.sample_id)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            source_image = (
                args.data_root / sample.trajectory / f"{sample.current_frame}.jpg"
            )
            with Image.open(source_image) as opened:
                image = opened.convert("RGB")
            if image.size != (640, 480):
                raise ValueError(
                    f"sample {sample.sample_id} input is {image.size}, expected (640, 480)"
                )
            trajectory = load_sample_trajectory(args.data_root, sample)
            poses, intrinsics = build_action_arrays(
                trajectory, frame_num=args.frame_num
            )
            with tempfile.TemporaryDirectory(prefix="lingbot_recon_action_") as action_dir:
                action_path = Path(action_dir)
                np.save(action_path / "poses.npy", poses)
                np.save(action_path / "intrinsics.npy", intrinsics)
                print(
                    f"[{ordinal}/{len(pending)}] sample={sample.sample_id} "
                    f"trajectory={sample.trajectory} frame={sample.current_frame} "
                    f"seed={seed}",
                    flush=True,
                )
                # On the first call, offload_model moves UMT5 back to CPU after
                # caching this fixed prompt.  It also moves DiT to CPU at the end;
                # restore DiT once, then keep it resident for the remaining calls.
                video = pipe.generate(
                    args.prompt,
                    image,
                    action_path=str(action_path),
                    chunk_size=args.chunk_size,
                    max_area=args.max_area,
                    frame_num=args.frame_num,
                    shift=5.0,
                    seed=seed,
                    offload_model=first_generation,
                    max_attention_size=None,
                )
            output = args.output_dir / f"id_{sample.sample_id}" / "4.png"
            frames, generated_width, generated_height = save_prediction(
                video, output, args.output_size
            )
            del video
            # The cache is per-video and is rebuilt at the start of every
            # generate() call. Drop it now to avoid holding the previous
            # 65-frame cache while allocating the next one.
            if hasattr(pipe, "self_kv_cache"):
                del pipe.self_kv_cache
            if first_generation:
                pipe.model.to(pipe.device)
                first_generation = False
            torch.cuda.empty_cache()
            completed.append(
                {
                    "sample_id": sample.sample_id,
                    "trajectory": sample.trajectory,
                    "current_frame": sample.current_frame,
                    "target_frame": sample.target_frame,
                    "seed": seed,
                    "output": str(output.resolve()),
                    "generated_frames": frames,
                    "generated_resolution": [generated_width, generated_height],
                    "resumed": False,
                }
            )
            completed.sort(key=lambda item: int(item["sample_id"]))
            manifest["completed_samples"] = completed
            manifest["completed_count"] = len(completed)
            manifest["updated_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        raise
    manifest["status"] = "complete"
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lingbot-repo", type=Path, default=DEFAULT_LINGBOT_REPO)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--frame-num", type=int, default=65)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=480 * 832)
    parser.add_argument("--output-size", type=int, default=224)
    args = parser.parse_args()
    if args.frame_num != 65 or args.chunk_size != 1:
        raise ValueError("the confirmed 4-second protocol requires frame-num=65, chunk-size=1")
    if args.output_size != 224:
        raise ValueError("the existing RECON metric protocol requires output-size=224")
    return args


if __name__ == "__main__":
    run(parse_args())
