#!/usr/bin/env python3
"""Evaluate NWM image predictions with sample-weighted perceptual metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lpips
import torch
from dreamsim import dreamsim
from PIL import Image
from torcheval.metrics import FrechetInceptionDistance
from torchvision.transforms.functional import pil_to_tensor


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def sample_directories(root: Path) -> list[Path]:
    """Return one directory per sample name in deterministic natural order.

    The NAS can transiently return the same directory entry more than once after
    a large distributed write.  Those entries have the same name/path and must
    not turn an otherwise identical GT/prediction sample set into a false count
    mismatch.
    """
    samples = {
        path.name: path
        for path in root.iterdir()
        if path.is_dir()
    }
    return sorted(samples.values(), key=natural_key)


def parse_frames(value: str) -> dict[str, int]:
    frames: dict[str, int] = {}
    for item in value.split(","):
        label, index = item.split(":", maxsplit=1)
        frames[label] = int(index)
    if not frames:
        raise argparse.ArgumentTypeError("at least one label:index frame is required")
    return frames


def load_rgb(path: Path) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    tensor = pil_to_tensor(image).float().div_(255.0)
    return image, tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=parse_frames, required=True)
    parser.add_argument("--dataset", default="recon")
    parser.add_argument("--eval-type", choices=("time", "rollout"), required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--rollout-fps", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dreamsim-cache", type=Path, required=True)
    parser.add_argument("--fid", action="store_true")
    args = parser.parse_args()

    gt_samples = sample_directories(args.gt_dir)
    pred_samples = sample_directories(args.pred_dir)
    gt_names = [path.name for path in gt_samples]
    pred_names = [path.name for path in pred_samples]
    if gt_names != pred_names:
        missing = sorted(set(gt_names) - set(pred_names), key=natural_key)
        extra = sorted(set(pred_names) - set(gt_names), key=natural_key)
        raise ValueError(f"sample mismatch: missing={missing[:10]}, extra={extra[:10]}")

    device = torch.device(args.device)
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    dreamsim_model, dreamsim_preprocess = dreamsim(
        pretrained=True,
        device=device,
        cache_dir=str(args.dreamsim_cache),
    )
    dreamsim_model.eval()

    metrics: dict[str, dict[str, float | int]] = {}
    with torch.inference_mode():
        for label, frame_index in args.frames.items():
            lpips_sum = 0.0
            dreamsim_sum = 0.0
            psnr_sum = 0.0
            sample_count = 0
            fid_metric = (
                FrechetInceptionDistance(feature_dim=2048).to(device)
                if args.fid
                else None
            )

            for start in range(0, len(gt_samples), args.batch_size):
                gt_batch: list[torch.Tensor] = []
                pred_batch: list[torch.Tensor] = []
                gt_dreamsim: list[torch.Tensor] = []
                pred_dreamsim: list[torch.Tensor] = []
                for gt_sample, pred_sample in zip(
                    gt_samples[start : start + args.batch_size],
                    pred_samples[start : start + args.batch_size],
                ):
                    gt_image, gt_tensor = load_rgb(gt_sample / f"{frame_index}.png")
                    pred_image, pred_tensor = load_rgb(pred_sample / f"{frame_index}.png")
                    if gt_tensor.shape != pred_tensor.shape:
                        raise ValueError(
                            f"shape mismatch for {gt_sample.name}/{frame_index}.png: "
                            f"{tuple(gt_tensor.shape)} != {tuple(pred_tensor.shape)}"
                        )
                    gt_batch.append(gt_tensor)
                    pred_batch.append(pred_tensor)
                    gt_dreamsim.append(dreamsim_preprocess(gt_image))
                    pred_dreamsim.append(dreamsim_preprocess(pred_image))

                gt = torch.stack(gt_batch).to(device)
                pred = torch.stack(pred_batch).to(device)
                lpips_values = lpips_model(gt.mul(2).sub(1), pred.mul(2).sub(1)).flatten()
                dreamsim_values = dreamsim_model(
                    torch.cat(gt_dreamsim).to(device),
                    torch.cat(pred_dreamsim).to(device),
                ).flatten()
                mse = (gt - pred).square().flatten(1).mean(1)
                psnr_values = 10.0 * torch.log10(1.0 / mse.clamp_min(torch.finfo(mse.dtype).tiny))

                lpips_sum += lpips_values.double().sum().item()
                dreamsim_sum += dreamsim_values.double().sum().item()
                psnr_sum += psnr_values.double().sum().item()
                sample_count += gt.shape[0]
                if fid_metric is not None:
                    fid_metric.update(gt, is_real=True)
                    fid_metric.update(pred, is_real=False)

            label_metrics: dict[str, float | int] = {
                "sample_count": sample_count,
                "lpips_alex": lpips_sum / sample_count,
                "dreamsim": dreamsim_sum / sample_count,
                "psnr": psnr_sum / sample_count,
            }
            if fid_metric is not None:
                label_metrics["fid"] = float(fid_metric.compute().item())
            metrics[label] = label_metrics

    result = {
        "dataset": args.dataset,
        "eval_type": args.eval_type,
        "eval_name": args.eval_name,
        "rollout_fps": args.rollout_fps,
        "sample_count": len(gt_samples),
        "sample_order": "natural_sorted",
        "gt_eval_dir": str(args.gt_dir.resolve()),
        "pred_eval_dir": str(args.pred_dir.resolve()),
        "device": str(device),
        "frame_indices": args.frames,
        "aggregation": {
            "lpips_alex": "arithmetic mean of per-sample distances",
            "dreamsim": "arithmetic mean of per-sample distances",
            "psnr": "arithmetic mean of per-sample RGB PSNR (dB), data_range=1",
            "fid": "global Inception feature statistics" if args.fid else "not computed",
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
