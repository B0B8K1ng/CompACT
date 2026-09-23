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
    parser.add_argument("--inference-backend")
    parser.add_argument("--sampler")
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-ids-file", type=Path,
                        help="JSON array of original split positions to evaluate")
    parser.add_argument("--sample-ids-by-frame-file", type=Path,
                        help="JSON mapping from frame label to original split positions")
    parser.add_argument("--reuse-frame-metrics-file", type=Path,
                        help="Verified per-frame metrics to include without recomputing")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--split-sha256")
    args = parser.parse_args()
    inference_fields = (
        args.inference_backend,
        args.sampler,
        args.sampling_steps,
        args.seed,
    )
    if any(value is not None for value in inference_fields) and not all(
        value is not None for value in inference_fields
    ):
        raise ValueError(
            "inference provenance requires --inference-backend, --sampler, "
            "--sampling-steps, and --seed together"
        )

    gt_samples = sample_directories(args.gt_dir)
    pred_samples = sample_directories(args.pred_dir)
    if args.sample_ids_file is not None and args.sample_ids_by_frame_file is not None:
        raise ValueError("Choose one sample ID filter")
    by_frame = None
    if args.sample_ids_by_frame_file is not None:
        by_frame = json.loads(args.sample_ids_by_frame_file.read_text(encoding="utf-8"))
        if set(by_frame) != set(args.frames):
            raise ValueError("Per-frame sample ID labels must match --frames")
    if args.sample_ids_file is not None:
        requested = json.loads(args.sample_ids_file.read_text(encoding="utf-8"))
        if not isinstance(requested, list) or len(requested) != len(set(requested)) or not requested:
            raise ValueError("--sample-ids-file must contain a nonempty unique JSON array")
        wanted = {f"id_{int(value)}" for value in requested}
        gt_samples = [path for path in gt_samples if path.name in wanted]
        pred_samples = [path for path in pred_samples if path.name in wanted]
        if len(gt_samples) != len(wanted):
            raise ValueError("ground truth does not contain every requested sample ID")
    gt_names = [path.name for path in gt_samples]
    pred_names = [path.name for path in pred_samples]
    if by_frame is None and gt_names != pred_names:
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

    reused = (json.loads(args.reuse_frame_metrics_file.read_text(encoding="utf-8"))
              if args.reuse_frame_metrics_file is not None else {})
    if not isinstance(reused, dict) or not set(reused).issubset(args.frames):
        raise ValueError("Reused frame labels must be a subset of --frames")
    metrics: dict[str, dict[str, float | int]] = dict(reused)
    with torch.inference_mode():
        for label, frame_index in args.frames.items():
            if label in reused:
                continue
            if by_frame is not None:
                values = by_frame[label]
                if not isinstance(values, list) or not values or len(values) != len(set(values)):
                    raise ValueError(f"Invalid sample IDs for {label}")
                wanted = {f"id_{int(value)}" for value in values}
                selected_gt = [path for path in gt_samples if path.name in wanted]
                selected_pred = [path for path in pred_samples if path.name in wanted]
                if len(selected_gt) != len(wanted) or [p.name for p in selected_gt] != [p.name for p in selected_pred]:
                    raise ValueError(f"GT/prediction sample mismatch for {label}")
            else:
                selected_gt, selected_pred = gt_samples, pred_samples
            lpips_sum = 0.0
            dreamsim_sum = 0.0
            psnr_sum = 0.0
            sample_count = 0
            fid_metric = (
                FrechetInceptionDistance(feature_dim=2048).to(device)
                if args.fid
                else None
            )

            for start in range(0, len(selected_gt), args.batch_size):
                gt_batch: list[torch.Tensor] = []
                pred_batch: list[torch.Tensor] = []
                gt_dreamsim: list[torch.Tensor] = []
                pred_dreamsim: list[torch.Tensor] = []
                for gt_sample, pred_sample in zip(
                    selected_gt[start : start + args.batch_size],
                    selected_pred[start : start + args.batch_size],
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
        "sample_count": len(gt_samples) if by_frame is None else None,
        "sample_order": "natural_sorted",
        "gt_eval_dir": str(args.gt_dir.resolve()),
        "pred_eval_dir": str(args.pred_dir.resolve()),
        "device": str(device),
        "frame_indices": args.frames,
        "sample_ids": requested if args.sample_ids_file is not None else None,
        "sample_ids_by_frame": by_frame,
        "aggregation": {
            "lpips_alex": "arithmetic mean of per-sample distances",
            "dreamsim": "arithmetic mean of per-sample distances",
            "psnr": "arithmetic mean of per-sample RGB PSNR (dB), data_range=1",
            "fid": "global Inception feature statistics" if args.fid else "not computed",
        },
        "metrics": metrics,
        "reused_metric_frames": (str(args.reuse_frame_metrics_file) if reused else None),
    }
    if all(value is not None for value in inference_fields):
        result["inference"] = {
            "backend": args.inference_backend,
            "sampler": args.sampler,
            "sampling_steps": args.sampling_steps,
            "seed": args.seed,
            "checkpoint_sha256": args.checkpoint_sha256,
            "split_sha256": args.split_sha256,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
