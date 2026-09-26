#!/usr/bin/env python3
"""Evaluate RAE-NWM with CompACT's fixed CEM80 navigation protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.raenwm_infer import (
    DATASET_LAYOUTS,
    EXPECTED_SOURCE_REVISION,
    build_models,
    configure_imports,
    sha256_file,
    source_revision,
)
from scripts.benchmark_reproducibility import samplewise_randn


DEFAULT_EXPECTED_SAMPLE_COUNT = 100
EXPECTED_SPLIT_SHA256 = {
    "recon": "c62cd08be9f124cbeec48d914460da8630e089bf0bdb84c5018013a82d12ec54",
    "scand": "8acb4062561cbf1549e27f39a6e80241b97294a8c0e55c345787ae6a47be55ce",
    "huron": "89e07bb2b934d7fe4e8ab0bddf51e28a4beb36d2415ad83c166350241ef4e9fd",
    "tartan_drive": "77bc38808b8df24b330fc4f9a4a17ed0de35a1c1bef0ff2283fc461b3ab16435",
    "go_stanford": "5013d8e2defbbee4d9652f7ae4569816113a8d44dd2af8b5e135d9f5cea3e27c",
    "planetary_rover": "17a83bbc5994d30700a9d4c181b0205e5771066a3f5d547190d41bf1ee2ba54b",
}
HORIZON_STEPS = 8
CONTEXT_SIZE = 4
LOCAL_ACTION_STATS = {
    "min": torch.tensor([-2.5, -4.0]),
    "max": torch.tensor([5.0, 4.0]),
}
RAENWM_ACTION_STATS = {
    "min": torch.tensor([-64.0, -64.0]),
    "max": torch.tensor([64.0, 64.0]),
}
EVAL_WAYPOINT_SPACING = {
    "recon": 0.25,
    "scand": 0.38,
    "huron": 0.255,
    "tartan_drive": 0.72,
    "go_stanford": 0.12,
    "planetary_rover": 1.0,
}
RAENWM_WAYPOINT_SPACING = {
    "recon": 0.25,
    "scand": 0.36,
    "huron": 0.255,
    "tartan_drive": 0.72,
    "go_stanford": 0.12,
    "planetary_rover": 1.0,
}
PLAN_DISTRIBUTIONS = {
    "recon": {"mu": [-0.1, 0.0, 0.0], "sigma": [0.02, 0.1, 0.1]},
    "scand": {"mu": [-0.25, 0.0, 0.0], "sigma": [0.04, 0.1, 0.1]},
    "huron": {"mu": [-0.33, 0.0, 0.0], "sigma": [0.03, 0.1, 0.1]},
    "tartan_drive": {"mu": [0.5, 0.0, 0.0], "sigma": [0.07, 0.1, 0.1]},
    "go_stanford": {"mu": [-0.1, 0.0, 0.0], "sigma": [0.1, 0.15, 0.1]},
    "planetary_rover": {"mu": [-0.1, 0.0, 0.0], "sigma": [0.1, 0.15, 0.1]},
}
OOD_PLAN_DISTRIBUTION = {
    "mu": [-0.1, 0.0, 0.0],
    "sigma": [0.1, 0.15, 0.1],
}
RESULT_STEM = "CEM_N80_K5_RS1_rep3_OPT1_COST-lpips-RECON-True"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_data(data: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    minimum = stats["min"].to(data)
    maximum = stats["max"].to(data)
    return (data - minimum) / (maximum - minimum) * 2.0 - 1.0


def unnormalize_data(data: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    minimum = stats["min"].to(data)
    maximum = stats["max"].to(data)
    return (data + 1.0) * 0.5 * (maximum - minimum) + minimum


def calculate_delta_yaw(unnormalized_xy: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(unnormalized_xy[..., 1], unnormalized_xy[..., 0]).unsqueeze(-1)
    previous = torch.cat((torch.zeros_like(yaw[:, :1]), yaw), dim=1)
    return previous[:, 1:] - previous[:, :-1]


def local_deltas_to_raenwm(
    local_normalized_xy: torch.Tensor, dataset_name: str, terminal_yaw: torch.Tensor
) -> torch.Tensor:
    """Convert the shared planner trajectory to RAE-NWM's training coordinates."""
    local_units = unnormalize_data(local_normalized_xy, LOCAL_ACTION_STATS)
    metric_xy = local_units * EVAL_WAYPOINT_SPACING[dataset_name]
    raenwm_units = metric_xy / RAENWM_WAYPOINT_SPACING[dataset_name]
    raenwm_xy = normalize_data(raenwm_units, RAENWM_ACTION_STATS)
    delta_yaw = calculate_delta_yaw(local_units)
    delta_yaw[:, -1, 0] += terminal_yaw * math.pi
    return torch.cat((raenwm_xy, delta_yaw), dim=-1)


def trajectory_metrics(
    ground_truth: torch.Tensor, prediction: torch.Tensor
) -> tuple[float, float]:
    """Match EVO's unaligned translation APE and frame-delta RPE for identity poses."""
    ground_truth = ground_truth.to(dtype=torch.float64, device="cpu")
    prediction = prediction.to(dtype=torch.float64, device="cpu")
    ate = (prediction - ground_truth).square().sum(dim=-1).mean().sqrt()
    gt_steps = ground_truth[1:] - ground_truth[:-1]
    pred_steps = prediction[1:] - prediction[:-1]
    rpe = (pred_steps - gt_steps).square().sum(dim=-1).mean().sqrt()
    return float(ate), float(rpe)


def distributed_context(seed: int) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch_dist.init_process_group(backend="nccl", init_method="env://")
    # The existing navigation benchmark initializes every rank with the same
    # top-level seed. Preserve that contract instead of introducing rank offsets.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return world_size, rank, torch.device("cuda", local_rank)


def configure_dataset_contract(args: argparse.Namespace, dataset_name: str) -> None:
    """Load generated OOD navigation contracts without weakening legacy pins."""

    if dataset_name in EXPECTED_SPLIT_SHA256:
        return
    report_path = args.data_root / dataset_name / "dataset_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset") != dataset_name:
        raise ValueError(f"Dataset report identity mismatch: {report_path}")
    reported_count = report.get(
        "navigation_sample_count", report.get("navigation_samples")
    )
    if reported_count != args.expected_sample_count:
        raise ValueError(
            f"{dataset_name} report does not pin "
            f"{args.expected_sample_count} navigation samples"
        )
    split_metadata = report.get("splits", {}).get("navigation_eval.pkl")
    if not split_metadata or not split_metadata.get("sha256"):
        raise ValueError(f"{dataset_name} report has no pinned navigation split")
    spacing = float(report["metric_waypoint_spacing"])
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Invalid {dataset_name} waypoint spacing: {spacing}")
    EXPECTED_SPLIT_SHA256[dataset_name] = split_metadata["sha256"]
    EVAL_WAYPOINT_SPACING[dataset_name] = spacing
    # OOD actions use their own measured waypoint scale before entering the
    # shared RAE-NWM action normalization.
    RAENWM_WAYPOINT_SPACING[dataset_name] = spacing
    PLAN_DISTRIBUTIONS[dataset_name] = dict(OOD_PLAN_DISTRIBUTION)


def build_dataset(args: argparse.Namespace, dataset_name: str, dataset_cls: Any, transform: Any) -> Any:
    layout = DATASET_LAYOUTS[dataset_name]
    split_root = args.project_root / "data_splits" / layout["split"] / "test"
    split = split_root / "navigation_eval.pkl"
    if not split.is_file():
        raise FileNotFoundError(split)
    actual_split_sha256 = sha256_file(split)
    if actual_split_sha256 != EXPECTED_SPLIT_SHA256[dataset_name]:
        raise RuntimeError(
            f"{dataset_name} navigation split SHA-256 mismatch: expected "
            f"{EXPECTED_SPLIT_SHA256[dataset_name]}, got {actual_split_sha256}"
        )
    dataset = dataset_cls(
        data_folder=str(args.data_root / layout["data"]),
        data_split_folder=str(split_root),
        dataset_name=layout["loader"],
        image_size=224,
        min_dist_cat=8,
        max_dist_cat=8,
        len_traj_pred=HORIZON_STEPS,
        traj_stride=8,
        context_size=CONTEXT_SIZE,
        normalize=True,
        transform=transform,
        predefined_index=str(split),
        traj_names=(
            "rollout_traj_names.txt"
            if (split_root / "rollout_traj_names.txt").is_file()
            else "traj_names.txt"
        ),
    )
    # Metrics and candidate trajectories must use the exact local benchmark
    # coordinate system. RAE-NWM's model-specific spacing is applied only by
    # local_deltas_to_raenwm immediately before model inference.
    dataset.dataset_name = dataset_name
    dataset.data_config = {
        "metric_waypoint_spacing": EVAL_WAYPOINT_SPACING[dataset_name]
    }
    if len(dataset) != args.expected_sample_count:
        raise RuntimeError(
            f"Expected {args.expected_sample_count} {dataset_name} navigation "
            f"samples, got {len(dataset)}"
        )
    return dataset


class Planner:
    def __init__(
        self,
        args: argparse.Namespace,
        model: Any,
        sampler: Any,
        rae: Any,
        lpips_model: Any,
        device: torch.device,
    ) -> None:
        self.args = args
        self.model = model
        self.sampler = sampler
        self.rae = rae
        self.lpips = lpips_model
        self.device = device
        self.latent_size = 224 // 14
        self.sample_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return self.rae.encode(images.to(self.device) * 0.5 + 0.5)

    def decode(self, latents: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            decoded = self.rae.decode(latents).float().nan_to_num()
        if decoded.shape[-2:] != target_size:
            decoded = F.interpolate(
                decoded, size=target_size, mode="bicubic", align_corners=False
            )
        return decoded.clamp(0, 1)

    def rollout_final(
        self,
        initial_latents: torch.Tensor,
        actions: torch.Tensor,
        sample_keys: list[str],
        noise_stream: str,
    ) -> torch.Tensor:
        if len(sample_keys) != actions.shape[0]:
            raise ValueError("sample_keys must identify every CEM candidate")
        outputs = []
        microbatch = min(self.args.microbatch_size, actions.shape[0])
        for start in range(0, actions.shape[0], microbatch):
            stop = min(start + microbatch, actions.shape[0])
            current = initial_latents.expand(stop - start, *initial_latents.shape[1:]).clone()
            action_batch = actions[start:stop].to(self.device)
            for step in range(action_batch.shape[1]):
                noise = samplewise_randn(
                    sample_keys[start:stop],
                    (self.rae.latent_dim, self.latent_size, self.latent_size),
                    base_seed=self.args.seed,
                    stream=f"{noise_stream}/rollout_step={step}",
                    device=self.device,
                )
                relative_time = torch.full(
                    (stop - start,), 1.0 / 128.0, device=self.device
                )
                with torch.inference_mode(), torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    prediction = self.sample_fn(
                        noise,
                        self.model,
                        y=action_batch[:, step],
                        x_cond=current[:, :CONTEXT_SIZE],
                        rel_t=relative_time,
                    )[-1]
                if getattr(self.args, "image_feedback", False):
                    if step + 1 < action_batch.shape[1]:
                        feedback_rgb = self.decode(prediction, (224, 224))
                        encoded_rgb = self.encode(feedback_rgb.mul(2).sub(1))
                        current = torch.cat((current[:, 1:], encoded_rgb.unsqueeze(1)), dim=1)
                else:
                    current = torch.cat((current[:, 1:], prediction.unsqueeze(1)), dim=1)
            outputs.append(prediction)
        return torch.cat(outputs, dim=0)

    def lpips_cost(self, predictions: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        values = []
        microbatch = min(self.args.microbatch_size, predictions.shape[0])
        for start in range(0, predictions.shape[0], microbatch):
            stop = min(start + microbatch, predictions.shape[0])
            decoded = self.decode(predictions[start:stop], tuple(goal.shape[-2:]))
            target = goal.expand(stop - start, -1, -1, -1)
            with torch.inference_mode():
                values.append(
                    self.lpips(decoded.mul(2).sub(1), target.mul(2).sub(1)).flatten()
                )
        return torch.cat(values)

    def evaluate_sample(
        self,
        dataset_name: str,
        observations: torch.Tensor,
        goal_image: torch.Tensor,
        ground_truth_actions: torch.Tensor,
        goal_position: torch.Tensor,
        sample_id: int,
    ) -> dict[str, float]:
        observations = observations[:, -CONTEXT_SIZE:]
        batch, context = observations.shape[:2]
        if batch != 1 or context != CONTEXT_SIZE:
            raise ValueError(f"Navigation evaluation requires [1,4,...], got {observations.shape}")
        initial_latents = self.encode(observations.flatten(0, 1)).unflatten(0, (1, context))
        goal_latent = self.encode(goal_image.flatten(0, 1))
        # Match compute_cost_with_recon=true: compare to this model's own
        # tokenizer reconstruction, not the raw goal pixels.
        reconstructed_goal = self.decode(goal_latent, tuple(goal_image.shape[-2:]))

        distribution = PLAN_DISTRIBUTIONS[dataset_name]
        mu = torch.tensor(distribution["mu"], device=self.device)
        sigma = torch.tensor(distribution["sigma"], device=self.device)
        for optimization_step in range(self.args.opt_steps):
            candidate_keys = [
                f"{dataset_name}/navigation/sample={sample_id}/opt={optimization_step}/candidate={index}"
                for index in range(self.args.num_samples)
            ]
            candidates = samplewise_randn(
                candidate_keys,
                (3,),
                base_seed=self.args.seed,
                stream=f"{dataset_name}/navigation/candidates",
                device=self.device,
            ) * sigma + mu
            local_xy = candidates[:, :2].unsqueeze(1).repeat(1, HORIZON_STEPS, 1)
            model_actions = local_deltas_to_raenwm(
                local_xy, dataset_name, candidates[:, 2]
            )
            repeated_costs = []
            for repetition in range(self.args.num_repeat_eval):
                final_latents = self.rollout_final(
                    initial_latents,
                    model_actions,
                    candidate_keys,
                    f"{dataset_name}/navigation/sample={sample_id}/opt={optimization_step}/repeat={repetition}",
                )
                repeated_costs.append(self.lpips_cost(final_latents, reconstructed_goal))
            costs = torch.stack(repeated_costs).mean(dim=0)
            topk = torch.argsort(costs)[: self.args.topk]
            # Preserve the existing local planner's update rule. Its terminal
            # yaw is already multiplied by pi before fitting the final mean.
            fitted = torch.cat((local_xy, model_actions[..., 2:]), dim=-1)
            final_steps = fitted[topk, -1]
            mu = final_steps.mean(dim=0)
            sigma = final_steps.std(dim=0)

        final_local_xy = mu[:2].view(1, 1, 2).repeat(1, HORIZON_STEPS, 1)
        final_local_units = unnormalize_data(final_local_xy, LOCAL_ACTION_STATS)
        predicted_actions = torch.cumsum(final_local_units, dim=1)[0]
        final_yaw_deltas = calculate_delta_yaw(final_local_units)
        final_yaw_deltas[:, -1, 0] += mu[-1] * math.pi
        predicted_yaw = final_yaw_deltas.sum()

        gt = ground_truth_actions[0, :, :2]
        if getattr(self.args, "save_planned_trajectories", False):
            atomic_json(self.args.output_root / dataset_name / RESULT_STEM / "trajectories" / f"{sample_id:06d}.json", {
                "sample_id": sample_id,
                "predicted_xy_waypoint_units": predicted_actions.float().cpu().tolist(),
                "gt_xy_waypoint_units": gt.float().cpu().tolist(),
                "goal_pose_waypoint_units": goal_position[0].float().cpu().tolist(),
            })
        ate, rpe_trans = trajectory_metrics(gt, predicted_actions)
        final_goal = goal_position[0, 0]
        pos_diff = torch.linalg.vector_norm(predicted_actions[-1].cpu() - final_goal[:2])
        yaw_diff = predicted_yaw.cpu() - final_goal[-1]
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
        return {
            "ate": ate,
            "rpe_trans": rpe_trans,
            "pos_diff_norm": float(pos_diff),
            "yaw_diff_norm": float(yaw_diff),
        }


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    dataset: Any,
    planner: Planner,
    world_size: int,
    rank: int,
) -> None:
    output = args.output_root / dataset_name / RESULT_STEM
    samples = output / "sample_metrics"
    samples.mkdir(parents=True, exist_ok=True)
    requested = args.sample_indices if args.sample_indices is not None else list(range(len(dataset)))
    assigned = requested[rank::world_size]

    started = time.monotonic()
    for offset, sample_index in enumerate(assigned):
        sample_path = samples / f"{sample_index:06d}.json"
        if args.resume and sample_path.is_file():
            continue
        indices, observations, goal, gt_actions, goal_position = dataset[sample_index]
        actual_index = int(indices.view(-1)[0].item())
        if actual_index != sample_index:
            raise RuntimeError(f"Dataset index mismatch: expected {sample_index}, got {actual_index}")
        metrics = planner.evaluate_sample(
            dataset_name,
            observations.unsqueeze(0),
            goal.unsqueeze(0),
            gt_actions.unsqueeze(0),
            goal_position.unsqueeze(0),
            sample_index,
        )
        atomic_json(sample_path, metrics)
        print(
            f"[rae-nwm-plan] rank={rank} dataset={dataset_name} "
            f"sample={sample_index} progress={offset + 1}/{len(assigned)} "
            f"elapsed_seconds={time.monotonic() - started:.1f} metrics={metrics}",
            flush=True,
        )

    elapsed = torch.tensor(
        time.monotonic() - started, dtype=torch.float64, device=planner.device
    )
    if world_size > 1:
        torch_dist.all_reduce(elapsed, op=torch_dist.ReduceOp.MAX)
        torch_dist.barrier()
    if rank == 0 and args.write_aggregate:
        records = []
        for sample_index in range(len(dataset)):
            path = samples / f"{sample_index:06d}.json"
            if not path.is_file():
                raise RuntimeError(f"Missing navigation sample metric: {path}")
            records.append(json.loads(path.read_text(encoding="utf-8")))
        result = {
            f"{dataset_name}_{key}": sum(float(row[key]) for row in records) / len(records)
            for key in ("ate", "rpe_trans", "pos_diff_norm", "yaw_diff_norm")
        }
        result.update(
            {
                "sample_count": len(records),
                "total_time": float(elapsed.cpu()),
                "inference": {
                    "backend": "rae-nwm",
                    "sampler": f"{args.sampling_method}_ode",
                    "sampling_steps": args.num_steps,
                },
            }
        )
        atomic_json(args.output_root / f"{dataset_name}_{RESULT_STEM}.json", result)
    if world_size > 1:
        torch_dist.barrier()


def write_manifest(args: argparse.Namespace, world_size: int) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "rae-nwm",
        "source_revision": source_revision(args.source),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
        },
        "datasets": {
            name: {
                "sample_count": args.expected_sample_count,
                "split": str(
                    args.project_root
                    / "data_splits"
                    / DATASET_LAYOUTS[name]["split"]
                    / "test/navigation_eval.pkl"
                ),
                "split_sha256": sha256_file(
                    args.project_root
                    / "data_splits"
                    / DATASET_LAYOUTS[name]["split"]
                    / "test/navigation_eval.pkl"
                ),
                "expected_split_sha256": EXPECTED_SPLIT_SHA256[name],
                "evaluation_waypoint_spacing": EVAL_WAYPOINT_SPACING[name],
                "model_waypoint_spacing": RAENWM_WAYPOINT_SPACING[name],
            }
            for name in args.datasets
        },
        "protocol": {
            "population": args.num_samples,
            "topk": args.topk,
            "rollout_stride": 1,
            "repetitions": args.num_repeat_eval,
            "optimization_steps": args.opt_steps,
            "horizon_steps": HORIZON_STEPS,
            "cost": "lpips_alex_on_rae_reconstruction",
            "trajectory_sampler": "line_constant_delta",
            **({"feedback": "decoded_image_reencoded"} if getattr(args, "image_feedback", False) else {}),
            "seed": args.seed,
            "sampler": f"{args.sampling_method}_ode",
            "sampling_steps": args.num_steps,
            "world_size": world_size,
            "microbatch_size": args.microbatch_size,
            "sample_partition": "split_position[rank::world_size]",
            "noise_policy": "sha256(seed,dataset,sample_id,candidate,repetition,rollout_step)",
            "topology_independent": True,
        },
    }
    manifest_path = args.output_root / "planning_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in (
            "schema_version",
            "model",
            "source_revision",
            "checkpoint",
            "protocol",
        ):
            if existing.get(field) != manifest[field]:
                raise RuntimeError(
                    f"Cannot merge incompatible planning manifest field {field}: "
                    f"{manifest_path}"
                )
        existing_datasets = existing.get("datasets")
        if not isinstance(existing_datasets, dict):
            raise RuntimeError(
                f"Existing planning manifest has invalid datasets: {manifest_path}"
            )
        manifest["created_at"] = existing.get("created_at", manifest["created_at"])
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["datasets"] = {**existing_datasets, **manifest["datasets"]}
    atomic_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--opt-steps", type=int, default=1)
    parser.add_argument("--num-repeat-eval", type=int, default=3)
    parser.add_argument("--microbatch-size", type=int, default=80)
    parser.add_argument("--sample-indices", nargs="+", type=int)
    parser.add_argument("--write-aggregate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampling-method", default="euler")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=DEFAULT_EXPECTED_SAMPLE_COUNT,
        help="Exact registered navigation split size for every dataset in this invocation.",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-planned-trajectories", action="store_true")
    parser.add_argument("--image-feedback", action="store_true")
    args = parser.parse_args()
    if (args.num_samples, args.topk, args.opt_steps, args.num_repeat_eval) != (80, 5, 1, 3):
        raise ValueError("The registered navigation protocol is fixed at CEM N80/K5/OPT1/rep3")
    if args.microbatch_size < 1:
        raise ValueError("--microbatch-size must be positive")
    if args.expected_sample_count < 1:
        raise ValueError("--expected-sample-count must be positive")
    if args.sample_indices is not None and (
        len(args.sample_indices) != len(set(args.sample_indices))
        or any(i < 0 or i >= args.expected_sample_count for i in args.sample_indices)
    ):
        raise ValueError("--sample-indices must be unique valid split positions")
    return args


def main() -> None:
    global RESULT_STEM
    args = parse_args()
    if args.image_feedback:
        RESULT_STEM += "_PIXEL-FEEDBACK"
    for field in (
        "source",
        "checkpoint",
        "decoder",
        "normalization_stats",
        "dino_model",
        "project_root",
        "data_root",
        "output_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    if source_revision(args.source) != EXPECTED_SOURCE_REVISION:
        raise RuntimeError("RAE-NWM source revision does not match the pinned benchmark revision")
    required = (
        args.checkpoint,
        args.decoder,
        args.normalization_stats,
        args.dino_model / "model.safetensors",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RAE-NWM assets: {missing}")
    for dataset_name in args.datasets:
        configure_dataset_contract(args, dataset_name)
    configure_imports(args.source)
    import lpips
    import misc as official_misc
    from datasets import TrajectoryEvalDataset

    world_size, rank, device = distributed_context(args.seed)
    model, sampler, rae = build_models(args, device)
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    planner = Planner(args, model, sampler, rae, lpips_model, device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args, world_size)
    for dataset_name in args.datasets:
        dataset = build_dataset(
            args, dataset_name, TrajectoryEvalDataset, official_misc.transform
        )
        run_dataset(args, dataset_name, dataset, planner, world_size, rank)
    if world_size > 1:
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
