#!/usr/bin/env python3
"""Maintain one extensible JSON registry for NWM benchmark results."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/benchmark_results.json"
)
EVAL_SEED = int(os.environ.get("NWM_EVAL_SEED", "0"))
OOD_DATA_ROOT = "/file_system/nas/algorithm/dujun.nie/nwm/data"

OOD_DATASET_CONTRACTS = {
    "planetary_rover": {
        "version": "real_native_color_camera_endpoint_expanded_239_20260921",
        "position_units": "sequence_median_step_units",
        "metric_navigation_evaluation_allowed": False,
        "prediction_sample_count": 10,
        "navigation_sample_count": 47,
        "rollout_sample_count": 0,
        "scene": (
            "15 Chang'e-4/Yutu-2 lunar clips and "
            "13 Tianwen-1/Zhurong Mars clips; real native-colour "
            "forward terrain with horizon"
        ),
        "report": {
            "path": f"{OOD_DATA_ROOT}/planetary_rover/dataset_report.json",
            "sha256": "a8bf9534cd8b025eedcd7e553a684ab0e7033937ab7d299735251a1ae6867fbb",
        },
        "prediction_split": {
            "path": "data_splits/planetary_rover/test/time.pkl",
            "sha256": "6e022d19ee5788539c0d1d021be1d64e01086f6f7cf21dd1219fdff65bd5a8ff",
        },
        "navigation_split": {
            "path": "data_splits/planetary_rover/test/navigation_eval.pkl",
            "sha256": "99cdf0b934c2b275e4f80dc2a923268bb785d2ef336beee7a05cee3439e4ad77",
        },
        "metric_waypoint_spacing": 1.0,
        "trajectory_cadence": "irregular_real_frame_sequence",
        "temporal_semantics": "irregular_real_frame_index",
    },
    "unitree_go2": {
        "scene": "locally recorded Unitree Go2 quadruped videos",
        "report": {
            "path": f"{OOD_DATA_ROOT}/unitree_go2/dataset_report.json",
            "sha256": "857142dfa00167fd19231c583b7c5af14ec2badfb19b44505aad8aafecff0030",
        },
        "prediction_split": {
            "path": "data_splits/unitree_go2/test/time.pkl",
            "sha256": "ec25e7b7811057a97a8a59ae5dadd2c683a48c8b8b2e861927e826a7478d1ccd",
        },
        "navigation_split": {
            "path": "data_splits/unitree_go2/test/navigation_eval.pkl",
            "sha256": "84963c5d3c14fb7de719e0f3b9669d6416e90fe607c7d8bcc9957c40ccb09dde",
        },
        "metric_waypoint_spacing": 0.07610250112606785,
        "trajectory_cadence": "fixed_4hz",
        "temporal_semantics": "physical_4_seconds",
    },
    "tum_rgbd": {
        "scene": "handheld indoor TUM RGB-D trajectories",
        "report": {
            "path": f"{OOD_DATA_ROOT}/tum_rgbd/dataset_report.json",
            "sha256": "1bdd4ccdbf2ecbda6611c71fe8166148d204c8fd93b9ce2a0de6bbed796f4a3e",
        },
        "prediction_split": {
            "path": "data_splits/tum_rgbd/test/time.pkl",
            "sha256": "172cb60ed65294f8b82fc27605d8ca01ff52d4a9a17036b325567fc53a13c68d",
        },
        "navigation_split": {
            "path": "data_splits/tum_rgbd/test/navigation_eval.pkl",
            "sha256": "be6d8f7b325e289b3eec728876d1e60636d936401cbe98ad9d59a49f87e31eb5",
        },
        "metric_waypoint_spacing": 0.05016063266653285,
        "trajectory_cadence": "fixed_4hz",
        "temporal_semantics": "physical_4_seconds",
    },
    "uzh_fpv": {
        "scene": "first-person indoor drone racing",
        "report": {
            "path": f"{OOD_DATA_ROOT}/uzh_fpv/dataset_report.json",
            "sha256": "449209b4e070b305b6ffcfa1fecce771c36a1e0911221e3c9cb8bbf3bc23ce2e",
        },
        "prediction_split": {
            "path": "data_splits/uzh_fpv/test/time.pkl",
            "sha256": "9cc88322e98f2a4216a7e8f98507c4a58d7e8500d5cfb831c293e615b62ce6db",
        },
        "navigation_split": {
            "path": "data_splits/uzh_fpv/test/navigation_eval.pkl",
            "sha256": "88eb250b60388709d3ce56775b6d596f9c16369ed249da4abd70944f3766e181",
        },
        "metric_waypoint_spacing": 1.1438262345654386,
        "trajectory_cadence": "fixed_4hz",
        "temporal_semantics": "physical_4_seconds",
    },
}

# Canonical identities for every dataset supported by the unified benchmark.
# ``split_name`` and ``data_name`` keep the public HuRoN/TartanDrive names while
# preserving the historical on-disk SACSoN/Tartan layout used by NWM.
DATASET_CONTRACTS = {
    "recon": {
        "data_name": "recon",
        "split_name": "recon",
        "metric_waypoint_spacing": 0.25,
        "temporal_semantics": "physical_4_seconds",
        "splits": {
            "time": "3914b6687f34f34b42a5d61838f59e4a80d8f19cdc17584c0441d51eb2e11178",
            "rollout": "29f6b87eb870f661e5ce4bdce2317247fa4744fa6557711113d2efcad077ea18",
            "navigation_eval": "c62cd08be9f124cbeec48d914460da8630e089bf0bdb84c5018013a82d12ec54",
        },
    },
    "scand": {
        "data_name": "scand",
        "split_name": "scand",
        "metric_waypoint_spacing": 0.38,
        "temporal_semantics": "physical_4_seconds",
        "splits": {
            "time": "0b0d293d800e4786acbfdc9cc7647a15d4f55234edd5c62dff01ff5498dac756",
            "rollout": "938d37ac8008b44eff43ea72c15171118df69c780eb5f9df8f8b22966fa6e585",
            "navigation_eval": "8acb4062561cbf1549e27f39a6e80241b97294a8c0e55c345787ae6a47be55ce",
        },
    },
    "huron": {
        "data_name": "sacson",
        "split_name": "sacson",
        "metric_waypoint_spacing": 0.255,
        "temporal_semantics": "physical_4_seconds",
        "splits": {
            "time": "9d1c5239dfbf0f73d4aaee29c5abda31f28e97bbaf02f5bfc76cfd4f7f63953e",
            "rollout": "3058126a3c8fea9776c364f41c394d00bff1bb8f9a4d092fa8d1433554ec6991",
            "navigation_eval": "89e07bb2b934d7fe4e8ab0bddf51e28a4beb36d2415ad83c166350241ef4e9fd",
        },
    },
    "tartan_drive": {
        "data_name": "tartan",
        "split_name": "tartan_drive",
        "metric_waypoint_spacing": 0.72,
        "temporal_semantics": "physical_4_seconds",
        "splits": {
            "time": "b753d2af89aba1cd3ba8bd85baba82af3ce87c7ba9488d70ee73b0883a639d3f",
            "rollout": "853eeea2f95a6355b69526a9a1ca9f95c0a554e6e844a1021440d9dd260ff23a",
            "navigation_eval": "77bc38808b8df24b330fc4f9a4a17ed0de35a1c1bef0ff2283fc461b3ab16435",
        },
    },
    "go_stanford": {
        "data_name": "go_stanford",
        "split_name": "go_stanford",
        "metric_waypoint_spacing": 0.12,
        "temporal_semantics": "physical_4_seconds",
        "splits": {
            "time": "51245d6fb46580b26c43748fab77760014ef34ea0d6f5d9509de242141644414",
            "rollout": "e48af806e991d465f8b04f4dd106ff9b8db55cd65303831ee31e71dbef95cc23",
            "navigation_eval": "5013d8e2defbbee4d9652f7ae4569816113a8d44dd2af8b5e135d9f5cea3e27c",
        },
    },
    **{
        name: {
            "data_name": name,
            "split_name": name,
            "metric_waypoint_spacing": contract["metric_waypoint_spacing"],
            "temporal_semantics": contract["temporal_semantics"],
            "prediction_sample_count": contract.get("prediction_sample_count", 500),
            "navigation_sample_count": contract.get("navigation_sample_count", 100),
            "rollout_sample_count": contract.get("rollout_sample_count", 150),
            "splits": {
                "time": contract["prediction_split"]["sha256"],
                "rollout": {
                    "planetary_rover": "ec0a6ccf9debf1c16781445c4b9106080d00478b0559469336db7c7b7b9711c8",
                    "unitree_go2": "0b618aa4cbbfd1e2b18d8fc32b19c6311c5e6e519842590ecc67df88a9d8a9b3",
                    "tum_rgbd": "d6f5b4e5077a3f220e8957b602f34341e0727b7eae7a98f738b570988765d62c",
                    "uzh_fpv": "45af1a0579b203a2803c8f3c206d737562a1c238873606b8345c9866045956d6",
                }[name],
                "navigation_eval": contract["navigation_split"]["sha256"],
            },
        }
        for name, contract in OOD_DATASET_CONTRACTS.items()
    },
}

ALL_DATASETS = tuple(DATASET_CONTRACTS)
ROLLOUT_DATASETS = tuple(
    name
    for name, contract in DATASET_CONTRACTS.items()
    if int(contract.get("rollout_sample_count", 150)) > 0
)


def dataset_sample_count(
    protocol: dict[str, Any], dataset: str, evaluation: str
) -> int:
    """Resolve a protocol count while preserving its dataset-level overrides."""

    if evaluation == "navigation":
        override_key = "navigation_sample_count"
        default = protocol.get("navigation_sample_count", protocol.get("sample_count"))
    elif evaluation == "time":
        override_key = "prediction_sample_count"
        default = protocol.get("sample_counts", {}).get(
            evaluation, protocol.get("sample_count")
        )
    elif evaluation.startswith("rollout"):
        override_key = "rollout_sample_count"
        default = protocol.get("sample_counts", {}).get(
            evaluation, protocol.get("sample_count")
        )
    else:
        raise ValueError(f"Unknown evaluation kind for sample count: {evaluation}")

    datasets = protocol.get("datasets", {})
    contract = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    split = protocol.get("splits", {}).get(dataset, {})
    value = contract.get(override_key, split.get("sample_count", default))
    if value is None:
        raise ValueError(
            f"Protocol has no sample-count contract for {dataset}/{evaluation}"
        )
    count = int(value)
    if count < 0:
        raise ValueError(
            f"Protocol has a negative sample count for {dataset}/{evaluation}: {count}"
        )
    return count

MODELS = {
    "rae-nwm": {
        "display_name": "RAE-NWM",
        "backend": "raenwm",
        "architecture": "CDiT-B/2 + DINOv2-B RAE",
        "source_dir": str(Path(__file__).resolve().parents[1] / "third_party/raenwm"),
        "assets_root": "/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/rae-nwm",
        "checkpoint_id": "raenwm_b",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/rae-nwm/checkpoints/raenwm_b.pth.tar",
        "checkpoint_step": None,
        "sha256": "97244579618eb0e376355157a5c1df7a83e534cb1ca35d8ff4e43c4fb4f6a9d0",
        "training_datasets": ["recon", "sacson", "scand"],
        "provenance": {
            "repository": "20robo/raenwm",
            "revision": "0219ce41c44d515f86719dd763c1efe7c7f72519",
            "weights_repository": "zmkun20/raenwm",
            "weights_revision": "3d21560bdbdbc8cc3d4a796e1e110d60e920d273",
            "sampling_method": "euler",
            "sampling_steps": 50,
        },
    },
    "nwm-base": {
        "architecture": "CDiT-B/2 + SD-VAE",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260820_164519_compact_nwm_cdit_b_sdvae_bs16",
        "checkpoint_id": "0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260820_164519_compact_nwm_cdit_b_sdvae_bs16/checkpoints/0200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "76b2de9a4e6efba008583057eea6a81ae90429bea988f45cbc8f5bcab10f120b",
        "training_datasets": ["recon", "huron_public_sacson_key", "scand"],
    },
    "nwm-real": {
        "display_name": "NWM 200k step",
        "architecture": "CDiT-B/2 + SD-VAE + real-motion adapter",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260823_160459_nwm_real_recon_scand_tartan_huron_bs16",
        "checkpoint_id": "0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260823_160459_nwm_real_recon_scand_tartan_huron_bs16/checkpoints/0200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "757a511cb5efc44bca53cbd0d5d19add019c95fd229cf723538de228cbe1b846",
        "training_datasets": [
            "recon",
            "huron_public_sacson_key",
            "scand",
            "tartan_drive",
        ],
    },
    "nwm-release": {
        "architecture": "CDiT-B/2 + SD-VAE",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/nwm-release",
        "checkpoint_id": "0100000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/weights/cdit_b_100000.pth.tar",
        "checkpoint_step": 100000,
        "sha256": "2a41c71eabd20946f61bb5d1d2490264246bff59b9ffca177eee672e855d5261",
        "training_datasets": ["recon", "huron", "scand", "tartan_drive"],
        "provenance": {
            "repository": "facebook/nwm",
            "revision": "0821a1a7b1ae938539e32f13f5ad82465e0f3fde",
            "checkpoint_train_steps_field": 100000,
        },
    },
    "nwm-ego4d": {
        "display_name": "NWM + Ego4D",
        "architecture": "CDiT-XL/2 + SD-VAE",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/nwm-ego4d",
        "checkpoint_id": "0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/weights/cdit_xl_ego4d_200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "6686fff6505ccc3b8e93a368b1bd3bf0051dfb76c2d3c280b77313035ef1d2f8",
        "training_datasets": ["recon", "huron", "scand", "tartan_drive", "ego4d"],
        "provenance": {
            "repository": "facebook/nwm",
            "revision": "0821a1a7b1ae938539e32f13f5ad82465e0f3fde",
            "checkpoint_filename": "cdit_xl_ego4d_200000.pth.tar",
            "download_url": "https://huggingface.co/facebook/nwm/resolve/main/cdit_xl_ego4d_200000.pth.tar?download=true",
            "checkpoint_train_steps_field": 200000,
        },
    },
    "nwm-latent": {
        "architecture": "CDiT-B/2 + SD-VAE + real/latent-motion adapters",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260824_163328_nwm-latent-badlam_bs16",
        "checkpoint_id": "0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/20260824_163328_nwm-latent-badlam_bs16/checkpoints/0200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "744ca3e025907977a42994709102234a4acad4faf63a59bdfa49e9922ee17664",
        "training_datasets": [
            "recon",
            "huron_public_sacson_key",
            "scand",
            "tartan_drive",
        ],
    },
    "nwm-timept-ft": {
        "architecture": "CDiT-B/2 + SD-VAE + TimePT adapter reset fine-tune",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_timept_ft/nwm-nav1-timept-finetune",
        "checkpoint_id": "joint_0100000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_timept_ft/nwm-nav1-timept-finetune/checkpoints/joint_0100000.pth.tar",
        "checkpoint_step": 110000,
        "sha256": "34332c5b130eb0bceab3ccdb21489d1e61bb1bd30e1bfca0e04b3050285842aa",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "stage1": "NavAnywhere-v1 TimePT",
            "fine_tune_scheme": "adapter reset; 10k warmup + 100k joint steps",
        },
    },
    "nwm-geopt-ft": {
        "architecture": "CDiT-B/2 + SD-VAE + GeoPT adapter reset fine-tune",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_geopt_ft/nwm-nav1-geopt-finetune",
        "checkpoint_id": "joint_0100000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_geopt_ft/nwm-nav1-geopt-finetune/checkpoints/joint_0100000.pth.tar",
        "checkpoint_step": 110000,
        "sha256": "4915124400ae8042c6616c645d0ab296ee508090fb18258feace417bd15b041a",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "stage1": "NavAnywhere-v1 GeoPT",
            "stage1_checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywhere_stage1/nwm-geopt/checkpoints/latest.pth.tar",
            "fine_tune_scheme": "adapter reset; 10k warmup + 100k joint steps",
            "training_complete_at": "2026-09-07T03:32:21Z",
        },
    },
    "nwm-latentpt-ft": {
        "architecture": "CDiT-B/2 + SD-VAE + LatentPT adapter reset fine-tune",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_latentpt_ft_pixel_action_l20/nwm-nav1-latentpt-finetune-pixel-action-l20",
        "checkpoint_id": "joint_0100000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_latentpt_ft_pixel_action_l20/nwm-nav1-latentpt-finetune-pixel-action-l20/checkpoints/joint_0100000.pth.tar",
        "checkpoint_step": 110000,
        "sha256": "f5d5b3b70cd8a483ce4783b06ceb4405a8af1ff1a843e521cdd65e51f076df35",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "stage1": "NavAnywhere-v1 LatentPT",
            "stage1_checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywhere_stage1/nwm-latentpt/checkpoints/latest.pth.tar",
            "fine_tune_scheme": "adapter reset; 10k warmup + 100k joint steps",
            "training_complete_at": "2026-09-10T18:23:44Z",
        },
    },
    "nwm-latentpt-ft-align": {
        "architecture": "CDiT-B/2 + SD-VAE + LatentPT-aligned real-motion adapter fine-tune",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_latentpt_ft_align_pixel_action_l20/nwm-nav1-latentpt-finetune-align-pixel-action-l20",
        "checkpoint_id": "joint_0100000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_latentpt_ft_align_pixel_action_l20/nwm-nav1-latentpt-finetune-align-pixel-action-l20/checkpoints/joint_0100000.pth.tar",
        "checkpoint_step": 110000,
        "sha256": "c8639c99e3f8dca8059f31a3c34b3cef93b11f16141cf002d84577e738550b1c",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "stage1": "NavAnywhere-v1 LatentPT",
            "stage1_checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywhere_stage1/nwm-latentpt/checkpoints/latest.pth.tar",
            "fine_tune_scheme": "adapter reset; 10k warmup + 100k joint steps with LatentPT alignment loss",
            "training_complete_at": "2026-09-11T20:34:16Z",
        },
    },
    "nwm-latentpt-ft-action2latent": {
        "architecture": "CDiT-B/2 + SD-VAE + LatentPT Action2Latent adapter fine-tune",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_latentpt_ft_action2latent_pixel_action_l20/nwm-nav1-latentpt-finetune-action2latent-pixel-action-l20",
        "checkpoint_id": "joint_0100000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywherev1_latentpt_ft_action2latent_pixel_action_l20/nwm-nav1-latentpt-finetune-action2latent-pixel-action-l20/checkpoints/joint_0100000.pth.tar",
        "checkpoint_step": 110000,
        "sha256": "c4973fb135bba5b261b94f3bf822cc158654a21a22993ac9f54b0835c70c6b6c",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "stage1": "NavAnywhere-v1 LatentPT",
            "stage1_checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywhere_stage1/nwm-latentpt/checkpoints/latest.pth.tar",
            "fine_tune_scheme": "real-to-latent action mapping; 10k warmup + 100k joint steps",
            "training_complete_at": "2026-09-12T13:27:27Z",
        },
    },
    "nwm-latentpt-reset-nwm-real-recipe": {
        "display_name": "OpenNWM",
        "architecture": "CDiT-B/2 + SD-VAE + LatentPT-initialized reset real-motion adapter",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/latentpt_reset_nwm_real_recipe/nwm-latentpt-reset-nwm-real-recipe",
        "checkpoint_id": "joint_0200000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/latentpt_reset_nwm_real_recipe/nwm-latentpt-reset-nwm-real-recipe/checkpoints/joint_0200000.pth.tar",
        "checkpoint_step": 200000,
        "sha256": "314ea1724ada64f69d78e9b3d15aa8754d8150fb5e5fca77f404cdacef0b27a4",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "stage1": "NavAnywhere-v1 LatentPT",
            "stage1_checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywhere_stage1/nwm-latentpt/checkpoints/latest.pth.tar",
            "fine_tune_scheme": "real-motion adapter reset; 0 warmup + 200000 joint steps",
            "training_complete_at": "2026-09-17T18:03:28Z",
        },
    },
    "nwm-latentpt-reset-nwm-real-recipe-180k": {
        "display_name": "OpenNWM 180k step",
        "architecture": "CDiT-B/2 + SD-VAE + LatentPT-initialized reset real-motion adapter",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/latentpt_reset_nwm_real_recipe/nwm-latentpt-reset-nwm-real-recipe",
        "checkpoint_id": "joint_0180000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/latentpt_reset_nwm_real_recipe/nwm-latentpt-reset-nwm-real-recipe/checkpoints/joint_0180000.pth.tar",
        "checkpoint_step": 180000,
        "sha256": "0e3e9e49a059fc05706fc6d7b457d66df2392b18b2303631c5e04195c72a6170",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "stage1": "NavAnywhere-v1 LatentPT",
            "stage1_checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/navanywhere_stage1/nwm-latentpt/checkpoints/latest.pth.tar",
            "fine_tune_scheme": "real-motion adapter reset; 0 warmup + 180000 joint steps (selected checkpoint from 200000-step run)",
            "source_run_complete_at": "2026-09-17T18:03:28Z",
        },
    },
    "nwm-no-pretrain": {
        "architecture": "CDiT-B/2 + SD-VAE + real-motion adapter, trained from scratch",
        "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/no_pretrain_ft/nwm-no-pretrain-finetune",
        "checkpoint_id": "joint_0110000",
        "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/no_pretrain_ft/nwm-no-pretrain-finetune/checkpoints/joint_0110000.pth.tar",
        "checkpoint_step": 110000,
        "sha256": "b5c914d39cf1d9ce059fb39131bf514716f2136cc7a3d9b6db1a177ceeba8787",
        "training_datasets": ["recon", "sacson", "scand", "tartan_drive"],
        "provenance": {
            "initialization": "fresh CDiT and real-action adapter weights; no NWM checkpoint loaded",
            "fine_tune_scheme": "0 warmup + 110k joint steps",
            "training_complete_at": "2026-09-06T20:45:11Z",
        },
    },
}

PROTOCOLS = {
    "recon_prediction_v1": {
        "category": "recon_prediction",
        "dataset": "recon",
        "sample_counts": {"time": 500, "rollout_1fps": 150, "rollout_4fps": 150},
        "horizons_seconds": [1, 2, 4, 8, 16],
        "reference_nwm_diffusion_steps": 250,
        "seed": EVAL_SEED,
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "go_stanford_unseen_v1": {
        "category": "unseen_generalization",
        "dataset": "go_stanford",
        "evaluation": "time",
        "sample_count": 500,
        "horizons_seconds": [1, 2, 4, 8, 16],
        "paper_comparison_horizon_seconds": 4,
        "reference_nwm_diffusion_steps": 250,
        "seed": EVAL_SEED,
        "paper_comparison_caveat": "NWM paper reports five-sample mean for CDiT-XL; local models are CDiT-B single-seed checkpoints",
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "go_stanford_unseen_rollout_v1": {
        "category": "unseen_generalization",
        "dataset": "go_stanford",
        "evaluation": ["rollout_1fps", "rollout_4fps"],
        "split": "data_splits/go_stanford/test/rollout.pkl",
        "split_sha256": "e48af806e991d465f8b04f4dd106ff9b8db55cd65303831ee31e71dbef95cc23",
        "source_evaluations": {
            "rollout_1fps": "rollout_1fps",
            "rollout_4fps": "rollout_4fps",
        },
        "source_sample_count": 150,
        "sample_indices": list(range(150)),
        "sample_count": 150,
        "input_fps": 4,
        "rollout_fps": [1, 4],
        "horizons_seconds": [1, 2, 4, 8, 16],
        "diffusion_steps": 250,
        "seed": EVAL_SEED,
        "autoregressive": True,
        "execution": {
            "batch_size_per_rank": 64,
            "use_efficient_rollout": True,
            "sample_partition": "split_position[rank::world_size]",
            "topology_independent": True,
        },
        "visualization_sample_ids": [0, 21, 43, 64, 86, 107, 129, 149],
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "direct_4s_v1": {
        "category": "direct_prediction",
        "datasets": list(ALL_DATASETS),
        "data_root": OOD_DATA_ROOT,
        "evaluation": "time",
        "sample_count": 500,
        "input_fps": 4,
        "context_frames": 4,
        "future_frames": 16,
        "horizons_seconds": [4],
        "frame_indices": {"4s": 4},
        "seed": EVAL_SEED,
        "splits": {
            name: {
                "path": f"data_splits/{contract['split_name']}/test/time.pkl",
                "sha256": contract["splits"]["time"],
                "sample_count": contract.get("prediction_sample_count", 500),
            }
            for name, contract in DATASET_CONTRACTS.items()
        },
        "inference": {
            "nwm": {"sampler": "ddpm", "sampling_steps": 250},
            "rae-nwm": {"sampler": "euler_ode", "sampling_steps": 50},
        },
        "reproducibility": {
            "sample_partition": "split_position[rank::world_size]",
            "randomness": "sha256(seed,dataset,evaluation,sample_id,step)",
            "topology_independent": True,
        },
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "rollout_v1": {
        "category": "rollout_prediction",
        "datasets": list(ROLLOUT_DATASETS),
        "data_root": OOD_DATA_ROOT,
        "evaluation": ["rollout_1fps", "rollout_4fps"],
        "sample_count": 150,
        "input_fps": 4,
        "context_frames": 4,
        "future_frames": 64,
        "rollout_fps": [1, 4],
        "horizons_seconds": [1, 2, 4, 8, 16],
        "seed": EVAL_SEED,
        "autoregressive": True,
        "splits": {
            name: {
                "path": f"data_splits/{contract['split_name']}/test/rollout.pkl",
                "sha256": contract["splits"]["rollout"],
            }
            for name, contract in DATASET_CONTRACTS.items()
            if name in ROLLOUT_DATASETS
        },
        "inference": {
            "nwm": {"sampler": "ddpm", "sampling_steps": 250},
            "rae-nwm": {"sampler": "euler_ode", "sampling_steps": 50},
        },
        "reproducibility": {
            "sample_partition": "split_position[rank::world_size]",
            "randomness": "sha256(seed,dataset,evaluation,sample_id,rollout_step)",
            "topology_independent": True,
        },
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "ood_direct_4s_v1": {
        "category": "ood_direct_prediction",
        "datasets": OOD_DATASET_CONTRACTS,
        "data_root": OOD_DATA_ROOT,
        "evaluation": "time",
        "sample_count": 500,
        "navigation_sample_count": 100,
        "input_fps": 4,
        "context_frames": 4,
        "future_frames": 16,
        "horizons_seconds": [4],
        "frame_indices": {"4s": 4},
        "seed": 0,
        "reference_dataset": "go_stanford",
        "inference": {
            "nwm": {
                "sampler": "ddpm",
                "sampling_steps": 250,
                "batch_size_per_rank": 64,
            },
            "rae-nwm": {
                "sampler": "euler_ode",
                "sampling_steps": 50,
                "batch_size_per_rank": 16,
            },
        },
        "execution": {"distributed_world_size": 4},
        "metrics": {
            "lpips_alex": "lower",
            "dreamsim": "lower",
            "psnr": "higher",
        },
    },
    "navigation_cem80_v1": {
        "category": "navigation_planning",
        "datasets": list(ALL_DATASETS),
        "sample_count": 100,
        "splits": {
            "recon": {
                "path": "data_splits/recon/test/navigation_eval.pkl",
                "sha256": "c62cd08be9f124cbeec48d914460da8630e089bf0bdb84c5018013a82d12ec54",
                "metric_waypoint_spacing": 0.25,
            },
            "scand": {
                "path": "data_splits/scand/test/navigation_eval.pkl",
                "sha256": "8acb4062561cbf1549e27f39a6e80241b97294a8c0e55c345787ae6a47be55ce",
                "metric_waypoint_spacing": 0.38,
            },
            "huron": {
                "path": "data_splits/sacson/test/navigation_eval.pkl",
                "sha256": "89e07bb2b934d7fe4e8ab0bddf51e28a4beb36d2415ad83c166350241ef4e9fd",
                "metric_waypoint_spacing": 0.255,
            },
            "tartan_drive": {
                "path": "data_splits/tartan_drive/test/navigation_eval.pkl",
                "sha256": "77bc38808b8df24b330fc4f9a4a17ed0de35a1c1bef0ff2283fc461b3ab16435",
                "metric_waypoint_spacing": 0.72,
            },
            "go_stanford": {
                "path": "data_splits/go_stanford/test/navigation_eval.pkl",
                "sha256": "5013d8e2defbbee4d9652f7ae4569816113a8d44dd2af8b5e135d9f5cea3e27c",
                "metric_waypoint_spacing": 0.12,
            },
            **{
                dataset: {
                    "path": contract["navigation_split"]["path"],
                    "sha256": contract["navigation_split"]["sha256"],
                    "sample_count": contract.get("navigation_sample_count", 100),
                    "metric_waypoint_spacing": contract[
                        "metric_waypoint_spacing"
                    ],
                    "trajectory_cadence": contract["trajectory_cadence"],
                    "temporal_semantics": contract["temporal_semantics"],
                }
                for dataset, contract in OOD_DATASET_CONTRACTS.items()
            },
        },
        "population": 80,
        "topk": 5,
        "rollout_stride": 1,
        "repetitions": 3,
        "optimization_steps": 1,
        "horizon_steps": 8,
        "seconds_per_step": 0.25,
        "cost": "lpips_alex_on_model_tokenizer_reconstruction",
        "seed": 42 + EVAL_SEED,
        "metrics": {
            "ate": "lower",
            "rpe_trans": "lower",
            "pos_diff_norm": "lower",
            "yaw_diff_norm": "lower",
        },
        "comparability": {
            "compact_paper": "exact CEM population/protocol",
            "nwm_paper": "NWM reports population 120; compare with protocol caveat",
            "planetary_rover": "H=8 is a spatial waypoint horizon, not a physical two-second horizon",
        },
    },
    "navigation_cem80_fast10_v1": {
        "category": "navigation_planning_fast",
        "datasets": ["recon", "scand"],
        "sample_count": 100,
        "population": 80,
        "topk": 5,
        "rollout_stride": 1,
        "repetitions": 3,
        "optimization_steps": 1,
        "horizon_steps": 8,
        "seconds_per_step": 0.25,
        "diffusion_steps": 10,
        "cost": "lpips_alex_on_vae_reconstruction",
        "seed": 42 + EVAL_SEED,
        "metrics": {"ate": "lower", "rpe_trans": "lower"},
        "comparability": {
            "paper": "accelerated local protocol; diffusion step count differs from the 250-step local exact run"
        },
    },
}

PAPER_BASELINES = {
    "nwm_paper": {
        "title": "Navigation World Models",
        "source": "https://arxiv.org/html/2412.03572",
        "reported_model": "CDiT-XL (1B parameters)",
        "results": {
            "navigation_planning": {
                "protocol": "CEM population 120, H=8, M=3, I=1",
                "recon": {
                    "ate": 1.13,
                    "ate_std": 0.02,
                    "rpe_trans": 0.35,
                    "rpe_trans_std": 0.01,
                },
                "scand": {
                    "ate": 1.28,
                    "ate_std": 0.02,
                    "rpe_trans": 0.33,
                    "rpe_trans_std": 0.01,
                },
            },
            "unseen_generalization": {
                "dataset": "go_stanford",
                "horizon_seconds": 4,
                "in_domain_data": {
                    "lpips_alex": 0.658,
                    "lpips_std": 0.002,
                    "dreamsim": 0.478,
                    "dreamsim_std": 0.001,
                    "psnr": 11.031,
                    "psnr_std": 0.036,
                },
                "plus_ego4d": {
                    "lpips_alex": 0.652,
                    "lpips_std": 0.003,
                    "dreamsim": 0.464,
                    "dreamsim_std": 0.003,
                    "psnr": 11.083,
                    "psnr_std": 0.064,
                },
            },
        },
    },
    "compact_paper": {
        "title": "Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model",
        "source": "https://arxiv.org/html/2603.05438",
        "results": {
            "navigation_planning": {
                "protocol": "CEM population 80, H=8, M=3, I=1",
                "sd_vae": {
                    "recon": {"ate": 1.262, "rpe_trans": 0.354},
                    "scand": {"ate": 1.065, "rpe_trans": 0.291},
                },
                "compact_16_tokens": {
                    "recon": {"ate": 1.330, "rpe_trans": 0.390},
                    "scand": {"ate": 1.358, "rpe_trans": 0.336},
                },
                "compact_8_tokens": {
                    "recon": {"ate": 1.373, "rpe_trans": 0.401},
                    "scand": {"ate": 1.391, "rpe_trans": 0.346},
                },
            }
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def new_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": utc_now(),
        "protocols": PROTOCOLS,
        "paper_baselines": PAPER_BASELINES,
        "models": {
            name: {**metadata, "results": {}} for name, metadata in MODELS.items()
        },
    }


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_registry()
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported registry schema: {registry.get('schema_version')}"
        )
    registry["protocols"] = PROTOCOLS
    registry["paper_baselines"] = PAPER_BASELINES
    for name, metadata in MODELS.items():
        previous = registry.setdefault("models", {}).get(name, {})
        results = previous.get("results", {})
        # The temporary 10-sample Go Stanford rollout was superseded by the
        # complete fixed 150-sample protocol. Never carry those stale rows into
        # a newly loaded registry.
        for category in results.values():
            if not isinstance(category, dict):
                continue
            for dataset_results in category.values():
                if not isinstance(dataset_results, dict):
                    continue
                for evaluation in list(dataset_results):
                    if evaluation.startswith("rollout_10_"):
                        del dataset_results[evaluation]
        registry["models"][name] = {**metadata, "results": results}
    return registry


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    registry["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_model(registry: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return registry["models"][name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}; register it first") from exc


def register_model(
    registry: dict[str, Any],
    name: str,
    exp_dir: Path,
    checkpoint_id: str,
    checkpoint: Path,
    checkpoint_step: int | None,
    sha256: str | None,
    training_datasets: list[str],
) -> None:
    previous = registry.setdefault("models", {}).get(name, {})
    registry["models"][name] = {
        "exp_dir": str(exp_dir.resolve()),
        "checkpoint_id": checkpoint_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "sha256": sha256,
        "training_datasets": training_datasets,
        "results": previous.get("results", {}),
    }


def import_prediction(
    registry: dict[str, Any],
    model_name: str,
    dataset: str,
    evaluation: str,
    audit: Path,
    protocol: str | None = None,
) -> None:
    payload = json.loads(audit.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("eval_name") != evaluation:
        raise ValueError(
            f"Audit identity mismatch: expected {dataset}/{evaluation}, got "
            f"{payload.get('dataset')}/{payload.get('eval_name')}"
        )
    model = require_model(registry, model_name)
    if protocol is None:
        if dataset == "recon":
            protocol = "recon_prediction_v1"
        elif evaluation.startswith("rollout_"):
            protocol = "go_stanford_unseen_rollout_v1"
        else:
            protocol = "go_stanford_unseen_v1"
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown prediction protocol: {protocol}")
    protocol_config = PROTOCOLS[protocol]
    category = protocol_config["category"]
    if "dataset" in protocol_config and protocol_config["dataset"] != dataset:
        raise ValueError(f"{dataset} is not the dataset pinned by {protocol}")
    if "datasets" in protocol_config and dataset not in protocol_config["datasets"]:
        raise ValueError(f"{dataset} is not part of {protocol}")
    protocol_evaluations = protocol_config.get("evaluation")
    if isinstance(protocol_evaluations, str):
        protocol_evaluations = [protocol_evaluations]
    if protocol_evaluations is not None and evaluation not in protocol_evaluations:
        raise ValueError(f"{evaluation} is not part of {protocol}")
    expected_count = dataset_sample_count(protocol_config, dataset, evaluation)
    if protocol_config.get("frame_indices") is not None:
        expected_frames = protocol_config["frame_indices"]
    elif evaluation == "time":
        expected_frames = {"1s": 1, "2s": 2, "4s": 4, "8s": 8, "16s": 16}
    elif evaluation.endswith("_1fps"):
        expected_frames = {"1s": 0, "2s": 1, "4s": 3, "8s": 7, "16s": 15}
    elif evaluation.endswith("_4fps"):
        expected_frames = {"1s": 3, "2s": 7, "4s": 15, "8s": 31, "16s": 63}
    else:
        raise ValueError(f"Unknown prediction evaluation: {evaluation}")
    if payload.get("sample_count") != expected_count:
        raise ValueError(
            f"Audit sample count does not match {protocol}: "
            f"expected {expected_count}, got {payload.get('sample_count')}"
        )
    if payload.get("frame_indices") != expected_frames:
        raise ValueError(
            f"Audit frame indices do not match {protocol}/{evaluation}: "
            f"expected {expected_frames}, got {payload.get('frame_indices')}"
        )
    inference_contracts = protocol_config.get("inference")
    if inference_contracts is not None:
        inference = payload.get("inference")
        if inference is None:
            raise ValueError(f"Audit is missing inference provenance required by {protocol}")
        expected_backend = "rae-nwm" if model.get("backend") == "raenwm" else "nwm"
        if inference.get("backend") != expected_backend:
            raise ValueError(
                f"Audit backend does not match {model_name}: expected "
                f"{expected_backend}, got {inference.get('backend')}"
            )
        expected_inference = {
            "backend": expected_backend,
            "sampler": inference_contracts[expected_backend]["sampler"],
            "sampling_steps": inference_contracts[expected_backend]["sampling_steps"],
            "seed": protocol_config["seed"],
        }
        if inference != expected_inference:
            raise ValueError(
                f"Audit inference provenance does not match {protocol}: expected "
                f"{expected_inference}, got {inference}"
            )
    result = {
        "protocol": protocol,
        "source_audit": str(audit.resolve()),
        "sample_count": payload["sample_count"],
        "frame_indices": payload["frame_indices"],
        "metrics": payload["metrics"],
        "imported_at": utc_now(),
    }
    inference = payload.get("inference")
    if inference is None and model.get("backend") == "raenwm":
        provenance = model["provenance"]
        inference = {
            "backend": "rae-nwm",
            "sampler": f"{provenance['sampling_method']}_ode",
            "sampling_steps": provenance["sampling_steps"],
        }
    if inference is not None:
        result["inference"] = inference
    model["results"].setdefault(category, {}).setdefault(dataset, {})[
        evaluation
    ] = result


def import_planning(
    registry: dict[str, Any],
    model_name: str,
    dataset: str,
    metrics_path: Path,
    protocol: str = "navigation_cem80_v1",
) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    prefix = f"{dataset}_"
    required = ("ate", "rpe_trans", "pos_diff_norm", "yaw_diff_norm")
    missing = [key for key in required if prefix + key not in payload]
    if missing:
        raise ValueError(f"Planning result is missing {missing}: {metrics_path}")
    if protocol not in registry["protocols"]:
        raise ValueError(f"Unknown planning protocol: {protocol}")
    protocol_config = registry["protocols"][protocol]
    if dataset not in protocol_config["datasets"]:
        raise ValueError(f"{dataset} is not part of {protocol}")
    category = protocol_config["category"]
    model = require_model(registry, model_name)
    expected_count = dataset_sample_count(protocol_config, dataset, "navigation")
    if payload.get("sample_count", expected_count) != expected_count:
        raise ValueError(
            f"Planning sample count does not match {protocol}: expected "
            f"{expected_count}, got {payload.get('sample_count')}"
        )
    result = {
        "protocol": protocol,
        "source_metrics": str(metrics_path.resolve()),
        "sample_count": expected_count,
        "metrics": {key: payload[prefix + key] for key in required},
        "total_time_seconds": payload.get("total_time"),
        "imported_at": utc_now(),
    }
    if payload.get("inference") is not None:
        result["inference"] = payload["inference"]
    model["results"].setdefault(category, {})[dataset] = result


def sync_existing(registry: dict[str, Any]) -> None:
    comparison = Path(
        "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_comparison_20260825"
    )
    release = Path(
        "/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/nwm_cdit_b"
    )
    roots = {
        "nwm-base": comparison / "nwm_base_200k",
        "nwm-real": comparison / "nwm_real_200k",
        "nwm-release": release,
    }
    for model_name, root in roots.items():
        for evaluation in ("time", "rollout_1fps", "rollout_4fps"):
            audit = root / f"recon_{evaluation}_audit.json"
            if audit.exists():
                import_prediction(registry, model_name, "recon", evaluation, audit)
    release_planning = Path(
        "/file_system/nas/algorithm/dujun.nie/nwm/results/release_eval_20260820/"
        "planning_compact80/nwm_cdit_b/recon_CEM_N80_K5_RS1_rep3_OPT1.json"
    )
    if release_planning.exists():
        import_planning(registry, "nwm-release", "recon", release_planning)


def metric_mean(metrics: dict[str, Any], key: str) -> float:
    return sum(float(item[key]) for item in metrics.values()) / len(metrics)


def iter_ood_direct_rows(registry: dict[str, Any]):
    protocol = registry["protocols"].get("ood_direct_4s_v1")
    if protocol is None:
        return
    for model_name, model in registry["models"].items():
        results = model.get("results", {}).get("ood_direct_prediction", {})
        for dataset in protocol["datasets"]:
            result = results.get(dataset, {}).get("time")
            if result is not None:
                yield model_name, dataset, result["metrics"]["4s"], result


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def render_ood_latex(registry: dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{\textbf{Out-of-domain direct visual prediction at the nominal "
            r"4\,s / 16-step horizon.} Rows use each dataset's pinned sample count. "
            r"Planetary Rover is a spatial waypoint sequence without physical fixed-rate "
            r"timestamps. DS: DreamSim.}"
        ),
        r"\label{tab:ood_direct_4s}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Dataset} & LPIPS$\downarrow$ & DS$\downarrow$ & PSNR$\uparrow$ \\",
        r"\midrule",
    ]
    rows = list(iter_ood_direct_rows(registry))
    for model_name, dataset, metrics, _ in rows:
        display_name = registry["models"][model_name].get("display_name", model_name)
        lines.append(
            f"{latex_escape(display_name)} & {latex_escape(dataset)} & "
            f"{metrics['lpips_alex']:.3f} & {metrics['dreamsim']:.3f} & "
            f"{metrics['psnr']:.3f} \\\\"
        )
    if not rows:
        lines.append(r"\multicolumn{5}{c}{Results pending} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def render_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# NWM benchmark registry",
        "",
        f"Updated: `{registry['updated_at']}`",
        "",
        "Lower is better for LPIPS, DreamSim, ATE and RPE; higher is better for PSNR.",
        "",
        "## RECON prediction (mean over 1/2/4/8/16 seconds)",
        "",
        "| Model | Mode | LPIPS | DreamSim | PSNR |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name, model in registry["models"].items():
        evaluations = (
            model.get("results", {}).get("recon_prediction", {}).get("recon", {})
        )
        for evaluation in ("time", "rollout_1fps", "rollout_4fps"):
            if evaluation not in evaluations:
                continue
            metrics = evaluations[evaluation]["metrics"]
            lines.append(
                f"| {model_name} | {evaluation} | {metric_mean(metrics, 'lpips_alex'):.6f} "
                f"| {metric_mean(metrics, 'dreamsim'):.6f} | {metric_mean(metrics, 'psnr'):.6f} |"
            )

    lines.extend(
        [
            "",
            "### RECON prediction details",
            "",
            "| Model | Mode | Horizon | LPIPS | DreamSim | PSNR | Samples |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        evaluations = (
            model.get("results", {}).get("recon_prediction", {}).get("recon", {})
        )
        for evaluation in ("time", "rollout_1fps", "rollout_4fps"):
            if evaluation not in evaluations:
                continue
            result = evaluations[evaluation]
            for horizon in ("1s", "2s", "4s", "8s", "16s"):
                metrics = result["metrics"][horizon]
                lines.append(
                    f"| {model_name} | {evaluation} | {horizon} | "
                    f"{metrics['lpips_alex']:.6f} | {metrics['dreamsim']:.6f} | "
                    f"{metrics['psnr']:.6f} | {metrics['sample_count']} |"
                )

    lines.extend(
        [
            "",
            "## Navigation planning (CEM-80)",
            "",
            "| Model/baseline | Dataset | ATE | RPE | Protocol note |",
            "|---|---|---:|---:|---|",
        ]
    )
    navigation_protocol = registry["protocols"]["navigation_cem80_v1"]
    for model_name, model in registry["models"].items():
        results = model.get("results", {}).get("navigation_planning", {})
        for dataset in navigation_protocol["datasets"]:
            if dataset in results:
                metrics = results[dataset]["metrics"]
                temporal_semantics = navigation_protocol["splits"][dataset].get(
                    "temporal_semantics"
                )
                note = "measured, N=80"
                if temporal_semantics == "spatial_index":
                    note += ", spatial H=8"
                lines.append(
                    f"| {model_name} | {dataset} | {metrics['ate']:.6f} | "
                    f"{metrics['rpe_trans']:.6f} | {note} |"
                )
    compact = registry["paper_baselines"]["compact_paper"]["results"][
        "navigation_planning"
    ]
    for baseline in ("sd_vae", "compact_16_tokens", "compact_8_tokens"):
        for dataset, metrics in compact[baseline].items():
            lines.append(
                f"| CompACT paper: {baseline} | {dataset} | {metrics['ate']:.3f} | "
                f"{metrics['rpe_trans']:.3f} | paper, N=80 |"
            )
    nwm = registry["paper_baselines"]["nwm_paper"]["results"]["navigation_planning"]
    for dataset in ("recon", "scand"):
        metrics = nwm[dataset]
        lines.append(
            f"| NWM paper | {dataset} | {metrics['ate']:.3f} | "
            f"{metrics['rpe_trans']:.3f} | paper, N=120 |"
        )

    lines.extend(
        [
            "",
            "### Navigation planning fast-10step",
            "",
            "CEM settings remain N=80, K=5, M=3 and H=8; diffusion sampling is reduced from 250 to 10 steps, so these rows are not paper-protocol equivalents.",
            "",
            "| Model | Dataset | ATE | RPE | Protocol |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model_name, model in registry["models"].items():
        results = model.get("results", {}).get("navigation_planning_fast", {})
        for dataset in ("recon", "scand"):
            if dataset in results:
                metrics = results[dataset]["metrics"]
                lines.append(
                    f"| {model_name} | {dataset} | {metrics['ate']:.6f} | "
                    f"{metrics['rpe_trans']:.6f} | fast-10step |"
                )

    lines.extend(
        [
            "",
            "## Go Stanford unseen (all measured horizons)",
            "",
            "| Model | Mode | Horizon | LPIPS | DreamSim | PSNR | Samples |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        evaluations = (
            model.get("results", {})
            .get("unseen_generalization", {})
            .get("go_stanford", {})
        )
        for evaluation in (
            "time",
            "rollout_1fps",
            "rollout_4fps",
        ):
            result = evaluations.get(evaluation)
            if not result:
                continue
            for horizon in ("1s", "2s", "4s", "8s", "16s"):
                metrics = result["metrics"][horizon]
                lines.append(
                    f"| {model_name} | {evaluation} | {horizon} | "
                    f"{metrics['lpips_alex']:.6f} | "
                    f"{metrics['dreamsim']:.6f} | {metrics['psnr']:.6f} | "
                    f"{metrics['sample_count']} |"
                )

    lines.extend(
        [
            "",
            "### Go Stanford paper comparison at 4 seconds",
            "",
            "| Model/baseline | LPIPS | DreamSim | PSNR |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        result = (
            model.get("results", {})
            .get("unseen_generalization", {})
            .get("go_stanford", {})
            .get("time")
        )
        if result:
            metrics = result["metrics"]["4s"]
            lines.append(
                f"| {model_name} | {metrics['lpips_alex']:.6f} | "
                f"{metrics['dreamsim']:.6f} | {metrics['psnr']:.6f} |"
            )
    unseen = registry["paper_baselines"]["nwm_paper"]["results"][
        "unseen_generalization"
    ]
    for name in ("in_domain_data", "plus_ego4d"):
        metrics = unseen[name]
        lines.append(
            f"| NWM paper: {name} | {metrics['lpips_alex']:.3f} | "
            f"{metrics['dreamsim']:.3f} | {metrics['psnr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Unified direct visual prediction at 4 seconds",
            "",
            (
                "All rows use `direct_4s_v1` with each dataset's pinned sample count, "
                "sample-ID keyed randomness, "
                "250-step DDPM for NWM or 50-step Euler for RAE-NWM."
            ),
            "",
            "| Model | Dataset | LPIPS | DreamSim | PSNR | Samples |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        results = model.get("results", {}).get("direct_prediction", {})
        for dataset in ALL_DATASETS:
            result = results.get(dataset, {}).get("time")
            if not result:
                continue
            metrics = result["metrics"]["4s"]
            lines.append(
                f"| {model_name} | {dataset} | {metrics['lpips_alex']:.6f} | "
                f"{metrics['dreamsim']:.6f} | {metrics['psnr']:.6f} | "
                f"{metrics['sample_count']} |"
            )

    lines.extend(
        [
            "",
            "## Unified autoregressive rollout",
            "",
            (
                "All rows use `rollout_v1`: all 150 fixed samples and exact 1/4-fps "
                "autoregressive rollout at 1/2/4/8/16 seconds."
            ),
            "",
            "| Model | Dataset | Mode | Horizon | LPIPS | DreamSim | PSNR | Samples |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, model in registry["models"].items():
        results = model.get("results", {}).get("rollout_prediction", {})
        for dataset in ALL_DATASETS:
            evaluations = results.get(dataset, {})
            for evaluation in ("rollout_1fps", "rollout_4fps"):
                result = evaluations.get(evaluation)
                if not result:
                    continue
                for horizon in ("1s", "2s", "4s", "8s", "16s"):
                    metrics = result["metrics"][horizon]
                    lines.append(
                        f"| {model_name} | {dataset} | {evaluation} | {horizon} | "
                        f"{metrics['lpips_alex']:.6f} | {metrics['dreamsim']:.6f} | "
                        f"{metrics['psnr']:.6f} | {metrics['sample_count']} |"
                    )
    lines.extend(
        [
            "",
            "## Out-of-domain direct visual prediction at 4 seconds",
            "",
            (
                "Each measured row uses the pinned `ood_direct_4s_v1` protocol and its "
                "dataset-specific fixed sample count. The horizon is physical 4 s for "
                "the fixed-4-Hz datasets; `planetary_rover` uses the same 16-step spatial "
                "horizon because physical fixed-rate timestamps are unavailable."
            ),
            "",
            "| Model | Dataset | LPIPS | DreamSim | PSNR | Samples |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_name, dataset, metrics, result in iter_ood_direct_rows(registry):
        lines.append(
            f"| {model_name} | {dataset} | {metrics['lpips_alex']:.6f} | "
            f"{metrics['dreamsim']:.6f} | {metrics['psnr']:.6f} | "
            f"{result['sample_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("initialize")
    subparsers.add_parser("sync-existing")

    register = subparsers.add_parser("register-model")
    register.add_argument("--model", required=True)
    register.add_argument("--exp-dir", type=Path, required=True)
    register.add_argument("--checkpoint-id", required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--checkpoint-step", type=int)
    register.add_argument("--sha256")
    register.add_argument("--training-datasets", type=csv_list, default=[])

    prediction = subparsers.add_parser("import-prediction")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--dataset", required=True)
    prediction.add_argument("--evaluation", required=True)
    prediction.add_argument("--audit", type=Path, required=True)
    prediction.add_argument("--protocol")

    planning = subparsers.add_parser("import-planning")
    planning.add_argument("--model", required=True)
    planning.add_argument("--dataset", required=True)
    planning.add_argument("--metrics", type=Path, required=True)
    planning.add_argument("--protocol", default="navigation_cem80_v1")

    render = subparsers.add_parser("render")
    render.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.registry.with_suffix(args.registry.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        registry = load_registry(args.registry)
        if args.command == "sync-existing":
            sync_existing(registry)
        elif args.command == "register-model":
            register_model(
                registry,
                args.model,
                args.exp_dir,
                args.checkpoint_id,
                args.checkpoint,
                args.checkpoint_step,
                args.sha256,
                args.training_datasets,
            )
        elif args.command == "import-prediction":
            import_prediction(
                registry,
                args.model,
                args.dataset,
                args.evaluation,
                args.audit,
                args.protocol,
            )
        elif args.command == "import-planning":
            import_planning(
                registry,
                args.model,
                args.dataset,
                args.metrics,
                args.protocol,
            )
        elif args.command == "render":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(render_markdown(registry), encoding="utf-8")
            latex_output = args.output.with_name("ood_direct_4s_table.tex")
            latex_output.write_text(render_ood_latex(registry), encoding="utf-8")
            print(args.output)
            print(latex_output)
            return
        save_registry(args.registry, registry)
        if args.command in {"import-prediction", "import-planning"}:
            rendered = render_markdown(registry)
            for filename in ("benchmark_results.md", "benchmark_comparison.md"):
                output = args.registry.with_name(filename)
                output.write_text(rendered, encoding="utf-8")
            args.registry.with_name("ood_direct_4s_table.tex").write_text(
                render_ood_latex(registry), encoding="utf-8"
            )
        print(args.registry)


if __name__ == "__main__":
    main()
