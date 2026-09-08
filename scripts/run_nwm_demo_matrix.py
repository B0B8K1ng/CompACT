#!/usr/bin/env python3
"""Run the requested NWM demo matrix with resumable per-case outputs."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo_nwm_rollout.py"
NAS = Path("/file_system/nas/algorithm/dujun.nie/nwm")
DATA = NAS / "data"
LEGACY_DATASETS = NAS / "demo_outputs/figure11_15_4datasets_20260907"
LEGACY_HONGYU = NAS / "demo_outputs/hongyu_3actions_20260907"
DEFAULT_OUTPUT = NAS / "demo_outputs/nwm_rollout_matrix_20260907_v2"
REALESTATE10K = DATA / "NavAnywhere/RealEstate10K"
REAL_WORLD_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CHECKPOINTS = {
    "nwm-timept-ft": NAS
    / "compact/runs/navanywherev1_timept_ft/nwm-nav1-timept-finetune/checkpoints/joint_0100000.pth.tar",
    "nwm-real": NAS
    / "compact/runs/20260823_160459_nwm_real_recon_scand_tartan_huron_bs16/checkpoints/0200000.pth.tar",
    "nwm-no-pretrain": NAS
    / "compact/runs/no_pretrain_ft/nwm-no-pretrain-finetune/checkpoints/joint_0110000.pth.tar",
}


@dataclass(frozen=True)
class DatasetCase:
    name: str
    folder: str
    trajectory: str
    current_time: int
    spacing: float
    existing_actions_name: str | None = None
    seconds: float = 16.0

    @property
    def trajectory_dir(self) -> Path:
        return DATA / self.folder / self.trajectory


DATASETS = (
    DatasetCase(
        "recon", "recon", "jackal_2019-10-24-13-35-07_2_r02", 123, 0.25, "recon"
    ),
    DatasetCase(
        "scand",
        "scand",
        "random_mdps_A_Jackal_AHG_Library_Thu_Oct_28_1_0",
        891,
        0.38,
        "scand",
    ),
    DatasetCase(
        "huron",
        "sacson",
        "Feb-16-2023-cory1-intloss_00000007_2",
        131,
        0.255,
        "huron",
    ),
    DatasetCase(
        "tartandrive",
        "tartan",
        "20210903_heightmaps_9_20210903_229_0",
        243,
        0.72,
        "tartandrive",
    ),
    DatasetCase(
        "go_stanford",
        "go_stanford",
        "no29vc_35_0",
        3,
        0.12,
        seconds=4.0,
    ),
)
PRESETS = ("forward", "forward_then_left", "forward_then_right")
REALESTATE10K_SAMPLES = (
    "0005c41463fe5e01",
    "00066b3649cc07e5",
    "0006e8e3eaa8cd39",
)


def _delta_actions(case: DatasetCase) -> np.ndarray:
    """Reproduce BaseDataset._compute_actions followed by get_delta_np."""
    frame_count = round(case.seconds * 4)
    with (case.trajectory_dir / "traj_data.pkl").open("rb") as handle:
        trajectory = pickle.load(handle)
    positions = np.asarray(trajectory["position"], dtype=np.float64)[
        case.current_time : case.current_time + frame_count + 1
    ]
    yaw = np.asarray(trajectory["yaw"], dtype=np.float64)[
        case.current_time : case.current_time + frame_count + 1
    ].reshape(-1)
    if len(positions) != frame_count + 1 or len(yaw) != frame_count + 1:
        raise ValueError(
            f"{case.name} does not contain {frame_count} future frames"
        )
    angle = float(yaw[0])
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    local_positions = (positions[:, :2] - positions[0, :2]).dot(rotation)
    local_positions /= case.spacing
    local_yaw = yaw - yaw[0]
    local_yaw -= 2 * np.pi * np.floor((local_yaw + np.pi) / (2 * np.pi))
    cumulative = np.concatenate([local_positions, local_yaw[:, None]], axis=1)[1:]
    return np.diff(
        np.concatenate([np.zeros((1, 3)), cumulative], axis=0), axis=0
    ).astype(np.float32)


def _actions_path(case: DatasetCase, output_root: Path) -> Path:
    if case.existing_actions_name is not None:
        return LEGACY_DATASETS / "inputs" / case.existing_actions_name / "actions.json"
    path = output_root / "inputs" / case.name / "actions.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("actions", [])) == round(case.seconds * 4):
            return path
        same_source = (
            payload.get("source_dataset_key") == case.folder
            and payload.get("trajectory") == case.trajectory
            and payload.get("current_time") == case.current_time
        )
        if not same_source:
            raise ValueError(f"Refusing to replace unrelated actions file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = _delta_actions(case)
    payload = {
        "fps": 4,
        "rollout_seconds": case.seconds,
        "translation_unit": "waypoint_spacing_units",
        "yaw_unit": "radians",
        "coordinate_frame": "first_image",
        "dataset": "Go Stanford",
        "source_dataset_key": case.folder,
        "trajectory": case.trajectory,
        "current_time": case.current_time,
        "metric_waypoint_spacing_meters": case.spacing,
        "actions": actions.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _real_world_images() -> list[Path]:
    image_dir = NAS / "real_world"
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in REAL_WORLD_EXTENSIONS
    )


def _run(
    command: list[str], destination: Path, expected_seconds: float = 16.0
) -> None:
    if (destination / "manifest.json").is_file():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if float(manifest.get("seconds", -1)) == expected_seconds:
            print(f"SKIP complete: {destination}", flush=True)
            return
        raise FileExistsError(
            f"Existing output has seconds={manifest.get('seconds')}, expected "
            f"{expected_seconds}: {destination}"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Incomplete non-empty output needs inspection: {destination}")
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def _dataset_jobs(args: argparse.Namespace, checkpoint: Path) -> None:
    for case in DATASETS:
        if args.datasets and case.name not in args.datasets:
            continue
        destination = args.output_root / args.model / "datasets" / case.name
        common = [
            sys.executable,
            str(DEMO),
            "--output-dir",
            str(destination),
            "--gt-frames-dir",
            str(case.trajectory_dir),
            "--gt-start-index",
            str(case.current_time + 1),
            "--seconds",
            str(case.seconds),
        ]
        reusable = (
            args.model == "nwm-timept-ft"
            and case.existing_actions_name is not None
            and (LEGACY_DATASETS / case.existing_actions_name / "manifest.json").is_file()
        )
        if args.reuse_only and not reusable:
            print(f"SKIP inference-only case: {destination}", flush=True)
            continue
        if reusable:
            command = common + [
                "--postprocess-from",
                str(LEGACY_DATASETS / case.existing_actions_name),
            ]
        else:
            command = common + [
                "--checkpoint",
                str(checkpoint),
                "--first-image",
                str(case.trajectory_dir / f"{case.current_time}.jpg"),
                "--actions",
                str(_actions_path(case, args.output_root)),
                "--device",
                args.device,
                "--feedback-mode",
                "pixel",
            ]
        _run(command, destination, expected_seconds=case.seconds)


def _real_world_jobs(args: argparse.Namespace, checkpoint: Path) -> None:
    images = _real_world_images()
    if not images:
        raise FileNotFoundError(f"No jpg images found under {NAS / 'real_world'}")
    for image in images:
        image_name = "hongyu" if image.stem == "hongyu_test" else image.stem
        if args.real_images and image_name not in args.real_images:
            continue
        for preset in PRESETS:
            if args.presets and preset not in args.presets:
                continue
            destination = args.output_root / args.model / "real_world" / image_name / preset
            reusable = (
                args.model == "nwm-timept-ft"
                and image.stem == "hongyu_test"
                # The old 16 s turning presets stayed straight for their first
                # four seconds, so only the straight rollout can be truncated.
                and preset == "forward"
                and (LEGACY_HONGYU / preset / "manifest.json").is_file()
            )
            if args.reuse_only and not reusable:
                print(f"SKIP inference-only case: {destination}", flush=True)
                continue
            if reusable:
                command = [
                    sys.executable,
                    str(DEMO),
                    "--output-dir",
                    str(destination),
                    "--postprocess-from",
                    str(LEGACY_HONGYU / preset),
                    "--seconds",
                    "4",
                ]
            else:
                command = [
                    sys.executable,
                    str(DEMO),
                    "--checkpoint",
                    str(checkpoint),
                    "--first-image",
                    str(image),
                    "--preset",
                    preset,
                    "--seconds",
                    "4",
                    "--output-dir",
                    str(destination),
                    "--device",
                    args.device,
                    "--feedback-mode",
                    "pixel",
                ]
            _run(command, destination, expected_seconds=4.0)


def _realestate10k_jobs(args: argparse.Namespace, checkpoint: Path) -> None:
    if args.model != "nwm-timept-ft":
        raise ValueError("RealEstate10K demo jobs are only defined for nwm-timept-ft")
    for sample in REALESTATE10K_SAMPLES:
        image = REALESTATE10K / sample / "0.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        for preset in PRESETS:
            if args.presets and preset not in args.presets:
                continue
            destination = (
                args.output_root / args.model / "realestate10k" / sample / preset
            )
            command = [
                sys.executable,
                str(DEMO),
                "--checkpoint",
                str(checkpoint),
                "--first-image",
                str(image),
                "--preset",
                preset,
                "--seconds",
                "4",
                "--output-dir",
                str(destination),
                "--device",
                args.device,
                "--feedback-mode",
                "pixel",
            ]
            _run(command, destination, expected_seconds=4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scope",
        choices=("datasets", "real_world", "realestate10k", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Only re-render compatible existing timept outputs; do not run a model",
    )
    parser.add_argument(
        "--datasets",
        help="Optional comma-separated dataset case names",
    )
    parser.add_argument(
        "--real-images",
        help="Optional comma-separated image stems from the real_world directory",
    )
    parser.add_argument(
        "--presets",
        help="Optional comma-separated preset names",
    )
    args = parser.parse_args()
    args.datasets = set(filter(None, (args.datasets or "").split(",")))
    args.real_images = set(filter(None, (args.real_images or "").split(",")))
    args.presets = set(filter(None, (args.presets or "").split(",")))
    unknown_datasets = args.datasets - {case.name for case in DATASETS}
    known_images = {
        "hongyu" if image.stem == "hongyu_test" else image.stem
        for image in _real_world_images()
    }
    unknown_images = args.real_images - known_images
    unknown_presets = args.presets - set(PRESETS)
    if unknown_datasets or unknown_images or unknown_presets:
        parser.error(
            f"unknown filters: datasets={sorted(unknown_datasets)}, "
            f"real_images={sorted(unknown_images)}, presets={sorted(unknown_presets)}"
        )
    return args


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    checkpoint = CHECKPOINTS[args.model]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.scope in {"datasets", "all"}:
        _dataset_jobs(args, checkpoint)
    if args.scope in {"real_world", "all"}:
        _real_world_jobs(args, checkpoint)
    if args.scope == "realestate10k":
        _realestate10k_jobs(args, checkpoint)


if __name__ == "__main__":
    main()
