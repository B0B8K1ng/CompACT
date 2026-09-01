#!/usr/bin/env python3
"""Validate VGGT-Omega raw poses and TartanDrive geometry-action caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry_action.validation import (
    ValidationOptions,
    ValidationThresholds,
    validate_dataset,
)


def _optional_float(value: str) -> float:
    if value.lower() in {"none", "null", "off"}:
        return None  # type: ignore[return-value]
    return float(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict, resumable validation of per-trajectory VGGT-Omega camera poses "
            "and OfflineMotionStore geometry actions. GT is read only for metrics."
        )
    )
    parser.add_argument("--raw-pose-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument(
        "--data-root", help="TartanDrive root containing <traj>/traj_data.pkl"
    )
    parser.add_argument("--dataset-name", default="tartan_drive")
    parser.add_argument(
        "--split-files",
        nargs="+",
        required=True,
        help="One or more traj_names.txt files; duplicates are removed in first-seen order",
    )
    parser.add_argument(
        "--output", required=True, help="Incremental machine-readable JSON report"
    )
    parser.add_argument("--raw-pattern", default="{dataset_name}/{trajectory_name}.pt")
    parser.add_argument(
        "--geometry-pattern", default="{dataset_name}/{trajectory_name}.pt"
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-frame-hashes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--use-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-algebra-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--max-composition-checks", type=int, default=10_000)
    parser.add_argument("--gt-rpe-delta", type=int, default=1)
    parser.add_argument(
        "--waypoint-spacing",
        type=float,
        help=(
            "Known meters per waypoint, if available. If omitted, GT action metrics use "
            "a validation-only per-trajectory median adjacent-step normalization."
        ),
    )

    defaults = ValidationThresholds()
    maximums = (
        ("so3-orthogonality", defaults.max_so3_orthogonality),
        ("so3-det-error", defaults.max_so3_det_error),
        ("identity-translation", defaults.max_identity_translation),
        ("identity-yaw-deg", defaults.max_identity_yaw_deg),
        ("inverse-translation", defaults.max_inverse_translation),
        ("inverse-yaw-deg", defaults.max_inverse_yaw_deg),
        ("composition-translation", defaults.max_composition_translation),
        ("composition-yaw-deg", defaults.max_composition_yaw_deg),
        ("overlap-center-rmse", defaults.max_overlap_center_rmse),
        ("overlap-rotation-rmse-deg", defaults.max_overlap_rotation_rmse_deg),
        ("pose-action-translation", defaults.max_pose_action_translation),
        ("pose-action-yaw-deg", defaults.max_pose_action_yaw_deg),
        ("gt-ate-rmse", defaults.max_gt_ate_rmse),
        ("gt-rpe-translation-rmse", defaults.max_gt_rpe_translation_rmse),
        ("gt-rpe-yaw-mae-deg", defaults.max_gt_rpe_yaw_mae_deg),
        ("gt-rpe-rotation-mae-deg", defaults.max_gt_rpe_rotation_mae_deg),
        ("action-translation-rmse", defaults.max_action_translation_rmse),
        ("action-yaw-mae-deg", defaults.max_action_yaw_mae_deg),
    )
    for name, default in maximums:
        parser.add_argument(
            f"--max-{name}",
            type=_optional_float,
            default=default,
            help="Use 'none' to report this metric without failing on it.",
        )
    parser.add_argument(
        "--min-action-direction-cosine",
        type=_optional_float,
        default=defaults.min_action_direction_cosine,
    )
    parser.add_argument(
        "--min-action-yaw-sign-agreement",
        type=_optional_float,
        default=defaults.min_action_yaw_sign_agreement,
    )
    return parser


def _read_names(paths: list[str]) -> list[str]:
    names: list[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            names.extend(line.strip() for line in handle if line.strip())
    return list(dict.fromkeys(names))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_frame_hashes and not args.data_root:
        _parser().error("--verify-frame-hashes requires --data-root")
    if args.use_gt and not args.data_root:
        _parser().error("--use-gt requires --data-root")
    if args.max_composition_checks < 1 or args.gt_rpe_delta < 1:
        _parser().error("composition checks and GT RPE delta must be positive")

    thresholds = ValidationThresholds(
        max_so3_orthogonality=args.max_so3_orthogonality,
        max_so3_det_error=args.max_so3_det_error,
        max_identity_translation=args.max_identity_translation,
        max_identity_yaw_deg=args.max_identity_yaw_deg,
        max_inverse_translation=args.max_inverse_translation,
        max_inverse_yaw_deg=args.max_inverse_yaw_deg,
        max_composition_translation=args.max_composition_translation,
        max_composition_yaw_deg=args.max_composition_yaw_deg,
        max_overlap_center_rmse=args.max_overlap_center_rmse,
        max_overlap_rotation_rmse_deg=args.max_overlap_rotation_rmse_deg,
        max_pose_action_translation=args.max_pose_action_translation,
        max_pose_action_yaw_deg=args.max_pose_action_yaw_deg,
        max_gt_ate_rmse=args.max_gt_ate_rmse,
        max_gt_rpe_translation_rmse=args.max_gt_rpe_translation_rmse,
        max_gt_rpe_yaw_mae_deg=args.max_gt_rpe_yaw_mae_deg,
        max_gt_rpe_rotation_mae_deg=args.max_gt_rpe_rotation_mae_deg,
        max_action_translation_rmse=args.max_action_translation_rmse,
        max_action_yaw_mae_deg=args.max_action_yaw_mae_deg,
        min_action_direction_cosine=args.min_action_direction_cosine,
        min_action_yaw_sign_agreement=args.min_action_yaw_sign_agreement,
    )
    options = ValidationOptions(
        strict=args.strict,
        verify_frame_hashes=args.verify_frame_hashes,
        require_algebra_coverage=args.require_algebra_coverage,
        max_composition_checks=args.max_composition_checks,
        gt_rpe_delta=args.gt_rpe_delta,
        waypoint_spacing=args.waypoint_spacing,
        thresholds=thresholds,
    )
    document = validate_dataset(
        _read_names(args.split_files),
        raw_pose_root=args.raw_pose_root,
        geometry_root=args.geometry_root,
        dataset_name=args.dataset_name,
        output_path=args.output,
        data_root=args.data_root,
        raw_pattern=args.raw_pattern,
        geometry_pattern=args.geometry_pattern,
        use_gt=args.use_gt,
        resume=args.resume,
        flush_every=args.flush_every,
        options=options,
    )
    print(json.dumps(document["summary"], sort_keys=True, allow_nan=False))
    return 0 if document["summary"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
