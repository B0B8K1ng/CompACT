#!/usr/bin/env python3
"""Build a manifest and extract VGGT-Omega TartanDrive geometry actions."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from geometry_action.tartandrive_vggt_omega import (
    DEFAULT_MODEL_REVISION,
    PINNED_CODE_REVISION,
    VGGTOmegaCameraExtractor,
    atomic_json_dump,
    atomic_torch_save,
    build_geometry_payload,
    build_input_manifest,
    build_raw_pose_payload,
    deterministic_shard,
    frame_paths_from_manifest,
    geometry_payload_matches,
    geometry_policy_descriptor,
    raw_extraction_descriptor,
    raw_pose_matches,
    safe_torch_load,
    sha256_file,
    validate_input_manifest,
)

EXPECTED_ALL_TRAJECTORIES = 1251
EXPECTED_ALL_FRAMES = 62884


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument(
        "--split", choices=("train", "test", "all"), default="all"
    )
    parser.add_argument("--dataset-name", default="tartan_drive")
    parser.add_argument(
        "--trajectory",
        action="append",
        default=None,
        help="Restrict to one or more named trajectories (repeatable).",
    )


def _split_names(args: argparse.Namespace) -> list[str]:
    if getattr(args, "splits", None):
        if args.split != "all":
            raise ValueError("Use either --split or --splits, not both")
        return list(dict.fromkeys(args.splits))
    return ["train", "test"] if args.split == "all" else [args.split]


def _git_revision(directory: str | os.PathLike[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_matrix(specification: str | None) -> np.ndarray | None:
    if specification is None:
        return None
    candidate = Path(specification)
    text = (
        candidate.read_text(encoding="utf-8")
        if candidate.is_file()
        else specification
    )
    value = json.loads(text)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("--camera-to-navigation must encode a JSON 4x4 matrix")
    return matrix


def _checkpoint_receipt(
    *,
    checkpoint_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str] | None,
    model_revision: str,
    expected_code_revision: str,
    expected_checkpoint_sha256: str | None,
) -> tuple[str, str, Path]:
    """Read the fetcher's atomic completion receipt without rehashing 4.6 GiB."""

    checkpoint = Path(checkpoint_path).resolve()
    receipt_path = (
        Path(manifest_path).resolve()
        if manifest_path
        else checkpoint.parent / "checkpoint_manifest.json"
    )
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint completion manifest is required: {receipt_path}. "
            "Run scripts/fetch_vggt_omega.py --verify-only once before extraction."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != 1:
        raise ValueError(f"Invalid checkpoint completion manifest: {receipt_path}")
    source = receipt.get("source")
    official_code = receipt.get("official_code")
    if not isinstance(source, Mapping) or source.get("resolved_revision") != model_revision:
        raise ValueError(
            "Checkpoint manifest resolved_revision does not match --model-revision"
        )
    if (
        not isinstance(official_code, Mapping)
        or official_code.get("commit") != expected_code_revision
    ):
        raise ValueError("Checkpoint manifest official code revision mismatch")
    entries = receipt.get("files")
    if not isinstance(entries, list):
        raise TypeError("Checkpoint manifest files must be a list")
    entry = next(
        (
            value
            for value in entries
            if isinstance(value, Mapping) and value.get("path") == checkpoint.name
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"Checkpoint manifest has no entry for {checkpoint.name}")
    if not checkpoint.is_file() or checkpoint.stat().st_size != entry.get("bytes"):
        raise ValueError("Checkpoint is missing or its byte count differs from the receipt")
    digest = entry.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError("Checkpoint manifest contains an invalid SHA256")
    if expected_checkpoint_sha256 and digest != expected_checkpoint_sha256:
        raise ValueError("Checkpoint receipt SHA256 differs from the expected SHA256")
    return digest, sha256_file(receipt_path), receipt_path


def _manifest_command(args: argparse.Namespace) -> int:
    manifest = build_input_manifest(
        data_root=args.data_root,
        split_root=args.split_root,
        splits=_split_names(args),
        dataset_name=args.dataset_name,
        selected_trajectories=args.trajectory,
        workers=args.workers,
    )
    if (
        set(_split_names(args)) == {"train", "test"}
        and len(_split_names(args)) == 2
        and not args.trajectory
        and not args.skip_expected_count_check
    ):
        actual = (manifest["num_trajectories"], manifest["num_frames"])
        expected = (args.expected_trajectories, args.expected_frames)
        if actual != expected:
            raise ValueError(
                f"Full TartanDrive count mismatch: actual={actual}, expected={expected}"
            )
    atomic_json_dump(manifest, args.manifest_path)
    result = {
        "status": "complete",
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "manifest_sha256": sha256_file(args.manifest_path),
        "num_trajectories": manifest["num_trajectories"],
        "num_frames": manifest["num_frames"],
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _validate_manifest_frame_files(
    trajectory: Mapping[str, Any],
    *,
    data_root: str | os.PathLike[str],
    verify_hashes: bool,
) -> list[Path]:
    paths = frame_paths_from_manifest(trajectory, data_root)
    for path, record in zip(paths, trajectory["frame_records"]):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest frame disappeared: {path}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Manifest frame size changed: {path}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"Manifest frame content changed: {path}")
    return paths


def _select_manifest_items(
    manifest: Mapping[str, Any], args: argparse.Namespace
) -> list[Mapping[str, Any]]:
    items = list(manifest["trajectories"])
    if args.split != "all":
        items = [item for item in items if item["split"] == args.split]
    if args.trajectory:
        requested = set(args.trajectory)
        known = {str(item["trajectory_name"]) for item in items}
        missing = sorted(requested.difference(known))
        if missing:
            raise ValueError(f"Requested trajectories are absent from selection: {missing}")
        items = [item for item in items if item["trajectory_name"] in requested]
    if args.limit is not None:
        items = sorted(items, key=lambda item: item["trajectory_name"])[: args.limit]
    if not items:
        raise ValueError("The selected extraction shard has no trajectories before sharding")
    return deterministic_shard(
        items,
        args.rank,
        args.world_size,
        window_size=args.window_size,
        overlap=args.overlap,
        cost_power=args.shard_cost_power,
    )


def _output_paths(
    args: argparse.Namespace, dataset_name: str, trajectory_name: str
) -> tuple[Path, Path]:
    output_root = Path(args.output_root).resolve()
    raw_root = (
        Path(args.raw_pose_root).resolve()
        if args.raw_pose_root
        else output_root / "raw_pose"
    )
    motion_root = (
        Path(args.motion_root).resolve()
        if args.motion_root
        else output_root / "geometry_motion"
    )
    return (
        raw_root / dataset_name / f"{trajectory_name}.pt",
        motion_root / dataset_name / f"{trajectory_name}.pt",
    )


def _extract_command(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_input_manifest(manifest)
    manifest_sha256 = sha256_file(manifest_path)
    dataset_name = str(manifest["dataset_name"])
    data_root = str(Path(manifest["data_root"]).resolve())
    if args.data_root and Path(args.data_root).resolve() != Path(data_root):
        raise ValueError("--data-root differs from the immutable manifest data_root")

    third_party_root = Path(args.third_party_root).resolve()
    code_revision = _git_revision(third_party_root)
    if code_revision != args.expected_code_revision:
        raise RuntimeError(
            f"VGGT-Omega code revision {code_revision} != "
            f"{args.expected_code_revision}"
        )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (
        checkpoint_sha256,
        checkpoint_manifest_sha256,
        checkpoint_manifest_path,
    ) = _checkpoint_receipt(
        checkpoint_path=args.checkpoint,
        manifest_path=args.checkpoint_manifest,
        model_revision=args.model_revision,
        expected_code_revision=args.expected_code_revision,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )

    selected = _select_manifest_items(manifest, args)
    camera_to_navigation = _load_matrix(args.camera_to_navigation)
    if args.alignment_policy == "fixed" and (
        camera_to_navigation is None or args.meters_per_model_unit is None
    ):
        raise ValueError(
            "fixed alignment requires --camera-to-navigation and "
            "--meters-per-model-unit"
        )
    if (
        args.alignment_policy == "tartandrive_forward_camera"
        and args.translation_unit != "waypoint_spacing_units"
    ):
        raise ValueError(
            "tartandrive_forward_camera requires waypoint_spacing_units"
        )
    policy_descriptor = (
        geometry_policy_descriptor(
            alignment_policy=args.alignment_policy,
            degenerate_scale_policy=args.degenerate_scale_policy,
            nonzero_epsilon=args.nonzero_step_epsilon,
            camera_to_navigation=camera_to_navigation,
            meters_per_model_unit=args.meters_per_model_unit,
            translation_unit=args.translation_unit,
            waypoint_spacing_meters=args.waypoint_spacing_meters,
            min_offset=args.min_offset,
            max_offset=args.max_offset,
            context_size=args.context_size,
            len_traj_pred=args.len_traj_pred,
        )
        if args.alignment_policy != "raw_only"
        else None
    )
    extraction_descriptor = raw_extraction_descriptor(
        resolution=args.resolution,
        resize_mode=args.resize_mode,
        window_size=args.window_size,
        overlap=args.overlap,
        inference_path=args.inference_path,
        dtype=args.dtype,
        allow_tf32=args.allow_tf32,
        allow_degenerate_window_scale=args.allow_degenerate_window_scale,
        seed=args.seed,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    extractor: VGGTOmegaCameraExtractor | None = None
    comparison_pending = args.compare_full_fast_frames > 0 and (
        args.world_size == 1 or args.rank == 0
    )
    comparison_result: Mapping[str, Any] | None = None
    comparison_seconds = 0.0
    inference_seconds = 0.0
    inferred_frames = 0
    successful: list[str] = []
    skipped: list[str] = []
    failures: list[dict[str, str]] = []
    started = time.monotonic()

    for trajectory in selected:
        trajectory_name = str(trajectory["trajectory_name"])
        raw_path, motion_path = _output_paths(
            args, dataset_name, trajectory_name
        )
        try:
            raw_valid = raw_pose_matches(
                raw_path,
                dataset_name=dataset_name,
                trajectory_name=trajectory_name,
                frame_list_sha256=str(trajectory["frame_list_sha256"]),
                checkpoint_sha256=checkpoint_sha256,
                code_revision=code_revision,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                model_revision=args.model_revision,
                input_manifest_sha256=manifest_sha256,
                extraction_fingerprint=str(extraction_descriptor["fingerprint"]),
            )
            motion_requested = args.alignment_policy != "raw_only"
            motion_valid = (
                geometry_payload_matches(
                    motion_path,
                    dataset_name=dataset_name,
                    trajectory_name=trajectory_name,
                    frame_list_sha256=str(trajectory["frame_list_sha256"]),
                    checkpoint_sha256=checkpoint_sha256,
                    alignment_policy=args.alignment_policy,
                    policy_fingerprint=str(policy_descriptor["fingerprint"]),
                )
                if motion_requested
                else True
            )
            if args.resume and raw_valid and motion_valid:
                skipped.append(trajectory_name)
                print(
                    json.dumps(
                        {"trajectory": trajectory_name, "status": "skipped"},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            if raw_path.exists() and not raw_valid and not args.overwrite:
                raise FileExistsError(
                    f"Existing raw artifact does not match this run: {raw_path}; "
                    "pass --overwrite only after auditing it"
                )
            frame_paths = _validate_manifest_frame_files(
                trajectory,
                data_root=data_root,
                verify_hashes=args.verify_input_hashes,
            )
            raw_rewritten = False
            if not (args.resume and raw_valid):
                if extractor is None:
                    extractor = VGGTOmegaCameraExtractor(
                        third_party_root=third_party_root,
                        checkpoint_path=args.checkpoint,
                        device=args.device,
                        dtype=args.dtype,
                        expected_code_revision=args.expected_code_revision,
                        retain_dense_head=(
                            args.inference_path == "full" or comparison_pending
                        ),
                        allow_tf32=args.allow_tf32,
                    )
                local_comparison: Mapping[str, Any] = {"enabled": False}
                if comparison_pending:
                    comparison_started = time.monotonic()
                    compare_count = min(
                        args.compare_full_fast_frames, len(frame_paths)
                    )
                    compare_images = extractor.preprocess(
                        frame_paths[:compare_count],
                        resize_mode=args.resize_mode,
                        resolution=args.resolution,
                    )
                    comparison_result = extractor.compare_full_and_fast(compare_images)
                    del compare_images
                    if comparison_result.get("allclose") is not True:
                        raise RuntimeError(
                            f"Full-vs-fast camera mismatch: {comparison_result}"
                        )
                    local_comparison = comparison_result
                    comparison_pending = False
                    if args.inference_path == "fast":
                        extractor.discard_dense_head()
                    comparison_seconds += time.monotonic() - comparison_started

                inference_started = time.monotonic()
                (
                    extrinsics,
                    intrinsics,
                    image_size_hw,
                    windows,
                    window_alignment,
                ) = extractor.extract_trajectory(
                    image_paths=frame_paths,
                    window_size=args.window_size,
                    overlap=args.overlap,
                    resize_mode=args.resize_mode,
                    resolution=args.resolution,
                    full=args.inference_path == "full",
                    allow_degenerate_window_scale=args.allow_degenerate_window_scale,
                )
                inference_seconds += time.monotonic() - inference_started
                inferred_frames += len(frame_paths)
                raw_payload = build_raw_pose_payload(
                    dataset_name=dataset_name,
                    trajectory_name=trajectory_name,
                    frame_list_sha256=str(trajectory["frame_list_sha256"]),
                    extrinsics_w2c=extrinsics,
                    intrinsics=intrinsics,
                    image_size_hw=image_size_hw,
                    windows=windows,
                    window_alignment=window_alignment,
                    checkpoint_sha256=checkpoint_sha256,
                    code_revision=code_revision,
                    model_revision=args.model_revision,
                    resize_mode=args.resize_mode,
                    resolution=args.resolution,
                    inference_path=args.inference_path,
                    dtype=args.dtype,
                    fast_full_comparison=local_comparison,
                    input_manifest_sha256=manifest_sha256,
                    checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                    extraction_descriptor=extraction_descriptor,
                )
                atomic_torch_save(raw_payload, raw_path)
                raw_rewritten = True
            else:
                raw_payload = safe_torch_load(raw_path)

            if motion_requested and (
                raw_rewritten or not (args.resume and motion_valid)
            ):
                if motion_path.exists() and not motion_valid and not args.overwrite:
                    raise FileExistsError(
                        f"Existing motion artifact does not match this run: "
                        f"{motion_path}; pass --overwrite only after auditing it"
                    )
                source_pose_sha256 = sha256_file(raw_path)
                geometry_payload = build_geometry_payload(
                    raw_payload,
                    source_pose_sha256=source_pose_sha256,
                    alignment_policy=args.alignment_policy,
                    degenerate_scale_policy=args.degenerate_scale_policy,
                    nonzero_epsilon=args.nonzero_step_epsilon,
                    camera_to_navigation=camera_to_navigation,
                    meters_per_model_unit=args.meters_per_model_unit,
                    translation_unit=args.translation_unit,
                    waypoint_spacing_meters=args.waypoint_spacing_meters,
                    min_offset=args.min_offset,
                    max_offset=args.max_offset,
                    context_size=args.context_size,
                    len_traj_pred=args.len_traj_pred,
                )
                atomic_torch_save(geometry_payload, motion_path)

            successful.append(trajectory_name)
            print(
                json.dumps(
                    {
                        "trajectory": trajectory_name,
                        "status": "complete",
                        "num_frames": trajectory["num_frames"],
                        "raw_path": str(raw_path),
                        "motion_path": str(motion_path) if motion_requested else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        # One bad trajectory is recorded without discarding work from the rest
        # of the deterministic shard; --fail-fast remains available for smoke.
        except Exception as exc:  # noqa: BLE001
            failure = {
                "trajectory": trajectory_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(json.dumps({"status": "failed", **failure}, sort_keys=True), flush=True)
            if args.print_traceback:
                traceback.print_exc()
            if args.fail_fast:
                break

    elapsed = time.monotonic() - started
    cuda_peak = {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
    }
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        cuda_peak = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    summary = {
        "schema_version": 1,
        "artifact_type": "vggt_omega_extraction_shard_summary",
        "complete": not failures and len(successful) + len(skipped) == len(selected),
        "rank": args.rank,
        "world_size": args.world_size,
        "selected_trajectories": len(selected),
        "successful": successful,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": elapsed,
        "inferred_frames": inferred_frames,
        "inference_seconds": inference_seconds,
        "inference_frames_per_second": (
            inferred_frames / inference_seconds if inference_seconds > 0 else None
        ),
        "comparison_seconds": comparison_seconds,
        "cuda_peak_memory": cuda_peak,
        "checkpoint_hash_seconds": 0.0,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "code_revision": code_revision,
        "input_manifest_sha256": manifest_sha256,
        "raw_extraction_descriptor": extraction_descriptor,
        "alignment_policy": args.alignment_policy,
        "inference_path": args.inference_path,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "fast_full_comparison": dict(
            comparison_result or {"enabled": False}
        ),
    }
    logs_root = (
        Path(args.logs_root).resolve()
        if args.logs_root
        else output_root / "logs"
    )
    summary_path = logs_root / (
        f"shard_{args.rank:05d}_of_{args.world_size:05d}.json"
    )
    atomic_json_dump(summary, summary_path)
    if summary["complete"]:
        success_path = logs_root / (
            f"_SHARD_{args.rank:05d}_OF_{args.world_size:05d}_SUCCESS.json"
        )
        atomic_json_dump(summary, success_path)
    print(
        json.dumps(
            {
                "status": "complete" if summary["complete"] else "failed",
                "summary_path": str(summary_path),
                "successful": len(successful),
                "skipped": len(skipped),
                "failed": len(failures),
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible VGGT-Omega extraction for processed TartanDrive"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest", help="Content-hash input images once before GPU extraction"
    )
    _add_data_arguments(manifest)
    manifest.add_argument("--manifest-path", required=True)
    manifest.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Trajectory-level hashing threads; output is byte-identical to --workers 1.",
    )
    manifest.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Compatibility form for custom smoke splits or '--splits train test'.",
    )
    manifest.add_argument(
        "--expected-trajectories", type=int, default=EXPECTED_ALL_TRAJECTORIES
    )
    manifest.add_argument("--expected-frames", type=int, default=EXPECTED_ALL_FRAMES)
    manifest.add_argument("--skip-expected-count-check", action="store_true")
    manifest.set_defaults(handler=_manifest_command)

    extract = subparsers.add_parser(
        "extract", help="Run camera extraction and write canonical motion caches"
    )
    extract.add_argument("--manifest-path", required=True)
    extract.add_argument("--data-root", default=None)
    extract.add_argument(
        "--split", choices=("train", "test", "all"), default="all"
    )
    extract.add_argument("--trajectory", action="append", default=None)
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument("--third-party-root", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument(
        "--checkpoint-manifest",
        default=None,
        help="Defaults to checkpoint_manifest.json beside --checkpoint.",
    )
    extract.add_argument("--output-root", required=True)
    extract.add_argument("--raw-pose-root", default=None)
    extract.add_argument("--motion-root", default=None)
    extract.add_argument("--logs-root", default=None)
    extract.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    extract.add_argument(
        "--expected-code-revision", default=PINNED_CODE_REVISION
    )
    extract.add_argument("--expected-checkpoint-sha256", default=None)
    extract.add_argument("--device", default="cuda")
    extract.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    extract.add_argument(
        "--inference-path", choices=("fast", "full"), default="fast"
    )
    extract.add_argument("--resolution", type=int, default=512)
    extract.add_argument(
        "--resize-mode", choices=("balanced", "max_size"), default="max_size"
    )
    extract.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="0 runs one joint forward over the complete trajectory.",
    )
    extract.add_argument("--overlap", type=int, default=32)
    extract.add_argument("--allow-degenerate-window-scale", action="store_true")
    extract.add_argument("--compare-full-fast-frames", type=int, default=0)
    extract.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    extract.add_argument(
        "--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1"))
    )
    extract.add_argument("--shard-cost-power", type=float, default=2.0)
    extract.add_argument("--seed", type=int, default=0)
    extract.add_argument(
        "--alignment-policy",
        choices=("raw_only", "tartandrive_forward_camera", "fixed"),
        default="tartandrive_forward_camera",
    )
    extract.add_argument(
        "--degenerate-scale-policy",
        choices=("error", "empty_only", "unit"),
        default="empty_only",
    )
    extract.add_argument("--nonzero-step-epsilon", type=float, default=1e-6)
    extract.add_argument("--camera-to-navigation", default=None)
    extract.add_argument("--meters-per-model-unit", type=float, default=None)
    extract.add_argument(
        "--translation-unit",
        choices=("meters", "waypoint_spacing_units"),
        default="waypoint_spacing_units",
    )
    extract.add_argument("--waypoint-spacing-meters", type=float, default=None)
    extract.add_argument("--min-offset", type=int, default=-64)
    extract.add_argument("--max-offset", type=int, default=64)
    extract.add_argument("--context-size", type=int, default=4)
    extract.add_argument("--len-traj-pred", type=int, default=64)
    extract.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument(
        "--verify-input-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify image content against the manifest (disable only for a measured fast run).",
    )
    extract.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )
    extract.add_argument("--fail-fast", action="store_true")
    extract.add_argument("--print-traceback", action="store_true")
    extract.set_defaults(handler=_extract_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
