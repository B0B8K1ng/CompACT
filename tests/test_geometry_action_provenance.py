import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from geometry_action.tartandrive_vggt_omega import (
    atomic_torch_save,
    build_geometry_payload,
    build_raw_pose_payload,
    c2w_to_w2c,
    raw_extraction_descriptor,
    sha256_file,
)
from scripts.audit_tartandrive_geometry_provenance import (
    EXPECTED_CHECKPOINT_MANIFEST_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CODE_REVISION,
    EXPECTED_EXTRACTION_CONFIGURATION,
    EXPECTED_EXTRACTION_FINGERPRINT,
    EXPECTED_INPUT_MANIFEST_SHA256,
    EXPECTED_MODEL_REVISION,
    main,
)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _formal_artifacts(tmp_path: Path) -> dict[str, Path]:
    name = "trajectory_a"
    num_frames = 68
    raw_root = tmp_path / "raw_pose"
    geometry_root = tmp_path / "geometry_motion"
    raw_directory = raw_root / "tartan_drive"
    geometry_directory = geometry_root / "tartan_drive"
    raw_directory.mkdir(parents=True)
    geometry_directory.mkdir(parents=True)

    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    c2w[:, 2, 3] = np.arange(num_frames, dtype=np.float64) * 0.1
    extrinsics = c2w_to_w2c(c2w)
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None], num_frames, axis=0)
    descriptor = raw_extraction_descriptor(
        resolution=384,
        resize_mode="max_size",
        window_size=0,
        overlap=64,
        inference_path="fast",
        dtype="bfloat16",
        allow_tf32=True,
        allow_degenerate_window_scale=False,
        seed=0,
    )
    windows = [
        {
            "window_index": 0,
            "start_index": 0,
            "end_index_exclusive": num_frames,
            "frame_ids": torch.arange(num_frames, dtype=torch.int64),
            "extrinsics_w2c_local": torch.as_tensor(extrinsics, dtype=torch.float32),
            "intrinsics_local": torch.as_tensor(intrinsics, dtype=torch.float32),
            "image_size_hw": torch.tensor([288, 384], dtype=torch.int64),
            "alignment_to_global": {
                "scale": 1.0,
                "rotation": torch.eye(3),
                "translation": torch.zeros(3),
                "degenerate_scale": False,
            },
            "overlap_frame_ids": torch.empty(0, dtype=torch.int64),
            "overlap_center_rmse": 0.0,
            "overlap_rotation_rmse_deg": 0.0,
        }
    ]
    raw = build_raw_pose_payload(
        dataset_name="tartan_drive",
        trajectory_name=name,
        frame_list_sha256="f" * 64,
        extrinsics_w2c=extrinsics,
        intrinsics=intrinsics,
        image_size_hw=(288, 384),
        windows=windows,
        window_alignment={
            "policy": "single_window_identity",
            "overlap": 0,
            "min_overlap": 3,
            "pose_selection": "first_prediction_wins_overlap",
            "scale_fallback": "error",
        },
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        code_revision=EXPECTED_CODE_REVISION,
        model_revision=EXPECTED_MODEL_REVISION,
        resize_mode="max_size",
        resolution=384,
        inference_path="fast",
        dtype="bfloat16",
        input_manifest_sha256=EXPECTED_INPUT_MANIFEST_SHA256,
        checkpoint_manifest_sha256=EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        extraction_descriptor=descriptor,
    )
    raw_path = raw_directory / f"{name}.pt"
    atomic_torch_save(raw, raw_path)
    geometry = build_geometry_payload(
        raw,
        source_pose_sha256=sha256_file(raw_path),
        waypoint_spacing_meters=0.72,
    )
    geometry_path = geometry_directory / f"{name}.pt"
    atomic_torch_save(geometry, geometry_path)
    split_path = tmp_path / "traj_names.txt"
    split_path.write_text(name + "\n", encoding="utf-8")
    return {
        "raw_root": raw_root,
        "geometry_root": geometry_root,
        "raw_path": raw_path,
        "geometry_path": geometry_path,
        "split_path": split_path,
        "receipt_path": tmp_path / "provenance_receipt.json",
    }


def _run(paths: dict[str, Path]) -> tuple[int, dict]:
    code = main(
        [
            "--raw-pose-root",
            str(paths["raw_root"]),
            "--geometry-root",
            str(paths["geometry_root"]),
            "--split-files",
            str(paths["split_path"]),
            "--expected-count",
            "1",
            "--workers",
            "1",
            "--output",
            str(paths["receipt_path"]),
        ]
    )
    return code, json.loads(paths["receipt_path"].read_text(encoding="utf-8"))


def _issue_codes(receipt: dict) -> set[str]:
    return {
        issue["code"]
        for record in receipt["trajectories"]
        for issue in record["issues"]
    }


def test_formal_provenance_audit_passes_and_recomputes_descriptor(tmp_path) -> None:
    paths = _formal_artifacts(tmp_path)
    code, receipt = _run(paths)

    assert code == 0
    assert receipt["status"] == "pass"
    assert receipt["complete"] is True
    assert receipt["read_only_audit"] is True
    assert receipt["counts"] == {
        "audited_trajectories": 1,
        "expected_trajectories": 1,
        "failed_trajectories": 0,
        "frame_pairs": 68,
        "frames": 68,
        "geometry_files": 1,
        "passed_trajectories": 1,
        "raw_files": 1,
        "required_trajectories": 1,
    }
    descriptor = receipt["protocol"]["extraction_descriptor"]
    assert descriptor["configuration"] == EXPECTED_EXTRACTION_CONFIGURATION
    assert descriptor["fingerprint"] == EXPECTED_EXTRACTION_FINGERPRINT
    assert descriptor["recomputed_fingerprint"] == EXPECTED_EXTRACTION_FINGERPRINT
    assert _canonical_sha256(descriptor["configuration"]) == descriptor["fingerprint"]
    assert not list(tmp_path.glob(".provenance_receipt.json.*.tmp"))


def test_audit_rejects_self_consistent_descriptor_tamper(tmp_path) -> None:
    paths = _formal_artifacts(tmp_path)
    raw = torch.load(paths["raw_path"], map_location="cpu", weights_only=True)
    raw = copy.deepcopy(raw)
    configuration = raw["extraction_descriptor"]["configuration"]
    configuration["inference"]["allow_tf32"] = False
    raw["extraction_descriptor"]["fingerprint"] = _canonical_sha256(configuration)
    atomic_torch_save(raw, paths["raw_path"])

    # Keep the source binding valid so this test isolates formal-protocol drift.
    geometry = torch.load(paths["geometry_path"], map_location="cpu", weights_only=True)
    geometry["source_pose_sha256"] = sha256_file(paths["raw_path"])
    atomic_torch_save(geometry, paths["geometry_path"])
    code, receipt = _run(paths)

    assert code == 1
    assert receipt["status"] == "fail"
    codes = _issue_codes(receipt)
    assert "raw.extraction_descriptor.fingerprint.self_consistency" not in codes
    assert "raw.extraction_descriptor.fingerprint.formal_protocol" in codes
    assert "raw.extraction_descriptor.configuration.formal_protocol" in codes


def test_audit_rejects_geometry_to_raw_binding_tamper(tmp_path) -> None:
    paths = _formal_artifacts(tmp_path)
    geometry = torch.load(paths["geometry_path"], map_location="cpu", weights_only=True)
    geometry["source_pose_sha256"] = "0" * 64
    atomic_torch_save(geometry, paths["geometry_path"])
    code, receipt = _run(paths)

    assert code == 1
    assert receipt["status"] == "fail"
    assert "geometry.source_pose_sha256" in _issue_codes(receipt)
