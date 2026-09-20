import importlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.nwm_benchmark_registry import (
    DATASET_CONTRACTS,
    MODELS,
    OOD_DATASET_CONTRACTS,
    PROTOCOLS,
    import_planning,
    import_prediction,
    load_registry,
    new_registry,
    render_markdown,
)
from scripts.raenwm_infer import compose_se2
from scripts.raenwm_planning_eval import (
    LOCAL_ACTION_STATS,
    RAENWM_ACTION_STATS,
    local_deltas_to_raenwm,
    normalize_data,
    trajectory_metrics,
)
from scripts.visualize_nwm_rollouts import (
    frame_index,
    frame_paths,
    render_contact_sheet,
    render_video_frame,
)


def test_unseen_rollout_protocol_is_fully_pinned() -> None:
    protocol = PROTOCOLS["go_stanford_unseen_rollout_v1"]

    assert protocol["sample_count"] == 150
    assert protocol["source_sample_count"] == 150
    assert protocol["sample_indices"] == list(range(150))
    assert protocol["split_sha256"] == (
        "e48af806e991d465f8b04f4dd106ff9b8db55cd65303831ee31e71dbef95cc23"
    )
    assert protocol["rollout_fps"] == [1, 4]
    assert protocol["horizons_seconds"] == [1, 2, 4, 8, 16]
    assert protocol["diffusion_steps"] == 250
    assert protocol["seed"] == 0
    assert protocol["execution"]["topology_independent"] is True
    assert protocol["visualization_sample_ids"] == [0, 21, 43, 64, 86, 107, 129, 149]


def test_no_pretrain_model_is_pinned_to_completed_checkpoint() -> None:
    model = MODELS["nwm-no-pretrain"]

    assert model["checkpoint_id"] == "joint_0110000"
    assert model["checkpoint_step"] == 110000
    assert model["sha256"] == (
        "b5c914d39cf1d9ce059fb39131bf514716f2136cc7a3d9b6db1a177ceeba8787"
    )
    assert model["provenance"]["initialization"].endswith("no NWM checkpoint loaded")


def test_open_nwm_model_is_pinned_to_completed_200k_checkpoint() -> None:
    model = MODELS["nwm-latentpt-reset-nwm-real-recipe"]

    assert model["display_name"] == "OpenNWM"
    assert model["checkpoint_id"] == "joint_0200000"
    assert model["checkpoint_step"] == 200000
    assert model["sha256"] == (
        "314ea1724ada64f69d78e9b3d15aa8754d8150fb5e5fca77f404cdacef0b27a4"
    )
    assert model["provenance"]["training_complete_at"] == "2026-09-17T18:03:28Z"


def test_open_nwm_180k_model_is_pinned_to_requested_checkpoint() -> None:
    model = MODELS["nwm-latentpt-reset-nwm-real-recipe-180k"]

    assert model["display_name"] == "OpenNWM 180k step"
    assert model["checkpoint_id"] == "joint_0180000"
    assert model["checkpoint_step"] == 180000
    assert model["sha256"] == (
        "0e3e9e49a059fc05706fc6d7b457d66df2392b18b2303631c5e04195c72a6170"
    )


def test_nwm_ego4d_model_is_pinned_to_official_200k_checkpoint() -> None:
    model = MODELS["nwm-ego4d"]

    assert model["display_name"] == "NWM + Ego4D"
    assert model["architecture"] == "CDiT-XL/2 + SD-VAE"
    assert model["checkpoint_id"] == "0200000"
    assert model["checkpoint_step"] == 200000
    assert model["sha256"] == (
        "6686fff6505ccc3b8e93a368b1bd3bf0051dfb76c2d3c280b77313035ef1d2f8"
    )
    assert model["provenance"]["checkpoint_train_steps_field"] == 200000


def test_geopt_finetune_model_is_pinned_to_completed_checkpoint() -> None:
    model = MODELS["nwm-geopt-ft"]

    assert model["checkpoint_id"] == "joint_0100000"
    assert model["checkpoint_step"] == 110000
    assert model["sha256"] == (
        "4915124400ae8042c6616c645d0ab296ee508090fb18258feace417bd15b041a"
    )
    assert model["provenance"]["stage1"] == "NavAnywhere-v1 GeoPT"
    assert model["provenance"]["fine_tune_scheme"] == (
        "adapter reset; 10k warmup + 100k joint steps"
    )


def test_latentpt_pixel_finetune_model_is_pinned_to_modelscope_checkpoint() -> None:
    model = MODELS["nwm-latentpt-ft-pixel"]

    assert model["display_name"] == "NWM-LatentPT-FT-Pixel"
    assert model["checkpoint_id"] == "joint_0100000"
    assert model["checkpoint_step"] == 110000
    assert model["sha256"] == (
        "03507c990230bbe329d8aca9e6e739aadefe4905ff4e5442ee307d9f7c2c12d2"
    )
    assert model["provenance"]["repository"] == (
        "LittleBoss/nwm-nav1-latentpt-pixel-ft-reset"
    )
    assert model["provenance"]["revision"] == (
        "c3dfa2533da976be47f75cc94dcf0eba07c90644"
    )


def test_raenwm_model_and_sampling_protocol_are_pinned() -> None:
    model = MODELS["rae-nwm"]

    assert model["backend"] == "raenwm"
    assert model["checkpoint_id"] == "raenwm_b"
    assert model["sha256"] == (
        "97244579618eb0e376355157a5c1df7a83e534cb1ca35d8ff4e43c4fb4f6a9d0"
    )
    assert model["provenance"]["revision"] == (
        "0219ce41c44d515f86719dd763c1efe7c7f72519"
    )
    assert model["provenance"]["sampling_method"] == "euler"
    assert model["provenance"]["sampling_steps"] == 50


def test_raenwm_se2_composition_uses_body_frame() -> None:
    deltas = torch.tensor([[[1.0, 0.0, torch.pi / 2], [1.0, 0.0, -torch.pi / 2]]])

    composed = compose_se2(deltas)

    torch.testing.assert_close(
        composed, torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-6, rtol=0
    )


def test_navigation_protocol_pins_splits_and_cem_settings() -> None:
    protocol = PROTOCOLS["navigation_cem80_v1"]

    assert protocol["sample_count"] == 100
    assert protocol["population"] == 80
    assert protocol["topk"] == 5
    assert protocol["repetitions"] == 3
    assert protocol["optimization_steps"] == 1
    assert protocol["horizon_steps"] == 8
    assert protocol["splits"]["recon"]["sha256"] == (
        "c62cd08be9f124cbeec48d914460da8630e089bf0bdb84c5018013a82d12ec54"
    )
    assert protocol["splits"]["scand"]["sha256"] == (
        "8acb4062561cbf1549e27f39a6e80241b97294a8c0e55c345787ae6a47be55ce"
    )


def test_unified_prediction_protocols_pin_all_nine_datasets() -> None:
    expected = [
        "recon",
        "scand",
        "huron",
        "tartan_drive",
        "go_stanford",
        "planetary_rover",
        "unitree_go2",
        "tum_rgbd",
        "uzh_fpv",
    ]
    direct = PROTOCOLS["direct_4s_v1"]
    rollout = PROTOCOLS["rollout_v1"]

    assert list(DATASET_CONTRACTS) == expected
    assert direct["datasets"] == expected
    assert direct["sample_count"] == 500
    assert direct["horizons_seconds"] == [4]
    assert direct["reproducibility"]["topology_independent"] is True
    assert rollout["datasets"] == expected
    assert rollout["sample_count"] == 150
    assert rollout["rollout_fps"] == [1, 4]
    assert rollout["reproducibility"]["topology_independent"] is True
    assert PROTOCOLS["navigation_cem80_v1"]["datasets"] == expected
    assert "go_stanford_unseen_rollout_10_v1" not in PROTOCOLS


def test_ood_direct_protocol_is_fully_pinned() -> None:
    protocol = PROTOCOLS["ood_direct_4s_v1"]
    expected_datasets = [
        "planetary_rover",
        "unitree_go2",
        "tum_rgbd",
        "uzh_fpv",
    ]

    assert list(protocol["datasets"]) == expected_datasets
    assert protocol["data_root"] == "/file_system/nas/algorithm/dujun.nie/nwm/data"
    assert protocol["evaluation"] == "time"
    assert protocol["sample_count"] == 500
    assert protocol["navigation_sample_count"] == 100
    assert protocol["context_frames"] == 4
    assert protocol["future_frames"] == 16
    assert protocol["input_fps"] == 4
    assert protocol["horizons_seconds"] == [4]
    assert protocol["frame_indices"] == {"4s": 4}
    assert protocol["seed"] == 0
    assert protocol["execution"]["distributed_world_size"] == 4
    assert protocol["inference"]["nwm"] == {
        "sampler": "ddpm",
        "sampling_steps": 250,
        "batch_size_per_rank": 64,
    }
    assert protocol["inference"]["rae-nwm"] == {
        "sampler": "euler_ode",
        "sampling_steps": 50,
        "batch_size_per_rank": 16,
    }
    expected_hashes = {
        "planetary_rover": (
            "cec1b5d0e9a7de2f1bac564721f81f17bb50819b6f96fc9e6751fbfb36897b43",
            "3ef7bbabc0244d1e18fb50bb47c231ca3993eacd976e209838f30b32bf4350e6",
            "7d3f27f694e5f554c944714a564b8d7bb354ca8fa604c1c2aab9e5664b6fd16f",
        ),
        "unitree_go2": (
            "857142dfa00167fd19231c583b7c5af14ec2badfb19b44505aad8aafecff0030",
            "ec25e7b7811057a97a8a59ae5dadd2c683a48c8b8b2e861927e826a7478d1ccd",
            "84963c5d3c14fb7de719e0f3b9669d6416e90fe607c7d8bcc9957c40ccb09dde",
        ),
        "tum_rgbd": (
            "1bdd4ccdbf2ecbda6611c71fe8166148d204c8fd93b9ce2a0de6bbed796f4a3e",
            "172cb60ed65294f8b82fc27605d8ca01ff52d4a9a17036b325567fc53a13c68d",
            "be6d8f7b325e289b3eec728876d1e60636d936401cbe98ad9d59a49f87e31eb5",
        ),
        "uzh_fpv": (
            "449209b4e070b305b6ffcfa1fecce771c36a1e0911221e3c9cb8bbf3bc23ce2e",
            "9cc88322e98f2a4216a7e8f98507c4a58d7e8500d5cfb831c293e615b62ce6db",
            "88eb250b60388709d3ce56775b6d596f9c16369ed249da4abd70944f3766e181",
        ),
    }
    assert {
        dataset: (
            contract["report"]["sha256"],
            contract["prediction_split"]["sha256"],
            contract["navigation_split"]["sha256"],
        )
        for dataset, contract in OOD_DATASET_CONTRACTS.items()
    } == expected_hashes
    for contract in OOD_DATASET_CONTRACTS.values():
        for key in ("report", "prediction_split", "navigation_split"):
            sha256 = contract[key]["sha256"]
            assert len(sha256) == 64
            int(sha256, 16)


def test_ood_navigation_uses_the_same_pinned_contracts() -> None:
    protocol = PROTOCOLS["navigation_cem80_v1"]

    assert protocol["datasets"][-4:] == list(OOD_DATASET_CONTRACTS)
    for dataset, contract in OOD_DATASET_CONTRACTS.items():
        split = protocol["splits"][dataset]
        assert split["path"] == contract["navigation_split"]["path"]
        assert split["sha256"] == contract["navigation_split"]["sha256"]
        assert split["metric_waypoint_spacing"] == contract[
            "metric_waypoint_spacing"
        ]
        assert split["temporal_semantics"] == contract["temporal_semantics"]


def test_ood_prediction_import_accepts_only_the_registered_4s_frame(
    tmp_path: Path,
) -> None:
    registry = new_registry()
    audit = tmp_path / "audit.json"
    payload = {
        "dataset": "unitree_go2",
        "eval_name": "time",
        "sample_count": 500,
        "frame_indices": {"4s": 4},
        "metrics": {
            "4s": {
                "sample_count": 500,
                "lpips_alex": 0.5,
                "dreamsim": 0.4,
                "psnr": 12.0,
            }
        },
        "inference": {
            "backend": "nwm",
            "sampler": "ddpm",
            "sampling_steps": 250,
            "seed": 0,
        },
    }
    audit.write_text(json.dumps(payload), encoding="utf-8")

    import_prediction(
        registry,
        "nwm-real",
        "unitree_go2",
        "time",
        audit,
        "ood_direct_4s_v1",
    )
    result = registry["models"]["nwm-real"]["results"][
        "ood_direct_prediction"
    ]["unitree_go2"]["time"]
    assert result["frame_indices"] == {"4s": 4}

    payload["frame_indices"] = {"4s": 15}
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frame indices"):
        import_prediction(
            registry,
            "nwm-real",
            "unitree_go2",
            "time",
            audit,
            "ood_direct_4s_v1",
        )


def test_raenwm_navigation_action_bridge_preserves_physical_motion_and_yaw() -> None:
    local_units = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    local_normalized = normalize_data(local_units, LOCAL_ACTION_STATS)
    terminal_yaw = torch.tensor([0.2])

    recon = local_deltas_to_raenwm(local_normalized, "recon", terminal_yaw)
    scand = local_deltas_to_raenwm(local_normalized, "scand", terminal_yaw)

    expected_recon_xy = normalize_data(local_units, RAENWM_ACTION_STATS)
    expected_scand_xy = normalize_data(
        local_units * (0.38 / 0.36), RAENWM_ACTION_STATS
    )
    torch.testing.assert_close(recon[..., :2], expected_recon_xy)
    torch.testing.assert_close(scand[..., :2], expected_scand_xy)
    torch.testing.assert_close(recon[..., 2], scand[..., 2])
    torch.testing.assert_close(
        recon[..., 2], torch.tensor([[torch.pi / 4, 0.2 * torch.pi]])
    )


def test_trajectory_metrics_match_evo() -> None:
    evo = pytest.importorskip("evo")
    del evo
    from evo.core import metrics, sync
    from evo.core.metrics import PoseRelation
    from evo.core.trajectory import PoseTrajectory3D
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe

    ground_truth = torch.tensor([[0.1, 0.0], [1.0, 0.2], [2.0, 0.8], [2.8, 1.1]])
    prediction = torch.tensor([[0.0, 0.2], [1.2, 0.1], [1.8, 1.0], [3.1, 0.9]])

    def as_evo(points: torch.Tensor) -> PoseTrajectory3D:
        positions = torch.zeros((len(points), 3), dtype=torch.float64)
        positions[:, :2] = points.to(torch.float64)
        quaternions = torch.zeros((len(points), 4), dtype=torch.float64)
        quaternions[:, -1] = 1.0
        return PoseTrajectory3D(
            positions_xyz=positions.numpy(),
            orientations_quat_wxyz=quaternions.numpy(),
            timestamps=torch.arange(len(points), dtype=torch.float64).numpy(),
        )

    reference, estimate = sync.associate_trajectories(
        as_evo(ground_truth), as_evo(prediction)
    )
    evo_ate = main_ape.ape(
        reference,
        estimate,
        pose_relation=PoseRelation.translation_part,
        align=False,
        correct_scale=False,
    ).stats["rmse"]
    evo_rpe = main_rpe.rpe(
        reference,
        estimate,
        pose_relation=PoseRelation.translation_part,
        align=False,
        correct_scale=False,
        delta=1.0,
        delta_unit=metrics.Unit.frames,
        rel_delta_tol=0.1,
    ).stats["rmse"]

    ate, rpe = trajectory_metrics(ground_truth, prediction)
    assert ate == pytest.approx(evo_ate, abs=1e-12)
    assert rpe == pytest.approx(evo_rpe, abs=1e-12)


def test_loading_stale_registry_adds_new_static_models(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"schema_version": 1, "models": {}}), encoding="utf-8"
    )

    registry = load_registry(registry_path)

    assert registry["models"]["nwm-no-pretrain"]["checkpoint_id"] == "joint_0110000"
    assert registry["models"]["nwm-geopt-ft"]["checkpoint_id"] == "joint_0100000"


def test_rollout_audit_uses_full_trajectory_protocol(tmp_path: Path) -> None:
    registry = new_registry()
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "dataset": "go_stanford",
                "eval_name": "rollout_4fps",
                "sample_count": 150,
                "frame_indices": {
                    "1s": 3,
                    "2s": 7,
                    "4s": 15,
                    "8s": 31,
                    "16s": 63,
                },
                "metrics": {
                    horizon: {
                        "sample_count": 150,
                        "lpips_alex": 0.5,
                        "dreamsim": 0.4,
                        "psnr": 12.0,
                    }
                    for horizon in ("1s", "2s", "4s", "8s", "16s")
                },
            }
        ),
        encoding="utf-8",
    )

    import_prediction(
        registry,
        "nwm-real",
        "go_stanford",
        "rollout_4fps",
        audit,
        "go_stanford_unseen_rollout_v1",
    )

    result = registry["models"]["nwm-real"]["results"]["unseen_generalization"][
        "go_stanford"
    ]["rollout_4fps"]
    assert result["protocol"] == "go_stanford_unseen_rollout_v1"


def test_visualization_frame_mapping_and_rendering(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    for index in range(16):
        Image.new("RGB", (32, 32), (index, 0, 0)).save(sequence / f"{index}.png")

    paths = frame_paths(sequence, 16)
    assert frame_index(16, 1) == 15
    assert frame_index(4, 4) == 15
    frame = render_video_frame(
        [paths[0], paths[1]], ["Ground truth", "model"], "t = 1s", (32, 32)
    )
    assert frame.size == (64, 80)

    sheet = tmp_path / "sheet.png"
    horizons = [paths[frame_index(horizon, 1)] for horizon in (1, 2, 4, 8, 16)]
    render_contact_sheet([("Ground truth", horizons), ("model", horizons)], sheet, 1)
    assert sheet.is_file()
    assert Image.open(sheet).size == (1070, 402)


def test_markdown_renders_unseen_rollout_mode() -> None:
    registry = new_registry()
    metrics = {
        horizon: {
            "sample_count": 150,
            "lpips_alex": 0.5,
            "dreamsim": 0.4,
            "psnr": 12.0,
        }
        for horizon in ("1s", "2s", "4s", "8s", "16s")
    }
    registry["models"]["nwm-real"]["results"] = {
        "unseen_generalization": {
            "go_stanford": {"rollout_1fps": {"metrics": metrics}}
        }
    }

    markdown = render_markdown(registry)

    assert "| nwm-real | rollout_1fps | 16s |" in markdown


def test_markdown_renders_ood_navigation_results() -> None:
    registry = new_registry()
    registry["models"]["nwm-real"]["results"] = {
        "navigation_planning": {
            "unitree_go2": {
                "metrics": {
                    "ate": 1.0,
                    "rpe_trans": 0.2,
                    "pos_diff_norm": 0.8,
                    "yaw_diff_norm": 0.1,
                }
            }
        }
    }

    markdown = render_markdown(registry)

    assert "| nwm-real | unitree_go2 | 1.000000 | 0.200000 | measured, N=80 |" in markdown


def test_ground_truth_inference_does_not_create_double_gt_directory(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    runner.ensure_ground_truth(
        "go_stanford",
        "rollout",
        {"exp_dir": "/models/reference"},
        ["0", "1"],
        {},
        True,
    )

    assert f"output_dir={tmp_path}" in commands[0]
    assert f"output_dir={tmp_path / 'gt'}" not in commands[0]


def test_unseen_rollout_split_hash_is_checked(monkeypatch) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")

    runner.validate_unseen_rollout_split({"protocols": PROTOCOLS})

    changed = json.loads(json.dumps(PROTOCOLS))
    changed["go_stanford_unseen_rollout_v1"]["split_sha256"] = "0" * 64
    try:
        runner.validate_unseen_rollout_split({"protocols": changed})
    except RuntimeError as error:
        assert "split SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("a changed split hash must be rejected")


def test_full_trajectory_rollout_dry_run_is_isolated(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()
    runner.run_unseen_rollout_prediction(
        "nwm-real",
        registry["models"]["nwm-real"],
        ["0", "1"],
        {},
        tmp_path / "registry.json",
        registry,
        False,
        True,
    )

    flattened = [argument for command in commands for argument in command]
    indices_argument = next(
        value for value in flattened if value.startswith("eval_sample_indices=")
    )
    assert indices_argument == "eval_sample_indices=[" + ",".join(map(str, range(150))) + "]"
    assert "eval_expected_full_count=150" in flattened
    assert any("go_stanford_unseen_rollout_v1" in argument for argument in flattened)


def test_raenwm_prediction_dry_run_uses_dedicated_backend(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "ensure_ground_truth",
        lambda *args, **kwargs: tmp_path / "gt" / "recon",
    )

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()
    runner.run_prediction(
        "rae-nwm",
        registry["models"]["rae-nwm"],
        "recon",
        ["2", "6"],
        {"NWM_DATA_ROOT": "/datasets"},
        tmp_path / "registry.json",
        False,
        True,
        evaluations=("time",),
        raenwm_python=Path("/envs/raenwm/bin/python"),
    )

    inference = commands[0]
    assert inference[0] == "/envs/raenwm/bin/torchrun"
    assert "scripts/raenwm_infer.py" in inference
    assert "--num-steps" in inference
    assert inference[inference.index("--num-steps") + 1] == "50"
    assert "isolated_nwm_infer.py" not in inference


def test_raenwm_rollout_dry_run_uses_autoregressive_backend(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "ensure_ground_truth",
        lambda *args, **kwargs: tmp_path / "gt" / "go_stanford",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()

    runner.run_grouped_prediction_inference(
        "rae-nwm",
        registry["models"]["rae-nwm"],
        ["go_stanford"],
        ("rollout_1fps", "rollout_4fps"),
        ["5"],
        {"NWM_DATA_ROOT": "/datasets"},
        False,
        True,
        reference_model=registry["models"]["nwm-real"],
        raenwm_python=Path("/envs/raenwm/bin/python"),
        batch_size=4,
        time_horizons=(4,),
    )

    inference = next(command for command in commands if "scripts/raenwm_infer.py" in command)
    assert inference[0] == "/envs/raenwm/bin/torchrun"
    assert inference[inference.index("--eval-type") + 1] == "rollout"
    assert inference[inference.index("--rollout-fps") + 1 :] == ["1", "4"]
    assert inference[inference.index("--future-frames") + 1] == "64"
    assert inference[inference.index("--num-steps") + 1] == "50"


def test_local_direct_prediction_can_serialize_four_logical_ranks_on_one_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "ensure_ground_truth",
        lambda *args, **kwargs: tmp_path / "gt" / "go_stanford",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()

    runner.run_prediction(
        "nwm-real",
        registry["models"]["nwm-real"],
        "go_stanford",
        ["7"],
        {"NWM_DATA_ROOT": "/datasets"},
        tmp_path / "registry.json",
        False,
        True,
        evaluations=("time",),
        serialize_logical_ranks=True,
    )

    inference = [
        command for command in commands if "isolated_nwm_infer.py" in command
    ]
    assert len(inference) == 4
    for logical_rank, command in enumerate(inference):
        assert "--nproc-per-node=1" in command
        assert "eval_expected_full_count=500" in command
        indices_argument = next(
            argument
            for argument in command
            if argument.startswith("eval_sample_indices=")
        )
        indices = [
            int(value)
            for value in indices_argument.removeprefix("eval_sample_indices=[")
            .removesuffix("]")
            .split(",")
        ]
        assert indices == list(range(logical_rank, 500, 4))
        assert "seed=0" in command
        assert "batch_size=64" in command


def test_serialized_direct_prediction_rejects_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    registry = new_registry()

    with pytest.raises(ValueError, match="direct time prediction only"):
        runner.run_prediction(
            "nwm-real",
            registry["models"]["nwm-real"],
            "recon",
            ["7"],
            {"NWM_DATA_ROOT": "/datasets"},
            tmp_path / "registry.json",
            False,
            True,
            evaluations=("time", "rollout_1fps"),
            serialize_logical_ranks=True,
        )


def test_local_ood_direct_dry_run_uses_fixed_4s_ddpm_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "ensure_ood_ground_truth",
        lambda *args, **kwargs: tmp_path / "gt",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()

    runner.run_ood_direct_prediction(
        "nwm-real",
        registry["models"]["nwm-real"],
        ["unitree_go2"],
        ["0", "1", "2", "3"],
        {"NWM_DATA_ROOT": PROTOCOLS["ood_direct_4s_v1"]["data_root"]},
        tmp_path / "registry.json",
        registry,
        False,
        True,
    )

    inference = next(command for command in commands if "isolated_nwm_infer.py" in command)
    assert "datasets_to_eval=[unitree_go2]" in inference
    assert "eval_type=time" in inference
    assert "eval_len_traj_pred=16" in inference
    assert "time_horizons_seconds=[4]" in inference
    assert "eval_expected_full_count=500" in inference
    assert "eval_diffusion_steps=250" in inference
    metric = next(
        command for command in commands if "scripts/evaluate_nwm_predictions.py" in command
    )
    assert metric[metric.index("--frames") + 1] == "4s:4"


def test_local_ood_direct_can_serialize_four_logical_ranks_on_one_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "ensure_ood_ground_truth",
        lambda *args, **kwargs: tmp_path / "gt",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()

    runner.run_ood_direct_prediction(
        "nwm-real",
        registry["models"]["nwm-real"],
        ["unitree_go2"],
        ["7"],
        {"NWM_DATA_ROOT": PROTOCOLS["ood_direct_4s_v1"]["data_root"]},
        tmp_path / "registry.json",
        registry,
        False,
        True,
        serialize_logical_ranks=True,
    )

    inference = [
        command for command in commands if "isolated_nwm_infer.py" in command
    ]
    assert len(inference) == 4
    for logical_rank, command in enumerate(inference):
        assert "--nproc-per-node=1" in command
        indices_argument = next(
            argument
            for argument in command
            if argument.startswith("eval_sample_indices=")
        )
        indices = [
            int(value)
            for value in indices_argument.removeprefix("eval_sample_indices=[")
            .removesuffix("]")
            .split(",")
        ]
        assert indices == list(range(logical_rank, 500, 4))
        assert "seed=0" in command
        assert "batch_size=64" in command


def test_raenwm_ood_direct_dry_run_forces_full_euler50_recompute(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "ensure_ood_ground_truth",
        lambda *args, **kwargs: tmp_path / "gt",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()

    runner.run_ood_direct_prediction(
        "rae-nwm",
        registry["models"]["rae-nwm"],
        ["unitree_go2"],
        ["0", "1", "2", "3"],
        {"NWM_DATA_ROOT": PROTOCOLS["ood_direct_4s_v1"]["data_root"]},
        tmp_path / "registry.json",
        registry,
        False,
        True,
        raenwm_python=Path("/envs/raenwm/bin/python"),
    )

    inference = next(command for command in commands if "scripts/raenwm_infer.py" in command)
    assert inference[0] == "/envs/raenwm/bin/torchrun"
    assert inference[inference.index("--horizons") + 1] == "4"
    assert inference[inference.index("--future-frames") + 1] == "16"
    assert inference[inference.index("--num-steps") + 1] == "50"
    assert "--force" in inference


def test_ood_direct_rejects_changed_distributed_topology(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    registry = new_registry()

    with pytest.raises(ValueError, match="requires exactly 4 unique GPUs"):
        runner.run_ood_direct_prediction(
            "nwm-real",
            registry["models"]["nwm-real"],
            ["unitree_go2"],
            ["0", "1", "2"],
            {"NWM_DATA_ROOT": PROTOCOLS["ood_direct_4s_v1"]["data_root"]},
            tmp_path / "registry.json",
            registry,
            False,
            True,
        )


def test_ood_protocol_rejects_a_different_data_root(monkeypatch) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")

    with pytest.raises(RuntimeError, match="data root mismatch"):
        runner.validate_ood_data_root(
            PROTOCOLS["ood_direct_4s_v1"], {"NWM_DATA_ROOT": "/tmp/not-pinned"}
        )


def test_raenwm_navigation_dry_run_uses_shared_cem_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()
    runner.run_planning(
        "rae-nwm",
        registry["models"]["rae-nwm"],
        ["2", "6"],
        {"NWM_DATA_ROOT": "/datasets"},
        tmp_path / "registry.json",
        False,
        True,
        80,
        ["recon", "scand"],
        raenwm_python=Path("/envs/raenwm/bin/python"),
        raenwm_num_steps=50,
    )

    command = commands[0]
    assert command[0] == "/envs/raenwm/bin/torchrun"
    assert "scripts/raenwm_planning_eval.py" in command
    for flag, expected in (
        ("--num-samples", "80"),
        ("--topk", "5"),
        ("--opt-steps", "1"),
        ("--num-repeat-eval", "3"),
        ("--microbatch-size", "80"),
        ("--num-steps", "50"),
        ("--seed", "42"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert command[command.index("--datasets") + 1 : command.index("--num-samples")] == [
        "recon",
        "scand",
    ]
    assert "planning_eval.py" not in command


def test_local_ood_navigation_dry_run_uses_per_sample_seed(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    registry = new_registry()

    runner.run_planning(
        "nwm-real",
        registry["models"]["nwm-real"],
        ["0", "1", "2", "3"],
        {"NWM_DATA_ROOT": PROTOCOLS["ood_direct_4s_v1"]["data_root"]},
        tmp_path / "registry.json",
        False,
        True,
        80,
        ["unitree_go2"],
    )

    command = commands[0]
    assert "planning_eval.py" in command
    assert "datasets_to_eval=[unitree_go2]" in command
    assert "seed=42" in command
    assert "planning_sample_seed=42" in command
    assert "resume_planning_samples=true" in command


def test_planning_import_preserves_raenwm_inference_provenance(tmp_path: Path) -> None:
    registry = new_registry()
    metrics = tmp_path / "recon_planning.json"
    metrics.write_text(
        json.dumps(
            {
                "recon_ate": 1.0,
                "recon_rpe_trans": 0.2,
                "recon_pos_diff_norm": 0.8,
                "recon_yaw_diff_norm": 0.1,
                "sample_count": 100,
                "inference": {
                    "backend": "rae-nwm",
                    "sampler": "euler_ode",
                    "sampling_steps": 50,
                },
            }
        ),
        encoding="utf-8",
    )

    import_planning(registry, "rae-nwm", "recon", metrics)

    result = registry["models"]["rae-nwm"]["results"]["navigation_planning"][
        "recon"
    ]
    assert result["inference"]["sampling_steps"] == 50
    assert result["sample_count"] == 100


def test_rollout_visualization_reads_shared_ground_truth(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_nwm_benchmark")
    seed_root = tmp_path / "evaluation_seeds" / "seed1"
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", seed_root)
    monkeypatch.setattr(runner, "SHARED_BENCHMARK_ROOT", tmp_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda command, env, dry_run: commands.append(command),
    )
    runner.run_rollout_visualization(
        ["nwm-real"], seed_root / "benchmark_results.json", {}, True
    )

    command = commands[0]
    gt_index = command.index("--gt-root") + 1
    benchmark_index = command.index("--benchmark-root") + 1
    assert command[gt_index] == str(
        tmp_path / "protocol_runs" / runner.UNSEEN_ROLLOUT_PROTOCOL / "gt"
    )
    assert command[benchmark_index] == str(
        seed_root / "protocol_runs" / runner.UNSEEN_ROLLOUT_PROTOCOL
    )
