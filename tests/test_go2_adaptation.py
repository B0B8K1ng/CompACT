"""Go2 transfer and offline protocol contracts, using CPU-sized real models."""
import copy
import json
import pickle
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import run_go2_adaptation as runner
from two_stage_checkpoint import (build_checkpoint_metadata, capture_rng_state,
                                 load_finetune_weights, load_two_stage_resume,
                                 restore_rng_state)
from two_stage_nwm import validate_two_stage_config
from two_stage_training import save_two_stage_checkpoint
from test_two_stage_smoke import _config, _model, _ema, _optimizer, _batch, _diffusion, _step


def tiny_config():
    cfg = _config(training_stage="real_finetune", action_mode="real")
    cfg.dataset.image_size = 16
    cfg.training.ema_decay = .99
    cfg.finetune.init_checkpoint = "/source.pt"
    return cfg


@pytest.mark.parametrize("legacy", [False, True])
def test_complete_ema_transfer_keeps_all_weights_and_starts_fresh(legacy):
    cfg = tiny_config()
    source = _model(cfg)
    # Distinguish model and EMA so accidental model loading is caught.
    ema = _ema(source)
    with torch.no_grad():
        for parameter in ema.parameters():
            parameter.add_(.02)
    source_cfg = OmegaConf.to_container(cfg, resolve=True)
    if legacy:
        source_cfg.pop("training_stage")
        source_cfg.pop("action_mode")
        source_cfg.pop("finetune")
    checkpoint = {"model": source.state_dict(), "ema": ema.state_dict(),
                  "config": source_cfg, "train_steps": 110000, "opt": {"old": True}}
    if not legacy:
        checkpoint["two_stage_metadata"] = build_checkpoint_metadata(cfg, "joint", 10000, 100000, source)
    target = _model(cfg)
    rng_before = torch.get_rng_state().clone()
    summary = load_finetune_weights(target, checkpoint, cfg)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert summary["weight_key"] == "ema"
    assert summary["source_train_steps"] == 110000
    for key, value in ema.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    opt = _optimizer(target, training_stage="real_finetune", action_mode="real", scheme="reset", substage="warmup")
    assert not opt.state


@pytest.mark.parametrize("damage", ["missing", "shape", "architecture", "normalization", "stage", "no_ema"])
def test_transfer_rejects_incompatible_sources(damage):
    cfg = tiny_config()
    target = _model(cfg)
    checkpoint = {"ema": copy.deepcopy(target.state_dict()), "config": OmegaConf.to_container(cfg)}
    key = "motion_condition_encoder.real_action_adapter.net.0.weight"
    if damage == "missing":
        checkpoint["ema"].pop(key)
    elif damage == "shape":
        checkpoint["ema"][key] = torch.zeros(1)
    elif damage == "normalization":
        checkpoint["config"]["motion_condition"]["real"]["normalization"] = {"mode": "wrong"}
    elif damage == "architecture":
        checkpoint["config"]["model"]["generator"]["num_heads"] = 8
    elif damage == "stage":
        checkpoint["config"]["training_stage"] = "proxy_pretrain"
    else:
        checkpoint.pop("ema")
    with pytest.raises(ValueError):
        load_finetune_weights(target, checkpoint, cfg)


def test_transfer_rejects_changed_hash_without_changing_target(tmp_path):
    cfg = tiny_config()
    model = _model(cfg)
    state = copy.deepcopy(model.state_dict())
    checkpoint = {"ema": state, "config": OmegaConf.to_container(cfg)}
    path = tmp_path / 'source.pt'
    torch.save(checkpoint, path)
    cfg.finetune.init_sha256 = 'wrong'
    with pytest.raises(ValueError, match='SHA-256'):
        load_finetune_weights(model, path, cfg)
    for key in state:
        torch.testing.assert_close(state[key], model.state_dict()[key], rtol=0, atol=0)


def test_transfer_rejects_changed_normalizer_buffer():
    cfg = tiny_config()
    model = _model(cfg)
    normalizer = model.motion_condition_encoder.real_motion_normalizer
    normalizer.register_buffer('test_mean', torch.zeros(3))
    state = copy.deepcopy(model.state_dict())
    state['motion_condition_encoder.real_motion_normalizer.test_mean'].fill_(1)
    with pytest.raises(ValueError, match='normalization buffer'):
        load_finetune_weights(model, {'ema': state, 'config': OmegaConf.to_container(cfg)}, cfg)


def test_runner_accepts_legacy_diffusion_config(tmp_path):
    with initialize_config_dir(config_dir=str(runner.ROOT / 'conf'), version_base=None):
        cfg = compose(config_name='nwm')
    source_config = OmegaConf.to_container(cfg, resolve=False)
    del source_config['model']['diffusion']['eval_timestep_respacing']
    path = tmp_path / 'legacy.pt'
    torch.save({'config': source_config}, path)
    args = SimpleNamespace(data_root=tmp_path, output=tmp_path, run_id='test', smoke=False)
    protocol = {'training': {'warmup_steps': 200, 'joint_steps': 800}}
    training = runner.build_training_config(args, 'nwm-real',
                                            {'checkpoint': str(path), 'sha256': 'test'}, protocol)
    assert training.model.diffusion.eval_timestep_respacing == 250
    assert training.finetune.init_checkpoint == str(path)


def test_go2_opt_in_and_legacy_defaults():
    with initialize_config_dir(config_dir=str(runner.ROOT / "conf"), version_base=None):
        cfg = compose(config_name="nwm", overrides=["two_stage=go2_adapt", "finetune.init_checkpoint=/source.pt"])
        original = compose(config_name="nwm", overrides=["two_stage=time_reset"])
    assert validate_two_stage_config(cfg) == ("real_finetune", "real", "reset")
    assert cfg.training.batch_size == 8
    assert cfg.finetune.warmup_steps == 200 and cfg.finetune.joint_steps == 800
    assert original.training.get("ema_decay", .9999) == .9999
    assert original.finetune.get("dataset_protocol", "paper") == "paper"
    for key, value in [("stage1_checkpoint", "/other.pt"), ("random_init", True)]:
        bad = copy.deepcopy(cfg)
        bad.finetune[key] = value
        with pytest.raises(ValueError):
            validate_two_stage_config(bad)
    cfg.eval_at_first_step = True
    with pytest.raises(ValueError, match="test set"):
        validate_two_stage_config(cfg)


def test_go2_training_loader_never_requests_test():
    from two_stage_training import _prepare_loader
    cfg = OmegaConf.create({"training_stage": "real_finetune"})
    with patch("two_stage_training.prepare_datasets", return_value=(object(), None)) as prepare, \
         patch("two_stage_training.create_dataloader", return_value=(object(), object())):
        _prepare_loader(cfg, 0, "warmup", include_eval=False)
    assert prepare.call_args.kwargs["include_test"] is False


def test_warmup_joint_and_transition_resume_match(tmp_path):
    cfg = tiny_config()
    cfg.finetune.warmup_steps = 1
    model = _model(cfg)
    ema = _ema(model)
    original = copy.deepcopy(model.state_dict())
    opt = _optimizer(model, training_stage="real_finetune", action_mode="real", scheme="reset", substage="warmup")
    batch, diffusion = _batch(), _diffusion()
    _step(model=model, ema=ema, diffusion=diffusion, optimizer=opt, batch=batch,
          config=cfg, scheme="reset", substage="warmup", check_gradients=True)
    changed = {key for key, value in model.state_dict().items() if not torch.equal(original[key], value)}
    assert changed and all(key.startswith("motion_condition_encoder.real_") for key in changed)
    for key in changed:
        torch.testing.assert_close(ema.state_dict()[key], original[key] * .99 + model.state_dict()[key] * .01)
    path = str(tmp_path / "transition.pt")
    save_two_stage_checkpoint(path, model=model, ema=ema, optimizer=None, scheduler=None,
                              scaler=None, config=cfg, substage="transition", completed_warmup_steps=1,
                              completed_joint_steps=0, train_steps=1, epoch=0, include_optimizer=False)
    rng = capture_rng_state()
    opt = _optimizer(model, training_stage="real_finetune", action_mode="real", scheme="reset", substage="joint")
    _step(model=model, ema=ema, diffusion=diffusion, optimizer=opt, batch=batch,
          config=cfg, scheme="reset", substage="joint", check_gradients=True)
    resumed = _model(cfg)
    resumed_opt = _optimizer(resumed, training_stage="real_finetune", action_mode="real", scheme="reset", substage="joint")
    info = load_two_stage_resume(resumed, resumed_opt, path, expected_substage="joint", config=cfg)
    assert info["train_steps"] == 1 and not info["optimizer_loaded"]
    resumed_ema = _ema(resumed)
    resumed_ema.load_state_dict(torch.load(path, weights_only=False)["ema"])
    restore_rng_state(rng)
    _step(model=resumed, ema=resumed_ema, diffusion=diffusion, optimizer=resumed_opt,
          batch=batch, config=cfg, scheme="reset", substage="joint", check_gradients=True)
    for key in model.state_dict():
        torch.testing.assert_close(resumed.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(resumed_ema.state_dict()[key], ema.state_dict()[key], rtol=0, atol=0)
    assert not torch.equal(model.x_embedder.proj.weight, original["x_embedder.proj.weight"])


def fake_data(root):
    (root / "normalization.json").write_text(json.dumps({"metric_waypoint_spacing": runner.SPACING,
                                                       "fit_split": "train", "fps": 4.0}))
    for split, count in [("train", 5), ("test", 2)]:
        folder = root / "data_splits/go2" / split
        folder.mkdir(parents=True)
        episodes = [f"{split}_{i}" for i in range(count)]
        names = [f"{e}__seg00" for e in episodes]
        for filename, values in [("episode_names.txt", episodes), ("traj_names.txt", names), ("all_traj_names.txt", names)]:
            (folder / filename).write_text("\n".join(values))
        for name in names:
            directory = root / "go2" / name
            directory.mkdir(parents=True)
            traj = {"position": np.column_stack([np.arange(90) * .08, np.zeros(90)]), "yaw": np.zeros(90)}
            with (directory / "traj_data.pkl").open("wb") as stream:
                pickle.dump(traj, stream)


def test_protocol_is_fixed_train_only_and_has_no_cross_episode_windows(tmp_path):
    fake_data(tmp_path)
    protocol, _ = runner.prepare_protocol(tmp_path)
    again, _ = runner.prepare_protocol(tmp_path)
    assert protocol == again
    assert len({tuple(x) for x in protocol["index"]}) == 100
    assert {runner.episode_name(x[0]) for x in protocol["index"]} == {"test_0", "test_1"}
    assert all(3 <= row[1] < 82 and row[2:] == [8, 8] for row in protocol["index"])
    path = tmp_path / "go2/test_0__seg00/traj_data.pkl"
    with path.open("rb") as stream:
        traj = pickle.load(stream)
    traj["position"] *= 100
    with path.open("wb") as stream:
        pickle.dump(traj, stream)
    changed, _ = runner.prepare_protocol(tmp_path)
    assert changed["prior"] == protocol["prior"]
    assert changed["data_sha256"] != protocol["data_sha256"]
    assert protocol["prior"]["var_scale"] == [.02, .02, .05]


def test_ten_window_protocol_is_balanced_and_keeps_training_recipe(tmp_path):
    fake_data(tmp_path)
    full, _ = runner.prepare_protocol(tmp_path)
    small, _ = runner.prepare_protocol(tmp_path, eval_windows=10)
    assert small == runner.prepare_protocol(tmp_path, eval_windows=10)[0]
    assert len({tuple(x) for x in small["index"]}) == 10
    assert [sum(runner.episode_name(x[0]) == ep for x in small["index"])
            for ep in ["test_0", "test_1"]] == [5, 5]
    assert small["training"] == full["training"]
    assert small["prior"] == full["prior"]
    for count in [0, -2, 9]:
        with pytest.raises(ValueError, match="positive even"):
            runner.prepare_protocol(tmp_path, eval_windows=count)


def test_all_trains_every_model_before_evaluation(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr("sys.argv", ["runner", "all", "--run-id", "test", "--dry-run",
                                  "--models", "timept-ft,geopt-ft", "--eval-windows", "10"])
    monkeypatch.setattr(runner, "prepare_protocol", lambda *a, **k: (
        {"version": "test", "sample_count": 10, "index": [], "training": {"joint_steps": 800}}, {}))
    monkeypatch.setattr(runner, "code_identity", lambda: {})
    monkeypatch.setattr(runner, "load_source", lambda *a: {"checkpoint": str(tmp_path / "source")})
    monkeypatch.setattr(runner, "build_training_config", lambda *a: None)
    monkeypatch.setattr(runner, "train_model", lambda args, name, *a: events.append(("train", name)))
    monkeypatch.setattr(runner, "eval_model", lambda args, name, condition, *a: events.append((condition, name)))
    monkeypatch.setattr(runner, "summarize", lambda *a: None)
    runner.main()
    assert events == [("train", "timept-ft"), ("train", "geopt-ft"),
                      ("before", "timept-ft"), ("after", "timept-ft"),
                      ("before", "geopt-ft"), ("after", "geopt-ft")]


def test_training_reuse_checks_original_artifact_and_training_inputs(monkeypatch, tmp_path):
    previous, output = tmp_path / "old", tmp_path / "new"
    source = previous / "source/train.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixed training implementation\n")
    identity = {"train.py": runner.sha256(source)}
    monkeypatch.setattr(runner, "code_identity", lambda: identity)
    old = {"data_sha256": "data", "training": {"joint_steps": 800},
           "code_sha256": runner.digest_json(identity)}
    runner.write_json(previous / "protocol.json", old)
    cfg = OmegaConf.create({"training": {"run_name": "old", "seed": 0}})
    config_path = previous / "configs/timept-ft/train.yaml"
    config_path.parent.mkdir(parents=True)
    OmegaConf.save(cfg, config_path)
    checkpoint = previous / "train/timept-ft/checkpoints/joint_0000800.pth.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified checkpoint")
    runner.write_json(previous / "train/timept-ft/completed.json", {
        "contract": {"config_sha256": runner.digest_json(OmegaConf.to_container(cfg)),
                     "protocol_sha256": runner.digest_json(old)},
        "sha256": runner.sha256(checkpoint)})
    args = SimpleNamespace(reuse_training_from=previous, output=output, dry_run=False)
    cfg.training.run_name = "new"
    protocol = dict(old, sample_count=10)
    assert runner.reuse_training(args, "timept-ft", cfg, protocol)
    assert (output / "train/timept-ft").resolve() == checkpoint.parents[1]
    cfg.training.seed = 1
    with pytest.raises(ValueError, match="configuration differs"):
        runner.reuse_training(args, "timept-ft", cfg, protocol)
    cfg.training.seed = 0
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.reuse_training(args, "timept-ft", cfg, protocol)


def test_metric_units_and_incomplete_results(tmp_path):
    protocol = {"sample_count": 2, "index": [["a__seg00", 3, 8, 8], ["b__seg00", 4, 8, 8]]}
    folder = tmp_path / "go2/cem/sample_metrics"
    folder.mkdir(parents=True)
    row = dict(zip(runner.METRICS, [1., 2., 3., .4]))
    (folder / "000000.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Incomplete"):
        runner.collect_metrics(tmp_path, protocol)
    (folder / "000001.json").write_text(json.dumps(row))
    result = runner.collect_metrics(tmp_path, protocol)
    assert result["overall"]["ate_m"] == runner.SPACING
    assert result["overall"]["endpoint_yaw_rad"] == .4
    assert result["a"]["count"] == 1 and result["overall"]["count"] == 2


def test_immutable_run_contract(tmp_path):
    path = tmp_path / "contract.json"
    runner.lock_json(path, {"sha256": "a"})
    runner.lock_json(path, {"sha256": "a"})
    with pytest.raises(ValueError, match="Inputs changed"):
        runner.lock_json(path, {"sha256": "b"})


def test_planning_seed_is_independent_of_skipped_windows():
    import random
    from planning_eval import seed_planning_sample
    cfg = {"planning_sample_seed": 42}
    def draw(sample_id):
        seed_planning_sample(cfg, torch.tensor([[sample_id]]))
        return [random.random(), np.random.rand(), torch.rand(()).item()]
    continuous = [draw(i) for i in range(5)]
    assert draw(4) == continuous[4]
    assert draw(2) == continuous[2]
    with pytest.raises(ValueError, match="batch_size=1"):
        seed_planning_sample(cfg, torch.tensor([[0], [1]]))
    with patch("planning_eval.seed_everything") as seed:
        seed_planning_sample({}, torch.tensor([[0]]))
        seed.assert_not_called()


def test_required_wandb_is_online_and_resume_uses_original_id(tmp_path):
    from train import initialize_wandb
    cfg = OmegaConf.create({'training': {'wandb_required': True}})
    run = SimpleNamespace(id='go2-id', url='https://wandb.ai/team/compact-nwm/runs/go2-id',
                          path='team/compact-nwm/go2-id', settings=SimpleNamespace(mode='online'))
    with patch('train.wandb.init', return_value=run) as init:
        initialize_wandb(cfg, tmp_path)
        assert init.call_args.kwargs['mode'] == 'online'
        assert 'resume' not in init.call_args.kwargs
        cfg.training.from_checkpoint = '/go2/latest.pt'
        initialize_wandb(cfg, tmp_path)
        assert init.call_args.kwargs['id'] == 'go2-id'
        assert init.call_args.kwargs['resume'] == 'must'
    with patch('train.wandb.init', return_value=None):
        with pytest.raises(RuntimeError, match='online WandB'):
            initialize_wandb(cfg, tmp_path)
    with patch('train.wandb.init', side_effect=ConnectionError('offline')):
        with pytest.raises(ConnectionError):
            initialize_wandb(cfg, tmp_path)
    cfg.training.wandb_required = False
    with patch('train.wandb.init', return_value=run) as init:
        initialize_wandb(cfg, tmp_path)
        assert not {'id', 'resume', 'mode'} & init.call_args.kwargs.keys()
