import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

import two_stage_training as training


def test_reset_phase_learning_rates(monkeypatch):
    parameter = torch.nn.Parameter(torch.zeros(1))
    calls = []

    def groups(model, **kwargs):
        calls.append(kwargs)
        return [{"params": [parameter], "lr": kwargs["adapter_lr"]}]

    monkeypatch.setattr(training, "build_optimizer_param_groups", groups)
    config = OmegaConf.create({
        "training": {"optimizer": {"_target_": "torch.optim.AdamW", "lr": 2e-4, "weight_decay": 0.01}},
        "finetune": {"adapter_lr": 2e-4, "backbone_lr": 1e-4,
                     "warmup_adapter_lr": 2e-4, "joint_adapter_lr": 1e-4},
    })
    for phase, expected in [("warmup", 2e-4), ("joint", 1e-4)]:
        optimizer = training._make_optimizer(config, None, training_stage="real_finetune",
                                             scheme="reset", substage=phase)
        assert optimizer.param_groups[0]["lr"] == expected
        assert calls[-1]["backbone_lr"] == 1e-4


def test_step_budget_launcher_composes_without_sample_stopping(tmp_path):
    import json
    import sys
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"samples_per_epoch": 17284820}))
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "SAMPLING_RECIPE": str(recipe),
           "NWM_CONDA_ENV_PATH": str(Path(sys.executable).parent.parent),
           "BUDGET_MODE": "steps", "PRETRAIN_STEPS": "20000",
           "FINETUNE_STEPS": "10000", "FINETUNE_WARMUP_STEPS": "1667"}
    result = subprocess.run(["bash", str(root / "run_nwm_latentpt_nav1_80gb.sh"), "--dry-run"],
                            env=env, check=True, capture_output=True, text=True)
    configs = []
    for label, variant in [("Stage 1", "latentpt"), ("Stage 2", "latent_reset")]:
        args = shlex.split(result.stdout.split(label + " command:\n")[1].splitlines()[0])
        with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
            configs.append(compose(config_name="nwm", overrides=["two_stage=" + variant, *args[args.index("--")+1:]]))
    first, second = configs
    assert first.max_train_steps == 20000
    assert first.training.target_samples_per_rank is None
    assert second.finetune.warmup_steps == 1667
    assert second.finetune.joint_steps == 8333
    assert second.finetune.warmup_samples_per_rank is None
    assert second.finetune.joint_samples_per_rank is None
