"""Regression tests for the stage-2 resource and latent-cache defaults."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
STAGE2_VARIANTS = (
    "no_pretrain",
    "time_reset",
    "geo_reset",
    "idm_reset",
    "latent_reset",
    "latent_align",
    "latent_real_to_latent",
)
DEFAULT_VAE_LATENT_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/"
    "vae_latents_sd_vae_ft_ema_224_four_datasets"
)
DEFAULT_VAE_MODEL_PATH = Path(
    "/file_system/nas/algorithm/dujun.nie/huggingface/hub/"
    "models--stabilityai--sd-vae-ft-ema/snapshots/"
    "f04b2c4b98319346dad8c65879f680b1997b204a"
)
DEFAULT_DREAMSIM_CACHE = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models"
)


class Stage2DefaultTests(unittest.TestCase):
    def test_every_finetune_overlay_uses_eight_workers(self):
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
            for variant in STAGE2_VARIANTS:
                with self.subTest(variant=variant):
                    config = compose(
                        config_name="nwm", overrides=[f"two_stage={variant}"]
                    )
                    self.assertEqual(config.training.num_workers, 8)

    def test_every_finetune_overlay_inherits_legacy_eval_schedule(self):
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
            for variant in STAGE2_VARIANTS:
                with self.subTest(variant=variant):
                    config = compose(
                        config_name="nwm", overrides=[f"two_stage={variant}"]
                    )
                    self.assertEqual(config.eval_every, 5000)
                    self.assertTrue(config.eval_at_first_step)

    def test_real_dataset_uses_precomputed_latents_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = OmegaConf.load(ROOT / "conf" / "dataset" / "nwm_real.yaml")
            self.assertTrue(config.precomputed_latents.enabled)
            self.assertEqual(
                Path(str(config.precomputed_latents.root)), DEFAULT_VAE_LATENT_ROOT
            )

    def test_finetune_launcher_defaults_to_eight_gpus(self):
        environment = os.environ.copy()
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.update(
            {
                "CONDA_PREFIX": "/root/miniconda3",
                "CONDA_DEFAULT_ENV": "base",
                "CONDA_PROMPT_MODIFIER": "(base)",
                "CONDA_SHLVL": "1",
                "CONDA_EXE": "/root/miniconda3/bin/conda",
                "CONDA_PYTHON_EXE": "/root/miniconda3/bin/python",
            }
        )
        result = subprocess.run(
            [
                str(ROOT / "two_stage_nwm.sh"),
                "stage2",
                "time_reset",
                "--dry-run",
                "--stage1-checkpoint=/tmp/timept.pth.tar",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(
            "Conda environment=/file_system/vepfs/algorithm/dujun.nie/"
            "miniconda3/envs/nwm",
            result.stdout,
        )
        self.assertIn(f"SD-VAE model={DEFAULT_VAE_MODEL_PATH}", result.stdout)
        self.assertIn(f"DreamSim cache={DEFAULT_DREAMSIM_CACHE}", result.stdout)
        self.assertIn(
            f"model.tokenizer.model_path={DEFAULT_VAE_MODEL_PATH}", result.stdout
        )
        self.assertIn("nproc/node=8", result.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", result.stdout)
        self.assertIn("--nproc=8", result.stdout)


if __name__ == "__main__":
    unittest.main()
