"""Utility functions for hydra configuration and object instantiation."""

import glob
from time import time
from einops import repeat
import torch
import torch.distributed as dist
import logging
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate, get_original_cwd
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from motion_condition import flatten_motion_groups

logger = logging.getLogger(__name__)

# Global cache for evaluation model to prevent memory leaks
_eval_model_cache = None


def sample_precomputed_vae_posterior(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """Sample cached SD-VAE posteriors with Diffusers' native semantics."""
    if posterior_mean.shape != posterior_logvar.shape:
        raise ValueError(
            "Cached posterior mean/logvar shapes differ: "
            f"{tuple(posterior_mean.shape)} != {tuple(posterior_logvar.shape)}"
        )
    if posterior_mean.ndim != 5 or posterior_mean.shape[2] != 4:
        raise ValueError(
            "Cached posterior tensors must have shape [B,T,4,H,W]; got "
            f"{tuple(posterior_mean.shape)}"
        )
    if posterior_mean.dtype != torch.bfloat16 or posterior_logvar.dtype != torch.bfloat16:
        raise TypeError(
            "Cached posterior tensors must stay bfloat16 through collation and transfer; "
            f"got {posterior_mean.dtype} and {posterior_logvar.dtype}"
        )

    batch_size, sequence_length = posterior_mean.shape[:2]
    posterior_parameters = torch.cat(
        (
            posterior_mean.flatten(0, 1),
            posterior_logvar.flatten(0, 1),
        ),
        dim=1,
    )
    posterior = DiagonalGaussianDistribution(posterior_parameters)
    latents = posterior.sample().mul_(float(scaling_factor))
    return latents.unflatten(0, (batch_size, sequence_length))


def setup_tokenizer(config: DictConfig, device: torch.device):
    tokenizer_path = config.get("tokenizer_path", None)

    if tokenizer_path is not None:
        # Handle relative paths by resolving against original working directory
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.join(get_original_cwd(), tokenizer_path)

        ckp_paths = glob.glob(os.path.join(tokenizer_path, "checkpoints", "*.ckpt"))
        ckp_path = max(ckp_paths, key=os.path.getmtime)

        # Try different config locations
        config_path = os.path.join(tokenizer_path, ".hydra", "config.yaml")
        config_merged_path = os.path.join(
            tokenizer_path, ".hydra", "config_merged.yaml"
        )
        if os.path.exists(config_merged_path):
            config_path = config_merged_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find config file in {config_path}")

        logger.info(f"Loading compact tokenizer config from {config_path}")
        logger.info(f"Loading compact tokenizer from {ckp_path}")

        # Load the experiment config and extract tokenizer config
        experiment_config = OmegaConf.load(config_path)

        # Resolve all variable references in the config before extracting tokenizer config
        OmegaConf.resolve(experiment_config)
        tokenizer_config = experiment_config.model.tokenizer

        # override the tokenizer config with the training config if theres any (e.g. CFG weight, sampling nums, etc.)
        if config.model.get("tokenizer", None):
            tokenizer_config = OmegaConf.merge(tokenizer_config, config.model.tokenizer)

        # override the tokenizer config with the experiment config
        with open_dict(config):
            config.model.tokenizer = tokenizer_config

        # Extract normalization parameters from tokenizer experiment config
        # Update the CDiT training config to use the same normalization as the tokenizer
        exp_dataset = experiment_config.dataset
        assert exp_dataset.mean is not None and exp_dataset.std is not None, (
            "Tokenizer experiment config has null mean/std values"
        )
        # Update the CDiT training config with tokenizer's normalization
        with open_dict(config):
            if not hasattr(config, "dataset"):
                config.dataset = {}
            if experiment_config.dataset.get("use_dino_image_for_compact", False):
                config.dataset.mean = exp_dataset.dinov2_mean
                config.dataset.std = exp_dataset.dinov2_std
            else:
                config.dataset.mean = exp_dataset.mean
                config.dataset.std = exp_dataset.std
        logger.info(
            f"Updated CDiT config with tokenizer experiment normalization: mean={config.dataset.mean}, std={config.dataset.std}"
        )

        # Instantiate tokenizer from loaded config
        tokenizer = instantiate(tokenizer_config, _recursive_=True).to(device)

        # Load checkpoint weights
        state_dict = torch.load(ckp_path, map_location="cpu", weights_only=False)
        state_dict = {
            k[len("tokenizer.") :]: v
            for k, v in state_dict["state_dict"].items()
            if k.startswith("tokenizer.")
        }
        tokenizer.load_state_dict(state_dict, strict=True)

    else:
        # Fallback to config-based tokenizer (legacy behavior)
        tokenizer_config = config.model.tokenizer
        tokenizer = instantiate(tokenizer_config, _recursive_=True).to(device)

    return tokenizer


def setup_model(config: DictConfig, device: torch.device):
    """Setup CDiT model using hydra instantiation."""
    model_kwargs = {"context_size": config.dataset.context_size}
    generator_target = str(config.model.generator.get("_target_", ""))
    if generator_target.startswith("models.CDiT") and "motion_condition" in config:
        model_kwargs["motion_condition"] = config.motion_condition
        training_stage = str(config.get("training_stage", "legacy"))
        if training_stage != "legacy":
            model_kwargs.update(
                training_stage=training_stage,
                action_mode=str(config.get("action_mode", "none")),
                finetune=config.get("finetune", None),
            )
    model = instantiate(
        config.model.generator,
        **model_kwargs,
        _recursive_=False,  # Don't recursively instantiate nested configs
    ).to(device)

    return model


def validate_model_context_sizes(config: DictConfig, *context_keys: str) -> int:
    """Fail early when an evaluation view disagrees with the trained CDiT.

    CDiT owns one positional embedding per configured context frame, so the
    dataset, evaluator slicing, and checkpoint architecture must use the same
    length. Keeping this check at the entry points turns an otherwise obscure
    positional-embedding shape error into an actionable configuration error.
    """
    model_context = int(config.dataset.context_size)
    mismatches = {}
    for key in context_keys:
        if key not in config:
            raise ValueError(
                f"Missing required context-size setting {key!r}; expected "
                f"dataset.context_size={model_context}"
            )
        value = int(config.get(key))
        if value != model_context:
            mismatches[key] = value
    if mismatches:
        rendered = ", ".join(
            f"{key}={value}" for key, value in sorted(mismatches.items())
        )
        raise ValueError(
            "Evaluation context length must match the trained CDiT/checkpoint: "
            f"dataset.context_size={model_context}, {rendered}. Set every "
            "evaluation/planning context-size override to the training value."
        )
    return model_context


def setup_diffusion(config: DictConfig, for_eval: bool, device: torch.device):
    diffusion_config = config.model.diffusion
    diffusion = instantiate(diffusion_config, for_eval=for_eval, _recursive_=False)
    return diffusion


def setup_optimizer(config: DictConfig, model_params):
    """Setup optimizer using hydra instantiation."""
    # Param groups contain live ``torch.nn.Parameter`` objects.  Hydra's default
    # conversion wraps the surrounding dictionaries in ``DictConfig`` objects,
    # which PyTorch optimizers reject because parameter groups must be plain
    # dictionaries.  Partial conversion preserves the tensors while converting
    # their containers to native Python types.
    return instantiate(
        config.training.optimizer,
        params=model_params,
        _convert_="partial",
    )


def setup_scheduler(config: DictConfig, optimizer):
    """Setup learning rate scheduler if specified."""
    if config.scheduler.scheduler._target_ is not None:
        return instantiate(config.scheduler.scheduler, optimizer=optimizer)
    return None


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    # Build a lookup only when needed (for torch.compile compatibility)
    ema_params_dict = dict(ema_model.named_parameters())

    for name, param in model.named_parameters():
        # Handle torch.compile naming convention which adds "_orig_mod." prefix
        ema_name = name.replace("_orig_mod.", "")
        if ema_name in ema_params_dict:
            # In-place update of EMA parameter
            ema_params_dict[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)


def print_config(config: DictConfig):
    """Print the config in a readable format."""
    formatted_config = OmegaConf.to_yaml(config)
    print("\n" + "=" * 80)
    print("CONFIGURATION:")
    print(formatted_config)
    print("=" * 80 + "\n")


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_checkpoint(
    checkpoint_dir,
    config,
    model,
    ema,
    opt,
    scheduler=None,
    bfloat_enable=False,
    scaler=None,
):
    """Load model checkpoint if available."""
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print("Searching for model from ", checkpoint_dir)
    start_epoch = 0
    train_steps = 0

    if os.path.isfile(latest_path) or config.training.from_checkpoint:
        if os.path.isfile(latest_path) and config.training.from_checkpoint:
            raise ValueError(
                "Resuming from checkpoint, this might override latest.pth.tar!!"
            )

        latest_path = (
            latest_path
            if os.path.isfile(latest_path)
            else config.training.from_checkpoint
        )
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )

        if "model" in latest_checkpoint:
            model_ckp = {
                k.replace("_orig_mod.", ""): v
                for k, v in latest_checkpoint["model"].items()
            }
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            model_ckp = {
                k.replace("_orig_mod.", ""): v
                for k, v in latest_checkpoint["ema"].items()
            }
            res = ema.load_state_dict(model_ckp, strict=True)
            print("Loading EMA model weights", res)
        else:
            update_ema(
                ema, model, decay=0
            )  # Ensure EMA is initialized with synced weights

        if "opt" in latest_checkpoint:
            opt_ckp = {
                k.replace("_orig_mod.", ""): v
                for k, v in latest_checkpoint["opt"].items()
            }
            opt.load_state_dict(opt_ckp)
            print("Loading optimizer params")

        if "epoch" in latest_checkpoint:
            start_epoch = latest_checkpoint["epoch"] + 1

        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]

        if "scaler" in latest_checkpoint and bfloat_enable and scaler is not None:
            scaler.load_state_dict(latest_checkpoint["scaler"])

        if "scheduler" in latest_checkpoint and scheduler is not None:
            scheduler.load_state_dict(latest_checkpoint["scheduler"])

    return start_epoch, train_steps


def save_checkpoint(
    model,
    ema,
    opt,
    config,
    epoch,
    train_steps,
    bfloat_enable,
    scheduler,
    scaler,
    checkpoint_dir,
):
    """Save model checkpoint."""
    import gc

    checkpoint = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
        "epoch": epoch,
        "train_steps": train_steps,
    }
    if bfloat_enable and scheduler is not None:
        checkpoint.update(
            {"scaler": scaler.state_dict(), "scheduler": scheduler.state_dict()}
        )
    elif bfloat_enable:
        checkpoint.update({"scaler": scaler.state_dict()})
    elif scheduler is not None:
        checkpoint.update({"scheduler": scheduler.state_dict()})

    # Save latest checkpoint
    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
    torch.save(checkpoint, checkpoint_path)

    # Save numbered checkpoint if needed
    if train_steps % (10 * config.ckpt_every) == 0 and train_steps > 0:
        numbered_checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
        torch.save(checkpoint, numbered_checkpoint_path)
        checkpoint_path = numbered_checkpoint_path  # Return the numbered path

    # Explicitly delete checkpoint dict and force garbage collection
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    return checkpoint_path


@torch.no_grad()
def evaluate(
    model,
    vae,
    diffusion,
    eval_loader,
    rank,
    latent_size,
    device,
    save_dir,
    seed,
    bfloat_enable,
    num_cond,
    unnormalize_fn,
):
    """Evaluate model on test dataset."""
    from isolated_nwm_infer import model_forward_wrapper

    global _eval_model_cache

    # Use the pre-created evaluation dataloader
    loader = eval_loader
    # Update sampler epoch for proper distributed evaluation
    loader.sampler.set_epoch(seed)

    # Use cached evaluation model to prevent memory leaks
    if _eval_model_cache is None:
        from dreamsim import dreamsim
        from dreamsim.model import download_weights

        eval_model_cache = os.environ.get(
            "NWM_MODEL_CACHE", os.path.join(get_original_cwd(), "models")
        )
        os.makedirs(eval_model_cache, exist_ok=True)

        # DreamSim's own cache helper is not distributed-aware. Without this
        # guard every DDP rank downloads the same ~1.17 GiB archive and writes
        # into the same directory. Rank 0 populates the shared NAS cache once;
        # all ranks wait before constructing their local evaluation model.
        if rank == 0:
            logger.info(
                f"Preparing DreamSim ensemble weights in {eval_model_cache}"
            )
            download_weights(eval_model_cache, "ensemble")
        dist.barrier()

        _eval_model_cache, _ = dreamsim(
            pretrained=True, cache_dir=eval_model_cache
        )
        _eval_model_cache = _eval_model_cache.to(device)
        logger.info("Created and cached DreamSim evaluation model")

    eval_model = _eval_model_cache
    score = torch.tensor(0.0).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step. New datasets return a dictionary so heterogeneous
    # conditions are not padded; legacy configs retain their tuple contract.
    batch = next(iter(loader))
    motion = None
    if isinstance(batch, dict):
        if "video" not in batch:
            raise RuntimeError("Evaluation requires pixel video samples")
        x = batch["video"].to(device)
        y = None
        rel_t = batch["k"].to(device)
    else:
        x, y, rel_t = batch
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device)
    with torch.amp.autocast("cuda", enabled=bfloat_enable, dtype=torch.bfloat16):
        B, T = x.shape[:2]
        num_goals = T - num_cond
        if isinstance(batch, dict):
            motion = flatten_motion_groups(
                batch.get("motion"),
                batch_size=B,
                num_goals=num_goals,
                device=device,
            )
        rel_t = rel_t.flatten(0, 1)

        start_time = time()
        samples = model_forward_wrapper(
            (model, diffusion, vae),
            x,
            y,
            num_timesteps=None,
            latent_size=latent_size,
            device=device,
            num_cond=num_cond,
            num_goals=num_goals,
            rel_t=rel_t,
            motion=motion,
        )
        logger.info(
            f"Time taken for generating {samples.shape}: {time() - start_time:.2f} seconds"
        )

        x_start_pixels = x[:, num_cond:].flatten(0, 1)
        x_cond_pixels = (
            x[:, :num_cond]
            .unsqueeze(1)
            .expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4])
            .flatten(0, 1)
        )
        # Unnormalize pixels directly using tokenizer's unnormalization
        # samples = vae.unnormalize_image(samples)
        x_start_pixels = unnormalize_fn(x_start_pixels)
        x_cond_pixels = unnormalize_fn(x_cond_pixels)

        res = eval_model(x_start_pixels, samples)

        score += res.sum()
        n_samples += len(res)

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples.shape[0], 10)):
            fig, ax = plt.subplots(1, 3, dpi=256)
            ax[0].imshow(
                (x_cond_pixels[i, -1].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    "uint8"
                )
            )
            ax[1].imshow(
                (x_start_pixels[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
            ax[2].imshow(
                (samples[i].permute(1, 2, 0).cpu().float().numpy() * 255).astype(
                    "uint8"
                )
            )
            plt.savefig(f"{save_dir}/{i}.png")
            plt.clf()  # Clear current figure
            plt.close(fig)  # Close specific figure
            del fig  # Explicitly delete figure reference

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score / n_samples

    # Explicitly delete large tensors and clear cache
    import gc

    try:
        del x, y, rel_t, motion, samples, x_start_pixels, x_cond_pixels
        if "res" in locals():
            del res
        if "ax" in locals():
            del ax
    except Exception:
        pass

    # Force garbage collection and CUDA cache cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Additional cleanup for matplotlib backend
    if rank == 0:
        plt.close("all")  # Close any remaining matplotlib figures
        gc.collect()

    return sim_score


def train_step(
    model,
    ema,
    diffusion,
    tokenizer,
    batch,
    device,
    opt,
    scheduler,
    bfloat_enable,
    scaler,
    config,
):
    """Execute a single training step."""
    use_precomputed_latents = bool(
        config.dataset.get("precomputed_latents", {}).get("enabled", False)
    )
    uses_motion_framework = isinstance(batch, dict)
    motion = None

    if uses_motion_framework:
        rel_t = batch["k"].to(device, non_blocking=True)
        y = None
        if use_precomputed_latents:
            if "posterior_mean" not in batch or "video" in batch:
                raise RuntimeError(
                    "Precomputed latent mode is enabled, but the motion dataset "
                    "did not return posterior tensors."
                )
            posterior_mean = batch["posterior_mean"].to(
                device, non_blocking=True
            )
            posterior_logvar = batch["posterior_logvar"].to(
                device, non_blocking=True
            )
        else:
            if "video" not in batch or "posterior_mean" in batch:
                raise RuntimeError(
                    "Precomputed latent mode is disabled, but the motion dataset "
                    "did not return pixels."
                )
            x = batch["video"].to(device, non_blocking=True)
    elif use_precomputed_latents:
        if len(batch) != 4:
            raise RuntimeError(
                "Precomputed latent mode is enabled, but the training dataset "
                "returned pixels. Refusing to silently mix input modes."
            )
        posterior_mean, posterior_logvar, y, rel_t = batch
        posterior_mean = posterior_mean.to(device, non_blocking=True)
        posterior_logvar = posterior_logvar.to(device, non_blocking=True)
    else:
        if len(batch) != 3:
            raise RuntimeError(
                "Precomputed latent mode is disabled, but the training dataset "
                "returned posterior tensors. Refusing to silently mix input modes."
            )
        x, y, rel_t = batch
        x = x.to(device, non_blocking=True)
    if y is not None:
        y = y.to(device, non_blocking=True)
    if not uses_motion_framework:
        rel_t = rel_t.to(device, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=bfloat_enable, dtype=torch.bfloat16):
        with torch.no_grad():
            if use_precomputed_latents:
                # This is the same sampling implementation used by
                # AutoencoderKL.encode(...).latent_dist.sample(). Keeping the
                # cached posterior in bfloat16 and sampling inside the existing
                # autocast context preserves the original CDiT-B training path.
                x = sample_precomputed_vae_posterior(
                    posterior_mean,
                    posterior_logvar,
                    tokenizer.scaling_factor,
                )
                B, T = x.shape[:2]
            else:
                # Map input images to latent space + normalize latents:
                B, T = x.shape[:2]
                x = x.flatten(0, 1)
                x = tokenizer.encode(x)
                x = x.unflatten(0, (B, T))

        num_goals = T - config.dataset.context_size
        if rel_t.shape != (B, num_goals):
            raise ValueError(
                f"Temporal k must have shape {(B, num_goals)}, "
                f"got {tuple(rel_t.shape)}"
            )
        x_start = x[:, config.dataset.context_size :].flatten(0, 1)
        x_cond = repeat(
            x[:, : config.dataset.context_size], "b t ... -> (b g) t ...", g=num_goals
        )
        if uses_motion_framework:
            motion = flatten_motion_groups(
                batch.get("motion"),
                batch_size=B,
                num_goals=num_goals,
                device=device,
            )
        else:
            y = y.flatten(0, 1)
        rel_t = rel_t.flatten(0, 1)

        t = torch.randint(
            0, diffusion.num_timesteps, (x_start.shape[0],), device=device
        )
        if uses_motion_framework:
            model_kwargs = {"x_cond": x_cond, "rel_t": rel_t, "motion": motion}
        else:
            model_kwargs = {"y": y, "x_cond": x_cond, "rel_t": rel_t}
        loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
        loss = loss_dict["loss"].mean()

    if not bfloat_enable:
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:
        opt.zero_grad()
        scaler.scale(loss).backward()
        if config.training.grad_clip_val > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.training.grad_clip_val
            )
        scaler.step(opt)
        scaler.update()

    if scheduler is not None:
        scheduler.step()

    update_ema(ema, model.module)

    return loss.detach().item()
