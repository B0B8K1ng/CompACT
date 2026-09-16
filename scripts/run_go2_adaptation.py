#!/usr/bin/env python3
"""Reproducible Go2 adaptation; orchestrates the existing trainer and planner.

Run with an explicit environment, e.g. conda run -n nwm python -u
scripts/run_go2_adaptation.py prepare --run-id go2_v1. See docs/go2_adaptation.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nwm_benchmark_registry import MODELS
from scripts.prepare_go2_nwm import local_deltas

NAS = Path("/file_system/nas/algorithm/dujun.nie")
DATA_ROOT = NAS / "datasets/unitree-go2-data/nwm_4hz"
OUTPUT_ROOT = NAS / "nwm/compact/go2_adaptation"
SPACING = 0.07610250112606785
SOURCES = {
    name: dict(MODELS[key]) for name, key in {
        "timept-ft": "nwm-timept-ft", "geopt-ft": "nwm-geopt-ft",
        "no-pretrain": "nwm-no-pretrain", "nwm-real": "nwm-real",
    }.items()
}
LATENT_DIR = NAS / "nwm/compact/runs/navanywherev1_latentpt_ft_pixel_action_l20/nwm-nav1-latentpt-finetune-pixel-action-l20"
SOURCES["latentpt-ft"] = {
    "checkpoint": str(LATENT_DIR / "checkpoints/joint_0100000.pth.tar"),
    "checkpoint_id": "joint_0100000", "exp_dir": str(LATENT_DIR),
}
MODEL_ORDER = ["timept-ft", "geopt-ft", "latentpt-ft", "no-pretrain", "nwm-real"]
METRICS = ("ate", "rpe_trans", "pos_diff_norm", "yaw_diff_norm")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def lock_json(path, value):
    """An experiment ID names immutable inputs, not an overwriteable output."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Inputs changed for {path}; use a new --run-id")
    else:
        write_json(path, value)


def read_names(root, split, filename="traj_names.txt"):
    return (root / "data_splits/go2" / split / filename).read_text().split()


def episode_name(trajectory):
    return trajectory.rsplit("__seg", 1)[0]


def read_trajectory(root, name):
    with (root / "go2" / name / "traj_data.pkl").open("rb") as stream:
        traj = pickle.load(stream)
    position, yaw = np.asarray(traj["position"]), np.asarray(traj["yaw"]).reshape(-1)
    if len(position) != len(yaw) or not np.isfinite(position).all() or not np.isfinite(yaw).all():
        raise ValueError(f"Invalid Go2 trajectory: {name}")
    return position[:, :2], yaw


def prepare_protocol(root, *, smoke=False, eval_windows=100):
    """Read-only: create the fixed index and train-only CEM prior in memory."""
    normalization = json.loads((root / "normalization.json").read_text())
    if normalization != {"metric_waypoint_spacing": SPACING, "fit_split": "train", "fps": 4.0}:
        raise ValueError("Go2 spacing/FPS must match the fixed train-only normalization")
    names = {s: sorted(read_names(root, s)) for s in ("train", "test")}
    if any(len(value) != len(set(value)) for value in names.values()):
        raise ValueError("Duplicate Go2 segments in split manifests")
    episodes = {s: set(read_names(root, s, "episode_names.txt")) for s in names}
    if len(episodes["train"]) != 5 or len(episodes["test"]) != 2 or episodes["train"] & episodes["test"]:
        raise ValueError("Expected five train and two disjoint test episodes")
    vectors, candidates = [], defaultdict(list)
    files = {"normalization.json": sha256(root / "normalization.json")}
    for split in names:
        for filename in ("traj_names.txt", "episode_names.txt", "all_traj_names.txt"):
            path = root / "data_splits/go2" / split / filename
            files[str(path.relative_to(root))] = sha256(path)
        for name in names[split]:
            episode = episode_name(name)
            if episode not in episodes[split]:
                raise ValueError(f"Segment {name} not in its declared {split} episode manifest")
            position, yaw = read_trajectory(root, name)
            # Fingerprint actual model inputs; caches are deliberately excluded.
            for path in sorted((root / "go2" / name).iterdir()):
                if path.suffix in {".jpg", ".pkl", ".json", ".csv"}:
                    files[str(path.relative_to(root))] = sha256(path)
            for start in range(3, len(yaw) - 8):
                if split == "test":
                    candidates[episode].append((name, start, 8, 8))
            if split == "train":
                delta = local_deltas(position, yaw, offset=8)[3:]
                translation = delta[:, :2] / (8 * SPACING)
                normalized = 2 * (translation - [-2.5, -4]) / [7.5, 8] - 1
                heading = np.arctan2(delta[:, 1], delta[:, 0])
                residual = (delta[:, 2] - heading + np.pi) % (2 * np.pi) - np.pi
                vectors.append(np.column_stack([normalized, residual / np.pi]))
    rng = np.random.default_rng(42)
    index = []
    if eval_windows <= 0 or eval_windows % 2:
        raise ValueError("eval_windows must be a positive even number")
    count = 1 if smoke else eval_windows // 2
    for episode in sorted(episodes["test"]):
        windows = candidates[episode]
        if len(windows) < count:
            raise ValueError(f"Not enough test windows in {episode}: {len(windows)}")
        selected = sorted(rng.choice(len(windows), count, replace=False).tolist())
        index.extend(windows[i] for i in selected)
    vectors = np.concatenate(vectors)
    if not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("No finite train-only CEM fitting data")
    protocol = {
        "version": "go2_cem80_v1_smoke" if smoke else "go2_cem80_v1",
        "data_root": str(root), "data_sha256": digest_json(files),
        "spacing": SPACING, "fps": 4, "context_size": 4,
        "horizon_steps": 8, "sample_count": len(index), "samples_per_episode": count,
        "index": [list(x) for x in index], "index_seed": 42, "planning_sample_seed": 42,
        "num_samples": 80, "topk": 5, "opt_steps": 1, "num_repeat_eval": 3,
        "diffusion_steps": 250, "planning_microbatch_size": 40, "batch_size": 1,
        "cost_fn": "lpips", "compute_cost_with_recon": True,
        "prior": {"mu": vectors.mean(0).tolist(),
                  "var_scale": np.maximum(vectors.std(0), [.02, .02, .05]).tolist(),
                  "train_window_count": len(vectors)},
        "training": {"seed": 0, "warmup_steps": 2 if smoke else 200,
                     "joint_steps": 2 if smoke else 800, "batch_size": 8,
                     "adapter_lr": 1e-5, "backbone_lr": 1e-6, "ema_decay": .99},
    }
    return protocol, files


def source_files():
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    return sorted({name for name in output.decode().split("\0")
                   if name and Path(name).suffix in {".py", ".yaml"}
                   and not name.startswith(("tables/", "third_party/")) and (ROOT / name).is_file()})


def code_identity():
    return {name: sha256(ROOT / name) for name in source_files()}


def save_environment(output, identity):
    if (output / "environment.json").exists():
        return
    import importlib.metadata
    versions = {}
    for package in ("torch", "numpy", "wandb", "hydra-core", "diffusers", "lpips", "evo"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    write_json(output / "environment.json", {
        "python": sys.version, "executable": sys.executable, "packages": versions,
        "installed_packages": sorted(f"{d.metadata['Name']}=={d.version}"
                                     for d in importlib.metadata.distributions()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_sha256": digest_json(identity),
    })
    (output / "workspace.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
    for name in identity:
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)


def require_capacity(gpu=None, *, allow_busy=False, output=OUTPUT_ROOT):
    disk_path = Path(output)
    while not disk_path.exists():
        disk_path = disk_path.parent
    if shutil.disk_usage(disk_path).free < 30 * 1024**3:
        raise RuntimeError("Less than 30 GiB free for Go2 artifacts")
    if gpu is None:
        return
    result = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    free, utilization = map(int, result.strip().split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True).strip()
    print(f"GPU {gpu}: free={free} MiB, utilization={utilization}%", flush=True)
    if free < 24000 or ((utilization > 10 or processes) and not allow_busy):
        raise RuntimeError(f"GPU {gpu} is not available (need >=24000 MiB and idle GPU). "
                           "Existing experiments were not changed; select an available GPU later.")


def run_process(command, log, args):
    print(f"cwd={ROOT}\ncommand={shlex.join(command)}\nlog={log}", flush=True)
    if args.dry_run:
        return
    require_capacity(args.gpu, allow_busy=args.allow_busy_gpu, output=args.output)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               NWM_GO2_ROOT=str(args.data_root), NWM_INDEX_ROOT=str(args.output / "index_cache"),
               MPLBACKEND="Agg")
    env.setdefault("TORCH_HOME", str(NAS / "nwm/cache/torch"))
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(f"PID={process.pid}", flush=True)
        write_json(log.with_suffix(".process.json"), {"pid": process.pid, "command": command,
                    "cwd": str(ROOT), "log": str(log), "status": "running"})
        code = process.wait()
    write_json(log.with_suffix(".process.json"), {"pid": process.pid, "command": command,
               "cwd": str(ROOT), "log": str(log), "exit_code": code})
    print(f"exit_code={code}\n" + "\n".join(log.read_text(errors="replace").splitlines()[-12:]), flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def build_training_config(args, name, source, protocol):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import hydra_utils  # registers divide
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="nwm", overrides=["two_stage=go2_adapt"])
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    original = OmegaConf.create(checkpoint["config"])
    cfg.model = original.model
    cfg.motion_condition = original.motion_condition
    cfg.motion_condition.train_types = ["real"]
    cfg.motion_condition.eval_type = "real"
    cfg.motion_condition.dataset_motion_types = {"go2": "real"}
    # Keep the VAE identity, but use the pinned local cache when sources name HF.
    if cfg.model.tokenizer.get("model_path") == "stabilityai/sd-vae-ft-ema":
        cfg.model.tokenizer.model_path = str(NAS / "huggingface/hub/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a")
    OmegaConf.update(cfg, "model.diffusion.eval_timestep_respacing", 250, force_add=True)
    cfg.dataset.datasets.go2.data_folder = str(args.data_root / "go2")
    cfg.dataset.datasets.go2.train = str(args.data_root / "data_splits/go2/train")
    cfg.dataset.datasets.go2.test = str(args.data_root / "data_splits/go2/test")
    cfg.finetune.init_checkpoint = source["checkpoint"]
    cfg.finetune.init_sha256 = source["sha256"]
    cfg.finetune.warmup_steps = protocol["training"]["warmup_steps"]
    cfg.finetune.joint_steps = protocol["training"]["joint_steps"]
    cfg.max_train_steps = cfg.finetune.warmup_steps + cfg.finetune.joint_steps
    cfg.training.run_name = f"{args.run_id}-{name}"
    cfg.training.results_dir = str(args.output / "train")
    cfg.training.wandb_group = args.run_id
    cfg.training.wandb_notes = f"Go2 adaptation from {name}; protocol={digest_json(protocol)}"
    if args.smoke:
        cfg.log_every = 1
        cfg.ckpt_every = 1
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.hydra = {"run": {"dir": str(args.output / "train" / name)}, "job": {"chdir": True}}
    return cfg


def load_source(name, args):
    source = dict(SOURCES[name])
    path = Path(source["checkpoint"])
    if not path.is_file():
        print(f"PENDING {name}: final checkpoint not available: {path}", flush=True)
        return None
    print(f"Checking source SHA-256: {name}", flush=True)
    actual = sha256(path)
    if source.get("sha256") and actual != source["sha256"]:
        raise ValueError(f"Registered source checksum mismatch: {name}")
    source["sha256"] = actual
    if not args.dry_run:
        lock_json(args.output / "sources" / f"{name}.json", source)
    return source


def final_checkpoint(args, name, protocol):
    steps = protocol["training"]["joint_steps"]
    return args.output / "train" / name / "checkpoints" / f"joint_{steps:07d}.pth.tar"


def train_model(args, name, cfg, protocol):
    from omegaconf import OmegaConf
    exp = args.output / "train" / name
    final = final_checkpoint(args, name, protocol)
    final_record = exp / "completed.json"
    contract = {"config_sha256": digest_json(OmegaConf.to_container(cfg, resolve=True)),
                "protocol_sha256": digest_json(protocol)}
    if final_record.exists():
        record = json.loads(final_record.read_text())
        if record["contract"] != contract or not final.is_file() or sha256(final) != record["sha256"]:
            raise ValueError(f"Completed training inputs/checkpoint changed: {name}")
        print(f"SKIP completed training: {name}", flush=True)
        return
    config_dir = args.output / "configs" / name
    if not args.dry_run:
        lock_json(exp / "contract.json", contract)
        config_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dir / "train.yaml")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
               str(ROOT / "train.py"), f"--config-path={config_dir}", "--config-name=train"]
    latest = exp / "checkpoints/latest.pth.tar"
    if args.resume and latest.exists():
        command.append(f"training.from_checkpoint={latest}")
    elif latest.exists() or (exp / "wandb_state.json").exists():
        raise ValueError(f"Existing run {name}: pass --resume to continue")
    run_process(command, args.output / "logs" / f"train-{name}.log", args)
    if not args.dry_run:
        if not final.exists():
            raise FileNotFoundError(f"Trainer did not produce {final}")
        write_json(final_record, {"contract": contract, "sha256": sha256(final)})


def reuse_training(args, name, cfg, protocol):
    """Explicitly reuse a completed training run across evaluation-only changes."""
    from omegaconf import OmegaConf
    if args.reuse_training_from is None:
        return False
    previous = args.reuse_training_from.resolve()
    old_exp = previous / "train" / name
    record_path = old_exp / "completed.json"
    if not record_path.exists():
        return False
    old_protocol = json.loads((previous / "protocol.json").read_text())
    for key in ("data_sha256", "training"):
        if old_protocol[key] != protocol[key]:
            raise ValueError(f"Cannot reuse training with changed {key}: {name}")
    old_cfg = OmegaConf.load(previous / "configs" / name / "train.yaml")
    record = json.loads(record_path.read_text())
    if record["contract"] != {
        "config_sha256": digest_json(OmegaConf.to_container(old_cfg, resolve=True)),
        "protocol_sha256": digest_json(old_protocol),
    }:
        raise ValueError(f"Original training contract changed: {name}")
    def training_inputs(value):
        value = OmegaConf.to_container(value, resolve=True)
        value.pop("hydra", None)
        for key in ("run_name", "results_dir", "wandb_group", "wandb_notes"):
            value["training"].pop(key, None)
        return value
    if training_inputs(old_cfg) != training_inputs(cfg):
        raise ValueError(f"Training configuration differs: {name}")
    # Only orchestration and tests may change when importing completed training.
    old_identity = {str(path.relative_to(previous / "source")): sha256(path)
                    for path in (previous / "source").rglob("*") if path.is_file()}
    if digest_json(old_identity) != old_protocol["code_sha256"]:
        raise ValueError("Original training source snapshot changed")
    identity = code_identity()
    for path in set(old_identity) | set(identity):
        if path == "scripts/run_go2_adaptation.py" or path.startswith("tests/"):
            continue
        if old_identity.get(path) != identity.get(path):
            raise ValueError(f"Training source changed: {path}")
    checkpoint = old_exp / "checkpoints" / final_checkpoint(args, name, protocol).name
    if sha256(checkpoint) != record["sha256"]:
        raise ValueError(f"Completed checkpoint changed: {name}")
    print(f"REUSE completed training: {name} from {old_exp}", flush=True)
    if not args.dry_run:
        lock_json(args.output / "reused_training" / f"{name}.json", {
            "original_run": str(previous), "checkpoint_sha256": record["sha256"],
            "original_contract": record["contract"], "protocol_sha256": digest_json(protocol),
        })
        target = args.output / "train" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != old_exp.resolve():
                raise ValueError(f"Training reuse destination changed: {target}")
        elif target.exists():
            raise ValueError(f"Training reuse destination already exists: {target}")
        else:
            target.symlink_to(old_exp, target_is_directory=True)
    return True


def eval_model(args, name, condition, cfg, checkpoint, protocol):
    from omegaconf import OmegaConf
    if not checkpoint.is_file() and not args.dry_run:
        print(f"PENDING {name}/{condition}: {checkpoint}", flush=True)
        return
    output = args.output / "eval" / name / condition
    checkpoint_sha = sha256(checkpoint) if checkpoint.exists() else "pending-training"
    contract = {"checkpoint_sha256": checkpoint_sha, "protocol_sha256": digest_json(protocol),
                "weight_key": "ema"}
    done = output / "completed.json"
    if done.exists():
        if json.loads(done.read_text()) != contract:
            raise ValueError(f"Completed evaluation inputs changed: {name}/{condition}")
        collect_metrics(output, protocol)  # Verify all samples before skipping.
        print(f"SKIP completed evaluation: {name}/{condition}", flush=True)
        return
    view = args.output / "views" / name / condition
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    plan = OmegaConf.load(ROOT / "conf/plan_config.yaml")
    del plan["defaults"]
    cfg = OmegaConf.merge(cfg, plan)
    cfg.exp_dir = str(view)
    cfg.ckp = "source"
    cfg.output_dir = str(output)
    cfg.datasets_to_eval = ["go2"]
    cfg.num_workers = 4
    cfg.planning_sample_seed = 42
    cfg.plan_datasets_hyperparams = {"go2": {k: protocol["prior"][k] for k in ("mu", "var_scale")}}
    for key in ("num_samples", "topk", "opt_steps", "num_repeat_eval", "planning_microbatch_size", "batch_size"):
        cfg[key] = protocol[key]
    cfg.dataset.datasets.go2.navigation_index = str(args.output / "navigation_eval.pkl")
    cfg.dataset.datasets.go2.navigation_traj_names = "traj_names.txt"
    cfg.dataset.datasets.go2.navigation_sample_count = protocol["sample_count"]
    cfg.hydra = {"run": {"dir": str(output / "hydra")}, "job": {"chdir": True}}
    if not args.dry_run:
        lock_json(output / "contract.json", contract)
        (view / ".hydra").mkdir(parents=True, exist_ok=True)
        (view / "checkpoints").mkdir(exist_ok=True)
        link = view / "checkpoints/source.pth.tar"
        if link.is_symlink():
            if link.resolve() != checkpoint.resolve():
                raise ValueError(f"Checkpoint view changed: {link}")
        else:
            link.symlink_to(checkpoint)
        OmegaConf.save(cfg, view / ".hydra/config.yaml")
        OmegaConf.save(cfg, view / "go2_eval.yaml")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
               str(ROOT / "planning_eval.py"), f"--config-path={view}", "--config-name=go2_eval"]
    run_process(command, args.output / "logs" / f"eval-{name}-{condition}.log", args)
    if not args.dry_run:
        collect_metrics(output, protocol)
        write_json(done, contract)


def collect_metrics(output, protocol):
    folders = list(output.glob("go2/*/sample_metrics"))
    if len(folders) != 1:
        raise ValueError(f"Expected exactly one metric directory: {output}")
    folder = folders[0]
    expected = {f"{i:06d}.json" for i in range(protocol["sample_count"])}
    if {p.name for p in folder.glob("*.json")} != expected:
        raise ValueError(f"Incomplete/extra navigation samples: {folder}")
    groups = defaultdict(list)
    for i, entry in enumerate(protocol["index"]):
        value = json.loads((folder / f"{i:06d}.json").read_text())
        row = [float(value[k]) for k in METRICS]
        if not np.isfinite(row).all():
            raise ValueError(f"Nonfinite navigation metric at sample {i}")
        groups[episode_name(entry[0])].append(row)
        groups["overall"].append(row)
    results = {}
    for group, rows in groups.items():
        mean = np.mean(rows, axis=0)
        results[group] = {"count": len(rows), "normalized": dict(zip(METRICS, mean.tolist())),
                          "ate_m": float(mean[0] * SPACING), "rpe_trans_m": float(mean[1] * SPACING),
                          "endpoint_m": float(mean[2] * SPACING), "endpoint_yaw_rad": float(mean[3])}
    return results


def summarize(args, protocol):
    results, pending, wandb_runs = {}, [], {}
    for name in args.models:
        state = args.output / "train" / name / "wandb_state.json"
        if state.exists():
            wandb_runs[name] = json.loads(state.read_text())
            if not args.dry_run:
                try:
                    import wandb
                    run = wandb.Api(timeout=30).run(wandb_runs[name]["path"])
                    wandb_runs[name]["remote_state"] = run.state
                    wandb_runs[name]["remote_last_step"] = run.summary.get("_step")
                except Exception as exc:
                    wandb_runs[name]["remote_state"] = "unverified"
                    wandb_runs[name]["verification_error"] = str(exc)
        for condition in ("before", "after"):
            key = f"{name}/{condition}"
            output = args.output / "eval" / name / condition
            if (output / "completed.json").exists():
                contract = json.loads((output / "completed.json").read_text())
                checkpoint = Path(SOURCES[name]["checkpoint"]) if condition == "before" else final_checkpoint(args, name, protocol)
                if contract["protocol_sha256"] != digest_json(protocol) or sha256(checkpoint) != contract["checkpoint_sha256"]:
                    raise ValueError(f"Stale completed metrics: {key}")
                results[key] = collect_metrics(output, protocol)
            else:
                pending.append(key)
    summary = {"evaluation": "offline CEM planning; not closed-loop success rate",
               "protocol_sha256": digest_json(protocol), "results": results, "pending": pending,
               "wandb": wandb_runs}
    lines = ["# Go2 offline navigation", "", "| Model | Condition | Episode | N | ATE (m) | RPE (m) | Endpoint (m) | Yaw (rad) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, groups in results.items():
        name, condition = key.split("/")
        for group, metrics in groups.items():
            lines.append(f"| {name} | {condition} | {group} | {metrics['count']} | "
                         + " | ".join(f"{metrics[k]:.6f}" for k in ("ate_m", "rpe_trans_m", "endpoint_m", "endpoint_yaw_rad")) + " |")
    lines.extend(["", "Pending: " + (", ".join(pending) or "none"), ""])
    for name, state in wandb_runs.items():
        lines.append(f"- {name}: {state['url']} (upload: {state.get('remote_state', state['status'])})")
    if not args.dry_run:
        write_json(args.output / "summary.json", summary)
        (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "eval", "summarize", "all"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Separate protocol: 2+2 training steps and two test windows")
    parser.add_argument("--eval-windows", type=int, default=100, help="Total fixed test windows, equally split between two episodes")
    parser.add_argument("--reuse-training-from", type=Path, help="Reuse verified completed training from an earlier experiment directory")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Only use when the GPU owner explicitly permits sharing")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("--run-id must be a simple directory name")
    args.models = args.models.split(",")
    if len(set(args.models)) != len(args.models) or set(args.models) - set(SOURCES):
        parser.error(f"--models must select unique names from {MODEL_ORDER}")
    args.data_root = args.data_root.resolve()
    args.output = args.output_root.resolve() / args.run_id
    print(f"Checking Go2 input files: {args.data_root}", flush=True)
    protocol, files = prepare_protocol(args.data_root, smoke=args.smoke, eval_windows=args.eval_windows)
    if args.reuse_training_from:
        protocol["reuse_training_from"] = str(args.reuse_training_from.resolve())
    identity = code_identity()
    protocol["code_sha256"] = digest_json(identity)
    index_bytes = pickle.dumps([tuple(x) for x in protocol["index"]], protocol=4)
    protocol["index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
    print(f"Protocol={protocol['version']}, windows={protocol['sample_count']}, sha256={digest_json(protocol)}", flush=True)
    if not args.dry_run:
        require_capacity(output=args.output)
        lock_json(args.output / "protocol.json", protocol)
        lock_json(args.output / "data_files.json", files)
        index_path = args.output / "navigation_eval.pkl"
        if index_path.exists() and sha256(index_path) != protocol["index_sha256"]:
            raise ValueError("Saved navigation index changed")
        if not index_path.exists():
            index_path.write_bytes(index_bytes)
        save_environment(args.output, identity)
    if args.stage in {"prepare", "summarize"}:
        if args.stage == "summarize":
            summarize(args, protocol)
        return
    ready = []
    for name in args.models:
        source = load_source(name, args)
        if source is None:
            continue
        cfg = build_training_config(args, name, source, protocol)
        ready.append((name, source, cfg))
        if args.stage in {"train", "all"}:
            if not reuse_training(args, name, cfg, protocol):
                train_model(args, name, cfg, protocol)
    if args.stage in {"eval", "all"}:
        for name, source, cfg in ready:
            eval_model(args, name, "before", cfg, Path(source["checkpoint"]), protocol)
            eval_model(args, name, "after", cfg, final_checkpoint(args, name, protocol), protocol)
    summarize(args, protocol)


if __name__ == "__main__":
    main()
