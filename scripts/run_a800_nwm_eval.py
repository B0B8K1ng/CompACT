#!/usr/bin/env python3
"""Resumable, single-node eight-GPU NWM benchmark coordinator.

The plan command never writes. All other state lives under RUN_ROOT/run_id.
Only one coordinator may hold the flock for a run. Workers have independent
process groups; use stop to terminate both the coordinator and this run's workers.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import functools
import hashlib
import json
import math
import os
import pickle
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
from nwm_benchmark_registry import DATASET_CONTRACTS, MODELS, PROTOCOLS, dataset_sample_count

NAS = Path("/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark")
RUN_ROOT = NAS / "a800_full_evaluations"
DATA = Path("/file_system/nas/algorithm/dujun.nie/nwm/data")
NWM_PYTHON = Path("/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python")
RAE_PYTHON = Path("/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin/python")
DATASETS = ("recon", "scand", "huron", "tartan_drive", "go_stanford", "planetary_rover", "tum_rgbd", "unitree_go2")
ROLLOUT = tuple(name for name in DATASETS if name != "planetary_rover")
NAV = ("recon", "scand", "go_stanford", "unitree_go2")
MODELS_REQUESTED = ("nwm-release", "rae-nwm", "nwm-ego4d", "opennwm-finalLAM-100k",
                    "nwm-latentpt-action-ft", "nwm-latentpt-pixel-ft-reset")
STEM = "CEM_N80_K5_RS1_rep3_OPT1_COST-lpips-RECON-True"
OPEN = {
    "exp_dir": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/nav15_pa_step60000_latentpt_reset_ft/nwm-latentpt-pixel-action-finalLAM-ft-reset",
    "checkpoint_id": "joint_0100000",
    "checkpoint": "/file_system/nas/algorithm/dujun.nie/nwm/compact/runs/nav15_pa_step60000_latentpt_reset_ft/nwm-latentpt-pixel-action-finalLAM-ft-reset/checkpoints/joint_0100000.pth.tar",
    "sha256": "aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4",
}
MODELSCOPE = Path("/file_system/nas/algorithm/dujun.nie/nwm/weights/modelscope/LittleBoss")
MODEL_VIEWS = Path("/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models")
LOCAL_VAE = Path("/file_system/nas/algorithm/dujun.nie/huggingface/hub/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a")
ADDED = {
    "nwm-latentpt-action-ft": {
        "exp_dir": str(MODEL_VIEWS / "nwm-latentpt-action-ft"),
        "checkpoint_id": "joint_0100000",
        "checkpoint": str(MODELSCOPE / "nwm-latentpt-action-ft/joint_0100000.pth.tar"),
        "sha256": "1b95998bc30b163b599be3687a028a93bf927ef53fb7f886da3f219c44d6f852",
        "embedded_run_name": "nwm-latentpt-action-ft-reset",
    },
    "nwm-latentpt-pixel-ft-reset": {
        "exp_dir": str(MODEL_VIEWS / "nwm-latentpt-pixel-ft-reset"),
        "checkpoint_id": "joint_0100000",
        "checkpoint": str(MODELSCOPE / "nwm-nav1-latentpt-pixel-ft-reset/joint_0100000.pth.tar"),
        "sha256": "03507c990230bbe329d8aca9e6e739aadefe4905ff4e5442ee307d9f7c2c12d2",
        "embedded_run_name": "nwm-latentpt-pixel-reset",
    },
}
MODEL = {**{name: MODELS[name] for name in MODELS_REQUESTED if name in MODELS},
         "opennwm-finalLAM-100k": OPEN, **ADDED}
FRAME_SPECS = {"rollout_1fps": "1s:0,2s:1,4s:3,8s:7,16s:15", "rollout_4fps": "1s:3,2s:7,4s:15,8s:31,16s:63"}
METRIC_KEYS = ("ate", "rpe_trans", "pos_diff_norm", "yaw_diff_norm")
MAX_CONCURRENT_GT = 4


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model_view(model):
    """Make an inference view from the checkpoint's own training config.

    ModelScope archives have no .hydra/config.yaml or checkpoints directory.
    The view contains one small config and a symlink to the original weight.
    Existing files are checked rather than silently replaced on resume.
    """
    spec = ADDED[model]
    checkpoint = Path(spec["checkpoint"])
    view = Path(spec["exp_dir"])
    config_path = view / ".hydra/config.yaml"
    link = view / "checkpoints" / f"{spec['checkpoint_id']}.pth.tar"
    manifest_path = view / "modelscope_view.json"
    if not LOCAL_VAE.is_dir():
        raise FileNotFoundError(f"Local SD-VAE missing: {LOCAL_VAE}")
    manifest = read_json(manifest_path)
    if manifest is not None:
        if (manifest.get("checkpoint_sha256") != spec["sha256"]
                or manifest.get("config_sha256") != sha(config_path)
                or not link.is_symlink() or link.resolve() != checkpoint.resolve()):
            raise RuntimeError(f"ModelScope inference view changed: {view}")
        return

    import torch
    import yaml
    archive = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = archive.get("config")
    metadata = archive.get("two_stage_metadata") or {}
    if (not isinstance(config, dict) or archive.get("train_steps") != 110000
            or config.get("training", {}).get("run_name") != spec["embedded_run_name"]
            or config.get("model", {}).get("generator", {}).get("_target_") != "models.CDiT_B_2"
            or config.get("dataset", {}).get("context_size") != 4
            or metadata.get("finetune_scheme") != "reset"
            or metadata.get("action_mode") != "real"):
        raise RuntimeError(f"Unexpected embedded ModelScope training config: {checkpoint}")
    tokenizer = config["model"]["tokenizer"]
    if "sd-vae-ft-ema" not in str(tokenizer.get("model_path", "")):
        raise RuntimeError(f"Unexpected tokenizer in {checkpoint}")
    tokenizer["model_path"] = str(LOCAL_VAE)
    config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        if config_path.read_text() != config_text:
            raise RuntimeError(f"Existing model config differs from embedded checkpoint: {config_path}")
    else:
        temporary = config_path.with_name(f"config.yaml.tmp.{os.getpid()}")
        temporary.write_text(config_text)
        temporary.replace(config_path)
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != checkpoint.resolve():
            raise RuntimeError(f"Existing checkpoint view points elsewhere: {link}")
    else:
        temporary = link.with_name(f"{link.name}.tmp.{os.getpid()}")
        temporary.symlink_to(checkpoint)
        temporary.replace(link)
    atomic_json(manifest_path, {"model": model, "checkpoint": str(checkpoint),
                                "checkpoint_sha256": spec["sha256"],
                                "config_sha256": config_hash,
                                "local_vae": str(LOCAL_VAE)})


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def good_png(path):
    try:
        stat = path.stat()
        return verified_png(str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return False


@functools.lru_cache(maxsize=300000)
def verified_png(path, inode, size, mtime, ctime):
    """Recheck changed files, but avoid rereading unchanged PNGs from NAS."""
    try:
        with Image.open(path) as picture:
            picture.verify()
        return True
    except (OSError, ValueError):
        return False


def good_nav(path):
    row = read_json(path)
    return isinstance(row, dict) and all(
        isinstance(row.get(key), (float, int)) and math.isfinite(row[key]) for key in METRIC_KEYS)


def expected_split(dataset, kind):
    contract = DATASET_CONTRACTS[dataset]
    name = "navigation_eval" if kind == "navigation" else "rollout" if kind == "rollout" else "time"
    return ROOT / "data_splits" / contract["split_name"] / "test" / f"{name}.pkl", contract["splits"][name]


def count(dataset, kind):
    if dataset == "huron" and kind in ("time", "rollout"):
        # The immutable split has 500/150 rows, but the dataset loader drops
        # rows with missing or too-short trajectories before assigning IDs.
        return 329 if kind == "time" else 103
    protocol = PROTOCOLS["navigation_cem80_v1" if kind == "navigation" else "rollout_v1" if kind == "rollout" else "direct_4s_v1"]
    return dataset_sample_count(protocol, dataset, "navigation" if kind == "navigation" else "rollout_1fps" if kind == "rollout" else "time")


def actual_split_population(dataset, kind):
    """Apply the loader's trajectory-existence and range checks without writing."""
    split, _ = expected_split(dataset, kind)
    rows = pickle.loads(split.read_bytes())
    data_folder = DATA / DATASET_CONTRACTS[dataset]["data_name"]
    lengths = {}
    missing = short = 0
    valid_raw_rows = []
    effective_rows = []
    for raw_index, (trajectory, current, lower, upper, *_) in enumerate(rows):
        if trajectory not in lengths:
            path = data_folder / trajectory / "traj_data.pkl"
            lengths[trajectory] = len(pickle.loads(path.read_bytes())["position"]) if path.is_file() else None
        length = lengths[trajectory]
        if length is None:
            missing += 1
        elif current + lower < 0 or current + upper >= length:
            short += 1
        else:
            valid_raw_rows.append(raw_index)
            effective_rows.append([raw_index, trajectory, current, lower, upper, length])
    mapping = json.dumps(effective_rows, sort_keys=True, separators=(",", ":"))
    length_manifest = json.dumps(sorted(lengths.items()), separators=(",", ":"))
    return {"raw": len(rows), "valid": len(rows) - missing - short,
            "missing_trajectory_rows": missing, "out_of_range_rows": short,
            "effective_raw_row_indices": valid_raw_rows,
            "effective_mapping_sha256": hashlib.sha256(mapping.encode()).hexdigest(),
            "trajectory_lengths_sha256": hashlib.sha256(length_manifest.encode()).hexdigest(),
            "split_sha256": sha(split),
            "sample_id_semantics": "position_after_loader_filtering"}


@functools.lru_cache(maxsize=2)
def huron_population(kind):
    audit = actual_split_population("huron", kind)
    if audit["valid"] != count("huron", kind):
        raise RuntimeError(f"Huron/{kind} population changed: expected {count('huron', kind)}, got {audit['valid']}")
    return audit


@functools.lru_cache(maxsize=16)
def direct_ids(dataset):
    """Full split positions that can provide each endpoint. No split is rewritten."""
    total = count(dataset, "time")
    if dataset == "planetary_rover":
        return {1: list(range(total)), 2: list(range(total)), 4: list(range(total))}
    if dataset not in ("tum_rgbd", "unitree_go2"):
        return {h: list(range(total)) for h in (1, 2, 4, 8, 16)}
    split, _ = expected_split(dataset, "time")
    rows = pickle.loads(split.read_bytes())
    lengths = {}
    result = {}
    for horizon in (1, 2, 4, 8, 16):
        ids = []
        for index, (trajectory, current, *_rest) in enumerate(rows):
            if trajectory not in lengths:
                p = DATA / dataset / trajectory / "traj_data.pkl"
                lengths[trajectory] = len(pickle.loads(p.read_bytes())["position"])
            if current + horizon * 4 < lengths[trajectory]:
                ids.append(index)
        result[horizon] = ids
    return result


def contract(model, dataset, kind):
    _, split_hash = expected_split(dataset, kind)
    spec = MODEL[model]
    rae = model == "rae-nwm"
    result = {"checkpoint_sha256": spec["sha256"], "split_sha256": split_hash,
            "seed": 42 if kind == "navigation" else 0,
            "sampler": "euler_ode" if rae else "ddpm", "sampling_steps": 50 if rae else 250,
            "protocol": "navigation_cem80_v1" if kind == "navigation" else "rollout_v1" if kind == "rollout" else "direct_4s_v1"}
    if dataset == "huron" and kind in ("time", "rollout"):
        population = huron_population(kind)
        result.update(effective_sample_count=population["valid"],
                      effective_mapping_sha256=population["effective_mapping_sha256"])
    return result


@functools.lru_cache(maxsize=1)
def run_contract():
    source_files = ("isolated_nwm_infer.py", "planning_eval.py", "scripts/raenwm_infer.py",
                    "scripts/raenwm_planning_eval.py", "scripts/evaluate_nwm_predictions.py",
                    "conf/infer_config.yaml", "conf/plan_config.yaml")
    return {
        "schema_version": 2,
        "models": {model: MODEL[model]["sha256"] for model in MODELS_REQUESTED},
        "contracts": {f"{model}/{dataset}/{kind}": contract(model, dataset, kind)
                      for model in MODELS_REQUESTED for dataset in DATASETS
                      for kind in ("time", "rollout", "navigation")
                      if kind == "time" or kind == "rollout" and dataset in ROLLOUT or kind == "navigation" and dataset in NAV},
        "direct_sample_ids": {dataset: {str(horizon): ids for horizon, ids in direct_ids(dataset).items()}
                              for dataset in DATASETS},
        "effective_split_population": {kind: huron_population(kind) for kind in ("time", "rollout")},
        "entrypoint_sha256": {name: sha(ROOT / name) for name in source_files},
    }


def refresh_failed_gt_contract(run_dir, recorded, current):
    """Adopt a source fix when the first GT failed before writing any new data."""
    if recorded == current:
        return recorded
    if run_dir.name != "full_8xa800_seed0_v1":
        return recorded
    old_without_sources = {key: value for key, value in recorded.items() if key != "entrypoint_sha256"}
    new_without_sources = {key: value for key, value in current.items() if key != "entrypoint_sha256"}
    if old_without_sources != new_without_sources:
        return recorded
    old_sources = recorded.get("entrypoint_sha256", {})
    new_sources = current.get("entrypoint_sha256", {})
    changed = {name for name in old_sources if old_sources[name] != new_sources.get(name)}
    if changed != {"isolated_nwm_infer.py", "scripts/raenwm_infer.py", "conf/infer_config.yaml"}:
        return recorded
    finish = read_json(run_dir / "finish.json") or {}
    job_paths = list((run_dir / "jobs").glob("*.json"))
    if (finish.get("exit_code") != 1 or not str(finish.get("error", "")).startswith("GT failed:")
            or len(job_paths) != 1 or job_paths[0].name != "gt_gt_direct_recon_ab69f171.json"
            or (read_json(job_paths[0]) or {}).get("status") != "failed"
            or any(path.is_file() for path in (run_dir / "gt").rglob("*"))):
        return recorded
    backup = run_dir / "contract.pre_gt_device_fix.json"
    if backup.exists():
        if read_json(backup) != recorded:
            raise RuntimeError(f"Existing contract backup differs: {backup}")
    else:
        with backup.open("x") as stream:
            json.dump(recorded, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    atomic_json(run_dir / "contract_gt_device_fix_migration.json", {
        "at": now(), "reason": "First shared GT failed before producing artifacts; CUDA integer-device fix",
        "backup": str(backup),
        "entrypoint_sha256_changes": {name: {"before": old_sources[name], "after": new_sources[name]}
                                        for name in sorted(changed)},
    })
    atomic_json(run_dir / "contract.json", current)
    print(f"updated failed-GT source contract; backup={backup}", flush=True)
    return current


def migrate_huron_population_contract(run_dir, recorded, current):
    """Allow only this failed run's 500/150 -> 329/103 Huron correction."""
    if recorded == current or run_dir.name != "full_8xa800_seed0_v1":
        return recorded
    legacy = copy.deepcopy(current)
    legacy["schema_version"] = 1
    legacy.pop("effective_split_population")
    legacy["direct_sample_ids"]["huron"] = {
        str(horizon): list(range(500)) for horizon in (1, 2, 4, 8, 16)}
    for key, value in legacy["contracts"].items():
        if key.split("/")[1:] in (["huron", "time"], ["huron", "rollout"]):
            value.pop("effective_sample_count")
            value.pop("effective_mapping_sha256")
    if recorded != legacy:
        return recorded
    finish = read_json(run_dir / "finish.json") or {}
    expected_failure = str(run_dir / "logs" / "gt_gt_direct_huron_ab69f171.log")
    huron_jobs = list((run_dir / "jobs").glob("*huron*.json"))
    if (finish.get("exit_code") != 1 or finish.get("error") != f"GT failed: {expected_failure}"
            or len(huron_jobs) != 1 or huron_jobs[0].name != "gt_gt_direct_huron_ab69f171.json"):
        return recorded
    failed_job = read_json(huron_jobs[0]) or {}
    if (failed_job.get("status") != "failed" or failed_job.get("job", {}).get("kind") != "gt_direct"
            or failed_job.get("job", {}).get("dataset") != "huron"
            or failed_job.get("job", {}).get("ids") != list(range(500))):
        return recorded
    artifact_roots = [run_dir / "gt" / "huron"] + [
        run_dir / "predictions" / model / "huron" for model in MODELS_REQUESTED]
    if any(path.is_file() or path.is_symlink() for root in artifact_roots for path in root.rglob("*")):
        return recorded
    if any(path.is_file() or path.is_symlink()
           for subdir in ("metrics", "metric_filters") for model in MODELS_REQUESTED
           for path in (run_dir / subdir / model).glob("huron_*")):
        return recorded
    backup = run_dir / "contract.pre_huron_population_fix.json"
    if backup.exists():
        if read_json(backup) != recorded:
            raise RuntimeError(f"Existing contract backup differs: {backup}")
    else:
        with backup.open("x") as stream:
            json.dump(recorded, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    canonical_hash = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(run_dir / "contract_huron_population_migration.json", {
        "at": now(), "reason": "Huron loader filters missing and out-of-range trajectories before assigning sample IDs",
        "backup": str(backup), "old_contract_sha256": canonical_hash(recorded),
        "new_contract_sha256": canonical_hash(current),
        "effective_split_population": current["effective_split_population"],
    })
    atomic_json(run_dir / "contract.json", current)
    print(f"updated Huron population contract; backup={backup}", flush=True)
    return current


def prediction_sources(model, dataset, evaluation, c):
    """Only audited paths with a matching pinned checkpoint and protocol qualify."""
    if dataset == "huron":
        # The old 500/150-row results do not prove the new loader-filtered
        # sample-ID mapping. Regenerate them under this run's pinned mapping.
        return []
    candidates = []
    trusted_paths = set()
    # The old 4 s sweep records its checkpoint in a separate immutable manifest.
    if model == "opennwm-finalLAM-100k" and evaluation == "time":
        sweep = NAS / "finalLAM_reset_checkpoint_sweep_20260922"
        manifest = read_json(sweep / "manifest.json") or {}
        alias = "finalLAM-reset-joint0100000"
        registry = read_json(Path(manifest.get("source_registry", "/missing"))) or {}
        registered_split = registry.get("protocols", {}).get("direct_4s_v1", {}).get("splits", {}).get(dataset, {})
        if (manifest.get("protocol") == "direct_4s_v1" and dataset in manifest.get("datasets", [])
            and manifest.get("checkpoints", {}).get(alias, {}).get("sha256") == c["checkpoint_sha256"]
            and registered_split.get("sha256") == c["split_sha256"]):
            audit = sweep / "predictions" / alias / f"{dataset}_time_direct_4s_v1_audit.json"
            candidates.append(audit)
            trusted_paths.add(audit)
    # A previous run of this script carries a complete, per-job provenance file.
    ood_root = NAS / "protocol_runs/ood_direct_4s_v1"
    ood_manifest = read_json(ood_root / "run_manifest.json") or {}
    if (evaluation == "time" and ood_manifest.get("models", {}).get(model, {}).get("sha256") == c["checkpoint_sha256"]
        and ood_manifest.get("verified_inputs", {}).get(dataset, {}).get("prediction_split", {}).get("sha256") == c["split_sha256"]):
        trusted_paths.add(ood_root / "predictions" / model / f"{dataset}_time_audit.json")
    for p in (NAS / "predictions" / model / f"{dataset}_{evaluation}_audit.json",
              ood_root / "predictions" / model / f"{dataset}_{evaluation}_audit.json"):
        candidates.append(p)
    sources = []
    for audit_path in candidates:
        audit = read_json(audit_path)
        if not isinstance(audit, dict) or audit.get("dataset") != dataset or audit.get("eval_name") != evaluation:
            continue
        inf = audit.get("inference")
        if not isinstance(inf, dict):
            continue
        if any(inf.get(k) != c[k] for k in ("seed", "sampler", "sampling_steps")):
            continue
        if inf.get("checkpoint_sha256") not in (None, c["checkpoint_sha256"]):
            continue
        if inf.get("split_sha256") not in (None, c["split_sha256"]):
            continue
        # A missing hash may only be supplied by a trusted external manifest.
        trusted = inf.get("checkpoint_sha256") == c["checkpoint_sha256"] and inf.get("split_sha256") == c["split_sha256"]
        if not trusted:
            trusted = audit_path in trusted_paths
        if not trusted:
            continue
        if not isinstance(audit.get("pred_eval_dir"), str) or not isinstance(audit.get("frame_indices"), dict):
            continue
        source = Path(audit["pred_eval_dir"])
        if source.is_dir():
            sources.append((source, str(audit_path), set(audit["frame_indices"].values())))
    return sources


def nav_source(model, dataset, c):
    base = NAS / "navigation_largebatch_20260921/planning" / model
    if model == "rae-nwm":
        base /= "euler50"
        manifest = read_json(base / "planning_manifest.json") or {}
        proto = manifest.get("protocol", {})
        ds = manifest.get("datasets", {}).get(dataset, {})
        valid = (manifest.get("checkpoint", {}).get("sha256") == c["checkpoint_sha256"]
                 and ds.get("split_sha256") == c["split_sha256"] and
                 all(proto.get(k) == v for k, v in {"population": 80, "topk": 5, "repetitions": 3,
                   "optimization_steps": 1, "seed": 42, "sampler": "euler_ode", "sampling_steps": 50,
                   "microbatch_size": 80}.items()))
    else:
        # Hydra's persisted invocation proves the actual checkpoint identity and
        # navigation settings; the registry pins the checkpoint bytes.
        import yaml
        registry = read_json(NAS / "navigation_largebatch_20260921/benchmark_results.json") or {}
        registered_model = registry.get("models", {}).get(model, {})
        registered_split = registry.get("protocols", {}).get("navigation_cem80_v1", {}).get("splits", {}).get(dataset, {})
        configs = list(base.glob(f"planning_*{dataset}*_step1/.hydra/config.yaml"))
        valid = False
        for path in configs:
            try:
                cfg = yaml.safe_load(path.read_text())
                valid = all(cfg.get(k) == v for k, v in {
                    "exp_dir": MODEL[model]["exp_dir"], "ckp": MODEL[model]["checkpoint_id"],
                    "num_samples": 80, "topk": 5, "opt_steps": 1, "num_repeat_eval": 3,
                    "seed": 42, "planning_sample_seed": 42, "planning_microbatch_size": 80,
                    "cost_fn": "lpips", "compute_cost_with_recon": True}.items())
                if valid:
                    break
            except (OSError, ValueError):
                pass
        valid = (valid and registered_model.get("sha256") == c["checkpoint_sha256"]
                 and registered_split.get("sha256") == c["split_sha256"])
    path = base / dataset / STEM / "sample_metrics"
    return (path, str(base)) if valid and path.is_dir() else None


def item_key(kind, model, dataset, evaluation, sample, endpoint=None):
    suffix = f"_{endpoint}" if endpoint is not None else ""
    return f"{kind}/{model}/{dataset}/{evaluation}/{sample}{suffix}"


def build_plan(run_dir, *, verify_files=True):
    """Reconcile actual artifacts on every call. No prior running flag is trusted."""
    items = {}
    jobs = []
    own_trusted = read_json(run_dir / "contract.json") == run_contract()
    predictions = run_dir / "predictions"
    navigation = run_dir / "planning"
    for model in MODELS_REQUESTED:
        for dataset in DATASETS:
            horizons = direct_ids(dataset)
            c = contract(model, dataset, "time")
            old = prediction_sources(model, dataset, "time", c)
            missing_by_horizon = {}
            for horizon, ids in horizons.items():
                missing = []
                for sample in ids:
                    own = predictions / model / dataset / "time" / f"id_{sample}" / f"{horizon}.png"
                    source = own if own_trusted and good_png(own) else None
                    provenance = "current_run" if source else None
                    if source is not None and own.is_symlink():
                        for root, audit, allowed in old:
                            if horizon in allowed and own.resolve() == (root / f"id_{sample}" / f"{horizon}.png").resolve():
                                provenance = audit
                                break
                    if source is None:
                        for root, audit, allowed in old:
                            if horizon not in allowed:
                                continue
                            path = root / f"id_{sample}" / f"{horizon}.png"
                            if good_png(path):
                                source, provenance = path, audit
                                break
                    key = item_key("direct", model, dataset, "time", sample, horizon)
                    items[key] = {"status": "complete" if source else "pending", "source": str(source) if source else None,
                                  "provenance": provenance, "endpoint": horizon,
                                  "endpoint_unit": "observed_frames" if dataset == "planetary_rover" else "seconds"}
                    if source is None:
                        missing.append(sample)
                if missing:
                    missing_by_horizon[horizon] = missing
            # Group endpoints whose full loader length is safe for all samples.
            groups = {}
            for horizon, ids in missing_by_horizon.items():
                groups.setdefault(tuple(ids), []).append(horizon)
            for ids, endpoints in groups.items():
                jobs.append({"kind": "direct", "model": model, "dataset": dataset,
                             "evaluation": "time", "ids": list(ids), "endpoints": endpoints,
                             "priority": 20 if model == "nwm-ego4d" else 30})
            if dataset not in ROLLOUT:
                continue
            c = contract(model, dataset, "rollout")
            for fps in (1, 4):
                evaluation = f"rollout_{fps}fps"
                old = prediction_sources(model, dataset, evaluation, c)
                missing = []
                for sample in range(count(dataset, "rollout")):
                    own = predictions / model / dataset / evaluation / f"id_{sample}"
                    frames = range(16 * fps)
                    source = own if own_trusted and all(good_png(own / f"{f}.png") for f in frames) else None
                    provenance = "current_run" if source else None
                    if source is not None:
                        for root, audit, _allowed in old:
                            original = root / f"id_{sample}"
                            if all((own / f"{frame}.png").is_symlink()
                                   and (own / f"{frame}.png").resolve() == (original / f"{frame}.png").resolve()
                                   for frame in frames):
                                provenance = audit
                                break
                    if source is None:
                        for root, audit, _allowed in old:
                            path = root / f"id_{sample}"
                            if all(good_png(path / f"{f}.png") for f in frames):
                                source, provenance = path, audit
                                break
                    key = item_key("rollout", model, dataset, evaluation, sample)
                    items[key] = {"status": "complete" if source else "pending", "source": str(source) if source else None,
                                  "provenance": provenance, "frames": 16 * fps}
                    if source is None:
                        missing.append(sample)
                if missing:
                    jobs.append({"kind": "rollout", "model": model, "dataset": dataset,
                                 "evaluation": evaluation, "ids": missing, "priority": 40})
        for dataset in NAV:
            c = contract(model, dataset, "navigation")
            old = nav_source(model, dataset, c)
            missing = []
            for sample in range(count(dataset, "navigation")):
                own = navigation / model / dataset / STEM / "sample_metrics" / f"{sample:06d}.json"
                source = own if own_trusted and good_nav(own) else None
                provenance = "current_run" if source else None
                if source is not None and old and own.is_symlink() and own.resolve() == (old[0] / f"{sample:06d}.json").resolve():
                    provenance = old[1]
                if source is None and old:
                    path = old[0] / f"{sample:06d}.json"
                    if good_nav(path):
                        source, provenance = path, old[1]
                key = item_key("navigation", model, dataset, "cem80", sample)
                items[key] = {"status": "complete" if source else "pending", "source": str(source) if source else None,
                              "provenance": provenance}
                if source is None:
                    missing.append(sample)
            for start in range(0, len(missing), 25):
                ids = missing[start:start + 25]
                jobs.append({"kind": "navigation", "model": model, "dataset": dataset,
                             "evaluation": "cem80", "ids": ids,
                             "priority": 0 if model == "nwm-ego4d" else 10})
    jobs.sort(key=lambda j: (j["kind"] == "navigation", j["priority"], -len(j["ids"]), j["model"], j["dataset"]))
    return items, jobs


def job_name(job):
    identity = hashlib.sha256(json.dumps({"ids": job["ids"], "endpoints": job.get("endpoints")},
                                        sort_keys=True).encode()).hexdigest()[:8]
    return f"{job['kind']}_{job['model']}_{job['dataset']}_{job['evaluation']}_{job['ids'][0]}-{job['ids'][-1]}_{identity}"


def env_for(model, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), NWM_DATA_ROOT=str(DATA), PYTHONUNBUFFERED="1",
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    if model == "rae-nwm":
        assets = Path(MODEL[model]["assets_root"])
        env.update(HF_HOME=str(assets / "hf_cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def rae_assets(model):
    spec = MODEL[model]
    assets = Path(spec["assets_root"])
    return ["--source", spec["source_dir"], "--checkpoint", spec["checkpoint"],
            "--decoder", str(assets / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"),
            "--normalization-stats", str(assets / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"),
            "--dino-model", str(assets / "models/dinov2-with-registers-base"),
            "--project-root", str(ROOT), "--data-root", str(DATA)]


def reusable_frame_metrics(model, dataset, evaluation, run_dir):
    """Reuse aggregate rows only when every prediction is still the audited image."""
    if evaluation != "time":
        return {}
    result = {}
    c = contract(model, dataset, "time")
    for source_root, audit_path, allowed in prediction_sources(model, dataset, "time", c):
        audit = read_json(Path(audit_path)) or {}
        for label, horizon in (audit.get("frame_indices") or {}).items():
            if horizon not in allowed or horizon not in direct_ids(dataset):
                continue
            expected_label = f"{horizon * 4}obs" if dataset == "planetary_rover" else f"{horizon}s"
            row = (audit.get("metrics") or {}).get(label)
            ids = direct_ids(dataset)[horizon]
            if (label != expected_label or not isinstance(row, dict)
                or row.get("sample_count") != len(ids)
                or not all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                           for key in ("lpips_alex", "dreamsim", "psnr"))):
                continue
            target_root = run_dir / "predictions" / model / dataset / "time"
            if all((target_root / f"id_{sample}" / f"{horizon}.png").is_symlink()
                   and (target_root / f"id_{sample}" / f"{horizon}.png").resolve()
                   == (source_root / f"id_{sample}" / f"{horizon}.png").resolve()
                   and good_png(target_root / f"id_{sample}" / f"{horizon}.png")
                   for sample in ids):
                result[expected_label] = {**row, "reused_from": audit_path}
    return result


def command_for(job, run_dir, batch):
    model, dataset, kind = job["model"], job["dataset"], job["kind"]
    spec = MODEL[model]
    ids = job["ids"]
    if kind == "metric":
        evaluation = job["evaluation"]
        prediction = run_dir / "predictions" / model / dataset / evaluation
        gt = run_dir / "gt" / dataset / evaluation
        output = run_dir / "metrics" / model / f"{dataset}_{evaluation}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        filter_path = run_dir / "metric_filters" / model / f"{dataset}_{evaluation}.json"
        if evaluation == "time":
            frames = {f"{h * 4}obs" if dataset == "planetary_rover" else f"{h}s": h for h in direct_ids(dataset)}
            atomic_json(filter_path, {label: direct_ids(dataset)[h] for label, h in frames.items()})
            filter_args = ["--sample-ids-by-frame-file", str(filter_path)]
        else:
            frames = dict(part.split(":") for part in FRAME_SPECS[evaluation].split(","))
            atomic_json(filter_path, list(range(count(dataset, "rollout"))))
            filter_args = ["--sample-ids-file", str(filter_path)]
        kind_contract = "time" if evaluation == "time" else "rollout"
        c = contract(model, dataset, kind_contract)
        reused = reusable_frame_metrics(model, dataset, evaluation, run_dir)
        reuse_args = []
        if reused:
            reuse_path = run_dir / "metric_filters" / model / f"{dataset}_{evaluation}_reused.json"
            atomic_json(reuse_path, reused)
            reuse_args = ["--reuse-frame-metrics-file", str(reuse_path)]
        return [str(NWM_PYTHON), "scripts/evaluate_nwm_predictions.py", "--gt-dir", str(gt),
                "--pred-dir", str(prediction), "--output", str(output),
                "--frames", ",".join(f"{label}:{index}" for label, index in frames.items()),
                "--dataset", dataset, "--eval-type", kind_contract, "--eval-name", evaluation,
                "--batch-size", str(batch), "--device", "cuda", "--dreamsim-cache",
                "/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models",
                "--inference-backend", "rae-nwm" if model == "rae-nwm" else "nwm",
                "--sampler", c["sampler"], "--sampling-steps", str(c["sampling_steps"]),
                "--seed", str(c["seed"]), "--checkpoint-sha256", c["checkpoint_sha256"],
                "--split-sha256", c["split_sha256"],
                *(["--rollout-fps", evaluation.split("_")[1].removesuffix("fps")] if evaluation != "time" else []),
                *filter_args, *reuse_args]
    if kind == "navigation":
        if model == "rae-nwm":
            return [str(RAE_PYTHON.parent / "torchrun"), "--standalone", "--nproc-per-node=1",
                    "scripts/raenwm_planning_eval.py", *rae_assets(model), "--output-root", str(run_dir / "planning" / model),
                    "--datasets", dataset, "--sample-indices", *map(str, ids), "--no-write-aggregate",
                    "--num-samples", "80", "--topk", "5", "--opt-steps", "1", "--num-repeat-eval", "3",
                    "--microbatch-size", "80", "--sampling-method", "euler", "--num-steps", "50", "--seed", "42"]
        return [str(NWM_PYTHON.parent / "torchrun"), "--standalone", "--nproc-per-node=1", "planning_eval.py",
                f"exp_dir={spec['exp_dir']}", f"ckp={spec['checkpoint_id']}", f"datasets_to_eval=[{dataset}]",
                f"hydra.run.dir={run_dir / 'hydra' / job_name(job)}",
                f"output_dir={run_dir / 'planning' / model}", f"planning_sample_indices=[{','.join(map(str, ids))}]",
                "planning_write_aggregate=false", "batch_size=1", "num_workers=4", "num_samples=80", "topk=5",
                "rollout_stride=1", "opt_steps=1", "num_repeat_eval=3", "seed=42", "+planning_sample_seed=42",
                "planning_microbatch_size=80", "cost_fn=lpips", "compute_cost_with_recon=true", "save_preds=false",
                "plot=false", "resume_planning_samples=true",
                # Match the existing RAE/OOD CEM distribution. These fields
                # are absent from the Go Stanford Hydra dataset entry.
                *(["+plan_datasets_hyperparams.go_stanford.mu=[-0.1,0.0,0.0]",
                   "+plan_datasets_hyperparams.go_stanford.var_scale=[0.1,0.15,0.1]"]
                  if dataset == "go_stanford" else [])]
    if model == "rae-nwm":
        cmd = [str(RAE_PYTHON.parent / "torchrun"), "--standalone", "--nproc-per-node=1", "scripts/raenwm_infer.py",
               *rae_assets(model), "--output-root", str(run_dir / "predictions" / model), "--datasets", dataset,
               "--eval-type", "time" if kind == "direct" else "rollout", "--sample-indices", *map(str, ids)]
        # argparse nargs consumes IDs until the next option.
        cmd += ["--expected-sample-count", str(count(dataset, "time" if kind == "direct" else "rollout")),
                "--batch-size", str(batch), "--num-workers", "4", "--sampling-method", "euler", "--num-steps", "50", "--seed", "0"]
        if kind == "direct":
            cmd += ["--future-frames", str(max(job["endpoints"]) * 4), "--horizons", *map(str, sorted(job["endpoints"]))]
        else:
            cmd += ["--future-frames", "64", "--rollout-fps", job["evaluation"].split("_")[1].removesuffix("fps")]
        return cmd
    cmd = [str(NWM_PYTHON.parent / "torchrun"), "--standalone", "--nproc-per-node=1", "isolated_nwm_infer.py",
           f"exp_dir={spec['exp_dir']}", f"ckp={spec['checkpoint_id']}", f"output_dir={run_dir}",
           f"hydra.run.dir={run_dir / 'hydra' / job_name(job)}",
           f"prediction_dir={run_dir / 'predictions' / model}", f"datasets_to_eval=[{dataset}]",
           f"eval_type={'time' if kind == 'direct' else 'rollout'}",
           f"eval_expected_full_count={count(dataset, 'time' if kind == 'direct' else 'rollout')}",
           f"eval_sample_indices=[{','.join(map(str, ids))}]", "eval_diffusion_steps=250",
           f"batch_size={batch}", "num_workers=4", "pin_memory=false", "seed=0"]
    if kind == "direct":
        cmd += [f"eval_len_traj_pred={max(job['endpoints']) * 4}",
                f"time_horizons_seconds=[{','.join(map(str, sorted(job['endpoints'])))}]"]
    else:
        fps = job["evaluation"].split("_")[1].removesuffix("fps")
        cmd += [f"rollout_fps_values=[{fps}]", "use_efficient_rollout=true"]
    return cmd


def preflight(gpus):
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise RuntimeError("Exactly eight distinct GPU IDs are required")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, check=True).stdout
    rows = {}
    for line in query.splitlines():
        index, name, total, used = [part.strip() for part in line.split(",")]
        rows[index] = (name, int(total), int(used))
    for gpu in gpus:
        name, total, used = rows[gpu]
        if "A800" not in name or total < 79000 or used > 1024:
            raise RuntimeError(f"GPU {gpu} is not an idle 80GB A800: {rows[gpu]}")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                               text=True, capture_output=True, check=True).stdout.strip()
    if processes and "No running processes found" not in processes:
        raise RuntimeError(f"GPU compute processes are active: {processes[:500]}")
    for python in (NWM_PYTHON, RAE_PYTHON):
        if not python.is_file():
            raise FileNotFoundError(python)
        imports = "import torch, PIL, hydra" if python == NWM_PYTHON else "import torch, PIL"
        subprocess.run([str(python), "-c", imports],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for flag, path in zip(("decoder", "stats", "dino"),
            (Path(MODEL["rae-nwm"]["assets_root"]) / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
             Path(MODEL["rae-nwm"]["assets_root"]) / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt",
             Path(MODEL["rae-nwm"]["assets_root"]) / "models/dinov2-with-registers-base/model.safetensors")):
        if not path.is_file():
            raise FileNotFoundError(f"RAE {flag}: {path}")
    for model in MODELS_REQUESTED:
        path = Path(MODEL[model]["checkpoint"])
        if sha(path) != MODEL[model]["sha256"]:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {model}: {path}")
    for model in ADDED:
        prepare_model_view(model)
    for dataset in DATASETS:
        if not (DATA / DATASET_CONTRACTS[dataset]["data_name"]).is_dir():
            raise FileNotFoundError(DATA / DATASET_CONTRACTS[dataset]["data_name"])
        for kind in ("time", "rollout", "navigation"):
            if kind == "rollout" and dataset not in ROLLOUT or kind == "navigation" and dataset not in NAV:
                continue
            path, expected = expected_split(dataset, kind)
            if sha(path) != expected:
                raise RuntimeError(f"Split SHA-256 mismatch: {path}")
    for kind in ("time", "rollout"):
        fresh = actual_split_population("huron", kind)
        if fresh != run_contract()["effective_split_population"][kind]:
            raise RuntimeError(f"Huron/{kind} effective sample mapping changed since contract verification")
    free = shutil.disk_usage(RUN_ROOT.parent).free
    if free < 500 * 1024**3:
        raise RuntimeError(f"NAS free space too low: {free / 1024**3:.0f} GiB (require 500 GiB)")
    # A user assertion alone cannot prove the old L20 writer has stopped.
    watch_roots = [NAS / "navigation_largebatch_20260921/planning",
                   NAS / "protocol_runs/ood_direct_4s_v1/predictions",
                   NAS / "finalLAM_reset_checkpoint_sweep_20260922/predictions"]
    for watch in watch_roots:
        if watch.exists():
            recent = subprocess.run(["find", str(watch), "-type", "f", "-mmin", "-2", "-print", "-quit"],
                                    text=True, capture_output=True, check=True).stdout.strip()
            if recent:
                raise RuntimeError(f"Target results are still being updated: {recent}")
    return {"gpus": rows, "nas_free_gib": round(free / 1024**3, 1)}


def ensure_links(run_dir, items):
    """Expose valid old samples through symlinks while preserving original paths."""
    for key, item in items.items():
        if item["status"] != "complete" or item["provenance"] == "current_run":
            continue
        kind, model, dataset, evaluation, sample, *rest = key.split("/")
        source = Path(item["source"])
        if kind == "navigation":
            target = run_dir / "planning" / model / dataset / STEM / "sample_metrics" / source.name
        elif kind == "direct":
            target = run_dir / "predictions" / model / dataset / evaluation / f"id_{sample}" / source.name
        else:
            target = run_dir / "predictions" / model / dataset / evaluation / f"id_{sample}"
            if target.is_symlink():
                target.unlink()
            target.mkdir(parents=True, exist_ok=True)
            for frame in range(item["frames"]):
                linked = target / f"{frame}.png"
                if not linked.is_symlink() or linked.resolve() != (source / f"{frame}.png").resolve():
                    linked.unlink(missing_ok=True)
                    linked.symlink_to(source / f"{frame}.png")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink() or target.resolve() != source.resolve():
            target.unlink(missing_ok=True)
            target.symlink_to(source)


def set_phase(run_dir, phase, detail):
    atomic_json(run_dir / "phase.json", {"updated_at": now(), "pid": os.getpid(),
                                        "phase": phase, "detail": detail})
    print(f"[{now()}] {phase}: {detail}", flush=True)


def gt_jobs(run_dir):
    """Validate independent datasets concurrently; bound NAS read concurrency."""
    by_dataset = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(gt_jobs_for_dataset, run_dir, dataset): dataset for dataset in DATASETS}
        while pending:
            completed, _ = concurrent.futures.wait(pending, timeout=10,
                                                   return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                dataset = pending.pop(future)
                by_dataset[dataset] = future.result()
            set_phase(run_dir, "checking_gt", f"{len(by_dataset)}/{len(DATASETS)} datasets checked; "
                      f"remaining={','.join(pending.values()) or 'none'}")
    return [job for dataset in DATASETS for job in by_dataset[dataset]]


def gt_jobs_for_dataset(run_dir, dataset):
    jobs = []
    direct_groups = {}
    for horizon, ids in direct_ids(dataset).items():
        directory = run_dir / "gt" / dataset / "time"
        missing = [i for i in ids if not good_png(directory / f"id_{i}" / f"{horizon}.png")]
        if missing:
            direct_groups.setdefault(tuple(missing), []).append(horizon)
    for ids, endpoints in direct_groups.items():
        jobs.append({"kind": "gt_direct", "dataset": dataset, "ids": list(ids), "endpoints": endpoints})
    if dataset in ROLLOUT:
        for fps in (1, 4):
            directory = run_dir / "gt" / dataset / f"rollout_{fps}fps"
            missing = [i for i in range(count(dataset, "rollout")) if not all(
                good_png(directory / f"id_{i}" / f"{f}.png") for f in range(16 * fps))]
            if missing:
                jobs.append({"kind": "gt_rollout", "dataset": dataset, "ids": missing, "fps": fps})
    return jobs


def gt_command(job, run_dir):
    dataset = job["dataset"]
    is_direct = job["kind"] == "gt_direct"
    reference = MODEL["nwm-release"]
    cmd = [str(NWM_PYTHON.parent / "torchrun"), "--standalone", "--nproc-per-node=1", "isolated_nwm_infer.py",
           f"exp_dir={reference['exp_dir']}", f"output_dir={run_dir}", "gt=1", f"datasets_to_eval=[{dataset}]",
           f"hydra.run.dir={run_dir / 'hydra' / gt_job_name(job)}",
           f"eval_type={'time' if is_direct else 'rollout'}", "seed=0", "batch_size=64", "num_workers=8",
           "pin_memory=false", f"eval_expected_full_count={count(dataset, 'time' if is_direct else 'rollout')}",
           f"eval_sample_indices=[{','.join(map(str, job['ids']))}]"]
    if is_direct:
        endpoints = sorted(job["endpoints"])
        cmd += [f"eval_len_traj_pred={max(endpoints) * 4}",
                f"time_horizons_seconds=[{','.join(map(str, endpoints))}]"]
    else:
        cmd += [f"rollout_fps_values=[{job['fps']}]", "use_efficient_rollout=true"]
    return cmd


def gt_job_name(job):
    identity = hashlib.sha256(json.dumps({"ids": job["ids"], "endpoints": job.get("endpoints"),
                                          "fps": job.get("fps")}, sort_keys=True).encode()).hexdigest()[:8]
    return f"gt_{job['kind']}_{job['dataset']}_{identity}"


def gt_dependency(job):
    if job["kind"] == "gt_direct":
        return job["dataset"], "time"
    if job["kind"] == "gt_rollout":
        return job["dataset"], f"rollout_{job['fps']}fps"
    return job["dataset"], job["evaluation"]


def gt_progress(job, run_dir):
    """Count visible GT frames for live status; resume does full PNG validation."""
    dataset, evaluation = gt_dependency(job)
    folder = run_dir / "gt" / dataset / evaluation
    if job["kind"] == "gt_direct":
        frames = ((sample, horizon) for sample in job["ids"] for horizon in job["endpoints"])
    else:
        frames = ((sample, frame) for sample in job["ids"] for frame in range(16 * job["fps"]))
    total = len(job["ids"]) * (len(job["endpoints"]) if job["kind"] == "gt_direct" else 16 * job["fps"])
    complete = sum((folder / f"id_{sample}" / f"{frame}.png").is_file() for sample, frame in frames)
    return complete, total


def run_job(job, run_dir, gpu, state, *, gt=False):
    name = job_name(job) if not gt else gt_job_name(job)
    command = gt_command(job, run_dir) if gt else command_for(job, run_dir, state.get("batch", 16 if job["model"] == "rae-nwm" else 64))
    log = run_dir / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = env_for("nwm-release" if gt else job["model"], gpu)
    entry = {"name": name, "job": job, "gpu": gpu, "command": command, "log": str(log),
             "started_at": now(), "status": "running", "batch": state.get("batch")}
    atomic_json(run_dir / "jobs" / f"{name}.json", entry)
    log_offset = log.stat().st_size if log.exists() else 0
    with log.open("a") as stream:
        stream.write(f"[{now()}] command: {shlex.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        entry["pid"] = process.pid
        atomic_json(run_dir / "jobs" / f"{name}.json", entry)
        code = process.wait()
    entry.update(status="complete" if code == 0 else "failed", exit_code=code, finished_at=now(),
                 elapsed_seconds=time.time() - datetime.fromisoformat(entry["started_at"]).timestamp())
    with log.open(errors="replace") as stream:
        stream.seek(log_offset)
        attempt_log = stream.read()
        peaks = [float(match.group(1)) for line in attempt_log.splitlines()
                 if (match := re.search(r"peak_memory_gib=([0-9.]+)", line))]
    entry["peak_memory_gib"] = max(peaks) if peaks else None
    entry["oom"] = "out of memory" in attempt_log.lower()
    entry["samples_per_second"] = (len(job["ids"]) / entry["elapsed_seconds"]
                                   if entry["elapsed_seconds"] > 0 and not gt else None)
    atomic_json(run_dir / "jobs" / f"{name}.json", entry)
    return entry


def aggregate_navigation(run_dir, items):
    for model in MODELS_REQUESTED:
        for dataset in NAV:
            rows = []
            sources = []
            for sample in range(count(dataset, "navigation")):
                item = items[item_key("navigation", model, dataset, "cem80", sample)]
                if item["status"] != "complete":
                    break
                rows.append(read_json(Path(item["source"])))
                sources.append(item["source"])
            if len(rows) != count(dataset, "navigation"):
                continue
            result = {f"{dataset}_{key}": sum(row[key] for row in rows) / len(rows) for key in METRIC_KEYS}
            result.update(sample_count=len(rows), sample_sources=sources, protocol="navigation_cem80_v1",
                          inference=contract(model, dataset, "navigation"))
            atomic_json(run_dir / "metrics" / model / f"{dataset}_navigation.json", result)


def refresh_active_items(run_dir, items, running):
    """Read only active jobs' outputs so status advances within long jobs."""
    for job, _gpu, _state in running.values():
        model, dataset, evaluation = job["model"], job["dataset"], job["evaluation"]
        for sample in job["ids"]:
            if job["kind"] == "direct":
                for horizon in job["endpoints"]:
                    key = item_key("direct", model, dataset, evaluation, sample, horizon)
                    path = run_dir / "predictions" / model / dataset / "time" / f"id_{sample}" / f"{horizon}.png"
                    if items[key]["status"] != "complete" and good_png(path):
                        items[key].update(status="complete", source=str(path), provenance="current_run")
            elif job["kind"] == "rollout":
                key = item_key("rollout", model, dataset, evaluation, sample)
                folder = run_dir / "predictions" / model / dataset / evaluation / f"id_{sample}"
                if items[key]["status"] != "complete" and all(
                    good_png(folder / f"{frame}.png") for frame in range(items[key]["frames"])):
                    items[key].update(status="complete", source=str(folder), provenance="current_run")
            elif job["kind"] == "navigation":
                key = item_key("navigation", model, dataset, evaluation, sample)
                path = run_dir / "planning" / model / dataset / STEM / "sample_metrics" / f"{sample:06d}.json"
                if items[key]["status"] != "complete" and good_nav(path):
                    items[key].update(status="complete", source=str(path), provenance="current_run")


def metric_jobs(run_dir, items):
    jobs = []
    for model in MODELS_REQUESTED:
        for dataset in DATASETS:
            for evaluation in ("time", "rollout_1fps", "rollout_4fps"):
                if dataset == "planetary_rover" and evaluation != "time":
                    continue
                kind = "direct" if evaluation == "time" else "rollout"
                relevant = [v for k, v in items.items() if k.startswith(f"{kind}/{model}/{dataset}/{evaluation}/")]
                if not relevant or any(v["status"] != "complete" for v in relevant):
                    continue
                path = run_dir / "metrics" / model / f"{dataset}_{evaluation}.json"
                old = read_json(path)
                if not isinstance(old, dict):
                    old = {}
                c = contract(model, dataset, "time" if evaluation == "time" else "rollout")
                inference = old.get("inference")
                if not isinstance(inference, dict):
                    inference = {}
                expected_frames = ({f"{h * 4}obs" if dataset == "planetary_rover" else f"{h}s": h
                                    for h in direct_ids(dataset)} if evaluation == "time" else
                                   {label: int(frame) for label, frame in
                                    (part.split(":") for part in FRAME_SPECS[evaluation].split(","))})
                expected_counts = ({f"{h * 4}obs" if dataset == "planetary_rover" else f"{h}s": len(ids)
                                    for h, ids in direct_ids(dataset).items()} if evaluation == "time" else
                                   {label: count(dataset, "rollout") for label in expected_frames})
                current_sources = [Path(item["source"]) for item in relevant]
                if (old.get("frame_indices") == expected_frames
                    and set((old.get("metrics") or {})) == set(expected_frames)
                    and all(isinstance(old["metrics"][label], dict)
                            and old["metrics"][label].get("sample_count") == expected_counts[label]
                            for label in expected_frames)
                    and all(inference.get(key) == c[key] for key in
                            ("checkpoint_sha256", "split_sha256", "seed", "sampler", "sampling_steps"))
                    and all(source.stat().st_mtime_ns <= path.stat().st_mtime_ns for source in current_sources)):
                    continue
                jobs.append({"kind": "metric", "model": model, "dataset": dataset,
                             "evaluation": evaluation, "ids": [0], "priority": 50})
    return jobs


def write_report(run_dir, items):
    groups = {}
    for key, item in items.items():
        kind, model, dataset, evaluation, _sample, *_ = key.split("/")
        group = groups.setdefault(f"{kind}/{model}/{dataset}/{evaluation}",
                                  {"kind": kind, "model": model, "dataset": dataset,
                                   "evaluation": evaluation, "expected_items": 0,
                                   "complete_items": 0, "reused_old_items": 0,
                                   "incomplete_items": []})
        group["expected_items"] += 1
        if item["status"] == "complete":
            group["complete_items"] += 1
            group["reused_old_items"] += item.get("provenance") not in (None, "current_run")
        else:
            group["incomplete_items"].append(key)
    for group in groups.values():
        kind, model, dataset, evaluation = (group[k] for k in ("kind", "model", "dataset", "evaluation"))
        metric_path = (run_dir / "metrics" / model /
                       (f"{dataset}_navigation.json" if kind == "navigation" else f"{dataset}_{evaluation}.json"))
        group["metric_path"] = str(metric_path)
        metrics = read_json(metric_path)
        group["metrics"] = metrics if isinstance(metrics, dict) else None
        if kind == "direct":
            group["sample_counts_by_endpoint"] = {
                (f"{h * 4}_observed_frames" if dataset == "planetary_rover" else f"{h}s"): len(ids)
                for h, ids in direct_ids(dataset).items()}
        elif kind == "rollout":
            group["sample_count"] = count(dataset, "rollout")
        else:
            group["sample_count"] = count(dataset, "navigation")
        if dataset == "huron":
            population_kind = "time" if kind == "direct" else "rollout"
            population = huron_population(population_kind)
            group["population_note"] = (
                f"Current loader-valid Huron split: {population['valid']} of {population['raw']} raw rows; "
                "sample IDs refer to positions after filtering. Older 500/150-row Huron results are excluded."
            )
            group["effective_mapping_sha256"] = population["effective_mapping_sha256"]
    report = {"run_id": run_dir.name, "updated_at": now(),
              "schedule": "reconstruction_and_metrics_before_navigation",
              "protocols": ["direct_4s_v1", "rollout_v1", "navigation_cem80_v1"],
              "huron": {"note": "Current data only: 329 valid direct samples out of 500 raw rows and 103 valid rollout samples out of 150 raw rows. Older 500/150-row results are not reused.",
                        "time": huron_population("time"), "rollout": huron_population("rollout")},
              "planetary_rover": {"direct_endpoint_unit": "observed_frames", "direct_endpoints": [4, 8, 16],
                                   "longer_direct_endpoints": "no_valid_samples", "rollout": "no_valid_samples"},
              "groups": groups, "job_state_dir": str(run_dir / "jobs"), "per_sample_state": str(run_dir / "status.json")}
    atomic_json(run_dir / "report.json", report)
    return report


def publish_state(run_dir, items, jobs, active=None, started=None, baseline=0):
    done = sum(item["status"] == "complete" for item in items.values())
    elapsed = time.time() - started if started else None
    rate = (done - baseline) / elapsed if elapsed and elapsed > 0 and done > baseline else None
    live = []
    for row in active or []:
        row = dict(row)
        record = read_json(run_dir / "jobs" / f"{row['name']}.json") or {}
        job = record.get("job") or {}
        if job.get("kind") in ("direct", "rollout", "navigation"):
            keys = (item_key(job["kind"], job["model"], job["dataset"], job["evaluation"], sample, endpoint)
                    for sample in job["ids"]
                    for endpoint in (job.get("endpoints", [None])))
            checked = [items[key]["status"] == "complete" for key in keys]
            row["complete_items"] = sum(checked)
            row["total_items"] = len(checked)
        elif job.get("kind") in ("gt_direct", "gt_rollout"):
            row["complete_items"], row["total_items"] = gt_progress(job, run_dir)
        row["pid"] = record.get("pid")
        row["log"] = record.get("log")
        if row["log"] and Path(row["log"]).is_file():
            with Path(row["log"]).open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 4096))
                row["log_tail"] = stream.read().decode(errors="replace")[-1000:]
        live.append(row)
    state = {"updated_at": now(), "run_dir": str(run_dir), "total": len(items), "complete": done,
             "pending": len(items) - done, "jobs_pending": len(jobs), "active": active or [],
             "samples_per_second": rate, "eta_seconds": (len(items) - done) / rate if rate else None,
             "items": items}
    state["active"] = live
    atomic_json(run_dir / "status.json", state)
    return state


def coordinator(run_dir, gpus, lock_fd):
    # The launcher passes its already locked descriptor, keeping the lock held
    # continuously through the detached parent/child handoff.
    with os.fdopen(lock_fd, "a+") as lock:
        lock.seek(0)
        lock.truncate()
        (run_dir / "finish.json").unlink(missing_ok=True)
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        started = time.time()
        set_phase(run_dir, "reconciling", f"coordinator PID={os.getpid()}; GPUs={gpus}; checking saved predictions")
        items, jobs = build_plan(run_dir)
        baseline = sum(item["status"] == "complete" for item in items.values())
        ensure_links(run_dir, items)
        publish_state(run_dir, items, jobs, started=started, baseline=baseline)
        write_report(run_dir, items)
        set_phase(run_dir, "checking_gt", "validating saved GT with four CPU readers; no GPU inference yet")
        gt_queue = deque(gt_jobs(run_dir))
        set_phase(run_dir, "running", f"{len(gt_queue)} GT jobs; {len(jobs)} inference jobs; reconstruction and metrics before navigation")
        pending_gt = Counter(gt_dependency(job) for job in gt_queue)
        failed_gt = set()
        print(f"[{now()}] {len(gt_queue)} shared GT jobs and {len(jobs)} inference jobs; "
              f"up to {min(MAX_CONCURRENT_GT, len(gpus))} GT workers; GPUs={gpus}", flush=True)
        failures = []
        tuned_batches = {}
        # A small first slice measures memory and throughput before launching
        # the larger jobs for each model/backend. The sample IDs stay fixed.
        pilot_keys = set()
        pending_pilots = {}
        staged_jobs = []
        for job in jobs:
            key = (job["model"], job["kind"])
            pilot_size = 8 if job["model"] == "rae-nwm" else 32
            if job["kind"] in ("direct", "rollout") and key not in pilot_keys and len(job["ids"]) > 2 * pilot_size:
                pilot_keys.add(key)
                pilot = {**job, "ids": job["ids"][:pilot_size]}
                rest = {**job, "ids": job["ids"][pilot_size:]}
                pending_pilots[key] = job_name(pilot)
                staged_jobs.extend((pilot, rest))
            else:
                staged_jobs.append(job)
        jobs = staged_jobs
        submitted_metrics = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            free = list(gpus)
            running = {}
            while gt_queue or running or any(
                job["kind"] == "navigation" or
                (gt_dependency(job) not in failed_gt and pending_gt[gt_dependency(job)] == 0)
                for job in jobs
            ):
                ready_image_metrics = metric_jobs(run_dir, items)
                reconstruction_pending = (bool(gt_queue) or bool(ready_image_metrics)
                    or any(j["kind"] in ("direct", "rollout") for j in jobs)
                    or any(is_gt or j["kind"] in ("direct", "rollout", "metric")
                           for j, _gpu, _state, is_gt in running.values()))
                active_gt = sum(is_gt for _job, _gpu, _state, is_gt in running.values())
                while gt_queue and free and active_gt < min(MAX_CONCURRENT_GT, len(gpus)):
                    gt_job = gt_queue.popleft()
                    gpu = free.pop(0)
                    running[pool.submit(run_job, gt_job, run_dir, gpu, {}, gt=True)] = (gt_job, gpu, {}, True)
                    active_gt += 1
                while jobs and free:
                    eligible = next(
                        (index for index, candidate in enumerate(jobs)
                         if (candidate["kind"] != "navigation" or not reconstruction_pending)
                         and (candidate["kind"] == "navigation" or
                             (gt_dependency(candidate) not in failed_gt and
                              pending_gt[gt_dependency(candidate)] == 0))
                         and ((candidate["model"], candidate["kind"]) not in pending_pilots
                              or job_name(candidate) == pending_pilots[(candidate["model"], candidate["kind"])])),
                        None,
                    )
                    if eligible is None:
                        break
                    job = jobs.pop(eligible)
                    gpu = free.pop(0)
                    tuning_key = (job["model"], job["kind"])
                    state = {"batch": job.pop("retry_batch", tuned_batches.get(
                        tuning_key, 8 if job["model"] == "rae-nwm" else 32))}
                    future = pool.submit(run_job, job, run_dir, gpu, state)
                    running[future] = (job, gpu, state, False)
                if free:
                    ready_metrics = [candidate for candidate in ready_image_metrics
                                     if job_name(candidate) not in submitted_metrics
                                     and gt_dependency(candidate) not in failed_gt
                                     and pending_gt[gt_dependency(candidate)] == 0]
                    while ready_metrics and free:
                        job = ready_metrics.pop(0)
                        submitted_metrics.add(job_name(job))
                        gpu = free.pop(0)
                        state = {"batch": 16}
                        running[pool.submit(run_job, job, run_dir, gpu, state)] = (job, gpu, state, False)
                if not running:
                    break
                completed, _ = concurrent.futures.wait(running, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
                refresh_active_items(run_dir, items, {
                    future: (job, gpu, state)
                    for future, (job, gpu, state, is_gt) in running.items() if not is_gt
                })
                for future in completed:
                    job, gpu, state, is_gt = running.pop(future)
                    free.append(gpu)
                    try:
                        entry = future.result()
                    except Exception as exc:
                        failures.append(f"{gt_job_name(job) if is_gt else job_name(job)}: {exc}")
                        if is_gt:
                            failed_gt.add(gt_dependency(job))
                            pending_gt[gt_dependency(job)] -= 1
                        elif pending_pilots.get((job["model"], job["kind"])) == job_name(job):
                            pending_pilots.pop((job["model"], job["kind"]))
                        continue
                    if is_gt:
                        dependency = gt_dependency(job)
                        pending_gt[dependency] -= 1
                        if entry["exit_code"]:
                            failed_gt.add(dependency)
                            failures.append(f"{entry['name']}: exit {entry['exit_code']} ({entry['log']})")
                        print(f"[{now()}] {entry['name']} exit={entry['exit_code']} "
                              f"GT pending={sum(pending_gt.values())}", flush=True)
                        continue
                    if entry["exit_code"] and entry["oom"] and job["kind"] != "navigation" and state["batch"] > 1:
                        job["retry_batch"] = max(1, state["batch"] // 2)
                        tuned_batches[(job["model"], job["kind"])] = job["retry_batch"]
                        jobs.insert(0, job)
                    elif entry["exit_code"]:
                        failures.append(f"{job_name(job)}: exit {entry['exit_code']} ({entry['log']})")
                    elif (job["kind"] != "navigation" and entry.get("peak_memory_gib")
                          and len(job["ids"]) >= state["batch"]):
                        limit = 32 if job["model"] == "rae-nwm" else 64 if job["model"] == "nwm-ego4d" else 128
                        batch = state["batch"]
                        if entry["peak_memory_gib"] < 42 and batch < limit:
                            batch = min(limit, batch * 2)
                        elif entry["peak_memory_gib"] > 72 and batch > 1:
                            batch = max(1, batch // 2)
                        tuned_batches[(job["model"], job["kind"])] = batch
                    if pending_pilots.get((job["model"], job["kind"])) == job_name(job) and not (
                        entry["exit_code"] and job.get("retry_batch")):
                        pending_pilots.pop((job["model"], job["kind"]))
                    items, _ = build_plan(run_dir)
                    ensure_links(run_dir, items)
                    print(f"[{now()}] {entry['name']} exit={entry['exit_code']} complete={sum(i['status']=='complete' for i in items.values())}/{len(items)}", flush=True)
                active = [{"name": gt_job_name(job) if is_gt else job_name(job), "gpu": gpu}
                          for job, gpu, _state, is_gt in running.values()]
                publish_state(run_dir, items, list(gt_queue) + jobs, active=active,
                              started=started, baseline=baseline)
        items, remaining = build_plan(run_dir)
        aggregate_navigation(run_dir, items)
        metrics = metric_jobs(run_dir, items)
        print(f"[{now()}] image inference done; {len(metrics)} metric jobs", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            free = list(gpus)
            running = {}
            while metrics or running:
                while metrics and free:
                    job = metrics.pop(0)
                    gpu = free.pop(0)
                    running[pool.submit(run_job, job, run_dir, gpu, {"batch": 16})] = (job, gpu)
                completed, _ = concurrent.futures.wait(running, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    job, gpu = running.pop(future)
                    free.append(gpu)
                    try:
                        entry = future.result()
                        if entry["exit_code"]:
                            failures.append(f"{job_name(job)}: exit {entry['exit_code']} ({entry['log']})")
                    except Exception as exc:
                        failures.append(f"{job_name(job)}: {exc}")
                publish_state(run_dir, items, remaining + metrics,
                              active=[{"name": job_name(j), "gpu": gpu} for j, gpu in running.values()], started=started, baseline=baseline)
        missing_metrics = metric_jobs(run_dir, items)
        publish_state(run_dir, items, remaining + missing_metrics, started=started, baseline=baseline)
        write_report(run_dir, items)
        atomic_json(run_dir / "finish.json", {"finished_at": now(), "complete": sum(i["status"] == "complete" for i in items.values()),
                                                   "total": len(items), "failures": failures, "remaining_jobs": len(remaining),
                                                   "remaining_metrics": len(missing_metrics),
                                                   "exit_code": 1 if failures or remaining or missing_metrics else 0})
        if failures or remaining or missing_metrics:
            raise RuntimeError(f"Incomplete run: {len(failures)} failed jobs, {len(remaining)} inference jobs, {len(missing_metrics)} metrics")


def print_status(run_dir):
    phase = read_json(run_dir / "phase.json")
    if phase:
        print(f"phase={phase['phase']} updated={phase['updated_at']} {phase['detail']}")
    state = read_json(run_dir / "status.json")
    if not state:
        print(f"No status yet: {run_dir}")
        finish = read_json(run_dir / "finish.json")
        if finish:
            print(f"finished={finish.get('finished_at')} exit_code={finish.get('exit_code')} error={finish.get('error')}")
        return
    print(f"{state['updated_at']}  {state['complete']}/{state['total']} samples/endpoints complete  "
          f"pending={state['pending']} jobs={state['jobs_pending']} ETA={state.get('eta_seconds')}")
    finish = read_json(run_dir / "finish.json")
    lock_path = run_dir / "coordinator.lock"
    if lock_path.is_file():
        label = "last_coordinator_pid" if finish else "coordinator_pid"
        print(f"{label}={lock_path.read_text().strip()}")
    for item in state.get("active", []):
        # Shared GT runs before the worker pool and does not refresh status.json
        # while a GT command is active. Its job record may also be replaced by
        # a resumed attempt after status.json was last published.
        record = read_json(run_dir / "jobs" / f"{item['name']}.json") or {}
        if record:
            item = {**item, "pid": record.get("pid"), "log": record.get("log")}
        progress = (f" progress={item['complete_items']}/{item['total_items']}"
                    if "complete_items" in item else "")
        print(f"  GPU {item['gpu']}: {item['name']} PID={item.get('pid')}{progress} log={item.get('log')}")
        log_tail = item.get("log_tail")
        if item["name"].startswith("gt_") and item.get("log"):
            try:
                with Path(item["log"]).open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 4096))
                    log_tail = stream.read().decode(errors="replace")
            except OSError:
                pass
        if log_tail and log_tail.splitlines():
            print("    " + log_tail.splitlines()[-1][-300:])
    if finish:
        print(f"finished={finish['finished_at']} exit_code={finish['exit_code']} "
              f"failures={len(finish.get('failures', []))} remaining_jobs={finish.get('remaining_jobs')}")
        if finish.get("error"):
            print(f"error={finish['error']}")
        report = read_json(run_dir / "report.json")
        if report:
            metric_count = sum(group.get("metrics") is not None for group in report["groups"].values())
            print(f"report={run_dir / 'report.json'} metrics={metric_count}/{len(report['groups'])}")


def coordinator_log_offset(run_dir, path):
    """Default to this launch, including runs created before launch markers."""
    launch = read_json(run_dir / "launch.json") or {}
    preflight = read_json(run_dir / "preflight.json") or {}
    started = preflight.get("at")
    if launch.get("preflight_at") == started and isinstance(launch.get("log_offset"), int):
        return launch["log_offset"]
    if not started:
        return 0
    started_time = datetime.fromisoformat(started)
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                finish = read_json(run_dir / "finish.json") or {}
                if (finish.get("exit_code") and finish.get("finished_at")
                        and datetime.fromisoformat(finish["finished_at"]) >= started_time):
                    # Older coordinators can emit a bare traceback before their
                    # first timestamp. Preserve it rather than hiding a new error.
                    return 0
                return offset
            match = re.match(rb"\[([0-9]{4}-[^\]]+)\]", line)
            if match:
                try:
                    if datetime.fromisoformat(match[1].decode()) >= started_time:
                        return offset
                except ValueError:
                    pass


def tail_log(run_dir, job=None, lines=40, follow=False, history=False):
    path = run_dir / "logs" / (f"{job}.log" if job else "coordinator.log")
    offset = 0 if job or history else coordinator_log_offset(run_dir, path)
    with path.open("rb") as stream:
        stream.seek(offset)
        recent = deque(stream, maxlen=max(0, lines))
        print(b"".join(recent).decode(errors="replace"), end="", flush=True)
        if not recent and not job:
            print("本次启动尚无新的调度日志；以下为持久化状态（旧报错请用 --history 查看）：", flush=True)
            print_status(run_dir)
            sys.stdout.flush()
        if follow:
            while True:
                chunk = stream.read(65536)
                if chunk:
                    print(chunk.decode(errors="replace"), end="", flush=True)
                else:
                    time.sleep(1)


def process_args(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return []


def stop_run(run_dir):
    """Stop this local run, retaining artifacts and avoiding unrelated jobs."""
    lock_path = run_dir / "coordinator.lock"
    pid_text = lock_path.read_text().strip() if lock_path.exists() else ""
    if pid_text:
        pid = int(pid_text)
        args = process_args(pid)
        if args:
            expected = ["--run-id", run_dir.name]
            if ("_coordinate" not in args or str(Path(__file__).resolve()) not in args
                    or not any(args[i:i + 2] == expected for i in range(len(args)))):
                raise RuntimeError(f"PID {pid} is not this run's coordinator; refusing to stop")
            os.kill(pid, signal.SIGTERM)
            print(f"Sent TERM to coordinator PID={pid}", flush=True)
            for _ in range(30):
                if not process_args(pid):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"Coordinator PID={pid} has not exited; do not resume yet")
    groups = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        try:
            # Workers start separate sessions; the absolute output path binds
            # the session leader to this run, unlike a stale jobs.json PID.
            if (any(str(run_dir) + "/" in arg for arg in process_args(pid))
                    and os.getpgid(pid) == pid):
                os.killpg(pid, signal.SIGTERM)
                groups.append(pid)
                print(f"Sent TERM to worker process group={pid}", flush=True)
        except ProcessLookupError:
            pass
    for _ in range(30):
        remaining = []
        for pid in groups:
            try:
                os.killpg(pid, 0)
                remaining.append(pid)
            except ProcessLookupError:
                pass
        if not remaining:
            # A remote coordinator or one blocked in I/O must not be mistaken
            # for a stopped run merely because local PIDs are absent.
            if lock_path.exists():
                with lock_path.open("r+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("Run stopped; artifacts retained. Safe to resume.", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"Worker groups still present: {remaining}; do not resume yet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "start", "resume", "stop", "status", "tail", "_coordinate"):
        sub = commands.add_parser(action)
        sub.add_argument("--run-id", default="full_8xa800_seed0_v1")
        if action in ("start", "resume", "_coordinate"):
            sub.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
        if action == "_coordinate":
            sub.add_argument("--lock-fd", type=int, required=True)
        if action in ("start", "resume"):
            sub.add_argument("--detach", action="store_true")
            sub.add_argument("--l20-paused", action="store_true")
        if action == "status":
            sub.add_argument("--watch", nargs="?", type=int, const=10)
        if action == "tail":
            sub.add_argument("--job")
            sub.add_argument("--lines", type=int, default=40)
            sub.add_argument("--follow", action="store_true")
            sub.add_argument("--history", action="store_true", help="Include earlier coordinator attempts")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id may contain only letters, numbers, _ and -")
    run_dir = RUN_ROOT / args.run_id
    if args.action == "stop":
        stop_run(run_dir)
        return
    if args.action == "plan":
        print("Checkpoint manifest (new ModelScope inference views are prepared at start):")
        for model in MODELS_REQUESTED:
            spec = MODEL[model]
            print(f"  {model}: {spec['checkpoint']}  sha256={spec['sha256']}")
        items, jobs = build_plan(run_dir)
        groups = {}
        for key, item in items.items():
            kind, model, dataset = key.split("/")[:3]
            bucket = groups.setdefault((kind, model, dataset), [0, 0])
            bucket[0] += item["status"] == "complete"
            bucket[1] += 1
        for (kind, model, dataset), (done, total) in groups.items():
            print(f"{kind:10} {model:24} {dataset:17} {done:5}/{total}")
        print(f"total={len(items)} complete={sum(i['status']=='complete' for i in items.values())} jobs={len(jobs)}")
        print("planetary_rover: direct labels 4/8/16 observed frames; longer endpoints and rollout: no valid samples")
        print("huron: 329/500 valid direct rows and 103/150 valid rollout rows; prior 500/150-row results excluded")
        for dataset in ("tum_rgbd", "unitree_go2"):
            print(f"{dataset} direct sample counts: { {h: len(ids) for h, ids in direct_ids(dataset).items()} }")
        print("Data population audit (read-only; counts after trajectory filtering):")
        for dataset in DATASETS:
            for kind in ("time", "rollout"):
                if kind == "rollout" and dataset not in ROLLOUT:
                    continue
                audit = actual_split_population(dataset, kind)
                expected = count(dataset, kind)
                label = "OK" if audit["valid"] == expected else "MISMATCH"
                print(f"  {dataset}/{kind}: {audit['valid']}/{expected} {label} "
                      f"(missing trajectory rows={audit['missing_trajectory_rows']}, "
                      f"out-of-range rows={audit['out_of_range_rows']})")
                if label == "MISMATCH":
                    print("    This GT job cannot pass eval_expected_full_count until its data or run protocol is resolved.")
        return
    if args.action == "status":
        while True:
            print_status(run_dir)
            if args.watch is None or (run_dir / "finish.json").is_file():
                break
            time.sleep(args.watch)
        return
    if args.action == "tail":
        try:
            tail_log(run_dir, args.job, args.lines, args.follow, args.history)
        except KeyboardInterrupt:
            pass
        return
    gpus = args.gpus.split(",")
    if args.action == "_coordinate":
        try:
            coordinator(run_dir, gpus, args.lock_fd)
        except Exception as exc:
            finish = read_json(run_dir / "finish.json") or {}
            finish.update(finished_at=now(), exit_code=1, error=str(exc))
            set_phase(run_dir, "failed", str(exc))
            atomic_json(run_dir / "finish.json", finish)
            try:
                items, remaining = build_plan(run_dir)
                publish_state(run_dir, items, remaining)
                write_report(run_dir, items)
                finish.update(complete=sum(item["status"] == "complete" for item in items.values()),
                              total=len(items), remaining_jobs=len(remaining))
            except Exception as audit_error:
                finish["reconcile_error"] = str(audit_error)
            atomic_json(run_dir / "finish.json", finish)
            raise
        return
    if not args.l20_paused:
        raise RuntimeError("Start requires --l20-paused after the L20 benchmark writer has been paused")
    if args.action == "start" and (run_dir / "status.json").exists():
        raise RuntimeError("Run ID already has state; use resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    recorded_contract = read_json(run_dir / "contract.json")
    if args.action == "resume" and recorded_contract is None:
        raise RuntimeError("Run ID has no contract.json; cannot trust existing artifacts")
    if recorded_contract is None and any((run_dir / name).exists() for name in ("predictions", "planning", "metrics")):
        raise RuntimeError("Run ID contains artifacts without a verifiable contract; use a new run ID")
    with (run_dir / "coordinator.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A coordinator already holds this run lock") from exc
        if args.action == "resume" and recorded_contract is not None:
            recorded_contract = refresh_failed_gt_contract(run_dir, recorded_contract, run_contract())
            recorded_contract = migrate_huron_population_contract(run_dir, recorded_contract, run_contract())
        if recorded_contract is not None and recorded_contract != run_contract():
            raise RuntimeError("Run contract differs in checkpoint, split, sample IDs, or entrypoint hashes; use a new run ID")
        for record_path in (run_dir / "jobs").glob("*.json"):
            record = read_json(record_path) or {}
            if record.get("status") != "running" or not record.get("pid"):
                continue
            try:
                os.killpg(int(record["pid"]), 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f"Previous job process group is still active: PID={record['pid']} {record_path}")
        checks = preflight(gpus)
        if recorded_contract is None:
            atomic_json(run_dir / "contract.json", run_contract())
        # Avoid racing a second machine writing to one of the same prediction trees.
        items, jobs = build_plan(run_dir)
        atomic_json(run_dir / "preflight.json", {"at": now(), "checks": checks, "jobs": len(jobs)})
        (run_dir / "finish.json").unlink(missing_ok=True)
        command = [str(NWM_PYTHON), str(Path(__file__).resolve()), "_coordinate", "--run-id", args.run_id,
                   "--gpus", args.gpus, "--lock-fd", str(lock.fileno())]
        log_path = run_dir / "logs" / "coordinator.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as stream:
            offset = stream.tell()
            stream.write(f"[{now()}] launching {args.action}: {shlex.join(command)}\n")
            stream.flush()
        atomic_json(run_dir / "launch.json", {
            "preflight_at": read_json(run_dir / "preflight.json")["at"],
            "log_offset": offset, "command": command, "started_at": now()})
        if args.detach:
            with log_path.open("a") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=(lock.fileno(),))
            print(f"working_directory={ROOT}\ncommand={shlex.join(command)}\nPID={process.pid}\nlog={log_path}")
        else:
            print(f"working_directory={ROOT}\ncommand={shlex.join(command)}\nlog={log_path}")
            subprocess.run(command, cwd=ROOT, check=True, pass_fds=(lock.fileno(),))


if __name__ == "__main__":
    main()
