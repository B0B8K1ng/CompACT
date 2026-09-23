#!/usr/bin/env python3
"""Reproduce selected ID/OOD autoregressive visual comparisons.

Selection uses the largest OpenNWM minus best-baseline 4-second direct PSNR
margin among pinned rollout windows with an exact time-window match (ID and Go
Stanford), or the closest same-trajectory time window (other OOD datasets).
The direct score only nominates a sample; final claims use rollout frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

REPO = Path(__file__).resolve().parents[1]
NAS = Path("/file_system/nas/algorithm/dujun.nie/nwm")
ROOT = NAS / "results/nwm_benchmark/opennwm_rollout_showcase_20260923"
PY_NWM = Path("/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin")
PY_RAE = Path("/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/raenwm/bin")
RAE_ASSETS = NAS / "benchmark_models/rae-nwm"
MODELS = {
    "nwm-release": {
        "exp_dir": NAS / "benchmark_models/nwm-release",
        "checkpoint_id": "0100000",
        "checkpoint_sha256": "2a41c71eabd20946f61bb5d1d2490264246bff59b9ffca177eee672e855d5261",
    },
    "rae-nwm": {
        "checkpoint": RAE_ASSETS / "checkpoints/raenwm_b.pth.tar",
        "checkpoint_sha256": "97244579618eb0e376355157a5c1df7a83e534cb1ca35d8ff4e43c4fb4f6a9d0",
    },
    "opennwm-finalLAM-100k": {
        "exp_dir": NAS / "compact/runs/nav15_pa_step60000_latentpt_reset_ft/nwm-latentpt-pixel-action-finalLAM-ft-reset",
        "checkpoint_id": "joint_0100000",
        "checkpoint_sha256": "aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4",
    },
}
# Primary candidate and one reserve per dataset. The exact direct-window
# ranking is recorded in selection.json; reserve cases run only if requested.
CASES = {
    "scand": {"group": "ID", "ids": [73, 55], "seconds": 16, "split_dir": "scand"},
    "huron": {"group": "ID", "ids": [41, 76], "source_ids": [63, 110],
              "expected_count": 103, "seconds": 16, "split_dir": "sacson"},
    "tartan_drive": {"group": "ID", "ids": [127, 59], "seconds": 16, "split_dir": "tartan_drive"},
    "go_stanford": {"group": "OOD", "ids": [118, 64], "seconds": 4, "split_dir": "go_stanford"},
    "unitree_go2": {"group": "OOD", "ids": [21, 82], "seconds": 4, "split_dir": "unitree_go2"},
    "tum_rgbd": {"group": "OOD", "ids": [82, 53], "seconds": 4, "split_dir": "tum_rgbd"},
}


def direct_prediction_root(dataset: str, model: str) -> Path:
    benchmark = NAS / "results/nwm_benchmark"
    if model == "opennwm-finalLAM-100k":
        return benchmark / "finalLAM_reset_checkpoint_sweep_20260922/predictions/finalLAM-reset-joint0100000"
    if dataset in ("unitree_go2", "tum_rgbd"):
        return benchmark / "protocol_runs/ood_direct_4s_v1/predictions" / model
    return benchmark / "predictions" / model


def rank_direct_candidates(dataset: str) -> tuple[list[dict], dict]:
    """Recreate the frozen candidate ranking from saved direct-4s predictions."""
    split_dir = CASES[dataset]["split_dir"]
    time_split = REPO / "data_splits" / split_dir / "test/time.pkl"
    rollout_split = REPO / "data_splits" / split_dir / "test/rollout.pkl"
    time_entries = pickle.loads(time_split.read_bytes())
    rollout_entries = pickle.loads(rollout_split.read_bytes())
    exact = dataset in ("scand", "tartan_drive", "go_stanford")
    candidates = []
    for sample_id, entry in enumerate(rollout_entries):
        matches = [
            (abs(int(entry[1]) - int(other[1])), index)
            for index, other in enumerate(time_entries)
            if other[0] == entry[0] and (not exact or entry == other)
        ]
        if not matches:
            continue
        distance, time_id = min(matches)
        if distance > (0 if exact else 12):
            continue
        gt_path = NAS / "results/nwm_benchmark/gt" / dataset / "time" / f"id_{time_id}/4.png"
        pred_paths = {name: direct_prediction_root(dataset, name) / dataset / "time" /
                      f"id_{time_id}/4.png" for name in MODELS}
        if not gt_path.is_file() or any(not path.is_file() for path in pred_paths.values()):
            continue
        gt = np.asarray(Image.open(gt_path).convert("RGB"))
        scores = {name: psnr(gt, np.asarray(Image.open(path).convert("RGB")))
                  for name, path in pred_paths.items()}
        margin = scores["opennwm-finalLAM-100k"] - max(scores["nwm-release"], scores["rae-nwm"])
        candidates.append({"sample_id": sample_id, "time_sample_id": time_id,
                           "time_index_distance_frames": distance, "direct_4s_psnr": scores,
                           "direct_4s_margin_db": margin})
    candidates.sort(key=lambda item: (-item["direct_4s_margin_db"],
                                      item["time_index_distance_frames"], item["sample_id"]))
    return candidates, {"time_split": str(time_split), "time_split_sha256": sha256(time_split),
                        "rollout_split": str(rollout_split), "rollout_split_sha256": sha256(rollout_split)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(dataset: str, sample_id: int, model: str | None, root: Path) -> Path:
    prefix = root / ("gt" if model is None else f"predictions/{model}")
    return prefix / dataset / "rollout_4fps" / f"id_{sample_id}"


def command(dataset: str, sample_id: int, model: str | None, root: Path,
            single_image_pixel: bool = False) -> list[str]:
    case = CASES[dataset]
    if model == "rae-nwm":
        rae_sample_id = (
            case["source_ids"][case["ids"].index(sample_id)]
            if "source_ids" in case else sample_id
        )
        cmd = [
            str(PY_RAE / "torchrun"), "--standalone", "--nproc-per-node=1",
            "scripts/raenwm_infer.py", "--source", str(REPO / "third_party/raenwm"),
            "--checkpoint", str(MODELS[model]["checkpoint"]),
            "--decoder", str(RAE_ASSETS / "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"),
            "--normalization-stats", str(RAE_ASSETS / "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"),
            "--dino-model", str(RAE_ASSETS / "models/dinov2-with-registers-base"),
            "--project-root", str(REPO), "--data-root", str(NAS / "data"),
            "--output-root", str(root / f"predictions/{model}"), "--datasets", dataset,
            "--eval-type", "rollout", "--rollout-fps", "4", "--future-frames", str(4 * case["seconds"]),
            "--sample-indices", str(rae_sample_id), "--expected-sample-count", "150",
            "--batch-size", "1", "--num-workers", "0", "--sampling-method", "euler",
            "--num-steps", "50", "--seed", "0", "--no-compile",
        ]
        if single_image_pixel:
            cmd.extend(("--single-image-context", "--pixel-feedback"))
        return cmd
    exp = MODELS[model or "nwm-release"]
    cmd = [
        str(PY_NWM / "torchrun"), "--standalone", "--nproc-per-node=1",
        "isolated_nwm_infer.py", f"exp_dir={exp['exp_dir']}",
        f"ckp={exp['checkpoint_id']}", f"output_dir={root}",
        f"prediction_dir={root / f'predictions/{model}'}" if model else f"prediction_dir={root}",
        f"datasets_to_eval=[{dataset}]", "eval_type=rollout",
        f"eval_sample_indices=[{sample_id}]",
        f"eval_expected_full_count={case.get('expected_count', 150)}",
        f"eval_len_traj_pred={4 * case['seconds']}", "rollout_fps_values=[4]",
        "eval_diffusion_steps=250", "batch_size=1", "num_workers=0", "pin_memory=false",
        "seed=0", f"use_efficient_rollout={str(not single_image_pixel).lower()}",
        f"gt={int(model is None)}",
    ]
    if single_image_pixel and model is not None:
        cmd.extend(("single_image_context=true", "save_initial_image=true"))
    return cmd


def gpu_ready(index: int, minimum_free_mib: int) -> bool:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    values = {int(row.split(",")[0]): [int(value.strip()) for value in row.split(",")[1:]]
              for row in output.strip().splitlines()}
    free, utilization = values[index]
    return free >= minimum_free_mib and utilization <= 5


def run_case(dataset: str, sample_id: int, model: str | None, args: argparse.Namespace) -> None:
    output = paths(dataset, sample_id, model, args.output)
    frames = 4 * CASES[dataset]["seconds"]
    if all((output / f"{step}.png").is_file() for step in range(frames)):
        print(f"SKIP complete {dataset}/{sample_id}/{model or 'gt'}", flush=True)
        return
    if model is None and args.single_image_pixel:
        source = paths(dataset, sample_id, None, ROOT)
        output.mkdir(parents=True, exist_ok=True)
        for step in range(frames):
            original = source / f"{step}.png"
            if not original.is_file():
                raise FileNotFoundError(original)
            shutil.copy2(original, output / original.name)
        print(f"COPY GT {dataset}/{sample_id} from {source}", flush=True)
        return
    if model is not None and args.wait_for_gpu:
        while not gpu_ready(args.gpu, args.minimum_free_mib):
            print(f"WAIT GPU {args.gpu} for {dataset}/{sample_id}/{model}", flush=True)
            time.sleep(60)
    cmd = command(dataset, sample_id, model, args.output, args.single_image_pixel)
    log = args.output / "logs" / f"{dataset}_{sample_id}_{model or 'gt'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "NWM_DATA_ROOT": str(NAS / "data"),
        "NWM_INDEX_ROOT": str(NAS / "cache/dataset_indices"),
        "TORCH_HOME": "/file_system/vepfs/algorithm/dujun.nie/models",
        "HF_HOME": str(RAE_ASSETS / "hf_cache"), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
    })
    print(f"RUN cwd={REPO} log={log} command={' '.join(cmd)}", flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(f"PID {process.pid}", flush=True)
        status = process.wait()
    print(f"EXIT {status} {dataset}/{sample_id}/{model or 'gt'} log={log}", flush=True)
    if status:
        raise RuntimeError(f"Inference failed: {log}\n{''.join(log.read_text().splitlines(True)[-25:])}")
    if model == "rae-nwm" and "source_ids" in CASES[dataset]:
        case = CASES[dataset]
        source_id = case["source_ids"][case["ids"].index(sample_id)]
        raw_output = paths(dataset, source_id, model, args.output)
        output.mkdir(parents=True, exist_ok=True)
        for step in range(frames):
            shutil.copy2(raw_output / f"{step}.png", output / f"{step}.png")
    missing = [step for step in range(frames) if not (output / f"{step}.png").is_file()]
    if missing:
        raise RuntimeError(f"Missing rollout frames: {dataset}/{sample_id}/{model}: {missing}")


def write_selection(root: Path, reserve: bool) -> None:
    cases = []
    rankings = {}
    for dataset, case in CASES.items():
        ranked, inputs = rank_direct_candidates(dataset)
        candidate_ranks = {item["sample_id"]: rank for rank, item in enumerate(ranked)}
        source_ids = case.get("source_ids", case["ids"])
        if any(sample_id not in candidate_ranks for sample_id in source_ids):
            raise RuntimeError(f"Pinned candidate missing from direct ranking for {dataset}")
        if dataset != "huron" and [item["sample_id"] for item in ranked[:2]] != case["ids"]:
            raise RuntimeError(f"Pinned top-two ranking changed for {dataset}")
        if dataset == "huron" and (candidate_ranks[63], candidate_ranks[110]) != (2, 3):
            raise RuntimeError("Pinned valid HuRoN candidate rankings changed")
        rankings[dataset] = {"inputs": inputs, "top_candidates": ranked[:20]}
        split = REPO / "data_splits" / case["split_dir"] / "test/rollout.pkl"
        entries = pickle.loads(split.read_bytes())
        if dataset == "huron":
            cache = NAS / "cache/dataset_indices/sacson/test"
            missing_file = cache / "missing_trajectories_sacson.txt"
            invalid_file = cache / "invalid_indices_in_rollout.txt"
            missing_names = set(missing_file.read_text().splitlines())
            invalid_entries = {
                tuple(part.strip() for part in line.split(","))
                for line in invalid_file.read_text().splitlines()
            }
            filtered_to_source = [
                index for index, entry in enumerate(entries)
                if entry[0] not in missing_names
                and tuple(map(str, entry)) not in invalid_entries
            ]
            if (len(filtered_to_source) != case["expected_count"]
                    or [filtered_to_source[index] for index in case["ids"]] != source_ids):
                raise RuntimeError("HuRoN filtered index mapping changed")
            rankings[dataset]["filtered_to_source_split_index"] = filtered_to_source
            rankings[dataset]["availability_files"] = {
                "missing_trajectories": {"path": str(missing_file), "sha256": sha256(missing_file)},
                "invalid_indices": {"path": str(invalid_file), "sha256": sha256(invalid_file)},
            }
        for rank, sample_id in enumerate(case["ids"] if reserve else case["ids"][:1]):
            source_id = source_ids[rank]
            preselection_rank = candidate_ranks[source_id]
            cases.append({
                "group": case["group"], "dataset": dataset, "sample_id": sample_id,
                "source_split_index": source_id,
                "rae_sample_id": source_id,
                "candidate_rank": preselection_rank + 1, "seconds": case["seconds"], "fps": 4,
                "split": str(split), "split_sha256": sha256(split),
                "split_entry": entries[source_id],
                "preselection": ranked[preselection_rank],
            })
    payload = {
        "selection_rule": "Pinned candidates from the largest OpenNWM minus best-baseline direct 4-second PSNR margins on exact or nearest same-trajectory time windows. Each dataset uses the top two, except HuRoN: its two highest-ranked valid local entries (original indices 63 and 110). IDs were fixed before their rollout inference.",
        "qualification": "Direct-prediction scores nominate candidates only; rollout scores and images determine the final claim.",
        "seed": 0, "models": {name: {key: str(value) for key, value in item.items()}
                               for name, item in MODELS.items()},
        "cases": cases,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / "selection_rankings.json").write_text(json.dumps(rankings, indent=2) + "\n")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255 * 255 / mse)) if mse else float("inf")


def assemble(root: Path, reserve: bool) -> None:
    summary = []
    for dataset, case in CASES.items():
        for sample_id in case["ids"] if reserve else case["ids"][:1]:
            count = case["seconds"] * 4
            sources = {"GT": paths(dataset, sample_id, None, root), **{
                name: paths(dataset, sample_id, name, root) for name in MODELS}}
            missing = [(name, step) for name, folder in sources.items()
                       for step in range(count) if not (folder / f"{step}.png").is_file()]
            if missing:
                raise RuntimeError(f"Incomplete case {dataset}/{sample_id}: {missing[:8]}")
            out = root / "examples" / dataset / f"id_{sample_id}"
            horizons = [1, 2, 4] + ([8, 16] if case["seconds"] == 16 else [])
            scores = {name: [] for name in MODELS}
            for name, folder in sources.items():
                dest = out / "frames" / name
                dest.mkdir(parents=True, exist_ok=True)
                for step in range(count):
                    shutil.copy2(folder / f"{step}.png", dest / f"{step:03d}.png")
            for step in range(count):
                gt = np.asarray(Image.open(sources["GT"] / f"{step}.png").convert("RGB"))
                for name in MODELS:
                    pred = np.asarray(Image.open(sources[name] / f"{step}.png").convert("RGB"))
                    scores[name].append(psnr(gt, pred))
            width, height = 224 * len(horizons), 260 * len(sources)
            sheet = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(sheet)
            for col, sec in enumerate(horizons):
                for row, (name, folder) in enumerate(sources.items()):
                    image = Image.open(folder / f"{sec * 4 - 1}.png").convert("RGB")
                    sheet.paste(image, (col * 224, row * 260 + 24))
                    draw.text((col * 224 + 4, row * 260 + 4), f"{name}  t={sec}s", fill="black")
            sheet.save(out / "comparison.png")
            row = {"group": case["group"], "dataset": dataset, "sample_id": sample_id,
                   "seconds": case["seconds"], "fps": 4,
                   "endpoint_psnr": {name: values[-1] for name, values in scores.items()},
                   "mean_psnr": {name: float(np.mean(values)) for name, values in scores.items()},
                   "advantage_endpoint_db": scores["opennwm-finalLAM-100k"][-1] -
                       max(scores["nwm-release"][-1], scores["rae-nwm"][-1]),
                   "comparison": str(out / "comparison.png"), "frames": str(out / "frames")}
            (out / "metrics.json").write_text(json.dumps({**row, "frame_psnr": scores}, indent=2) + "\n")
            summary.append(row)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    baselines = ("nwm-release", "rae-nwm")
    open_model = "opennwm-finalLAM-100k"
    selected = []
    for row in summary:
        mean_margin = row["mean_psnr"][open_model] - max(
            row["mean_psnr"][name] for name in baselines
        )
        endpoint_margin = row["endpoint_psnr"][open_model] - max(
            row["endpoint_psnr"][name] for name in baselines
        )
        if mean_margin > 0 and endpoint_margin > 0:
            selected.append({**row, "advantage_mean_db": mean_margin,
                             "advantage_endpoint_db": endpoint_margin})
    (root / "selected.json").write_text(json.dumps({
        "rule": "OpenNWM PSNR strictly exceeds both baselines for both the arithmetic mean over all predicted frames and the final predicted frame.",
        "candidate_count": len(summary),
        "selected_count": len(selected),
        "selected": selected,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("gt", "predict", "assemble", "all"))
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--minimum-free-mib", type=int, default=24000)
    parser.add_argument("--reserve", action="store_true", help="Also run backup candidates")
    parser.add_argument("--single-image-pixel", action="store_true",
                        help="Repeat one initial image and re-encode each decoded prediction")
    parser.add_argument("--datasets", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.single_image_pixel:
        if args.output == ROOT:
            raise ValueError("Single-image rollout requires a separate --output directory")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "protocol.json").write_text(json.dumps({
            "context_initialization": "repeat_current_image_four_times",
            "feedback": "decoded_prediction_image_reencoded_each_step",
            "fps": 4,
            "seed": 0,
            "gt_source": str(ROOT),
            "selection_source": str(ROOT / "selection.json"),
        }, indent=2) + "\n")
    write_selection(args.output, args.reserve)
    if args.stage in ("gt", "all"):
        for dataset in args.datasets:
            case = CASES[dataset]
            for sample_id in case["ids"] if args.reserve else case["ids"][:1]:
                run_case(dataset, sample_id, None, args)
    if args.stage in ("predict", "all"):
        for model in args.models:
            for dataset in args.datasets:
                case = CASES[dataset]
                for sample_id in case["ids"] if args.reserve else case["ids"][:1]:
                    run_case(dataset, sample_id, model, args)
    if args.stage in ("assemble", "all"):
        assemble(args.output, args.reserve)


if __name__ == "__main__":
    main()
