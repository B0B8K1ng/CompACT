"""Add frame-level FID to existing A800 rollout LPIPS measurements.

Uses the torcheval Inception features used by the released RAE-NWM evaluator.
For N << 2048, the covariance square-root trace equals the nuclear norm of
the product of centered sample matrices. This computes the same FID in
float64 without diagonalizing a rank-deficient 2048 x 2048 covariance product.
No frames are pooled across horizons and no per-image FID is averaged.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torcheval.metrics import FrechetInceptionDistance
from torchvision.transforms.functional import pil_to_tensor


DATASETS = ["recon", "scand", "huron", "tartan_drive", "go_stanford", "tum_rgbd", "unitree_go2"]
ID_DATASETS = {"recon", "scand", "huron", "tartan_drive"}
MODELS = {"nwm-release": "NWM", "rae-nwm": "RAE-NWM", "opennwm-finalLAM-100k": "OpenNWM"}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def fid_from_features(real, fake):
    real, fake = real.double(), fake.double()
    rm, fm = real.mean(0), fake.mean(0)
    rc = (real - rm) / (len(real) - 1) ** 0.5
    fc = (fake - fm) / (len(fake) - 1) ** 0.5
    cross_trace = torch.linalg.svdvals(rc @ fc.T).sum()
    return float((rm - fm).square().sum() + rc.square().sum() + fc.square().sum() - 2 * cross_trace)


def image_tensor(path):
    with Image.open(path) as im:
        return pil_to_tensor(im.convert("RGB")).float().div_(255)


@torch.inference_mode()
def extract(paths, model, device, pool, batch_size):
    futures = [pool.submit(image_tensor, p) for p in paths]
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([f.result() for f in futures[start : start + batch_size]])
        chunks.append(model(batch.to(device)).cpu())
    return torch.cat(chunks)


def validate(real, fake, shared_model):
    """Independent dense torcheval calculation, with float64 statistics."""
    metric = FrechetInceptionDistance(model=shared_model, feature_dim=2048, device="cpu")
    for prefix, features in [("real", real), ("fake", fake)]:
        f = features.double()
        setattr(metric, prefix + "_sum", f.sum(0))
        setattr(metric, prefix + "_cov_sum", f.T @ f)
        setattr(metric, "num_" + prefix + "_images", torch.tensor(len(f)).int())
    expected = float(metric.compute())
    actual = fid_from_features(real, fake)
    identity = fid_from_features(real, real)
    reverse = fid_from_features(fake, real)
    assert abs(expected - actual) < 0.001, (expected, actual)
    assert abs(identity) < 1e-8, identity
    assert abs(reverse - actual) < 1e-8, (reverse, actual)
    return {"torcheval_dense_float64": expected, "low_rank_float64": actual,
            "absolute_difference": abs(expected - actual), "identity_fid": identity,
            "symmetry_absolute_difference": abs(reverse - actual)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.25)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contracts = json.loads((args.run_dir / "contract.json").read_text())
    lpips = {}
    with (args.analysis_dir / "rollout_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if row["metric"] == "lpips_alex":
                lpips[row["dataset"], int(row["fps"]), int(row["horizon_s"]), row["model_key"]] = row
    # Construct once and share frozen weights for every dataset/model/horizon.
    metric = FrechetInceptionDistance(feature_dim=2048, device=args.device)
    model = metric.model.eval()
    metadata = {"run_dir": str(args.run_dir), "pid": os.getpid(),
                "metrics": ["lpips_alex", "fid"], "seed": 0,
                "FID": {"feature_backend": "torcheval.metrics.image.fid.FIDInceptionV3",
                        "weights": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
                        "feature_dim": 2048, "input": "RGB float32 [0,1]",
                        "resize": "torcheval bilinear 299x299, align_corners=False",
                        "aggregation": "one image per sample at each horizon; unbiased sample covariance",
                        "arithmetic": "float64 exact low-rank covariance trace; verified against dense torcheval"},
                "ID_datasets": sorted(ID_DATASETS),
                "domain_note": "ID relative to NWM/OpenNWM navigation training; RAE-NWM training excludes TartanDrive",
                "allowed_endpoints": {"ID": [4, 8, 16], "OOD": [4]},
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(args.output_dir / "metadata.json", metadata)
    validated = (args.output_dir / "fid_validation.json").exists()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for dataset in DATASETS:
            count = 103 if dataset == "huron" else 150
            horizons = [1, 2, 4, 8, 16] if dataset in ID_DATASETS else [1, 2, 4]
            for fps in [1, 4]:
                for horizon in horizons:
                    output = args.output_dir / "points" / f"{dataset}_{fps}fps_{horizon}s.json"
                    if output.exists():
                        continue
                    begin = time.monotonic()
                    frame = horizon * fps - 1
                    evaluation = f"rollout_{fps}fps"
                    gt_root = args.run_dir / "gt" / dataset / evaluation
                    real = extract([gt_root / f"id_{i}" / f"{frame}.png" for i in range(count)],
                                   model, args.device, pool, args.batch_size)
                    results = {}
                    for key, label in MODELS.items():
                        row = lpips[dataset, fps, horizon, key]
                        ref = json.loads(Path(row["source"]).read_text())
                        c = contracts["contracts"][f"{key}/{dataset}/rollout"]
                        assert ref["sample_ids"] == list(range(count))
                        assert ref["frame_indices"][f"{horizon}s"] == frame
                        for k in ["checkpoint_sha256", "split_sha256", "seed", "sampler", "sampling_steps"]:
                            assert ref["inference"][k] == c[k], (dataset, key, k)
                        pred_root = args.run_dir / "predictions" / key / dataset / evaluation
                        fake = extract([pred_root / f"id_{i}" / f"{frame}.png" for i in range(count)],
                                       model, args.device, pool, args.batch_size)
                        value = fid_from_features(real, fake)
                        assert value >= -1e-8 and torch.isfinite(torch.tensor(value)), value
                        if not validated:
                            check = validate(real, fake, model)
                            model.to(args.device)
                            check.update(dataset=dataset, fps=fps, horizon_s=horizon, model=key)
                            atomic_json(args.output_dir / "fid_validation.json", check)
                            validated = True
                            print("VALIDATED", check, flush=True)
                        results[key] = {"label": label, "lpips_alex": float(row["value"]), "fid": value,
                                        "sample_count": count, "inference": ref["inference"],
                                        "lpips_source": row["source"], "pred_dir": str(pred_root)}
                    atomic_json(output, {"dataset": dataset, "domain": "ID" if dataset in ID_DATASETS else "OOD",
                                         "fps": fps, "horizon_s": horizon, "frame_index": frame,
                                         "sample_ids": list(range(count)), "gt_dir": str(gt_root), "models": results})
                    print("DONE", dataset, fps, horizon, {k: round(v["fid"], 4) for k, v in results.items()},
                          "seconds", round(time.monotonic() - begin, 2), flush=True)
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
