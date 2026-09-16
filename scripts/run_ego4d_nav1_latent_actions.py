#!/usr/bin/env python3
"""Dynamically use idle GPUs to extract planned Ego4D nav1 latent actions."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import precompute_navanywhere_nav1_latent_actions as pre  # noqa: E402


NAS_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm")
DEFAULT_RECIPE = NAS_ROOT / "compact/recipes/ego4d_balanced_seed20260901.json"
DEFAULT_PLAN = (
    NAS_ROOT
    / "compact/plans/ego4d_latent_action_seed20260901_ws8_bs16_steps200000.json"
)
DEFAULT_OUTPUT = (
    NAS_ROOT
    / "compact/cache/ego4d_nav1_pixel_action_step100000_ws8_bs16_steps200000"
)
DEFAULT_CHECKPOINT = (
    NAS_ROOT
    / "weights/navigation_lam/variant_4_pixel_action/nav1-pixel-action/"
    "checkpoints/step=100000.ckpt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "ec7d4c159a0bcd661167b35ea88a1c61ac42d73a771a0de5de660cced4325ac1"
)
DEFAULT_LAM_ROOT = (
    REPO_ROOT.parent / "DreamDojo/external/lam_project"
)
DEFAULT_DATA_ROOT = NAS_ROOT / "data/NavAnywhere"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def safe_component(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"unsafe cache identity component: {value!r}")
    return value


def task_key(task: dict[str, object]) -> str:
    return f"{safe_component(str(task['source_id']))}__{safe_component(str(task['trajectory_id']))}"


def record_path(state: dict[str, object], task: dict[str, object]) -> Path:
    return (
        Path(str(state["output_root"]))
        / "_records"
        / safe_component(str(task["source_id"]))
        / f"{safe_component(str(task['trajectory_id']))}.json"
    )


def claim_path(state: dict[str, object], task: dict[str, object]) -> Path:
    return Path(str(state["output_root"])) / "_claims" / task_key(task)


def failure_path(state: dict[str, object], task: dict[str, object]) -> Path:
    return Path(str(state["output_root"])) / "_failures" / f"{task_key(task)}.json"


def failure_attempts(state: dict[str, object], task: dict[str, object]) -> int:
    path = failure_path(state, task)
    if not path.is_file():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempts", 0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return int(state["max_attempts"])


def output_path(state: dict[str, object], task: dict[str, object]) -> Path:
    return pre.safe_path(
        Path(str(state["output_root"])),
        str(task["source_id"]),
        str(task["trajectory_id"]),
        ".pt",
    )


def build_state(args: argparse.Namespace) -> dict[str, object]:
    extraction_args = SimpleNamespace(
        data_root=str(args.data_root),
        sampling_recipe=str(args.sampling_recipe),
        training_pair_plan=str(args.training_pair_plan),
        output_root=str(args.output_root),
        reuse_root=[str(path) for path in (args.reuse_root or ())],
        lam_project_root=str(args.lam_project_root),
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=str(args.checkpoint_sha256),
        precision=str(args.precision),
        batch_size=int(args.batch_size),
        loader_threads=int(args.loader_threads),
        stream_frames=True,
        stream_pair_chunk_size=int(args.stream_pair_chunk_size),
        allow_future_training_plan=True,
        image_height=240,
        image_width=320,
        context_size=4,
        max_abs_frame_offset=8,
        max_trajectories=0,
        trajectory=None,
        log_every_trajectories=1,
        overwrite=False,
    )
    state = pre._build_state(extraction_args, world_size=1)
    state["runtime_args"] = vars(extraction_args)
    state["max_attempts"] = int(args.max_attempts)
    state["created_at_utc"] = pre.utc_now()
    for task in state["tasks"]:
        task.pop("assigned_rank", None)
    return state


def load_state(path: Path) -> dict[str, object]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("tasks"), list):
        raise TypeError(f"malformed extraction state: {path}")
    return state


def acquire_claim(state: dict[str, object], task: dict[str, object], gpu_label: str) -> Path | None:
    claim = claim_path(state, task)
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        return None
    pre.atomic_json_dump(
        {
            "pid": os.getpid(),
            "gpu": gpu_label,
            "task": task_key(task),
            "claimed_at_utc": pre.utc_now(),
        },
        claim / "owner.json",
    )
    return claim


def release_claim(claim: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (claim / "owner.json").unlink()
    with contextlib.suppress(FileNotFoundError, OSError):
        claim.rmdir()


def run_worker(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    runtime_args = SimpleNamespace(**state["runtime_args"])
    pair_bitmap = None
    if state.get("training_pair_plan") is not None:
        import numpy as np

        pair_bitmap = np.memmap(
            state["training_pair_plan"]["pair_bitmap_path"], dtype=np.uint8, mode="r"
        )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("worker has no visible CUDA device")
    torch.cuda.set_device(0)
    _, _, load_frame = pre._lam_api(state["lam_project_root"])
    adapter = None
    tasks = sorted(
        state["tasks"],
        key=lambda task: (
            -int(task.get("planned_pair_count", 0)),
            str(task["source_id"]),
            str(task["trajectory_id"]),
        ),
    )
    completed = 0
    completed_pairs = 0
    started = time.monotonic()
    for task in tasks:
        record_file = record_path(state, task)
        if record_file.is_file() and output_path(state, task).is_file():
            continue
        if failure_attempts(state, task) >= int(state["max_attempts"]):
            continue
        claim = acquire_claim(state, task, args.gpu_label)
        if claim is None:
            continue
        try:
            if record_file.is_file() and output_path(state, task).is_file():
                continue
            log(
                f"worker_gpu={args.gpu_label} start={task_key(task)} "
                f"planned_pairs={task.get('planned_pair_count', 0)}"
            )
            frames, indices, source_fingerprint = pre._scan_task(
                Path(str(state["data_root"])), task
            )
            cache_path = output_path(state, task)
            record = None
            if cache_path.is_file():
                try:
                    record = pre._cache_record(
                        cache_path,
                        state,
                        task,
                        indices,
                        source_fingerprint,
                        pair_bitmap,
                    )
                except Exception as exc:
                    log(f"worker_gpu={args.gpu_label} recompute={task_key(task)} reason={exc!r}")
            if record is None:
                target_pairs = pre._task_frame_pairs(
                    state, task, indices, pair_bitmap
                )
                reuse_result = pre._reuse_task_motion(
                    state, task, indices, source_fingerprint, target_pairs
                )
                if len(reuse_result["missing_rows"]) and adapter is None:
                    log(f"worker_gpu={args.gpu_label} loading navigation PixelActionLAM")
                    adapter = pre._load_adapter(state, 0)
                    log(f"worker_gpu={args.gpu_label} navigation PixelActionLAM ready")
                record = pre._compute_task(
                    adapter,
                    load_frame,
                    state,
                    task,
                    frames,
                    indices,
                    source_fingerprint,
                    pair_bitmap,
                    runtime_args,
                    rank=int(args.gpu_label),
                    reuse_result=reuse_result,
                )
            pre.atomic_json_dump(record, record_file)
            with contextlib.suppress(FileNotFoundError):
                failure_path(state, task).unlink()
            completed += 1
            completed_pairs += int(record["pair_count"])
            elapsed = max(time.monotonic() - started, 1e-6)
            log(
                f"worker_gpu={args.gpu_label} done={task_key(task)} "
                f"trajectories={completed} pairs={completed_pairs} "
                f"pairs_per_second={completed_pairs / elapsed:.1f}"
            )
        except Exception as exc:
            failure = failure_path(state, task)
            attempts = failure_attempts(state, task) + 1
            pre.atomic_json_dump(
                {
                    "attempts": attempts,
                    "gpu": args.gpu_label,
                    "pid": os.getpid(),
                    "task": task_key(task),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "failed_at_utc": pre.utc_now(),
                },
                failure,
            )
            log(
                f"worker_gpu={args.gpu_label} ERROR={task_key(task)} "
                f"attempt={attempts} exception={exc!r}"
            )
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
        finally:
            release_claim(claim)
    log(
        f"worker_gpu={args.gpu_label} exhausted_queue trajectories={completed} "
        f"pairs={completed_pairs}"
    )


def gpu_capacity() -> dict[int, tuple[int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in output.splitlines():
        index, free_mib, utilization = (int(value.strip()) for value in line.split(","))
        result[index] = (free_mib, utilization)
    return result


def clean_stale_claims(state: dict[str, object]) -> None:
    root = Path(str(state["output_root"])) / "_claims"
    if not root.is_dir():
        return
    for claim in root.iterdir():
        if not claim.is_dir():
            continue
        owner = claim / "owner.json"
        try:
            pid = int(json.loads(owner.read_text(encoding="utf-8"))["pid"])
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"removing stale generated claim {claim.name}")
            release_claim(claim)
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A live worker can be between mkdir and atomic owner publication.
            continue
        except PermissionError:
            continue


def progress_counts(state: dict[str, object]) -> tuple[int, int, int]:
    records = 0
    claims = 0
    eligible = 0
    for task in state["tasks"]:
        if record_path(state, task).is_file() and output_path(state, task).is_file():
            records += 1
        elif claim_path(state, task).is_dir():
            claims += 1
        elif failure_attempts(state, task) < int(state["max_attempts"]):
            eligible += 1
    return records, claims, eligible


def finalize(state: dict[str, object]) -> None:
    records = []
    for task in state["tasks"]:
        path = record_path(state, task)
        if not path.is_file():
            raise FileNotFoundError(f"missing completed record: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    pre._write_completion(state, records, partial=False)
    log(f"validated completion marker={Path(str(state['output_root'])) / '_SUCCESS.json'}")


def run_supervisor(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_root / ".dynamic-precompute.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output_root / "extraction_state.json"
    if state_path.is_file():
        state = load_state(state_path)
        log(f"resuming state={state_path}")
    else:
        state = build_state(args)
        pre.atomic_json_dump(state, state_path)
        log(f"created state={state_path} tasks={len(state['tasks'])}")
    clean_stale_claims(state)

    allowed_gpus = [int(value) for value in args.gpu_ids.split(",")]
    if len(set(allowed_gpus)) != len(allowed_gpus):
        raise ValueError("--gpu-ids contains duplicates")
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[subprocess.Popen[bytes], object]] = {}
    launches = 0
    last_progress_log = 0.0
    while True:
        for gpu, (process, handle) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[gpu]
            log(f"worker_exit gpu={gpu} pid={process.pid} status={status}")
        clean_stale_claims(state)
        records, claims, eligible = progress_counts(state)
        if records == len(state["tasks"]) and not active:
            finalize(state)
            return
        if not active and eligible == 0:
            raise RuntimeError(
                f"extraction blocked: records={records}/{len(state['tasks'])}, claims={claims}; "
                f"see {output_root / '_failures'}"
            )

        capacity = gpu_capacity()
        unclaimed_budget = eligible
        for gpu in allowed_gpus:
            if gpu in active or unclaimed_budget <= 0:
                continue
            free_mib, utilization = capacity[gpu]
            if free_mib < args.min_free_mib or utilization > args.max_utilization:
                continue
            launches += 1
            worker_log = logs / f"gpu_{gpu}_launch_{launches:03d}.log"
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "worker",
                "--state",
                str(state_path),
                "--gpu-label",
                str(gpu),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4",
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            handle = worker_log.open("ab", buffering=0)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, handle)
            unclaimed_budget -= 1
            log(
                f"worker_start gpu={gpu} pid={process.pid} log={worker_log} "
                f"command={shlex.join(command)}"
            )
        now = time.monotonic()
        if now - last_progress_log >= args.progress_seconds:
            gpu_state = " ".join(
                f"{gpu}:{capacity[gpu][0]}MiB/{capacity[gpu][1]}%" for gpu in allowed_gpus
            )
            log(
                f"progress records={records}/{len(state['tasks'])} claims={claims} "
                f"eligible={eligible} active={sorted(active)} gpu={gpu_state}"
            )
            last_progress_log = now
        time.sleep(args.poll_seconds)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    run.add_argument("--sampling-recipe", type=Path, default=DEFAULT_RECIPE)
    run.add_argument("--training-pair-plan", type=Path, default=DEFAULT_PLAN)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument(
        "--reuse-root",
        action="append",
        type=Path,
        default=None,
        help="Compatible prior cache used for exact frame-pair row reuse; repeatable.",
    )
    run.add_argument("--lam-project-root", type=Path, default=DEFAULT_LAM_ROOT)
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    run.add_argument("--precision", choices=("32", "16-mixed", "bf16-mixed"), default="bf16-mixed")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--loader-threads", type=int, default=16)
    run.add_argument("--stream-pair-chunk-size", type=int, default=4096)
    run.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    run.add_argument("--min-free-mib", type=int, default=30000)
    run.add_argument("--max-utilization", type=int, default=20)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--progress-seconds", type=float, default=120.0)
    run.add_argument("--max-attempts", type=int, default=3)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--state", type=Path, required=True)
    worker.add_argument("--gpu-label", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        run_supervisor(args)
    elif args.command == "worker":
        run_worker(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
