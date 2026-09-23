#!/usr/bin/env python3
"""Run the full ID/OOD checkpoint sweep, select a weight, then resume navigation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time


CONTROL_ROOT = Path(
    "/file_system/vepfs/algorithm/dujun.nie/code/CompACT/"
    ".runtime/final_lam_checkpoint_selection"
)
EVAL_ROOT = Path("/file_system/vepfs/algorithm/dujun.nie/code/CompACT-eval-huron-fix")
PYTHON = Path("/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python")
BASE = Path("/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark")
OUT = BASE / "finalLAM_reset_checkpoint_sweep_20260922"
SHARED = BASE
DATASETS = (
    "recon,scand,huron,tartan_drive,go_stanford,unitree_go2,tum_rgbd,uzh_fpv"
)
# Run sequentially on the four GPUs that completed the pinned 50k evaluation.
# GPU 0 has too little headroom for the observed ~14.7 GiB/rank peak, and two
# concurrent checkpoints would contend with unrelated GPU work on this host.
PAIRS = tuple(
    ((step, "5,6,7,3"),)
    for step in (10_000, 20_000, 30_000, 40_000, 60_000, 70_000, 80_000, 90_000, 100_000)
)


def model_name(step: int) -> str:
    return f"finalLAM-reset-joint{step:07d}"


def command(step: int, gpus: str) -> list[str]:
    return [
        str(PYTHON),
        ".runtime/final_lam_eval/run.py",
        "--models",
        model_name(step),
        "--metrics",
        "direct",
        "--datasets",
        DATASETS,
        "--gpus",
        gpus,
        "--batch-size",
        "32",
        "--metric-batch-size",
        "32",
        "--benchmark-root",
        str(OUT),
        "--shared-benchmark-root",
        str(SHARED),
    ]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    job_path = OUT / "job.json"
    state = {
        "state": "running",
        "coordinator_pid": os.getpid(),
        "cwd": str(EVAL_ROOT),
        "started_at": time.time(),
        "pairs": [],
        "navigation_resume_on_success": True,
    }
    atomic_json(job_path, state)
    env = os.environ.copy()
    env.update(
        PATH=f"{PYTHON.parent}:{env.get('PATH', '')}",
        PYTHONUNBUFFERED="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    try:
        for pair_index, pair in enumerate(PAIRS, start=1):
            running = []
            pair_state = {"pair": pair_index, "started_at": time.time(), "jobs": []}
            state["pairs"].append(pair_state)
            for step, gpus in pair:
                cmd = command(step, gpus)
                log_path = OUT / "logs" / f"{model_name(step)}.log"
                stream = log_path.open("a")
                process = subprocess.Popen(
                    cmd,
                    cwd=EVAL_ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                record = {
                    "joint_steps": step,
                    "gpus": gpus,
                    "pid": process.pid,
                    "cwd": str(EVAL_ROOT),
                    "command": cmd,
                    "shell_command": shlex.join(cmd),
                    "log": str(log_path),
                    "started_at": time.time(),
                }
                pair_state["jobs"].append(record)
                running.append((process, stream, record))
                print(
                    f"START pair={pair_index} step={step} gpus={gpus} "
                    f"pid={process.pid} log={log_path}",
                    flush=True,
                )
            atomic_json(job_path, state)
            failures = []
            for process, stream, record in running:
                returncode = process.wait()
                stream.close()
                record.update(exit_code=returncode, finished_at=time.time())
                if returncode:
                    failures.append(record)
                print(
                    f"DONE pair={pair_index} step={record['joint_steps']} "
                    f"exit={returncode}",
                    flush=True,
                )
                atomic_json(job_path, state)
            pair_state["finished_at"] = time.time()
            atomic_json(job_path, state)
            if failures:
                raise RuntimeError(
                    "Checkpoint evaluations failed: "
                    + ", ".join(
                        f"step={record['joint_steps']} exit={record['exit_code']}"
                        for record in failures
                    )
                )

        select_command = [str(PYTHON), str(CONTROL_ROOT / "select_checkpoint.py")]
        print(f"SELECT {shlex.join(select_command)}", flush=True)
        subprocess.run(select_command, cwd=EVAL_ROOT, env=env, check=True)
        resume_command = [str(PYTHON), str(CONTROL_ROOT / "resume_navigation.py")]
        print(f"RESUME {shlex.join(resume_command)}", flush=True)
        resume = subprocess.run(
            resume_command,
            cwd=EVAL_ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(resume.stdout, end="", flush=True)
        state.update(
            state="complete_navigation_resumed",
            finished_at=time.time(),
            selection=str(OUT / "selection.json"),
            navigation_resume=resume.stdout,
        )
        atomic_json(job_path, state)
    except BaseException as error:
        state.update(
            state="failed_navigation_paused",
            error=repr(error),
            finished_at=time.time(),
        )
        atomic_json(job_path, state)
        print(f"FAILED {error!r}; navigation remains paused", flush=True)
        raise


if __name__ == "__main__":
    main()
