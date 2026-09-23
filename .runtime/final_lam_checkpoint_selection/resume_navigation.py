#!/usr/bin/env python3
"""Resume only navigation jobs paused for the finalLAM checkpoint sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time


EVAL_ROOT = Path("/file_system/vepfs/algorithm/dujun.nie/code/CompACT-eval-huron-fix")
NAV_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/"
    "navigation_largebatch_20260921"
)
PYTHON = Path("/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python")
REASON = "user_requested_finalLAM_checkpoint_selection_20260922"


def alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return len(stat) > 2 and stat[2] != "Z"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    scheduler_path = NAV_ROOT / "scheduler.pid"
    if scheduler_path.exists():
        previous_pid = int(scheduler_path.read_text().strip())
        if alive(previous_pid):
            raise RuntimeError(f"Navigation scheduler already active: PID {previous_pid}")
    jobs_path = NAV_ROOT / "jobs.json"
    jobs = json.loads(jobs_path.read_text())
    resumed = []
    now = time.time()
    for job in jobs:
        if job.get("state") == "paused" and job.get("pause_reason") == REASON:
            job.update(state="pending", resumed_at=now)
            resumed.append(f"{job['model']}/{job['dataset']}")
    if not resumed:
        raise RuntimeError(f"No navigation jobs paused with reason {REASON}")
    atomic_json(jobs_path, jobs)
    command = [str(PYTHON), ".runtime/navlarge/schedule.py"]
    env = os.environ.copy()
    env.update(
        PATH=f"{PYTHON.parent}:{env.get('PATH', '')}",
        PYTHONUNBUFFERED="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    log_path = NAV_ROOT / "logs/scheduler.log"
    with log_path.open("a") as stream:
        process = subprocess.Popen(
            command,
            cwd=EVAL_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    scheduler_path.write_text(f"{process.pid}\n")
    print(
        json.dumps(
            {
                "state": "navigation_resumed",
                "cwd": str(EVAL_ROOT),
                "command": shlex.join(command),
                "scheduler_pid": process.pid,
                "log": str(log_path),
                "resumed_jobs": resumed,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
