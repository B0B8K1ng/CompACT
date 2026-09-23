#!/usr/bin/env python3
"""Pause the resumable navigation benchmark before the finalLAM checkpoint sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time


NAV_ROOT = Path(
    "/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/"
    "navigation_largebatch_20260921"
)
REASON = "user_requested_finalLAM_checkpoint_selection_20260922"


def alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return len(stat) > 2 and stat[2] != "Z"


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return ""


def terminate(pid: int, *, group: bool) -> dict:
    record = {"pid": pid, "cmdline": cmdline(pid), "group": group}
    if not alive(pid):
        record["result"] = "already_exited"
        return record
    target = -pid if group and os.getpgid(pid) == pid else pid
    os.kill(target, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if alive(pid):
        os.kill(target, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        record["result"] = "killed" if not alive(pid) else "still_alive"
    else:
        record["result"] = "terminated"
    return record


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    jobs_path = NAV_ROOT / "jobs.json"
    jobs = json.loads(jobs_path.read_text())
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    running = [job for job in jobs if job.get("state") == "running"]
    scheduler_pid = int((NAV_ROOT / "scheduler.pid").read_text().strip())
    scheduler_record = terminate(scheduler_pid, group=False)
    process_records = []
    for job in running:
        pid = int(job["pid"])
        command = cmdline(pid)
        if alive(pid) and ".runtime/run_benchmark_job.sh" not in command:
            raise RuntimeError(f"Refusing to stop unexpected PID {pid}: {command}")
        process_records.append(terminate(pid, group=True))
    for job in jobs:
        if job.get("state") != "running":
            continue
        job.update(
            state="paused",
            pause_reason=REASON,
            paused_at=now,
            paused_pid=job.get("pid"),
        )
    snapshot = {
        "paused_at": now,
        "reason": REASON,
        "scheduler": scheduler_record,
        "processes": process_records,
        "jobs_before_pause": running,
    }
    snapshot_path = NAV_ROOT / f"pause_snapshot_finalLAM_selection_{stamp}.json"
    atomic_json(snapshot_path, snapshot)
    atomic_json(jobs_path, jobs)
    print(
        json.dumps(
            {
                "state": "paused",
                "scheduler_pid": scheduler_pid,
                "jobs_paused": len(running),
                "snapshot": str(snapshot_path),
                "processes": process_records,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
