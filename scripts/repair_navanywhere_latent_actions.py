#!/usr/bin/env python3
"""Fill a small explicit missing-cache list using independent GPU workers.

Writes into a staging cache; the audited main cache remains untouched until
the CPU finalizer installs and validates the repairs. No distributed collectives.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time


def worker(gpu, tasks, state_path, staging):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    import precompute_navanywhere_nav1_latent_actions as pre

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    state = json.loads(Path(state_path).read_text())
    state['output_root'] = staging
    bitmap = np.memmap(state['training_pair_plan']['pair_bitmap_path'], dtype=np.uint8, mode='r')
    config = state['extraction']['configuration']
    args = argparse.Namespace(**config, loader_threads=16)
    pre.log(gpu, f'repair worker starts: {len(tasks)} trajectories, '
            f"{sum(t['planned_pair_count'] for t in tasks)} planned pairs")
    adapter = pre._load_adapter(state, 0)
    _, _, load_frame = pre._lam_api(state['lam_project_root'])
    original_encode = adapter.encode
    done = 0
    last_report = time.monotonic()

    def encode(videos):
        nonlocal done, last_report
        result = original_encode(videos)
        done += len(videos)
        now = time.monotonic()
        if now-last_report >= 20:
            pre.log(gpu, f'encoded_pairs={done} allocated_GiB={torch.cuda.memory_allocated()/2**30:.2f} '
                    f'peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}')
            last_report = now
        return result

    adapter.encode = encode
    for task in tasks:
        identity = f"{task['source_id']}/{task['trajectory_id']}"
        pre.log(gpu, f'scanning {identity}: frames={task["frame_count"]}, pairs={task["planned_pair_count"]}')
        frames, indices, source_fp = pre._scan_task(Path(state['data_root']), task)
        path = Path(staging) / task['source_id'] / (task['trajectory_id']+'.pt')
        if path.exists():
            try:
                pre._cache_record(path, state, task, indices, source_fp, bitmap)
                pre.log(gpu, f'reusing completed repair {identity}')
                continue
            except Exception as exc:
                pre.log(gpu, f'repair cache invalid: {exc}')
        pre.log(gpu, f'loading frames and encoding {identity}')
        record = pre._compute_task(adapter, load_frame, state, task, frames, indices, source_fp, bitmap, args, gpu)
        pre.log(gpu, f'repaired {identity}: pairs={record["pair_count"]}, '
                f'substitutions={len(record["invalid_frame_substitutions"])}')
        torch.cuda.empty_cache()
    pre.log(gpu, 'repair worker complete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--staging-root', required=True)
    p.add_argument('--devices', required=True)
    args = p.parse_args()
    tasks = json.loads(Path(args.tasks).read_text())
    devices = [int(x) for x in args.devices.split(',')]
    assigned = [[] for _ in devices]
    loads = [0 for _ in devices]
    for task in sorted(tasks, key=lambda t: -t['planned_pair_count']):
        i = min(range(len(devices)), key=lambda i: loads[i])
        assigned[i].append(task)
        loads[i] += task['planned_pair_count']
    Path(args.staging_root).mkdir(parents=True, exist_ok=True)
    processes = []
    context = mp.get_context('spawn')
    for device, selected in zip(devices, assigned):
        if selected:
            proc = context.Process(target=worker, args=(device, selected, args.state, args.staging_root))
            proc.start()
            processes.append(proc)
    failed = []
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failed.append((proc.pid, proc.exitcode))
    if failed:
        raise RuntimeError(f'Repair workers failed: {failed}')
    print('All requested repairs complete', flush=True)


if __name__ == '__main__':
    main()
