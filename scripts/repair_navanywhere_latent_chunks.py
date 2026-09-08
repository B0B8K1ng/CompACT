#!/usr/bin/env python3
"""Repair explicit trajectories in resumable, batch-aligned GPU chunks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def chunk_is_valid(payload, start, end, metadata, pairs):
    import torch
    return (payload.get('complete') is True and payload.get('start') == start
            and payload.get('end') == end and payload.get('metadata') == metadata
            and torch.equal(payload['frame_pairs'], torch.from_numpy(pairs.copy()))
            and payload['motion'].shape == (end-start, 32)
            and payload['motion'].dtype == torch.float32
            and bool(torch.isfinite(payload['motion']).all()))


def run_workers(jobs, devices, state_path, chunk_root):
    context = mp.get_context('spawn')
    work, results = context.Queue(), context.Queue()
    processes = []
    started = time.monotonic()
    try:
        for job in jobs:
            work.put(job)
        for _ in devices:
            work.put(None)
        for gpu in devices:
            proc = context.Process(target=worker, args=(gpu, work, results, state_path, chunk_root))
            proc.start()
            processes.append(proc)
        completed, pairs_done = 0, 0
        while completed < len(jobs):
            try:
                result = results.get(timeout=30)
            except queue.Empty:
                failed = [(proc.pid, proc.exitcode) for proc in processes if proc.exitcode not in (None, 0)]
                if failed or all(proc.exitcode is not None for proc in processes):
                    raise RuntimeError(f'Worker failures {failed}; unfinished chunks={len(jobs)-completed}')
                print(f'heartbeat completed_chunks={completed}/{len(jobs)} pairs={pairs_done} '
                      f'elapsed={time.monotonic()-started:.1f}s', flush=True)
                continue
            if not result['ok']:
                raise RuntimeError(f'Chunk repair failed: {result}')
            completed += 1
            pairs_done += result['pairs']
            print(f'completed_chunks={completed}/{len(jobs)} pairs={pairs_done}', flush=True)
        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f'Worker {proc.pid} exited {proc.exitcode}')
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join()
        work.cancel_join_thread()
        work.close()
        results.close()


def worker(gpu, work, results, state_path, chunk_root):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import numpy as np
    import torch
    import precompute_navanywhere_nav1_latent_actions as pre
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    state = json.loads(Path(state_path).read_text())
    adapter = pre._load_adapter(state, 0)
    _, _, load_frame = pre._lam_api(state['lam_project_root'])
    config = state['extraction']['configuration']
    bitmap = np.memmap(state['training_pair_plan']['pair_bitmap_path'], dtype=np.uint8, mode='r')
    cache = {}
    original_encode = adapter.encode
    encoded = 0
    last_report = time.monotonic()

    def encode(videos):
        nonlocal encoded, last_report
        output = original_encode(videos)
        encoded += len(videos)
        if time.monotonic()-last_report >= 20:
            pre.log(gpu, f'encoded_pairs={encoded} peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}')
            last_report = time.monotonic()
        return output

    adapter.encode = encode
    while True:
        job = work.get()
        if job is None:
            return
        task, start, end, path = job
        identity = f"{task['source_id']}/{task['trajectory_id']}"
        try:
            if identity not in cache:
                descriptor = json.loads((Path(chunk_root)/task['source_id']/(task['trajectory_id']+'.json')).read_text())
                indices = np.asarray([f[0] for f in descriptor['frames']], dtype=np.int64)
                frame_pairs = pre._task_frame_pairs(state, task, indices, bitmap)
                cache[identity] = descriptor, indices, frame_pairs
            descriptor, indices, frame_pairs = cache[identity]
            selected_pairs = frame_pairs[start:end]
            positions = pre.position_pairs(indices, selected_pairs)
            needed, inverse = np.unique(positions, return_inverse=True)
            with ThreadPoolExecutor(max_workers=12) as pool:
                loaded = list(pool.map(lambda position: pre._load_frame_with_fallback(
                    descriptor['frames'], int(position), load_frame,
                    image_height=config['image_height'], image_width=config['image_width']), needed))
            frames = torch.stack([item[0] for item in loaded])
            substitutions = [item[1] for item in loaded if item[1] is not None]
            del loaded
            motion = pre.encode_pair_batches(adapter, frames, inverse.reshape(-1, 2),
                                             batch_size=config['batch_size'], precision=config['precision'])
            del frames
            payload = dict(start=start, end=end, motion=motion, frame_pairs=torch.from_numpy(selected_pairs.copy()),
                           metadata=pre._expected_metadata(state, task, descriptor['source_fingerprint']),
                           invalid_frame_substitutions=substitutions, complete=True)
            pre.atomic_torch_save(payload, Path(path))
            pre.log(gpu, f'chunk complete {identity} [{start}:{end}]')
            results.put(dict(ok=True, path=path, pairs=end-start))
            torch.cuda.empty_cache()
        except Exception as exc:
            results.put(dict(ok=False, identity=identity, start=start, end=end,
                             error=f'{type(exc).__name__}: {exc}'))
            raise


def main():
    import numpy as np
    import torch
    import precompute_navanywhere_nav1_latent_actions as pre
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--staging-root', required=True)
    p.add_argument('--chunk-root', required=True)
    p.add_argument('--devices', required=True)
    p.add_argument('--chunk-size', type=int, default=4096)
    args = p.parse_args()
    torch.set_num_threads(1)
    state = json.loads(Path(args.state).read_text())
    tasks = json.loads(Path(args.tasks).read_text())
    if args.chunk_size < 1 or args.chunk_size % state['extraction']['configuration']['batch_size']:
        p.error('chunk size must be a positive multiple of the original encoder batch size')
    bitmap = np.memmap(state['training_pair_plan']['pair_bitmap_path'], dtype=np.uint8, mode='r')
    chunk_root = Path(args.chunk_root)
    staging = Path(args.staging_root)
    staging.mkdir(parents=True, exist_ok=True)
    jobs, descriptors, active = [], {}, []

    def describe(task):
        frames, indices, source_fp = pre._scan_task(Path(state['data_root']), task)
        pairs = pre._task_frame_pairs(state, task, indices, bitmap)
        return task, frames, indices, source_fp, pairs

    print('Scanning only the explicitly missing trajectories', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        described = list(pool.map(describe, tasks))
    for task, frames, indices, source_fp, pairs in described:
        identity = (task['source_id'], task['trajectory_id'])
        target = staging/identity[0]/(identity[1]+'.pt')
        if target.exists():
            try:
                pre._cache_record(target, state, task, indices, source_fp, bitmap)
            except Exception as exc:
                print(f'Rebuilding invalid staged repair {identity}: {exc}', flush=True)
            else:
                print(f'Reusing complete repair {identity}', flush=True)
                continue
        active.append(task)
        descriptors[identity] = (indices, source_fp, pairs)
        descriptor_path = chunk_root/identity[0]/(identity[1]+'.json')
        pre.atomic_json_dump(dict(frames=frames, source_fingerprint=source_fp), descriptor_path)
        for start in range(0, len(pairs), args.chunk_size):
            end = min(start+args.chunk_size, len(pairs))
            path = chunk_root/identity[0]/identity[1]/f'{start:09d}_{end:09d}.pt'
            if path.exists():
                try:
                    if chunk_is_valid(pre.safe_torch_load(path), start, end,
                                      pre._expected_metadata(state, task, source_fp), pairs[start:end]):
                        continue
                except Exception as exc:
                    print(f'Rebuilding unreadable chunk {path}: {exc}', flush=True)
            jobs.append((task, start, end, str(path)))
    print(f'Pending {len(jobs)} chunks / {sum(end-start for _,start,end,_ in jobs)} pairs', flush=True)
    if jobs:
        devices = [int(x) for x in args.devices.split(',')]
        run_workers(jobs, devices, args.state, args.chunk_root)
    state['output_root'] = str(staging)
    for task in active:
        identity = (task['source_id'], task['trajectory_id'])
        indices, source_fp, pairs = descriptors[identity]
        chunks, substitutions = [], {}
        for start in range(0, len(pairs), args.chunk_size):
            end = min(start+args.chunk_size, len(pairs))
            chunk = pre.safe_torch_load(chunk_root/identity[0]/identity[1]/f'{start:09d}_{end:09d}.pt')
            if not chunk_is_valid(chunk, start, end, pre._expected_metadata(state, task, source_fp), pairs[start:end]):
                raise ValueError('Invalid chunk during assembly')
            chunks.append(chunk['motion'])
            for item in chunk['invalid_frame_substitutions']:
                old = substitutions.setdefault(item['frame_index'], item)
                if old != item:
                    raise ValueError('Inconsistent image fallback across chunks')
        payload = dict(schema_version=pre.SCHEMA_VERSION, format=pre.FORMAT_NAME,
                       proxy_type='latent', motion_type='latent', action_mode='latent',
                       source_id=identity[0], trajectory_id=identity[1],
                       dataset_name=identity[0], trajectory_name=identity[1],
                       pair_direction='current_to_goal', normalization='raw', latent_value='z_mu', latent_dim=32,
                       frame_indices=torch.from_numpy(indices.copy()), frame_pairs=torch.from_numpy(pairs.copy()),
                       motion=torch.cat(chunks) if chunks else torch.empty((0,32), dtype=torch.float32),
                       invalid_frame_substitutions=list(substitutions.values()),
                       metadata=pre._expected_metadata(state, task, source_fp), complete=True,
                       repair_runtime=dict(batch_aligned_chunks=args.chunk_size,
                                           frame_loading='required_frames_per_chunk'))
        target = staging/identity[0]/(identity[1]+'.pt')
        pre.atomic_torch_save(payload, target)
        record = pre._cache_record(target, state, task, indices, source_fp, bitmap)
        print(f'Assembled and validated {identity}: pairs={record["pair_count"]}', flush=True)
    print('All requested repairs complete', flush=True)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}; saved chunks can be resumed')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    main()
