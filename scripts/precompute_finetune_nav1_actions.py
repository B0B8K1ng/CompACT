#!/usr/bin/env python3
"""Build and validate four-dataset LatentPT-compatible local action caches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from precompute_navanywhere_nav1_latent_actions import (
    _lam_api, atomic_json_dump, atomic_torch_save, encode_pair_batches,
    safe_torch_load, sha256_file, validate_checkpoint_metadata,
)

LAM_ROOT = REPO.parent / 'DreamDojo/external/lam_project'
DATASETS = {'recon': 'recon', 'sacson': 'sacson', 'scand': 'scand', 'tartan_drive': 'tartan'}


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def pairs_for(n):
    current = np.arange(3, max(3, n - 64), dtype=np.int64)
    targets = current[:, None] + np.arange(-8, 9, dtype=np.int64)[None, :]
    currents = np.broadcast_to(current[:, None], targets.shape)
    valid = (targets >= 0) & (targets < n)
    return np.stack((currents[valid], targets[valid]), axis=1)


def output_path(root, task):
    return root / task['dataset'] / (task['trajectory'] + '.pt')


def plan(args):
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    inspect, _, _ = _lam_api(args.lam_root)
    metadata = inspect(args.checkpoint)
    validate_checkpoint_metadata(metadata, image_height=240, image_width=320,
                                 max_abs_frame_offset=8)
    digest = sha256_file(args.checkpoint)
    reference = {
        'path': str(args.checkpoint.resolve()),
        'sha256': digest,
        'size_bytes': args.checkpoint.stat().st_size,
        'class_path': metadata['class_path'],
        'global_step': int(metadata['global_step']),
        'epoch': int(metadata['epoch']),
        'latent_dim': int(metadata['hparams']['lam_latent_dim']),
        'patch_size': int(metadata['hparams']['lam_patch_size']),
        'data_contract': {
            'image_height': int(metadata['datamodule_hparams']['image_height']),
            'image_width': int(metadata['datamodule_hparams']['image_width']),
            'max_frame_offset': int(metadata['datamodule_hparams']['max_frame_offset']),
        },
    }
    log('Verified checkpoint SHA256=' + digest)
    jobs = []
    split_hashes = {}
    for dataset, folder in DATASETS.items():
        names = {}
        for split in ('train', 'test'):
            path = args.split_root / dataset / split / 'traj_names.txt'
            split_hashes[f'{dataset}/{split}'] = sha256_file(path)
            for name in path.read_text().splitlines():
                name = name.strip()
                if not name:
                    continue
                if Path(name).name != name or name in ('.', '..'):
                    raise ValueError('Unsafe trajectory ' + name)
                names.setdefault(name, []).append(split)
        for name, splits in names.items():
            jobs.append({'dataset': dataset, 'trajectory': name, 'splits': splits,
                         'directory': str(args.data_root / folder / name)})

    def inspect(task):
        path = Path(task['directory']) / 'traj_data.pkl'
        if not path.is_file():
            return dict(task, missing=True)
        raw = path.read_bytes()
        data = pickle.loads(raw)
        n = len(data['position'])
        return dict(task, frames=n, pairs=len(pairs_for(n)),
                    trajectory_data_sha256=hashlib.sha256(raw).hexdigest())

    with ThreadPoolExecutor(max_workers=32) as pool:
        records = list(pool.map(inspect, jobs))
    missing = [r for r in records if r.get('missing')]
    tasks = [r for r in records if not r.get('missing')]
    counts = {d: {'trajectories': 0, 'frames': 0, 'pairs': 0, 'empty': 0} for d in DATASETS}
    for task in tasks:
        stats = counts[task['dataset']]
        for key, value in [('trajectories', 1), ('frames', task['frames']),
                           ('pairs', task['pairs']), ('empty', int(task['pairs'] == 0))]:
            stats[key] += value
    contract = {'checkpoint': reference, 'split_sha256': split_hashes,
                'pair_direction': 'current_to_goal', 'context_size': 4, 'len_traj_pred': 64,
                'max_abs_frame_offset': 8, 'latent_dim': 32, 'latent_value': 'z_mu',
                'normalization': 'raw', 'precision': 'bf16-mixed',
                'image_preprocessing': 'center_crop_4:3_then_bilinear_resize',
                'image_height': 240, 'image_width': 320}
    identity = hashlib.sha256(json.dumps({'contract': contract, 'tasks': tasks}, sort_keys=True).encode()).hexdigest()
    value = {'contract': contract, 'identity': identity, 'tasks': tasks,
             'missing_source_trajectories': missing, 'counts': counts}
    atomic_json_dump(value, root / 'plan.json')
    log(json.dumps({'counts': counts, 'missing_source_trajectories': len(missing)}))


def validate_payload(path, task, plan):
    payload = safe_torch_load(path)
    assert payload['complete'] is True and payload['plan_identity'] == plan['identity'], str(path)
    assert payload['dataset_name'] == task['dataset'] and payload['trajectory_name'] == task['trajectory']
    assert payload['checkpoint_sha256'] == plan['contract']['checkpoint']['sha256']
    pairs = torch.from_numpy(pairs_for(task['frames']))
    assert torch.equal(payload['frame_pairs'], pairs), str(path)
    motion = payload['motion']
    assert motion.shape == (len(pairs), 32) and motion.dtype == torch.float32
    assert torch.isfinite(motion).all(), str(path)
    return payload


def load_adapter(args):
    _, load, load_frame = _lam_api(args.lam_root)
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    # Restore strictly on CPU, then discard the pixel/action decoders before
    # moving to CUDA. Extraction only calls lam.encode(); pruning the unused
    # half of PixelActionLAM leaves z_mu bit-identical while reserving much
    # more memory for the largest safe inference batch.
    adapter = load(args.checkpoint, device=torch.device('cpu'), fused_attention=True)
    validate_checkpoint_metadata(adapter.checkpoint_metadata, image_height=240,
                                 image_width=320, max_abs_frame_offset=8)
    for name in ('decoder', 'patch_up', 'action_up'):
        if hasattr(adapter.model.lam, name):
            delattr(adapter.model.lam, name)
    if hasattr(adapter.model, 'action_decoder'):
        delattr(adapter.model, 'action_decoder')
    device = torch.device('cuda', gpu)
    adapter.model.to(device=device)
    adapter.device = device
    adapter.dtype = next(adapter.model.parameters()).dtype
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.compile_encoder:
        # RotaryEmbedding mutates a tiny sequence-length cache in eager mode.
        # The temporal length is only two here, so recomputing those frequencies
        # is negligible and removes graph-breaking scalar/item and copy_ calls.
        for module in adapter.model.lam.encoder.modules():
            if hasattr(module, 'cache_if_possible'):
                module.cache_if_possible = False
        adapter.model.lam.encoder = torch.compile(
            adapter.model.lam.encoder, dynamic=True, fullgraph=False
        )
    return adapter, load_frame


def benchmark(args):
    adapter, _ = load_adapter(args)
    for batch in [int(value) for value in args.batch_sizes.split(',')]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        frames = torch.rand(24, 240, 320, 3)
        base_pairs = np.array([[3, 4], [4, 3], [3, 3]], dtype=np.int64)
        warmup_pairs = np.resize(base_pairs, (batch, 2))
        pairs = np.resize(base_pairs, (batch * args.benchmark_steps, 2))
        try:
            encode_pair_batches(adapter, frames, warmup_pairs, batch_size=batch,
                                precision='bf16-mixed')
            torch.cuda.synchronize()
            start = time.monotonic()
            out = encode_pair_batches(adapter, frames, pairs, batch_size=batch, precision='bf16-mixed')
            torch.cuda.synchronize()
            log(f'benchmark batch={batch} status=ok pairs_per_sec={len(out)/(time.monotonic()-start):.1f} '
                f'peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}')
        except torch.cuda.OutOfMemoryError:
            log(f'benchmark batch={batch} status=oom')
            torch.cuda.empty_cache()
            break


def worker(args):
    plan_data = json.loads((args.output_root / 'plan.json').read_text())
    planned_checkpoint = Path(plan_data['contract']['checkpoint']['path']).resolve()
    if args.checkpoint != planned_checkpoint:
        raise RuntimeError(
            f'Worker checkpoint {args.checkpoint} differs from plan checkpoint '
            f'{planned_checkpoint}'
        )
    # Shared filesystem atomic claims balance differently loaded GPUs dynamically.
    claim_root = args.output_root / '_claims'
    claim_root.mkdir(exist_ok=True)
    tasks = sorted(plan_data['tasks'], key=lambda t: (-t['pairs'], t['dataset'], t['trajectory']))
    adapter, load_frame = load_adapter(args)
    if args.compile_encoder:
        compile_started = time.monotonic()
        warmup_frames = torch.zeros((2, 240, 320, 3), dtype=torch.float32)
        warmup_pairs = np.zeros((args.batch_size, 2), dtype=np.int64)
        encode_pair_batches(adapter, warmup_frames, warmup_pairs,
                            batch_size=args.batch_size, precision='bf16-mixed')
        torch.cuda.synchronize(adapter.device)
        del warmup_frames
        log(f'encoder compile/warmup seconds={time.monotonic()-compile_started:.1f}')
    start = time.monotonic()
    total = 0
    completed_tasks = 0
    total_read_wait_seconds = 0.0
    total_encode_seconds = 0.0
    failures = []
    with ThreadPoolExecutor(max_workers=args.loader_threads) as readers:
        for task in tasks:
            path = output_path(args.output_root, task)
            if path.exists():
                continue
            claim = claim_root / (task['dataset'] + '__' + task['trajectory'])
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            try:
                task_started = time.monotonic()
                log(f"start {task['dataset']}/{task['trajectory']} pairs={task['pairs']}")
                pairs = pairs_for(task['frames'])
                results = []
                # Small adjacent chunks bound memory and reuse each decoded frame
                # across its local pairs. No corrupted-frame substitutions.
                chunk_size = 4096
                chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]
                def prepare(pair_chunk):
                    indices, inverse = np.unique(pair_chunk, return_inverse=True)
                    def read(index):
                        return load_frame(Path(task['directory']) / f'{index}.jpg',
                                          image_height=240, image_width=320)
                    frames = torch.stack(list(readers.map(read, indices))).contiguous()
                    return frames, inverse.reshape(-1, 2)
                with ThreadPoolExecutor(max_workers=1) as prefetch:
                    future = prefetch.submit(prepare, chunks[0]) if chunks else None
                    for i in range(len(chunks)):
                        read_wait_started = time.monotonic()
                        frames, positions = future.result()
                        total_read_wait_seconds += time.monotonic() - read_wait_started
                        future = prefetch.submit(prepare, chunks[i+1]) if i+1 < len(chunks) else None
                        encode_started = time.monotonic()
                        results.append(encode_pair_batches(adapter, frames, positions,
                                       batch_size=args.batch_size, precision='bf16-mixed',
                                       gpu_frame_bank_limit_bytes=1024**3))
                        total_encode_seconds += time.monotonic() - encode_started
                motion = torch.cat(results) if results else torch.empty((0, 32), dtype=torch.float32)
                if motion.shape != (len(pairs), 32) or not torch.isfinite(motion).all():
                    raise RuntimeError('Invalid extracted motion')
                payload = {'complete': True, 'schema_version': 1, 'format': 'compact_offline_lam_action',
                           'dataset_name': task['dataset'], 'trajectory_name': task['trajectory'],
                           'source_id': task['dataset'], 'trajectory_id': task['trajectory'],
                           'proxy_type': 'latent', 'motion_type': 'latent', 'latent_dim': 32,
                           'pair_direction': 'current_to_goal', 'normalization': 'raw', 'latent_value': 'z_mu',
                           'plan_identity': plan_data['identity'],
                           'checkpoint_sha256': plan_data['contract']['checkpoint']['sha256'],
                           'frame_pairs': torch.from_numpy(pairs), 'motion': motion}
                atomic_torch_save(payload, path)
                total += len(pairs)
                completed_tasks += 1
                log(f"done {task['dataset']}/{task['trajectory']} pairs={len(pairs)} "
                    f'task_seconds={time.monotonic()-task_started:.1f} '
                    f'worker_pairs={total} pairs_per_sec={total/(time.monotonic()-start):.1f}')
            except Exception as exc:
                failures.append({'dataset': task['dataset'], 'trajectory': task['trajectory'], 'error': repr(exc)})
                log('ERROR ' + repr(failures[-1]))
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                claim.rmdir()
            if args.max_tasks and completed_tasks >= args.max_tasks:
                break
    elapsed = time.monotonic() - start
    worker_id = args.worker_id or str(args.gpu)
    if not worker_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError(f'Unsafe worker id: {worker_id!r}')
    atomic_json_dump({'gpu': args.gpu, 'worker_id': worker_id,
                      'batch_size': args.batch_size,
                      'loader_threads': args.loader_threads,
                      'completed_tasks': completed_tasks, 'pairs': total,
                      'elapsed_seconds': elapsed,
                      'pairs_per_second': total / elapsed if elapsed else 0.0,
                      'read_wait_seconds': total_read_wait_seconds,
                      'encode_seconds': total_encode_seconds,
                      'failures': failures},
                     args.output_root / f'worker_{worker_id}.json')
    if failures:
        raise RuntimeError(f'{len(failures)} trajectory failures')


def validate(args):
    from two_stage_data import OfflineProxyStore
    plan_data = json.loads((args.output_root / 'plan.json').read_text())
    store = OfflineProxyStore(root=args.output_root, proxy_type='latent', dim=32, strict_loading=True)
    def inspect(task):
        path = output_path(args.output_root, task)
        validate_payload(path, task, plan_data)
        return {'dataset': task['dataset'], 'trajectory': task['trajectory'],
                'pairs': task['pairs'], 'sha256': sha256_file(path), 'bytes': path.stat().st_size}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(inspect, plan_data['tasks']))
    checked = set()
    for task in plan_data['tasks']:
        d = task['dataset']
        if task['pairs'] and d not in checked:
            payload = validate_payload(output_path(args.output_root, task), task, plan_data)
            for index in (0, len(payload['frame_pairs'])//2, len(payload['frame_pairs'])-1):
                current, target = payload['frame_pairs'][index].tolist()
                result = store.lookup(d, task['trajectory'], current, target)
                assert result.valid and torch.equal(result.proxy_action, payload['motion'][index])
            checked.add(d)
    assert checked == set(DATASETS)
    result = {'complete': True, 'status': 'complete', 'contract': plan_data['contract'],
              'plan_identity': plan_data['identity'], 'counts': plan_data['counts'],
              'missing_source_trajectories': plan_data['missing_source_trajectories'],
              'records': records, 'strict_loader_datasets_checked': sorted(checked)}
    atomic_json_dump(result, args.output_root / 'metadata.json')
    atomic_json_dump({'complete': True, 'trajectories': len(records),
                      'pairs': sum(r['pairs'] for r in records),
                      'metadata_sha256': sha256_file(args.output_root / 'metadata.json')},
                     args.output_root / '_SUCCESS.json')
    log('Validation complete: ' + str(args.output_root / '_SUCCESS.json'))


def run(args):
    if not (args.output_root / 'plan.json').exists():
        plan(args)
    log('cwd=' + str(REPO))
    logs = args.output_root / 'logs'
    logs.mkdir(exist_ok=True)
    # A SIGKILL can leave empty atomic-claim directories behind. No workers
    # exist yet at this point, so clearing only those empty directories makes
    # interrupted runs safely resumable without touching completed payloads.
    claim_root = args.output_root / '_claims'
    if claim_root.is_dir():
        for claim in claim_root.iterdir():
            if claim.is_dir():
                claim.rmdir()
    processes = []
    import shlex
    capacity = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
    free_mib = {int(row.split(',')[0]): int(row.split(',')[1]) for row in capacity.splitlines()}
    for gpu in args.gpus.split(','):
        if free_mib[int(gpu)] < 9000:
            raise RuntimeError(f'GPU {gpu} has insufficient headroom: {free_mib[int(gpu)]} MiB')
    log('GPU free MiB=' + str(free_mib))
    for gpu in args.gpus.split(','):
        for slot in range(args.workers_per_gpu):
            worker_id = f'{gpu}-{slot}' if args.workers_per_gpu > 1 else gpu
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                       '--output-root', str(args.output_root), '--checkpoint', str(args.checkpoint),
                       '--data-root', str(args.data_root), '--split-root', str(args.split_root),
                       '--lam-root', str(args.lam_root), '--gpu', gpu, '--worker-id', worker_id,
                       '--batch-size', str(args.batch_size), '--loader-threads', str(args.loader_threads)]
            if args.compile_encoder:
                command.append('--compile-encoder')
            logfile = logs / f'gpu_{gpu}_worker_{slot}.log'
            with logfile.open('a') as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO)
            log(f'PID={process.pid} log={logfile} command={shlex.join(command)}')
            processes.append(process)
    while any(p.poll() is None for p in processes):
        time.sleep(30)
        log('Worker states=' + str({p.pid: p.poll() for p in processes}))
    statuses = [p.wait() for p in processes]
    atomic_json_dump({'pids': [p.pid for p in processes], 'exit_statuses': statuses},
                     args.output_root / 'process_status.json')
    log('Worker exit statuses=' + str(statuses))
    if any(statuses):
        raise RuntimeError('Extraction failed; see per-GPU logs')
    validate(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'benchmark', 'worker', 'run', 'validate'])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/data1/ndj/LAM-Data'))
    parser.add_argument('--split-root', type=Path, default=REPO / 'data_splits')
    parser.add_argument('--lam-root', type=Path, default=LAM_ROOT)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--worker-id', default='')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-sizes', default='16,24,32,40,48,64,80,96,128')
    parser.add_argument('--benchmark-steps', type=int, default=2)
    parser.add_argument('--compile-encoder', action='store_true')
    parser.add_argument('--loader-threads', type=int, default=16)
    parser.add_argument('--max-tasks', type=int, default=0,
                        help='Stop a worker after N completed trajectories; 0 runs to exhaustion.')
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        parser.error('--workers-per-gpu must be positive')
    args.output_root = args.output_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.split_root = args.split_root.expanduser().resolve()
    args.lam_root = args.lam_root.expanduser().resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
