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

NAS = Path('/file_system/nas/algorithm/dujun.nie/nwm')
DEFAULT_OUTPUT = NAS / 'compact/cache/finetune_nav1_pixel_action_step100000_four_datasets'
CHECKPOINT = NAS / 'weights/navigation_lam/variant_4_pixel_action/nav1-pixel-action/checkpoints/step=100000.ckpt'
REFERENCE = NAS / 'compact/cache/navanywhere_nav1_pixel_action_step100000/metadata.json'
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
    reference = json.loads(REFERENCE.read_text())['checkpoint']
    assert str(CHECKPOINT) == reference['path']
    digest = sha256_file(CHECKPOINT)
    if digest != reference['sha256']:
        raise RuntimeError('Navigation LAM checkpoint differs from LatentPT cache provenance')
    log('Verified checkpoint SHA256=' + digest)
    jobs = []
    split_hashes = {}
    for dataset, folder in DATASETS.items():
        names = {}
        for split in ('train', 'test'):
            path = REPO / 'data_splits' / dataset / split / 'traj_names.txt'
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
                         'directory': str(NAS / 'data' / folder / name)})

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


def load_adapter(gpu):
    _, load, load_frame = _lam_api(LAM_ROOT)
    torch.cuda.set_device(gpu)
    adapter = load(CHECKPOINT, device=torch.device('cuda', gpu), fused_attention=True)
    validate_checkpoint_metadata(adapter.checkpoint_metadata, image_height=240,
                                 image_width=320, max_abs_frame_offset=8)
    return adapter, load_frame


def benchmark(args):
    adapter, _ = load_adapter(args.gpu)
    for batch in (16, 32, 64):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        frames = torch.rand(24, 240, 320, 3)
        pairs = np.tile(np.array([[3, 4], [4, 3], [3, 3]], dtype=np.int64), (batch * 2, 1))
        start = time.monotonic()
        out = encode_pair_batches(adapter, frames, pairs, batch_size=batch, precision='bf16-mixed')
        torch.cuda.synchronize()
        log(f'benchmark batch={batch} pairs_per_sec={len(out)/(time.monotonic()-start):.1f} '
            f'peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}')


def worker(args):
    plan_data = json.loads((args.output_root / 'plan.json').read_text())
    # Shared filesystem atomic claims balance differently loaded GPUs dynamically.
    claim_root = args.output_root / '_claims'
    claim_root.mkdir(exist_ok=True)
    tasks = sorted(plan_data['tasks'], key=lambda t: (-t['pairs'], t['dataset'], t['trajectory']))
    adapter, load_frame = load_adapter(args.gpu)
    start = time.monotonic()
    total = 0
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
                        frames, positions = future.result()
                        future = prefetch.submit(prepare, chunks[i+1]) if i+1 < len(chunks) else None
                        results.append(encode_pair_batches(adapter, frames, positions,
                                       batch_size=args.batch_size, precision='bf16-mixed',
                                       gpu_frame_bank_limit_bytes=1024**3))
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
                log(f"done {task['dataset']}/{task['trajectory']} pairs={len(pairs)} "
                    f'worker_pairs={total} pairs_per_sec={total/(time.monotonic()-start):.1f}')
            except Exception as exc:
                failures.append({'dataset': task['dataset'], 'trajectory': task['trajectory'], 'error': repr(exc)})
                log('ERROR ' + repr(failures[-1]))
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                claim.rmdir()
    atomic_json_dump({'gpu': args.gpu, 'pairs': total, 'failures': failures},
                     args.output_root / f'worker_{args.gpu}.json')
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
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                   '--output-root', str(args.output_root), '--gpu', gpu,
                   '--batch-size', str(args.batch_size), '--loader-threads', str(args.loader_threads)]
        logfile = logs / f'gpu_{gpu}.log'
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
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', default='0,1,5,6')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--loader-threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
