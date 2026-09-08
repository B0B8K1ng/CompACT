#!/usr/bin/env python3
"""Audit every latent shard against the immutable recipe and training pair plan.

This checks cache contents, not current raw JPEG sizes/mtimes. It deliberately
avoids rescanning millions of immutable input images or initializing NCCL.
Only a fully valid inventory may receive completion markers.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import fcntl
import json
import multiprocessing as mp
import os
import stat
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import precompute_navanywhere_nav1_latent_actions as pre
from navanywhere_recipe import frame_indices_sha256

STATE = None
BITMAP = None


def share_group_read(path):
    info = path.stat()
    if info.st_uid == os.geteuid() and not info.st_mode & stat.S_IRGRP:
        path.chmod(stat.S_IMODE(info.st_mode) | stat.S_IRGRP)


def check_task(task):
    started = time.monotonic()
    path = Path(STATE['output_root']) / task['source_id'] / (task['trajectory_id'] + '.pt')
    identity = dict(source_id=task['source_id'], trajectory_id=task['trajectory_id'])
    try:
        before = path.stat()
        payload = pre.safe_torch_load(path)
        indices = payload['frame_indices'].numpy()
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise ValueError('invalid frame index tensor')
        if (len(indices) != task['frame_count']
                or frame_indices_sha256(indices) != task['frame_indices_sha256']):
            raise ValueError('frame inventory does not match immutable recipe')
        source_fp = payload['metadata']['source_fingerprint']
        if len(source_fp) != 64 or any(c not in '0123456789abcdef' for c in source_fp):
            raise ValueError('invalid recorded source fingerprint')
        record = pre._cache_record(path, STATE, task, indices, source_fp, BITMAP, payload=payload)
        share_group_read(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('cache changed during audit')
        return dict(ok=True, record=record, file_mtime_ns=after.st_mtime_ns,
                    elapsed_seconds=round(time.monotonic()-started, 3))
    except Exception as exc:
        return dict(ok=False, **identity, error=f'{type(exc).__name__}: {exc}')


def check_unchanged(item):
    record = item['record']
    path = Path(STATE['output_root']) / record['source_id'] / (record['trajectory_id'] + '.pt')
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (record['file_size_bytes'], item['file_mtime_ns']):
        raise ValueError(f'Validated file changed since audit: {path}')
    share_group_read(path)


def install_repairs(report_dir, staging):
    """Reuse the full audit, verifying unchanged files and all repaired contents."""
    global STATE, BITMAP
    STATE = json.loads((report_dir / 'state.json').read_text())
    if pre.sha256_file(STATE['sampling_recipe_path']) != STATE['sampling_recipe_sha256']:
        raise ValueError('Sampling recipe changed after audit')
    plan, BITMAP = pre._load_training_pair_plan(
        STATE['training_pair_plan']['path'],
        recipe_sha256=STATE['sampling_recipe_sha256'],
        max_abs_frame_offset=STATE['policy']['configuration']['max_abs_frame_offset'])
    if plan != STATE['training_pair_plan']:
        raise ValueError('Training plan changed after audit')
    items = [json.loads(line) for line in (report_dir/'validated.jsonl').read_text().splitlines()]
    problems = json.loads((report_dir/'problems.json').read_text())['problems']
    root = Path(STATE['output_root'])
    with (root / '.precompute.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'Verifying {len(items)} audited file sizes/mtimes', flush=True)
        with futures.ThreadPoolExecutor(max_workers=32) as pool:
            for _ in pool.map(check_unchanged, items):
                pass
        tasks = {(t['source_id'], t['trajectory_id']): t for t in STATE['tasks']}
        records = [item['record'] for item in items]
        for problem in problems:
            key = (problem['source_id'], problem['trajectory_id'])
            task = tasks[key]
            staged = Path(staging) / key[0] / (key[1]+'.pt')
            target = root / key[0] / (key[1]+'.pt')
            target.parent.mkdir(parents=True, exist_ok=True)
            # A repeated install can resume after already moved repairs.
            if staged.exists():
                payload = pre.safe_torch_load(staged)
                indices = payload['frame_indices'].numpy()
                if frame_indices_sha256(indices) != task['frame_indices_sha256']:
                    raise ValueError(f'Repair frame inventory mismatch: {staged}')
                pre._cache_record(staged, STATE, task, indices,
                                  payload['metadata']['source_fingerprint'], BITMAP, payload=payload)
                os.chmod(staged, stat.S_IMODE(staged.stat().st_mode) | stat.S_IRGRP)
                os.replace(staged, target)
            result = check_task(task)
            if not result['ok']:
                raise ValueError(f'Invalid repair: {result}')
            records.append(result['record'])
            print(f'Installed and validated {key[0]}/{key[1]}', flush=True)
        keys = [(r['source_id'], r['trajectory_id']) for r in records]
        if len(set(keys)) != len(keys) or set(keys) != set(tasks):
            raise ValueError('Completed cache does not cover exactly the recipe')
        if sum(r['planned_pair_count'] for r in records) != plan['unique_local_pairs']:
            raise ValueError('Completed cache does not cover every planned pair')
        pre._write_completion(STATE, records, partial=False)
        summary = dict(complete=True, validated_files=len(records), repaired_files=len(problems),
                       planned_pairs=sum(r['planned_pair_count'] for r in records),
                       stored_pairs=sum(r['pair_count'] for r in records),
                       invalid_frame_substitutions=sum(len(r['invalid_frame_substitutions']) for r in records),
                       raw_source_files_rescanned=False,
                       completed_at_utc=pre.utc_now())
        pre.atomic_json_dump(summary, report_dir/'final_summary.json')
        print(json.dumps(summary, sort_keys=True), flush=True)


def recheck_problems(report_dir, workers):
    global STATE, BITMAP
    STATE = json.loads((report_dir/'state.json').read_text())
    BITMAP = np.memmap(STATE['training_pair_plan']['pair_bitmap_path'], dtype=np.uint8, mode='r')
    items = [json.loads(line) for line in (report_dir/'validated.jsonl').read_text().splitlines()]
    by_key = {(item['record']['source_id'], item['record']['trajectory_id']): item for item in items}
    bad = []
    started = time.monotonic()
    with (Path(STATE['output_root'])/'.precompute.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def still_valid(item):
            try:
                check_unchanged(item)
                return True
            except OSError:
                return False
            except ValueError:
                return False
        print(f'Checking size/mtime of {len(items)} previously validated files', flush=True)
        with futures.ThreadPoolExecutor(max_workers=32) as pool:
            for item, valid in zip(items, pool.map(still_valid, items)):
                if not valid:
                    by_key.pop((item['record']['source_id'], item['record']['trajectory_id']), None)
        tasks = [t for t in STATE['tasks'] if (t['source_id'], t['trajectory_id']) not in by_key]
        print(f'Rechecking {len(tasks)} previously inaccessible/missing files', flush=True)
        temporary = report_dir/f'.validated.{os.getpid()}.jsonl'
        with temporary.open('w') as out:
            for item in by_key.values():
                out.write(json.dumps(item, ensure_ascii=False)+'\n')
            with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
                pending = {pool.submit(check_task, task): task for task in tasks}
                processed = 0
                while pending:
                    done, _ = futures.wait(pending, timeout=15, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        task = pending.pop(future)
                        result = future.result()
                        processed += 1
                        if result['ok']:
                            out.write(json.dumps(result, ensure_ascii=False)+'\n')
                            by_key[(task['source_id'], task['trajectory_id'])] = result
                        else:
                            bad.append(result)
                    if not done or processed//1000 != (processed-len(done))//1000 or not pending:
                        out.flush()
                        print(f'rechecked={processed}/{len(tasks)} total_valid={len(by_key)} '
                              f'problems={len(bad)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        temporary.chmod(0o640)
        os.replace(temporary, report_dir/'validated.jsonl')
        pre.atomic_json_dump(dict(problems=bad), report_dir/'problems.json')
        summary = dict(complete=not bad, validated_files=len(by_key), problems=len(bad),
                       planned_pairs=sum(i['record']['planned_pair_count'] for i in by_key.values()),
                       expected_planned_pairs=STATE['training_pair_plan']['unique_local_pairs'],
                       raw_source_files_rescanned=False, elapsed_seconds=time.monotonic()-started)
        pre.atomic_json_dump(summary, report_dir/'summary.json')
        print(json.dumps(summary, sort_keys=True), flush=True)
        if bad:
            raise SystemExit(2)


def main():
    global STATE, BITMAP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--write-completion', action='store_true')
    parser.add_argument('--install-repairs', metavar='STAGING_ROOT')
    parser.add_argument('--recheck-problems', action='store_true')
    args, extraction_args = parser.parse_known_args()
    torch.set_num_threads(1)
    if args.recheck_problems:
        if extraction_args or args.install_repairs:
            parser.error('problem recheck uses only the recorded audit state')
        recheck_problems(Path(args.report_dir), args.workers)
        return
    if args.install_repairs:
        if extraction_args:
            parser.error('repair installation uses the recorded audit state')
        install_repairs(Path(args.report_dir), args.install_repairs)
        return
    sys.argv = [sys.argv[0], *extraction_args]
    extraction = pre.parse_args()
    if extraction.trajectory or extraction.max_trajectories:
        parser.error('finalization requires the entire recipe')
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (Path(extraction.output_root) / '.precompute.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Building validation state (CPU only)', flush=True)
        STATE = pre._build_state(extraction, 1)
        pre.atomic_json_dump(STATE, report_dir / 'state.json')
        BITMAP = np.memmap(STATE['training_pair_plan']['pair_bitmap_path'], dtype=np.uint8, mode='r')
        tasks = STATE['tasks']
        good, bad = [], []
        print(f'Checking {len(tasks)} files with {args.workers} CPU workers', flush=True)
        with (report_dir / 'validated.jsonl').open('w') as out:
            with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
                pending = {pool.submit(check_task, task): task for task in tasks}
                while pending:
                    done, _ = futures.wait(pending, timeout=15, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        task = pending.pop(future)
                        result = future.result()
                        if result['ok']:
                            good.append(result['record'])
                            out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        else:
                            bad.append(result)
                            print('PROBLEM ' + json.dumps(result, ensure_ascii=False), flush=True)
                    checked = len(good) + len(bad)
                    if not done or checked // 1000 != (checked-len(done)) // 1000 or not pending:
                        out.flush()
                        print(f'checked={checked}/{len(tasks)} valid={len(good)} problems={len(bad)} '
                              f'elapsed={time.monotonic()-started:.1f}s pending={len(pending)}', flush=True)
                        if len(pending) <= 20:
                            print('pending identities=' + json.dumps([
                                f"{t['source_id']}/{t['trajectory_id']}" for t in pending.values()]), flush=True)
        pre.atomic_json_dump(dict(problems=bad), report_dir / 'problems.json')
        total_pairs = sum(r['planned_pair_count'] for r in good)
        expected_pairs = STATE['training_pair_plan']['unique_local_pairs']
        complete = not bad and len(good) == len(tasks) and total_pairs == expected_pairs
        summary = dict(complete=complete, validated_files=len(good), problems=len(bad),
                       planned_pairs=total_pairs, expected_planned_pairs=expected_pairs,
                       stored_pairs=sum(r['pair_count'] for r in good),
                       full_domain_superset_files=sum(r['pair_coverage']=='full_domain_superset' for r in good),
                       invalid_frame_substitutions=sum(len(r['invalid_frame_substitutions']) for r in good),
                       raw_source_files_rescanned=False, elapsed_seconds=time.monotonic()-started)
        pre.atomic_json_dump(summary, report_dir / 'summary.json')
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.write_completion and complete:
            pre._write_completion(STATE, good, partial=False)
            print('Completion markers written', flush=True)
        if not complete:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
