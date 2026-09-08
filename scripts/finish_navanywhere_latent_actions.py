#!/usr/bin/env python3
"""Fast CPU audit, resumable GPU repair, and verified finalization of nav1 caches."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run(script, *args, allowed=(0,)):
    command = [sys.executable, '-u', str(REPO/'scripts'/script), *map(str, args)]
    print('+ '+shlex.join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode not in allowed:
        raise RuntimeError(f'{script} exited {result.returncode}')
    return result.returncode


def check_gpu_capacity(devices, minimum):
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    print('GPU index, free MiB, utilization %:\n'+result, flush=True)
    rows = {int(row.split(',')[0]): int(row.split(',')[1]) for row in result.splitlines()}
    for device in devices:
        if rows.get(device, -1) < minimum:
            raise RuntimeError(f'GPU {device} needs at least {minimum} MiB free; '
                               'choose GPU_IDS or retry after memory becomes available')


def check_running_repairs():
    # Older repair workers do not hold the newer pipeline lock. Never launch
    # duplicate model copies on top of them or stop another experiment here.
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            argv = (directory/'cmdline').read_bytes().split(b'\0')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        scripts = {Path(os.fsdecode(arg)).name for arg in argv if arg}
        if scripts & {'repair_navanywhere_latent_actions.py', 'repair_navanywhere_latent_chunks.py'}:
            raise RuntimeError(f'Existing latent repair PID {directory.name} is still running. '
                               'Finish or explicitly stop the earlier repair before launching this pipeline.')


def main():
    import precompute_navanywhere_nav1_latent_actions as pre
    base = Path(os.environ.get('COMPACT_NAS_ROOT', '/file_system/nas/algorithm/dujun.nie/nwm/compact'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', default=os.environ.get('OUTPUT_ROOT', str(base/'cache/navanywhere_nav1_pixel_action_step100000')))
    p.add_argument('--report-dir', default=os.environ.get('REPORT_DIR', str(base/'logs/navanywhere_nav1_latent_actions/finalize_20260908')))
    p.add_argument('--staging-root', default=os.environ.get('STAGING_ROOT', str(base/'cache/navanywhere_nav1_pixel_action_step100000_repairs_20260908')))
    p.add_argument('--chunk-root', default=os.environ.get('CHUNK_ROOT', str(base/'cache/navanywhere_nav1_pixel_action_step100000_chunks_20260908')))
    p.add_argument('--data-root', default=os.environ.get('NAVANYWHERE_ROOT', '/file_system/nas/algorithm/dujun.nie/nwm/data/NavAnywhere'))
    p.add_argument('--sampling-recipe', default=os.environ.get('SAMPLING_RECIPE', str(base/'recipes/navanywhere_balanced_seed20260901.json')))
    p.add_argument('--training-pair-plan', default=os.environ.get('TRAINING_PAIR_PLAN', str(base/'plans/navanywhere_latent_action_seed20260901_ws8_bs16_steps200000.json')))
    p.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', '/file_system/nas/algorithm/dujun.nie/nwm/weights/navigation_lam/variant_4_pixel_action/nav1-pixel-action/checkpoints/step=100000.ckpt'))
    p.add_argument('--lam-project-root', default=os.environ.get('LAM_PROJECT_ROOT', str(REPO.parent/'DreamDojo/external/lam_project')))
    p.add_argument('--workers', type=int, default=int(os.environ.get('AUDIT_WORKERS', '24')))
    p.add_argument('--devices', default=os.environ.get('GPU_IDS', '0,1,2,3,4,5,6,7'))
    p.add_argument('--min-free-gpu-mib', type=int, default=int(os.environ.get('MIN_FREE_GPU_MB', '6000')))
    p.add_argument('--fresh-audit', action='store_true', help='Re-read all caches instead of resuming the saved audit')
    args = p.parse_args()
    devices = [int(value) for value in args.devices.split(',')]
    if not devices or len(set(devices)) != len(devices) or args.workers < 1:
        p.error('GPU IDs must be unique and worker count positive')
    check_running_repairs()
    root, report = Path(args.output_root).resolve(), Path(args.report_dir).resolve()
    if not root.is_dir():
        p.error(f'Existing extraction cache not found: {root}')
    usage = shutil.disk_usage(root)
    print(f'Cache filesystem free: {usage.free/2**30:.1f} GiB', flush=True)
    if usage.free < 2*2**30:
        raise RuntimeError('At least 2 GiB free is required for repair staging')
    with (root/'.finish.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.fstat(lock.fileno()).st_uid == os.geteuid():
            os.fchmod(lock.fileno(), 0o660)
        recipe = json.loads(Path(args.sampling_recipe).read_text())
        if root.stat().st_uid == os.geteuid():
            root.chmod(stat.S_IMODE(root.stat().st_mode) | 0o070)
        # Probe every source directory before doing expensive work; this is the
        # permission failure encountered in the interrupted root-owned cache.
        for source in sorted({task['source_id'] for task in recipe['trajectories']}):
            directory = root/source
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory):
                pass
            mode = stat.S_IMODE(directory.stat().st_mode)
            if directory.stat().st_uid == os.geteuid():
                directory.chmod(mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)
        report.mkdir(parents=True, exist_ok=True)
        state_path = report/'state.json'
        saved_audit = all((report/name).exists() for name in ('state.json', 'validated.jsonl', 'problems.json'))
        if saved_audit and not args.fresh_audit:
            state = json.loads(state_path.read_text())
            for key, value in [('output_root', root), ('data_root', args.data_root),
                               ('sampling_recipe_path', args.sampling_recipe),
                               ('lam_project_root', args.lam_project_root)]:
                if Path(state[key]).resolve() != Path(value).resolve():
                    raise ValueError(f'Saved audit {key} differs; use a new REPORT_DIR')
            if (Path(state['checkpoint']['path']).resolve() != Path(args.checkpoint).resolve()
                or Path(state['training_pair_plan']['path']).resolve() != Path(args.training_pair_plan).resolve()):
                raise ValueError('Saved audit checkpoint/plan path differs; use a new REPORT_DIR')
            if pre.sha256_file(args.sampling_recipe) != state['sampling_recipe_sha256']:
                raise ValueError('Sampling recipe changed after the saved audit')
            plan, _ = pre._load_training_pair_plan(args.training_pair_plan,
                recipe_sha256=state['sampling_recipe_sha256'], max_abs_frame_offset=8)
            if plan != state['training_pair_plan']:
                raise ValueError('Training plan changed after the saved audit')
            run('finalize_navanywhere_latent_actions.py', '--report-dir', report,
                '--workers', args.workers, '--recheck-problems', allowed=(0, 2))
        else:
            run('finalize_navanywhere_latent_actions.py', '--report-dir', report,
                '--workers', args.workers, '--data-root', args.data_root,
                '--sampling-recipe', args.sampling_recipe, '--training-pair-plan', args.training_pair_plan,
                '--output-root', root, '--checkpoint', args.checkpoint,
                '--lam-project-root', args.lam_project_root, '--batch-size', '64',
                '--precision', 'bf16-mixed', '--loader-threads', '16', allowed=(0, 2))
            state = json.loads(state_path.read_text())
        problems = json.loads((report/'problems.json').read_text())['problems']
        permission_errors = [item for item in problems if item['error'].startswith('PermissionError:')]
        if permission_errors:
            raise PermissionError(f'{len(permission_errors)} cache files are inaccessible. '
                                  'Run in the environment that owns these files; no GPU repair was started.')
        by_key = {(t['source_id'], t['trajectory_id']): t for t in state['tasks']}
        repairs = [by_key[(item['source_id'], item['trajectory_id'])] for item in problems]
        task_file = report/'repair_tasks.json'
        pre.atomic_json_dump(repairs, task_file)
        print(f'Repair inventory: {len(repairs)} trajectories, '
              f'{sum(t["planned_pair_count"] for t in repairs)} planned pairs', flush=True)
        # Invalidate any stale marker before replacing bad shards.
        if problems:
            (root/'_SUCCESS.json').unlink(missing_ok=True)
            check_gpu_capacity(devices, args.min_free_gpu_mib)
            run('repair_navanywhere_latent_chunks.py', '--state', state_path,
                '--tasks', task_file, '--staging-root', args.staging_root,
                '--chunk-root', args.chunk_root, '--devices', args.devices)
        run('finalize_navanywhere_latent_actions.py', '--report-dir', report,
            '--install-repairs', args.staging_root)
        print(f'COMPLETE: {root/"_SUCCESS.json"}\nReport: {report/"final_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
