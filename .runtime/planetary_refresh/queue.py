import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path('/file_system/vepfs/algorithm/dujun.nie/code/CompACT')
NAV_ROOT = ROOT.with_name('CompACT-eval-huron-fix')
BASE = Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark')
OUT = BASE / 'planetary_refresh_20260922'
NAV = BASE / 'navigation_largebatch_20260921'
PYTHON = '/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python'
MODELS = ['nwm-release', 'nwm-ego4d', 'nwm-latentpt-ft', 'rae-nwm', 'nwm-latentpt-ft-pixel']
command = [PYTHON, 'scripts/run_nwm_benchmark.py', '--models', ','.join(MODELS),
           '--metrics', 'direct', '--datasets', 'planetary_rover', '--gpus', '5',
           '--batch-size', '10', '--metric-batch-size', '10',
           '--benchmark-root', str(OUT), '--shared-benchmark-root', str(OUT)]
env = os.environ.copy()
env.update(PATH=str(Path(PYTHON).parent) + ':' + env.get('PATH', ''),
           PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
state = dict(state='running', cwd=str(ROOT), command=command,
             shell_command=shlex.join(command), queue_pid=os.getpid(),
             log=str(OUT / 'logs/direct.log'), models=MODELS, started_at=time.time())
def save():
    tmp = OUT / 'job.tmp'
    tmp.write_text(json.dumps(state, indent=2)); tmp.replace(OUT / 'job.json')
try:
    print('START', shlex.join(command), flush=True)
    with (OUT / 'logs/direct.log').open('a') as log:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['pid'] = p.pid; save()
        rc = p.wait()
    state['exit_code'] = rc; save()
    if rc:
        raise RuntimeError(f'Planetary evaluation exited with {rc}')
    result = json.loads((OUT / 'benchmark_results.json').read_text())
    for model in MODELS:
        entry = result['models'][model]['results']['direct_prediction']['planetary_rover']['time']
        assert entry['sample_count'] == 10, (model, entry)
    sys.path.insert(0, str(ROOT / 'scripts'))
    import nwm_benchmark_registry as registry
    target = BASE / 'benchmark_results.json'
    with target.with_suffix('.json.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        combined = registry.load_registry(target)
        for model in MODELS:
            combined['models'][model]['results'].setdefault('direct_prediction', {})['planetary_rover'] = result['models'][model]['results']['direct_prediction']['planetary_rover']
        registry.save_registry(target, combined)
        for name in ('benchmark_results.md', 'benchmark_comparison.md'):
            (BASE / name).write_text(registry.render_markdown(combined))
        (BASE / 'ood_direct_4s_table.tex').write_text(registry.render_ood_latex(combined))
    jobs = json.loads((NAV / 'jobs.json').read_text())
    for job in jobs:
        if job['state'] == 'paused' and job.get('pause_reason') == 'user_requested_updated_space_before_navigation':
            job.update(state='pending', resumed_at=time.time())
    tmp = NAV / 'jobs.space_resume.tmp'; tmp.write_text(json.dumps(jobs, indent=2)); tmp.replace(NAV / 'jobs.json')
    with (NAV / 'logs/scheduler.log').open('a') as log:
        scheduler = subprocess.Popen([PYTHON, '.runtime/navlarge/schedule.py'], cwd=NAV_ROOT,
                                    env=env, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    (NAV / 'scheduler.pid').write_text(str(scheduler.pid) + '\n')
    state.update(state='complete_navigation_resumed', navigation_scheduler_pid=scheduler.pid, finished_at=time.time())
    (NAV / 'space_gate.json').write_text(json.dumps(state, indent=2))
    save(); print('COMPLETE exit=0 navigation scheduler PID', scheduler.pid, flush=True)
except Exception as error:
    state.update(state='failed_navigation_paused', error=repr(error), finished_at=time.time())
    save(); print('FAILED', repr(error), flush=True)
    raise
