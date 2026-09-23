#!/usr/bin/env python3
"""Direct, verified downloads; launch only u100l100 after its download."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TRAIN_ENV = os.environ.copy()
for key in list(os.environ):
    if key.lower().endswith('_proxy'):
        os.environ.pop(key)
os.environ['NO_PROXY'] = '*'
os.environ['MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS'] = '16'

from modelscope_hub import HubApi

ROOT = Path('/data1/ndj/checkpoints/modelscope_lam_nav_abl')
REPO = Path('/home/user/ndj/code/CompACT')


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def main():
    records = json.loads((ROOT / 'download_manifest.json').read_text())
    api = HubApi(endpoint='https://modelscope.cn')
    for tag in ('u100l100', 'u25l0', 'u25l50'):
        matches = [r for r in records if r['path'].startswith(f'pa_ablation_{tag}/')]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one step40000 checkpoint for {tag}')
        record = matches[0]
        for attempt in range(1, 4):
            try:
                log(f'Downloading {tag}, direct connection, attempt={attempt}')
                path = api.downloader.download_file(
                    repo_id='wzx6ccf9a/lam_nav_abl', repo_type='model',
                    file_path=record['path'], revision='master', local_dir=ROOT,
                    expected_sha256=record['sha256'], file_size=record['size'],
                )
                if path.stat().st_size != record['size']:
                    raise RuntimeError(f'Size mismatch for {tag}')
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
                if digest is None:
                    sha = hashlib.sha256()
                    with path.open('rb') as handle:
                        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                            sha.update(block)
                    digest = sha.hexdigest()
                if digest != record['sha256']:
                    raise RuntimeError(f'SHA256 mismatch for {tag}')
                break
            except Exception as exc:
                log(f'{tag}: {type(exc).__name__}; attempt {attempt} failed')
                if attempt == 3:
                    raise RuntimeError(f'Download failed for {tag}') from None
                time.sleep(10)
        alias = Path('/data1/ndj/checkpoints/hongyu') / f'LAM-{tag.upper()}.ckpt'
        if alias.exists() or alias.is_symlink():
            if alias.resolve() != path.resolve():
                raise RuntimeError(f'Refusing to replace existing {alias}')
        else:
            alias.symlink_to(path)
        receipt = dict(record, tag=tag, local_path=str(path), verified_sha256=digest)
        (ROOT / f'{tag}.verified.json').write_text(json.dumps(receipt, indent=2))
        log(f'Verified {tag}: {alias}; SHA256={digest}')
        if tag == 'u100l100':
            pidfile = ROOT / 'u100l100_pipeline.pid'
            if pidfile.exists():
                raise RuntimeError('Pipeline has already been launched; inspect PID before relaunching')
            with (ROOT / 'pipeline_bootstrap.log').open('a') as output:
                process = subprocess.Popen(
                    ['bash', str(REPO / 'scripts/run_u100l100_latentpt_local.sh')],
                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True, env=TRAIN_ENV,
                )
            pidfile.write_text(str(process.pid) + '\n')
            log(f'Launched u100l100 extraction -> pretrain -> finetune; PID={process.pid}')
    log('All three step40000 checkpoints verified and downloaded.')


if __name__ == '__main__':
    main()
