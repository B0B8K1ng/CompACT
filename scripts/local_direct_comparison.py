"""Read existing predictions; write an isolated local metric comparison."""
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from PIL import Image
import evaluate_nwm_predictions as evaluator
import run_a800_nwm_eval as protocol

ROOT = Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark')
SOURCE = ROOT / 'a800_full_evaluations/full_8xa800_seed0_v1'
OUT = ROOT / 'local_direct_comparison_20260924'
MODELS = ('nwm-latentpt-action-ft', 'nwm-latentpt-pixel-ft-reset')
torch.set_num_threads(4)
# Reuse immutable metric networks across datasets, without changing the evaluator.
evaluator.lpips.LPIPS = functools.lru_cache()(evaluator.lpips.LPIPS)
evaluator.dreamsim = functools.lru_cache()(evaluator.dreamsim)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def evaluate(model, dataset, frames, ids, gt, pred, provenance):
    tag = 'primary' if len(frames) == 1 else 'all'
    output = OUT / 'metrics' / model / f'{dataset}_{tag}.json'
    if output.exists():
        return
    filt = OUT / 'filters' / f'{model}_{dataset}_{tag}.json'
    save(filt, ids)
    args = [str(Path(evaluator.__file__)), '--gt-dir', str(gt), '--pred-dir', str(pred),
            '--output', str(output), '--frames', ','.join(f'{k}:{v}' for k,v in frames.items()),
            '--dataset', dataset, '--eval-type', 'time', '--eval-name', 'time',
            '--batch-size', '8', '--device', 'cuda', '--dreamsim-cache',
            '/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models',
            '--sample-ids-by-frame-file', str(filt), '--inference-backend', 'nwm',
            '--sampler', 'ddpm', '--sampling-steps', '250', '--seed', '0']
    for key in ('checkpoint_sha256', 'split_sha256'):
        if provenance.get(key):
            args += ['--' + key.replace('_','-'), provenance[key]]
    primary = OUT / 'metrics' / model / f'{dataset}_primary.json'
    if tag == 'all' and primary.exists():
        reuse = OUT / 'filters' / f'{model}_{dataset}_reuse.json'
        save(reuse, json.loads(primary.read_text())['metrics'])
        args += ['--reuse-frame-metrics-file', str(reuse)]
    save(OUT / 'commands' / f'{model}_{dataset}_{tag}.json', args)
    print(time.strftime('%FT%TZ', time.gmtime()), 'START', model, dataset, tag, flush=True)
    sys.argv = args
    log = OUT / 'logs' / f'{model}_{dataset}_{tag}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream, contextlib.redirect_stdout(stream):
        evaluator.main()
    print(time.strftime('%FT%TZ', time.gmtime()), 'DONE', model, dataset, tag,
          json.loads(output.read_text())['metrics'], flush=True)


def baseline_audits():
    model = json.loads((ROOT / 'benchmark_results.json').read_text())['models']['nwm-latentpt-ft']
    found = {}
    for category in ('recon_prediction', 'unseen_generalization', 'ood_direct_prediction', 'direct_prediction'):
        for ds, evaluations in model['results'].get(category, {}).items():
            if ds != 'huron' and 'time' in evaluations:
                found[ds] = Path(evaluations['time']['source_audit'])
    hp = ROOT / 'huron_valid329_20260924/results.json'
    if hp.exists():
        entry = json.loads(hp.read_text()).get('models', {}).get('nwm-latentpt-ft')
        if entry:
            found['huron'] = Path(entry['audit'])
    return found


def compare():
    audits = baseline_audits()
    records = []
    for ds in protocol.DATASETS:
        primary_label = '16obs' if ds == 'planetary_rover' else '4s'
        old = json.loads(audits[ds].read_text()) if ds in audits else None
        for model in MODELS:
            p = OUT / 'metrics' / model / f'{ds}_all.json'
            if not p.exists():
                p = OUT / 'metrics' / model / f'{ds}_primary.json'
            if not p.exists():
                continue
            new = json.loads(p.read_text())
            for label, row in new['metrics'].items():
                rec = dict(dataset=ds, model=model, horizon=label, metrics=row)
                oldlabel = '4s' if ds == 'planetary_rover' and label == '16obs' else label
                if old and oldlabel in old['frame_indices'] and oldlabel in old['metrics']:
                    h = new['frame_indices'][label]
                    oh = old['frame_indices'][oldlabel]
                    ids = new['sample_ids_by_frame'][label]
                    checkfile = OUT / 'gt_checks' / f'{ds}_{label}.json'
                    if checkfile.exists():
                        check = json.loads(checkfile.read_text())
                    else:
                        mismatches = []
                        for sample in ids:
                            a = Path(new['gt_eval_dir']) / f'id_{sample}' / f'{h}.png'
                            b = Path(old['gt_eval_dir']) / f'id_{sample}' / f'{oh}.png'
                            if not b.exists() or Image.open(a).convert('RGB').tobytes() != Image.open(b).convert('RGB').tobytes():
                                mismatches.append(sample)
                        check = dict(sample_count=len(ids), mismatches=mismatches, baseline_audit=str(audits[ds]))
                        save(checkfile, check)
                    compatible = not check['mismatches'] and row['sample_count'] == old['metrics'][oldlabel]['sample_count']
                    rec['baseline_gt_check'] = check
                    rec['comparable'] = compatible
                    if compatible:
                        rec['baseline'] = old['metrics'][oldlabel]
                        rec['delta'] = {k: row[k]-rec['baseline'][k] for k in ('lpips_alex','dreamsim','psnr')}
                records.append(rec)
    save(OUT / 'comparison.json', records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'run.json', dict(pid=os.getpid(), host=os.uname().nodename, source=str(SOURCE),
                              gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started=time.time()))
    # All inputs are read-only. No coordinator, registry, or A800 output is touched.
    for phase in ('primary', 'all'):
        for ds in protocol.DATASETS:
            indices = protocol.direct_ids(ds)
            if phase == 'primary':
                indices = {4: indices[4]}
            frames = {f'{h*4}obs' if ds == 'planetary_rover' else f'{h}s':h for h in indices}
            ids = {label:indices[h] for label,h in frames.items()}
            for model in MODELS:
                evaluate(model, ds, frames, ids, SOURCE/'gt'/ds/'time',
                         SOURCE/'predictions'/model/ds/'time', protocol.contract(model,ds,'time'))
            compare()
    compare()
    save(OUT / 'completed.json', dict(exit_code=0, finished=time.time()))
    print('COMPLETE exit_code=0', flush=True)


if __name__ == '__main__':
    main()
