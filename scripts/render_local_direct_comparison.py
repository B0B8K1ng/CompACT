"""Render the isolated local Direct metrics as CSV and Markdown."""
import csv
import json
from pathlib import Path

ROOT = Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/local_direct_comparison_20260924')
rows = json.loads((ROOT / 'comparison.json').read_text())
keys = ('lpips_alex', 'dreamsim', 'psnr')
flat = []
seen = set()
for row in rows:
    identity = (row['dataset'], row['horizon'])
    if row.get('comparable') and identity not in seen:
        flat.append(dict(dataset=row['dataset'], horizon=row['horizon'], model='nwm-latentpt-ft', **row['baseline']))
        seen.add(identity)
    flat.append(dict(dataset=row['dataset'], horizon=row['horizon'], model=row['model'],
                     **row['metrics'], **{'delta_'+k:v for k,v in row.get('delta', {}).items()}))
dataset_order = list(dict.fromkeys(row['dataset'] for row in rows))
model_order = ['nwm-latentpt-ft','nwm-latentpt-action-ft','nwm-latentpt-pixel-ft-reset']
flat.sort(key=lambda r: (dataset_order.index(r['dataset']),
                        int(r['horizon'].removesuffix('obs').removesuffix('s')),
                        model_order.index(r['model'])))
with (ROOT / 'comparison.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['dataset','horizon','model','sample_count',*keys,*['delta_'+k for k in keys]])
    writer.writeheader()
    writer.writerows(flat)

lines = ['# Local Direct prediction comparison', '',
         'Computed on local L20 GPU 0 from read-only A800 prediction/GT images. A800 state and metrics were not modified.', '',
         'Metrics: LPIPS-Alex ↓ / DreamSim ↓ / PSNR (dB) ↑. Baseline values are reused only after every target GT image matches pixel-for-pixel and sample counts match.', '',
         'Checkpoint: 100k for all three models. Sampler: DDPM, 250 steps, seed 0. Metric batch size: 8.', '',
         'Huron uses the corrected 329-sample population. Planetary endpoints are observed frames, not seconds. Other datasets use seconds.', '',
         '## Primary comparison (4 seconds; Planetary 16 observed frames)', '',
         '| Dataset | N | nwm-latentpt-ft | action-only | pixel-only |',
         '|---|---:|---|---|---|']
models = ('nwm-latentpt-ft','nwm-latentpt-action-ft','nwm-latentpt-pixel-ft-reset')
def triplet(row):
    return ' / '.join(f'{row[k]:.4f}' for k in keys) if row else '—'
for ds in dict.fromkeys(row['dataset'] for row in rows):
    label = '16obs' if ds == 'planetary_rover' else '4s'
    entries = {r['model']:r for r in flat if r['dataset']==ds and r['horizon']==label}
    n = next(iter(entries.values()))['sample_count']
    lines.append(f'| {ds} | {n} | ' + ' | '.join(triplet(entries.get(m)) for m in models) + ' |')
lines += ['', '## All available Direct endpoints', '',
          '| Dataset | Horizon | Model | N | LPIPS ↓ | DreamSim ↓ | PSNR ↑ | ΔLPIPS | ΔDreamSim | ΔPSNR |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in flat:
    lines.append('| '+ ' | '.join([r['dataset'],r['horizon'],r['model'],str(r['sample_count']),
                 *[f'{r[k]:.4f}' for k in keys],*[f'{r["delta_"+k]:+.4f}' if 'delta_'+k in r else '—' for k in keys]])+' |')
lines += ['', 'Deltas are model minus baseline. Missing baseline endpoints remain blank; no historical Huron 500-row aggregate is used.', '',
          'UZH FPV is not included because this A800 batch has no action-only Direct predictions for it.', '',
          'Raw metrics: `metrics/`. Pixel-level GT checks and baseline audit paths: `gt_checks/`. Exact evaluator arguments: `commands/`. Process log: `logs/runner.log`.']
(ROOT / 'comparison.md').write_text('\n'.join(lines)+'\n')
print(ROOT / 'comparison.md')
print('rows:',len(flat))
