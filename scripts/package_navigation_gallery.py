#!/usr/bin/env python3
"""Package measured navigation examples, inputs, and exact replay metadata."""
import hashlib
import csv
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from opennwm_navigation_gallery import ROOT, REPO, LABELS, CONFIG


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selected = json.loads((ROOT / 'selected_samples.json').read_text())
    audit = json.loads((ROOT / 'selection_audit.json').read_text())
    selected_keys={(r['dataset'],r['sample_id']) for r in selected}
    with (ROOT/'candidate_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['dataset','sample_id','selected','ATE_m','goal_error_m','normalized_ATE','normalized_goal_error','visual_rejection_reason','directory'])
        for r in audit:
            chosen=(r['dataset'],r['sample_id']) in selected_keys
            folder=('gallery' if chosen else 'all_candidates')+f"/{r['dataset']}_{r['sample_id']:03d}"
            writer.writerow([r['dataset'],r['sample_id'],chosen,r['ate_m'],r['goal_error_m'],r['normalized_ate'],r['normalized_goal_error'],r.get('visual_review',{}).get('reason',''),folder])
    cards = []
    for r in selected:
        folder = f"gallery/{r['dataset']}_{r['sample_id']:03d}"
        metadata = json.loads((ROOT / folder / 'sample.json').read_text())
        assert not metadata['identical_predicted_pairs'], folder
        assert len(list((ROOT / folder / 'GT').glob('*.png'))) == 9
        assert len(list((ROOT / folder / 'OpenNWM').glob('*.png'))) == 9
        probe = subprocess.run(['/usr/bin/ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,r_frame_rate', '-of', 'json', str(ROOT / folder / 'comparison.mp4')], check=True, capture_output=True, text=True)
        video = json.loads(probe.stdout)['streams'][0]
        assert video['nb_read_frames'] == '9' and video['r_frame_rate'] == '4/1', video
        cards.append(f'<article><h2>{html.escape(LABELS[r["dataset"]])} · {r["sample_id"]}</h2><a href="{folder}/navigation.pdf"><img src="{folder}/preview.png"></a><p><a href="{folder}/navigation.pdf">PDF</a> · <a href="{folder}/navigation.svg">SVG</a> · <a href="{folder}/navigation.png">PNG</a> · <a href="{folder}/trajectories_m.csv">Trajectory CSV</a> · <a href="{folder}/sample.json">Metadata</a></p><video controls preload="none" src="{folder}/comparison.mp4"></video></article>')
    (ROOT / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenNWM Navigation</title><style>body{font:16px system-ui;max-width:1440px;margin:32px auto;padding:0 24px;color:#222;background:#fafafa}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:white;padding:16px;border:1px solid #ddd;border-radius:6px}h2{font-size:18px}img,video{width:100%;height:auto}a{color:#168373}p{line-height:1.6}@media(max-width:900px){main{grid-template-columns:1fr}}</style><h1>OpenNWM Navigation</h1><p>Goal-image-conditioned offline planning with RGB feedback. GT frames follow recorded actions; predicted frames follow planned actions. Selected qualitative examples; no closed-loop execution was measured.</p><main>' + ''.join(cards) + '</main></html>')

    datasets = [ds for ds in LABELS if any(r['dataset'] == ds for r in selected)]
    overview = Image.new('RGB', (1600, 500 * len(datasets)), 'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    draw = ImageDraw.Draw(overview)
    for row, ds in enumerate(datasets):
        for col, r in enumerate(x for x in selected if x['dataset'] == ds):
            im = Image.open(ROOT / 'gallery' / f"{ds}_{r['sample_id']:03d}" / 'preview.png')
            im.thumbnail((790,460))
            overview.paste(im, (800*col+(800-im.width)//2, 500*row+35))
            draw.text((800*col+20,500*row+8),f"{LABELS[ds]} · {r['sample_id']}",font=font,fill='#222222')
    if datasets:overview.save(ROOT/'overview.jpg',quality=94)

    repro = ROOT / 'reproducibility'
    repro.mkdir(exist_ok=True)
    files = subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines()
    prior_repro=ROOT.parent/'goal_navigation_pixel_paper_20260924/reproducibility'
    files += list(json.loads((prior_repro/'source_tree_sha256.json').read_text()))
    files += ['scripts/opennwm_navigation_gallery.py','scripts/package_navigation_gallery.py','scripts/prepare_tum_camera_heading.py','tests/test_navigation_image_feedback.py','tests/test_navigation_source_indices.py']
    hashes = {}
    for name in sorted(set(files)):
        path = REPO / name
        if path.is_file() and path.suffix in {'.py','.yaml','.yml','.json','.md','.toml','.txt'} and path.stat().st_size < 5_000_000:
            dest = repro/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            hashes[name]=digest(dest)
    for ds in LABELS:
        name=CONFIG['evaluation_datasets'][ds]['navigation_index']
        dest=repro/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest);hashes[name]=digest(dest)
    (repro/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    (repro/'git_head.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO))
    (repro/'working_tree.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=REPO))
    (repro/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    previous=prior_repro/'checkpoint_sha256.json'
    shutil.copy2(previous,repro/'checkpoint_sha256.json')
    status={ds:dict(evaluated=sum(r['dataset']==ds for r in audit),selected=sum(r['dataset']==ds for r in selected)) for ds in LABELS}
    (ROOT/'validation.json').write_text(json.dumps(dict(status=status,checks=['GT and goal independently reconstructed from original odometry','ATE and goal error independently recomputed in meters','All selected cases pass normalized ATE <= 0.25 and normalized goal error <= 0.30','No identical consecutive predicted frames','Nine GT and nine predicted PNG frames per case','Video: nine frames, 4 FPS']),indent=2))
    (ROOT/'README.md').write_text('''# OpenNWM goal-conditioned navigation

Open `index.html` to browse the selected examples. Each sample directory contains
publication figures (PDF/SVG/PNG), complete GT and predicted frame sequences,
a comparison video, trajectories in meters, and hashed input metadata.
`gallery/` contains the recommended examples. `all_candidates/` preserves
the other completed examples with the same figures, PNG sequences and videos;
`candidate_results.csv` lists every result and its selection status.

## Protocol and interpretation

Four recorded context frames and a goal image eight steps ahead condition CEM
planning. Every imagined step decodes predicted latent to RGB, normalizes RGB,
and reencodes the updated observation window. Raw predicted latent is never
appended to the next observation window. No later GT frames are fed back.
The final fitted plan is rolled out again and its eight predictions are saved.

The horizon is eight steps at 4 FPS (2 seconds); step zero is the recorded start
image shared by both rows. CEM uses 80 candidates, top 5, one update, three
stochastic repetitions, 250 DDPM steps, seed 42 + original sample index, and
microbatch 80. Candidates use the inherited constant-planar-delta family; thus
these figures visualize short local plans rather than arbitrary curved routes.
LPIPS to the reconstructed goal image ranks candidates. GT future actions and
poses are used for assessment and qualitative selection, not optimization.
HuRoN uses a documented fixed wider prior; see `protocol.json`.
Its selected valid rows keep their original split positions; missing windows
are never allowed to renumber the sample IDs. Clearly unrelated predicted
scene content is rejected after visual inspection; see `visual_review.json`.

GT frames follow recorded actions; predictions follow planned actions. They
are not same-action reconstruction pairs. These are offline planned trajectories,
not executed closed-loop trajectories, and do not establish navigation success
rates. This is a selected qualitative gallery, not a dataset-average benchmark.
The complete selection audit retains every evaluated candidate and error.
Planetary is excluded because its poses lack metric navigation ground truth.

## Reproduction

Run from the CompACT repository with the explicit nwm Python environment.
`logs/*.json` contains the exact expanded command, runtime, working directory,
GPU, process ID, log path, and exit status for each completed job. The resolved
Hydra configuration is under `hydra/`. Preserve the exact job/sample grouping,
seed, runtime and microbatch 80:
the tokenizer samples latent codes, so its RNG call order must also be preserved.
Cross-archive checks matched RECON 3 and Office-go2 1 exactly; Office-go2 3
differed by up to 0.0355 m in planned coordinates under a different job grouping.
The cause is not established; bitwise identity across job variants is not claimed.
See `reproducibility/cross_archive_comparison.json` for the measured comparison.

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python scripts/opennwm_navigation_gallery.py prepare
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python scripts/opennwm_navigation_gallery.py run --gpu 2 --datasets recon --ids 3 0
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python scripts/opennwm_navigation_gallery.py render
/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python scripts/package_navigation_gallery.py
```

Other dataset/sample commands are recorded in `logs/`. The source snapshot,
split indices, environment and checkpoint hashes are in `reproducibility/`.
''')
    artifacts={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_sha256.json' and 'logs' not in p.relative_to(ROOT).parts}
    (ROOT/'artifact_sha256.json').write_text(json.dumps(artifacts,indent=2))
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
