#!/usr/bin/env python3
"""Freeze provenance and generate a browsable, auditable rollout gallery."""
import argparse, hashlib, html, json, os, shutil, subprocess
from pathlib import Path
from run_corrected_rollout_search import ROOT
from run_opennwm_rollout_showcase import REPO, MODELS
from render_rollout_academic import DATA, LAYOUT


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def freeze(root):
    out=root/'reproducibility';out.mkdir(exist_ok=True)
    files=subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines()
    files+= [str(p.relative_to(REPO)) for p in (REPO/'scripts').glob('*.py')]
    files+=['tests/test_rollout_actions.py']
    records={}
    for rel in sorted(set(files)):
        source=REPO/rel
        if source.is_file() and source.suffix in {'.py','.yaml','.yml','.pkl','.json','.toml','.txt'} and 'third_party/' not in rel:
            dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);records[rel]=digest(source)
    (out/'source_sha256.json').write_text(json.dumps(records,indent=2))
    (out/'working_tree.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=REPO))
    (out/'git_head.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO))
    for env in ['nwm','raenwm']:
        py=f'/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/{env}/bin/python'
        (out/f'{env}_pip_freeze.txt').write_bytes(subprocess.check_output([py,'-m','pip','freeze']))
    (out/'gpu.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
    (out/'models.json').write_text(json.dumps(MODELS,indent=2,default=str))
    inputs={}
    for c in json.loads((root/'search_plan.json').read_text()):
        ds=c['dataset'];split,folder=LAYOUT.get(ds,(ds,ds));name,start,*_=c['source_entry']
        paths=[REPO/'data_splits'/split/'test'/('time.pkl' if ds=='planetary_rover' else 'rollout.pkl'), DATA/folder/name/'traj_data.pkl',DATA/folder/name/'frame_metadata.jsonl']
        paths += [DATA/folder/name/f'{k}.jpg' for k in range(start,start+c['seconds']*4+1)]
        for p in paths:
            if p.is_file() and str(p) not in inputs:inputs[str(p)]=digest(p)
    (out/'input_sha256.json').write_text(json.dumps(inputs,indent=2))
    artifacts={str(p.relative_to(root)):digest(p) for pattern in ['examples/*/*/frames/*/*.png','initial/*/*.png','examples/*/*/sequence.*','examples/*/*/comparison.mp4'] for p in root.glob(pattern)}
    (out/'artifact_sha256.json').write_text(json.dumps(artifacts,indent=2))
    print('Frozen',len(records),'source files,',len(inputs),'inputs,',len(artifacts),'artifacts',flush=True)


def gallery(root):
    cases=json.loads((root/'summary.json').read_text())
    review_path=root/'visual_review.json';reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
    sections=[]
    for ds in dict.fromkeys(c['dataset'] for c in cases):
        cards=[]
        for c in [c for c in cases if c['dataset']==ds]:
            i=c['sample_id'];rel=f'examples/{ds}/id_{i}';review=reviews.get(f'{ds}/{i}',{})
            note=review.get('note','尚未人工审核；请勿仅凭指标认定优势。')
            cards.append(f'''<article><h3>id_{i} · {html.escape(review.get('status','未审核'))}</h3>
<p>OpenNWM 相对最强基线：平均 PSNR {c['mean_psnr_margin']:+.3f} dB；平均 SSIM {c['mean_ssim_margin']:+.4f}</p>
<p>{html.escape(note)}</p><a href="{rel}/sequence.png"><img loading="lazy" src="{rel}/sequence.png"></a>
<p><a href="{rel}/sequence.pdf">PDF</a> · <a href="{rel}/metrics.json">指标</a> · <a href="{rel}/trajectory.json">真实轨迹</a> · <a href="{rel}/frames/">逐帧 PNG</a></p>
<video controls preload="none" src="{rel}/comparison.mp4"></video></article>''')
        sections.append(f'<section id="{ds}"><h2>{ds}</h2>'+''.join(cards)+'</section>')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>单图自回归对比</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1400px;margin:40px auto;padding:0 24px;color:#222;background:#fff}h1,h2,h3{font-weight:550}article{border-top:1px solid #ddd;padding:20px 0 40px}img{width:100%;height:auto}video{width:min(100%,896px)}a{color:#32658c}nav{position:sticky;top:0;background:white;padding:12px 0}nav a{margin-right:16px}</style>
<h1>单图自回归预测对比</h1><p>GT · NWM · RAE-NWM · OpenNWM。单张真实初始图，后续回灌解码图像。ID：16 s；OOD：4 s；4 FPS。Planetary 的 t 是播放时间。</p>
<p>这是按优势筛选的定性候选集，不代表数据集总体性能。失败样本与所有已运行候选均保留。平均指标不含共享初始帧。</p><p><a href="README.md">协议与复现说明</a> · <a href="selection_audit.json">完整筛选审计</a> · <a href="visual_review.json">人工审核</a></p>'''
    document+='<nav>'+''.join(f'<a href="#{ds}">{ds}</a>' for ds in dict.fromkeys(c['dataset'] for c in cases))+'</nav>'
    (root/'index.html').write_text(document+''.join(sections)+'</html>')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    gallery(ROOT)
    if a.freeze:freeze(ROOT)
