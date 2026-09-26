#!/usr/bin/env python3
"""Audit every completed search candidate; publish clean figures and videos."""
import argparse,json,os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_corrected_rollout_search import ROOT
from run_opennwm_rollout_showcase import MODELS,paths,psnr
from render_rollout_academic import render
from make_opennwm_rollout_videos import make_video


def read(path):return np.asarray(Image.open(path).convert('RGB')).astype(np.float32)


def ssim(x,y):
    x,y=x.astype(np.float64),y.astype(np.float64)
    blur=lambda z:cv2.GaussianBlur(z,(11,11),1.5)[5:-5,5:-5]
    ux,uy=blur(x),blur(y)
    vx,vy=blur(x*x)-ux*ux,blur(y*y)-uy*uy
    vxy=blur(x*y)-ux*uy
    return float((((2*ux*uy+6.5025)*(2*vxy+58.5225))/((ux*ux+uy*uy+6.5025)*(vx+vy+58.5225))).mean())


def assemble(root,figures=True):
    cases=json.loads((root/'search_plan.json').read_text());complete=[];incomplete=[]
    for case in cases:
        ds,i=case['dataset'],case['sample_id'];count=case['seconds']*4
        folders={'GT':paths(ds,i,None,root),**{m:paths(ds,i,m,root) for m in MODELS}}
        if any(not (p/f'{k}.png').exists() for p in folders.values() for k in range(count)):
            incomplete.append([ds,i]);continue
        out=root/'examples'/ds/f'id_{i}';out.mkdir(parents=True,exist_ok=True)
        cache=out/'metrics.json'
        if cache.exists():result=json.loads(cache.read_text())
        else:
            gt=np.stack([read(folders['GT']/f'{k}.png') for k in range(count)])
            initial=read(root/'initial'/ds/f'id_{i}.png')
            gt_delta=np.diff(np.concatenate([initial[None],gt]),axis=0)
            scores={}
            for model in MODELS:
                pred=np.stack([read(folders[model]/f'{k}.png') for k in range(count)])
                delta=np.diff(np.concatenate([initial[None],pred]),axis=0)
                scores[model]=dict(mean_psnr=float(np.mean([psnr(a,b) for a,b in zip(gt,pred)])),endpoint_psnr=psnr(gt[-1],pred[-1]),mean_ssim=float(np.mean([ssim(a,b) for a,b in zip(gt,pred)])),temporal_difference_mae=float(abs(delta-gt_delta).mean()),adjacent_frame_mae=float(abs(delta).mean()))
            result={**case,'metrics':scores,'gt_adjacent_frame_mae':float(abs(gt_delta).mean())}
            op=scores['opennwm-finalLAM-100k'];base=[scores[m] for m in ['nwm-release','rae-nwm']]
            result['mean_psnr_margin']=op['mean_psnr']-max(v['mean_psnr'] for v in base)
            result['mean_ssim_margin']=op['mean_ssim']-max(v['mean_ssim'] for v in base)
            result['endpoint_psnr_margin']=op['endpoint_psnr']-max(v['endpoint_psnr'] for v in base)
            cache.write_text(json.dumps(result,indent=2))
        if figures:
            for name,folder in folders.items():
                dest=out/'frames'/name;dest.mkdir(parents=True,exist_ok=True)
                for k in range(count):
                    target=dest/f'{k:03d}.png'
                    if not target.exists():os.link(folder/f'{k}.png',target)
            if not (out/'sequence.png').exists():render(case,root)
            make_video(case,root,False)
        complete.append(result)
    (root/'summary.json').write_text(json.dumps(complete,indent=2))
    selected=[];best={}
    for c in complete:
        ds=c['dataset']
        if ds not in best or c['mean_ssim_margin']>best[ds]['mean_ssim_margin']:best[ds]=c
        if c['mean_psnr_margin']>0 and c['mean_ssim_margin']>0:selected.append(c)
    (root/'selection_audit.json').write_text(json.dumps(dict(rule='Mean PSNR and mean RGB SSIM both exceed both baselines; visual review still required, static collapse disqualifies a case.',complete_count=len(complete),incomplete=incomplete,selected=selected,best_per_dataset=best),indent=2))
    print(json.dumps(dict(complete=len(complete),incomplete=len(incomplete),best={k:{'id':v['sample_id'],'psnr_margin':v['mean_psnr_margin'],'ssim_margin':v['mean_ssim_margin']} for k,v in best.items()}),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--metrics-only',action='store_true');a=p.parse_args();assemble(a.root,not a.metrics_only)
