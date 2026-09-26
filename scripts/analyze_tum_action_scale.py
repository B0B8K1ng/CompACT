#!/usr/bin/env python3
"""Analyze frozen diagnostic runs; scaled actions are not benchmark results."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_corrected_rollout_search import ROOT
D=ROOT/'diagnostics/corrected_scale_sweep'
def read(p):return np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for ident in [124,81,20,127]:
 base=ROOT/f'examples/tum_rgbd/id_{ident}/frames'
 init=read(ROOT/f'initial/tum_rgbd/id_{ident}.png')
 gt=np.stack([init]+[read(base/'GT'/f'{k:03}.png') for k in range(16)])
 for scale in [0,1,2,4]:
  folder=D/f'scale_{scale}/predictions/opennwm-finalLAM-100k/tum_rgbd/rollout_4fps/id_{ident}'
  pred=np.stack([init]+[read(folder/f'{k}.png') for k in range(16)])
  dp=np.diff(pred,axis=0);dg=np.diff(gt,axis=0)
  r=dict(id=ident,scale=scale,pred_adjacent_mae_first2s=float(abs(dp[:8]).mean()),gt_adjacent_mae_first2s=float(abs(dg[:8]).mean()),pred_adjacent_mae_4s=float(abs(dp).mean()),gt_adjacent_mae_4s=float(abs(dg).mean()),temporal_difference_error=float(abs(dp-dg).mean()),mean_psnr=float(np.mean(10*np.log10(255**2/np.mean((pred[1:]-gt[1:])**2,axis=(1,2,3))))))
  if scale==1:r['baseline_hash_matches']=[digest(folder/f'{k}.png')==digest(base/'opennwm-finalLAM-100k'/f'{k:03}.png') for k in range(16)]
  records.append(r)
 # Clean diagnostic grid with common GT initial frame.
 w,h=init.shape[1],init.shape[0];left,top=125,35
 canvas=Image.new('RGB',(left+5*w,top+5*h),'white');draw=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 for col,t in enumerate([0,1,2,3,4]):draw.text((left+col*w+w//2-28,7),f't = {t} s',fill='black',font=font)
 for row,scale in enumerate([None,0,1,2,4]):
  draw.text((8,top+row*h+h//2-10),'GT' if scale is None else f'Action ×{scale}',fill='black',font=font)
  for col,t in enumerate([0,1,2,3,4]):
   p=ROOT/f'initial/tum_rgbd/id_{ident}.png' if t==0 else base/'GT'/f'{t*4-1:03}.png' if scale is None else D/f'scale_{scale}/predictions/opennwm-finalLAM-100k/tum_rgbd/rollout_4fps/id_{ident}/{t*4-1}.png'
   canvas.paste(Image.open(p).convert('RGB'),(left+col*w,top+row*h))
 canvas.save(D/f'id_{ident}_scale_comparison.png')
(D/'analysis.json').write_text(json.dumps({'pixel_range':[0,255],'initial_frame_in_adjacent_metrics':True,'scales_multiply':['forward','left','yaw'],'records':records},indent=2))
for r in records:print(r['id'],r['scale'],*[round(r[k],3) for k in ['pred_adjacent_mae_first2s','pred_adjacent_mae_4s','mean_psnr','temporal_difference_error']],all(r.get('baseline_hash_matches',[True])))
