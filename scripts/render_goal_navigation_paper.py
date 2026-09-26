#!/usr/bin/env python3
"""Render genuine goal-image-conditioned CEM plans with odometry validation."""
import argparse,csv,json,pickle,hashlib,shutil,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
parser=argparse.ArgumentParser();parser.add_argument('--result-root',type=Path,default=Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/goal_navigation_pixel_paper_20260924'));args=parser.parse_args()
R=args.result_root;OUT=R/'figures';OUT.mkdir(exist_ok=True)
DATA=Path('/file_system/nas/algorithm/dujun.nie/nwm/data');STEM='CEM_N80_K5_RS1_rep3_OPT1_COST-lpips-RECON-True_PIXEL-FEEDBACK'
MODELS=[('NWM','nwm-release','#d48535'),('RAE-NWM','rae-nwm','#507fb0'),('OpenNWM','opennwm-finalLAM-100k','#168373')]
SPACING={'recon':.25,'unitree_go2':.07610250112606785}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def load(ds,i):
 split=REPO/f'data_splits/{ds}/test/navigation_eval.pkl';name,t,lo,hi=pickle.loads(split.read_bytes())[i];assert lo==hi==8
 source=DATA/ds/name;pose=pickle.loads((source/'traj_data.pkl').read_bytes());yaw=float(np.asarray(pose['yaw'][t]).item());c,s=np.cos(yaw),np.sin(yaw)
 gt=(np.array(pose['position'][t:t+9])[:,:2]-np.array(pose['position'][t])[:2])@np.array([[c,-s],[s,c]])
 record={'dataset':ds,'sample_id':i,'trajectory':name,'observation_frames':list(range(t-3,t+1)),'goal_frame':t+8,'spacing_m':SPACING[ds],'gt_xy_m':gt.tolist(),'models':{},'input_sha256':{str(split):hashlib.sha256(split.read_bytes()).hexdigest()}}
 for label,m,color in MODELS:
  path=R/f'planning/{m}/{ds}/{STEM}/trajectories/{i:06d}.json';x=json.loads(path.read_text());pred=np.array(x['predicted_xy_waypoint_units'])*SPACING[ds]
  np.testing.assert_allclose(np.array(x['gt_xy_waypoint_units'])*SPACING[ds],gt[1:],atol=2e-5)
  np.testing.assert_allclose(np.array(x['goal_pose_waypoint_units'])[0,:2]*SPACING[ds],gt[-1],atol=2e-5)
  err=np.linalg.norm(pred-gt[1:],axis=1);ate=float(np.sqrt(np.mean(err**2)));fde=float(err[-1])
  metric=json.loads((path.parent.parent/f'sample_metrics/{i:06d}.json').read_text())
  np.testing.assert_allclose([ate,fde],np.array([metric['ate'],metric['pos_diff_norm']])*SPACING[ds],atol=2e-5)
  record['models'][m]={'planned_xy_m':np.vstack([np.zeros(2),pred]).tolist(),'ate_m':ate,'goal_error_m':fde,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 for p in [source/'traj_data.pkl']+[source/f'{k}.jpg' for k in range(t-3,t+9)]:record['input_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 record['observation_image']=str(source/f'{t}.jpg');record['goal_image']=str(source/f'{t+8}.jpg');return record
allcases=[]
for ds in ['recon','unitree_go2']:
 ids=sorted(int(p.stem) for p in (R/f'planning/opennwm-finalLAM-100k/{ds}/{STEM}/trajectories').glob('*.json'))
 for i in ids:
  try:r=load(ds,i)
  except FileNotFoundError:continue
  o=r['models']['opennwm-finalLAM-100k'];bases=[r['models'][m] for _,m,_ in MODELS[:2]]
  r['wins_both_metrics']=all(o[k]<b[k] for b in bases for k in ['ate_m','goal_error_m'])
  r['min_relative_gain']=min(1-o[k]/max(b[k],1e-8) for b in bases for k in ['ate_m','goal_error_m'])
  allcases.append(r)
(R/'selection_audit.json').write_text(json.dumps(allcases,indent=2))
selected=[]
for ds in ['recon','unitree_go2']:
 choices=[r for r in allcases if r['dataset']==ds and r['wins_both_metrics'] and np.linalg.norm(r['gt_xy_m'][-1])>.2]
 if not choices:raise RuntimeError(f'No completed dual-metric winner for {ds}')
 selected.append(max(choices,key=lambda x:x['min_relative_gain']))
fig=plt.figure(figsize=(9,6.1),facecolor='white');grid=fig.add_gridspec(3,2,height_ratios=[1.35,2.85,.63],left=.075,right=.975,top=.92,bottom=.13,wspace=.35,hspace=.30)
import misc
transform=misc.get_transform(224,[.5]*3,[.5]*3)
for col,r in enumerate(selected):
 ds=r['dataset'];dslabel='RECON' if ds=='recon' else 'Office-go2'
 with (OUT/f'{ds}_trajectories_m.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['step','GT_x','GT_y']+[f'{label}_{axis}' for label,_,_ in MODELS for axis in ['x','y']])
  for step,xy in enumerate(r['gt_xy_m']):writer.writerow([step]+xy+[v for _,m,_ in MODELS for v in r['models'][m]['planned_xy_m'][step]])
 inputs=OUT/'inputs'/ds;inputs.mkdir(parents=True,exist_ok=True)
 for frame in r['observation_frames']+[r['goal_frame']]:shutil.copy2(Path(r['observation_image']).parent/f'{frame}.jpg',inputs/f'{frame}.jpg')
 sg=grid[0,col].subgridspec(1,2,wspace=.10)
 for j,(key,title) in enumerate([('observation_image','Start'),('goal_image','Goal')]):
  ax=fig.add_subplot(sg[0,j]);tensor=transform(Image.open(r[key]).convert('RGB'));img=(tensor*.5+.5).clamp(0,1).permute(1,2,0).numpy();ax.imshow(img);ax.axis('off');ax.set_title(title,fontsize=10,pad=4)
  Image.fromarray((img*255).astype('uint8')).save(OUT/f'{ds}_{title.lower()}.png')
 bb=grid[0,col].get_position(fig);fig.text((bb.x0+bb.x1)/2,.977,dslabel,ha='center',va='top',fontsize=13,fontweight='medium')
 ax=fig.add_subplot(grid[1,col]);gt=np.array(r['gt_xy_m']);ax.plot(-gt[:,1],gt[:,0],color='#333333',ls='--',lw=1.9,zorder=5)
 for label,m,color in MODELS:
  p=np.array(r['models'][m]['planned_xy_m']);ax.plot(-p[:,1],p[:,0],color=color,lw=2,marker='o',ms=2.5,zorder=3)
 ax.scatter(0,0,s=36,facecolor='white',edgecolor='#333333',zorder=7)
 ax.scatter(-gt[-1,1],gt[-1,0],s=150,marker='*',facecolor='#333333',edgecolor='white',linewidth=.7,zorder=7)
 ax.set_aspect('equal',adjustable='datalim');ax.margins(.16);ax.set_xlabel('Lateral (m)',labelpad=3);ax.set_ylabel('Forward (m)',labelpad=3);ax.tick_params(labelsize=9,length=3);ax.grid(color='#eeeeee',lw=.65);ax.set_axisbelow(True)
 ax=fig.add_subplot(grid[2,col]);ax.axis('off')
 tab=ax.table(cellText=[[label]+[f"{r['models'][m][key]:.3f}" for _,m,_ in MODELS] for label,key in [('ATE (m)','ate_m'),('Goal error (m)','goal_error_m')]],colLabels=['']+[x[0] for x in MODELS],cellLoc='center',loc='center',colWidths=[.31,.21,.25,.23])
 tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.14)
 for (row,c),cell in tab.get_celld().items():
  cell.set_linewidth(0);cell.set_facecolor('white')
  if c>0 and row==0:cell.get_text().set_color(MODELS[c-1][2])
  if c==0:cell.get_text().set_ha('left')
  if c==3 and row>0:cell.get_text().set_weight('bold')
handles=[Line2D([0],[0],color='#333333',ls='--',lw=1.9,label='GT')]+[Line2D([0],[0],color=color,lw=2,label=label) for label,_,color in MODELS]+[Line2D([0],[0],marker='o',mfc='white',mec='#333333',ls='none',label='Start'),Line2D([0],[0],marker='*',color='#333333',ms=10,ls='none',label='Goal')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.018),ncol=6,frameon=False,columnspacing=1.5,handlelength=2)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'goal_navigation_comparison.{ext}',dpi=300,facecolor='white')
plt.close(fig);im=Image.open(OUT/'goal_navigation_comparison.png');im.thumbnail((1500,1100));im.save(OUT/'preview.png')
(OUT/'selected_samples.json').write_text(json.dumps(selected,indent=2));shutil.copy2(__file__,OUT/Path(__file__).name)
print([(r['dataset'],r['sample_id'],r['min_relative_gain']) for r in selected])
