#!/usr/bin/env python3
"""Paper figures from unchanged, frozen image-feedback rollout PNGs."""
import hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
R=Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/opennwm_pixel_corrected_20260924')
O=R/'paper_recon_go2';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'font.size':15})
ROWS=[('GT','GT'),('NWM','nwm-release'),('RAE-NWM','rae-nwm'),('OpenNWM','opennwm-finalLAM-100k')]
records={}
def render(name,panels):
 # Coordinate system in inches: every source image has exactly the same size.
 cell=1.55;gap=.035;label=1.30;between=.34;top=.94;bottom=.06
 width=sum(label+len(ts)*(cell+gap)-gap for _,_,_,ts in panels)+between*(len(panels)-1)
 height=top+4*cell+3*gap+bottom
 fig=plt.figure(figsize=(width,height),facecolor='white');xbase=0
 for title,ds,ident,times in panels:
  base=R/f'examples/{ds}/id_{ident}';meta=json.loads((base/'trajectory.json').read_text())
  local=np.array(meta['positions_local']);xy=np.column_stack((-local[:,1],-local[:,0]))
  # Same physical scale at every timestamp within this sample; origin bottom center.
  scale=min(.42/max(abs(xy[:,0]).max(),1e-8),.68/max(-xy[:,1].min(),1e-8),.15/max(xy[:,1].max(),1e-8))
  xy=xy*scale+[.5,.78];panelwidth=label+len(times)*(cell+gap)-gap
  fig.text((xbase+label+(panelwidth-label)/2)/width,1-.14/height,title,ha='center',va='center',fontsize=17,fontweight='medium')
  fig.text((xbase+label-.08)/width,1-.66/height,'GT path',ha='right',va='center',fontsize=12,color='#555555')
  for col,t in enumerate(times):
   x=xbase+label+col*(cell+gap)
   fig.text((x+cell/2)/width,1-.37/height,f'$t={t}$ s',ha='center',va='center',fontsize=14)
   ax=fig.add_axes([x/width,1-.91/height,cell/width,.42/height]);ax.set_xlim(0,1);ax.set_ylim(1,0);ax.set_aspect('equal');ax.axis('off')
   p=xy[:t*4+1];ax.plot(p[:,0],p[:,1],color='#406d88',lw=1.4);ax.plot(*xy[0],marker='o',ms=2.7,mfc='white',mec='#406d88',mew=.8);ax.plot(*p[-1],marker='o',ms=3,color='#406d88')
   for row,(text,folder) in enumerate(ROWS):
    y=height-top-(row+1)*cell-row*gap
    path=R/f'initial/{ds}/id_{ident}.png' if t==0 else base/f'frames/{folder}/{4*t-1:03}.png'
    records[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    ax=fig.add_axes([x/width,y/height,cell/width,cell/height]);ax.imshow(Image.open(path),interpolation='none');ax.axis('off')
    if col==0:fig.text((xbase+label-.08)/width,(y+cell/2)/height,text,ha='right',va='center',fontsize=14,fontweight='medium' if row==3 else 'normal')
  xbase+=panelwidth+between
 for ext in ['pdf','png']:fig.savefig(O/f'{name}.{ext}',dpi=300,facecolor='white',pad_inches=0)
 plt.close(fig)
recon=('(a) RECON','recon',25,[0,4,8,12,16]);go2=('(b) Unitree Go2','unitree_go2',81,[0,1,2,3,4])
render('navigation_comparison',[recon,go2])
render('recon_full',[('RECON','recon',25,list(range(0,17,2)))])
render('unitree_go2',[('Unitree Go2','unitree_go2',81,[0,1,2,3,4])])
manifest={'samples':[{'dataset':ds,'id':i,'main_times':ts,'metrics':json.loads((R/f'examples/{ds}/id_{i}/metrics.json').read_text()),'source':json.loads((R/f'examples/{ds}/id_{i}/trajectory.json').read_text())} for _,ds,i,ts in [recon,go2]],'input_sha256':records,'protocol':'Single initial image; decoded image feedback; original GT actions; 4 FPS; unchanged source images, no crop or enhancement.','selection_note':'Qualitative selected examples; RECON has higher PSNR but lower SSIM than NWM; does not measure closed-loop navigation success.'}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
(O/'caption.txt').write_text('Action-conditioned autoregressive visual prediction on RECON and Unitree Go2. Each rollout starts from a single observation and feeds decoded predictions back as images. Columns show matched timestamps; paths show the shared ground-truth motion. OpenNWM more closely preserves the late-horizon grass/sky layout in RECON and the corridor wall/floor appearance in Unitree Go2.\n')
(O/'README.md').write_text('# Paper figures\n\n- navigation_comparison.pdf / .png: main two-panel figure. RECON: 0,4,8,12,16 s; Go2: 0,1,2,3,4 s.\n- recon_full.pdf / .png: all requested RECON timestamps (0,2,…,16 s).\n- unitree_go2.pdf / .png: standalone Go2 panel.\n- PDF embeds original RGB images with vector text/paths; PNG exported at 300 dpi. No image crops, retouching, or action modifications.\n- Full frames and horizontal videos remain in ../examples/recon/id_25 and ../examples/unitree_go2/id_81.\n- manifest.json records sample origins, full-horizon metrics and source hashes.\n\nThese are selected visual-prediction examples, not closed-loop navigation evaluation. RECON mean PSNR: OpenNWM 14.351, NWM 13.259, RAE-NWM 12.746; SSIM: 0.3532, 0.3603, 0.3509. Go2 mean PSNR: 14.943, 12.696, 12.335; SSIM: 0.5432, 0.4659, 0.5325.\n\nReproduce from repository root:\n```bash\n/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python scripts/render_navigation_paper.py\n```\n')
shutil.copy2(__file__,O/'render_navigation_paper.py')
# Small preview only; publication images retain native embedded pixels.
im=Image.open(O/'navigation_comparison.png');im.thumbnail((2200,1200));im.save(O/'preview.png')
print(O)
