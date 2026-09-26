#!/usr/bin/env python3
"""Render saved rollouts with true cumulative pose paths and sparse time columns."""
import argparse
import json
import pickle
import shutil
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from make_opennwm_rollout_videos import make_video, PANELS, FONT_PATH

REPO = Path(__file__).resolve().parents[1]
DATA = Path('/file_system/nas/algorithm/dujun.nie/nwm/data')
LAYOUT = {'huron': ('sacson','sacson'), 'tartan_drive': ('tartan_drive','tartan')}


def link_saved_frame(source, destination):
    if not Path(destination).exists():
        os.link(source, destination)
    return destination


def trajectory(case):
    dataset = case['dataset']
    split, folder = LAYOUT.get(dataset, (dataset,dataset))
    if 'source_entry' in case:
        name, start, *_ = case['source_entry']
    else:
        rows = pickle.loads((REPO/'data_splits'/split/'test/rollout.pkl').read_bytes())
        if dataset == 'huron':
            from raenwm_infer import filter_huron_rows
            rows = filter_huron_rows(rows, DATA/folder)
        name, start, *_ = rows[case['sample_id']]
    poses = pickle.loads((DATA/folder/name/'traj_data.pkl').read_bytes())
    if dataset=='tum_rgbd':
        from prepare_tum_camera_heading import heading_from_metadata
        poses['yaw']=heading_from_metadata(str(DATA/folder/name/'frame_metadata.jsonl'))
    p = np.asarray(poses['position'][start:start+case['seconds']*4+1])[:, :2]
    theta = float(np.asarray(poses['yaw'][start]).item())
    c,s = np.cos(theta),np.sin(theta)
    local = (p-p[0]) @ np.array([[c,-s],[s,c]])
    return local, {'trajectory': name, 'initial_frame': int(start), 'positions_local': local.tolist(),
                   'source_pose_file': str(DATA/folder/name/'traj_data.pkl'),
                   'axes': 'local +x forward, +y left; display up forward, right negative local y',
                   'time_semantics': 'playback_time_4fps_irregular_source' if dataset=='planetary_rover' else '4fps_source_samples',
                   'sample_times_seconds': (np.arange(len(local))/4).tolist()}


def render(case, root):
    out = root/'examples'/case['dataset']/f"id_{case['sample_id']}"
    initial = root/'initial'/case['dataset']/f"id_{case['sample_id']}.png"
    seconds = list(range(0,case['seconds']+1,2 if case['group']=='ID' else 1))
    size,gap,left,header,traj_h = 224,8,112,32,150
    sheet=Image.new('RGB',(left+len(seconds)*(size+gap),header+traj_h+4*(size+gap)), 'white')
    draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(str(FONT_PATH),18)
    if case['dataset']=='planetary_rover':
        draw.text((left-12,5),'Playback',font=ImageFont.truetype(str(FONT_PATH),14),fill='#222222',anchor='rt')
    local, provenance=trajectory(case)
    # Robot forward is +x and left is +y: image up=-x, image right=-y.
    xy=np.column_stack((-local[:,1],-local[:,0]))
    origin=np.array([size*.5,traj_h*.76])
    bounds=np.array([size*.44,traj_h*.67])
    scale=min(bounds[0]/max(abs(xy[:,0]).max(),1e-8),bounds[1]/max(-xy[:,1].min(),1e-8),traj_h*.17/max(xy[:,1].max(),1e-8))
    points=xy*scale+origin
    draw.text((left-12,header+traj_h/2),'Trajectory',font=font,fill='#222222',anchor='rm')
    for col,t in enumerate(seconds):
        x=left+col*(size+gap)
        draw.text((x+size/2,5),f't={t}s',font=font,fill='#222222',anchor='mt')
        path=[(x+float(a),header+float(b)) for a,b in points[:t*4+1]]
        if len(path)>1:draw.line(path,fill='#32658c',width=3)
        px,py=path[0];draw.ellipse((px-3,py-3,px+3,py+3),outline='#32658c',width=1)
        px,py=path[-1];draw.ellipse((px-4,py-4,px+4,py+4),fill='#32658c')
        for row,(label,folder,_) in enumerate(PANELS):
            y=header+traj_h+row*(size+gap)
            frame=initial if t==0 else out/'frames'/folder/f'{t*4-1:03d}.png'
            with Image.open(frame) as im:sheet.paste(im.convert('RGB'),(x,y))
            if col==0:draw.text((left-12,y+size/2),label,font=font,fill='#222222',anchor='rm')
    sheet.save(out/'sequence.png',dpi=(200,200))
    sheet.save(out/'sequence.pdf',resolution=200)
    (out/'trajectory.json').write_text(json.dumps(provenance,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cases=json.loads((a.source/'summary.json').read_text())
    for case in cases:
        rel=Path('examples')/case['dataset']/f"id_{case['sample_id']}"
        shutil.copytree(a.source/rel/'frames',a.output/rel/'frames',dirs_exist_ok=True,copy_function=link_saved_frame)
    if (a.source/'initial').exists():shutil.copytree(a.source/'initial',a.output/'initial',dirs_exist_ok=True)
    if (a.source/'protocol.json').exists():shutil.copy2(a.source/'protocol.json',a.output/'protocol.json')
    videos=[]
    for case in cases:
        # Historical four-real-context runs have no saved t=0; reconstruct the
        # current GT observation from the same source preprocessing.
        initial=a.output/'initial'/case['dataset']/f"id_{case['sample_id']}.png"
        if not initial.exists():
            import sys
            sys.path.insert(0,str(REPO))
            import misc
            _,meta=trajectory(case);_,folder=LAYOUT.get(case['dataset'],(case['dataset'],case['dataset']))
            path=DATA/folder/meta['trajectory']/f"{meta['initial_frame']}.jpg"
            with Image.open(path) as im:tensor=misc.get_transform(224,[.5]*3,[.5]*3)(im.convert('RGB'))
            arr=((tensor*.5+.5).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
            initial.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(initial)
        render(case,a.output)
        videos.append(make_video(case,a.output,True))
    (a.output/'summary.json').write_text(json.dumps(cases,indent=2))
    (a.output/'video_index.json').write_text(json.dumps(videos,indent=2))
    print(f'DONE {len(cases)} figures and videos',flush=True)

if __name__=='__main__':main()
