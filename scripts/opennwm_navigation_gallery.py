#!/usr/bin/env python3
"""Reproducible OpenNWM goal planning and RGB-feedback qualitative gallery."""
import argparse
import csv
import hashlib
import json
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from run_a800_nwm_eval import command_for, env_for
from prepare_tum_camera_heading import heading_from_metadata

ROOT = Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/opennwm_navigation_gallery_20260924')
DATA = Path('/file_system/nas/algorithm/dujun.nie/nwm/data')
MODEL = 'opennwm-finalLAM-100k'
STEM = 'CEM_N80_K5_RS1_rep3_OPT1_COST-lpips-RECON-True_PIXEL-FEEDBACK'
CONFIG = yaml.safe_load((REPO / 'conf/plan_config.yaml').read_text())
LABELS = dict(recon='RECON', scand='SCAND', huron='HuRoN', tartan_drive='TartanDrive', go_stanford='Go Stanford', unitree_go2='Office-go2', tum_rgbd='TUM', uzh_fpv='UZH-FPV')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def source(dataset, sample):
    cfg = CONFIG['evaluation_datasets'][dataset]
    split = REPO / cfg['navigation_index']
    name, start, lo, hi = pickle.loads(split.read_bytes())[sample]
    assert lo == hi == 8
    directory = DATA / cfg['data_folder'].split('/')[-1] / name
    posefile = directory / 'traj_data.pkl'
    pose = pickle.loads(posefile.read_bytes())
    xy = np.asarray(pose['position'], dtype=float)[start:start + 9, :2]
    yaw = float(np.asarray(pose['yaw'][start]).item())
    if dataset == 'tum_rgbd':
        yaw = float(heading_from_metadata(str(directory / 'frame_metadata.jsonl'))[start])
    assert xy.shape == (9, 2) and start >= 3
    assert all((directory / f'{t}.jpg').exists() for t in range(start - 3, start + 9))
    c, s = np.cos(yaw), np.sin(yaw)
    xy = (xy - xy[0]) @ np.array([[c, -s], [s, c]])
    return dict(dataset=dataset, sample_id=sample, trajectory=name, start_frame=int(start), goal_frame=int(start + 8), source_dir=str(directory), split=str(split), spacing_m=cfg['metric_waypoint_spacing'], gt_xy_m=xy.tolist(), distance_m=float(np.linalg.norm(xy[-1])))


def prepare():
    candidates = {}
    for ds in LABELS:
        cfg = CONFIG['evaluation_datasets'][ds]
        mu = CONFIG['plan_datasets_hyperparams'][ds].get('mu', [-.1, 0, 0])[0]
        if ds == 'huron':
            mu = -.1
        expected = ((mu + 1) * 3.75 - 2.5) * 8 * cfg['metric_waypoint_spacing']
        rows, invalid = [], []
        for i in range(cfg['navigation_sample_count']):
            try:
                r = source(ds, i)
            except (OSError, ValueError, IndexError, AssertionError) as e:
                invalid.append(dict(sample_id=i, reason=str(e)))
                continue
            xy = np.array(r['gt_xy_m'])
            d = r['distance_m']
            straight = np.arange(9)[:, None] / 8 * xy[-1]
            shape_error = float(np.sqrt(np.mean(np.sum((xy - straight) ** 2, axis=1)))) / max(d, .01)
            r['geometry_score'] = abs(np.log(max(d, .01) / max(expected, .01))) + shape_error * 2 + abs(np.arctan2(xy[-1, 1], xy[-1, 0])) * .25
            r['eligible_geometry'] = bool(d > .15 and xy[-1, 0] > .1 and shape_error < .3)
            rows.append(r)
        ordered = sorted([r for r in rows if r['eligible_geometry']], key=lambda r: r['geometry_score'])
        picked = []
        if ds == 'recon':
            picked = [3]
        if ds == 'unitree_go2':
            picked = [1, 3]
        count = 3 if ds in ['huron', 'tum_rgbd'] else 2
        for diverse in [True, False]:
            for r in ordered:
                if len(picked) >= count:
                    break
                names = {x['trajectory'] for x in rows if x['sample_id'] in picked}
                if r['sample_id'] not in picked and (not diverse or r['trajectory'] not in names):
                    picked.append(r['sample_id'])
        candidates[ds] = dict(initial_ids=picked, ranked_ids=[r['sample_id'] for r in ordered], valid_windows=rows, invalid_windows=invalid)
    write(ROOT / 'candidates.json', candidates)
    write(ROOT / 'protocol.json', dict(model=MODEL, feedback='decoded RGB reencoded at every imagined step', context_frames=4, horizon=8, seed=42, CEM=dict(population=80, topk=5, updates=1, repetitions=3), huron_prior_override=dict(mu=[-.1,0,0], var_scale=[.1,.15,.1]), selection='Prefer normalized ATE <= 0.25 and normalized goal error <= 0.30; choose two with lowest maximum normalized error, favor different source trajectories. Keep all results.', frame_comparison='GT follows recorded actions; predictions follow the goal-conditioned planned actions. These are different paths, so no pixel reconstruction metric is claimed.', excluded=dict(planetary_rover='Positions are median-step units; metric navigation evaluation prohibited by data contract.')))
    protocol=json.loads((ROOT/'protocol.json').read_text())
    protocol.update(source_index_handling='HuRoN keeps original split positions via planning_preserve_source_indices=true; requested rows must be present in loader-valid entries.',
                    visual_screening='Reject clearly unrelated scene content after viewing unmodified predictions; reasons retained in visual_review.json.',
                    evaluation_scope='Offline goal-image-conditioned local planning; no closed-loop execution measured.',
                    candidate_family='Constant planar delta over eight steps',sampling_steps=250,fps=4,
                    replay='Preserve exact job grouping, runtime, sample seeds, microbatch 80 and resolved config. Floating-point changes can alter CEM top-k selection.')
    write(ROOT/'protocol.json',protocol)
    print({ds: v['initial_ids'] for ds, v in candidates.items()}, flush=True)


def run(args):
    candidates = json.loads((ROOT / 'candidates.json').read_text())
    for ds in args.datasets:
        ids = args.ids if args.ids is not None else candidates[ds]['initial_ids']
        job = dict(kind='navigation', model=MODEL, dataset=ds, evaluation='navigation', ids=ids)
        cmd = command_for(job, ROOT, 80)
        cmd += ['+planning_image_feedback=true', '+save_planned_trajectories=true', '+save_navigation_rollout=true']
        if ds == 'huron':
            cmd += ['+planning_preserve_source_indices=true', 'plan_datasets_hyperparams.huron.mu=[-0.1,0,0]', 'plan_datasets_hyperparams.huron.var_scale=[0.1,0.15,0.1]']
        env = env_for(MODEL, args.gpu)
        env.update(TORCH_HOME='/file_system/vepfs/algorithm/dujun.nie/models', NWM_INDEX_ROOT='/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices')
        tag = ds + '_' + '-'.join(map(str, ids))
        log = ROOT / 'logs' / (tag + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        attempt=1
        while log.exists():
            attempt+=1
            log=ROOT/'logs'/f'{tag}_attempt{attempt}.log'
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            metadata = dict(command=cmd, shell_command=shlex.join(cmd), cwd=str(REPO), gpu=args.gpu, pid=proc.pid, log=str(log))
            write(log.with_suffix('.json'), metadata)
            print(json.dumps(metadata), flush=True)
            metadata['exit_status'] = proc.wait()
            write(log.with_suffix('.json'), metadata)
            print(ds, 'EXIT', metadata['exit_status'], flush=True)
            if metadata['exit_status']:
                raise SystemExit(metadata['exit_status'])


def metrics(ds, sample):
    r = source(ds, sample)
    directory = ROOT / 'planning' / MODEL / ds / STEM
    traj = json.loads((directory / 'trajectories' / f'{sample:06d}.json').read_text())
    gt = np.array(r['gt_xy_m'])
    pred = np.vstack([np.zeros(2), np.array(traj['predicted_xy_waypoint_units']) * r['spacing_m']])
    np.testing.assert_allclose(np.array(traj['gt_xy_waypoint_units']) * r['spacing_m'], gt[1:], atol=2e-5)
    np.testing.assert_allclose(np.array(traj['goal_pose_waypoint_units'])[0, :2] * r['spacing_m'], gt[-1], atol=2e-5)
    error = np.linalg.norm(pred[1:] - gt[1:], axis=1)
    ate, goal = float(np.sqrt(np.mean(error ** 2))), float(error[-1])
    saved = json.loads((directory / 'sample_metrics' / f'{sample:06d}.json').read_text())
    np.testing.assert_allclose([ate, goal], np.array([saved['ate'], saved['pos_diff_norm']]) * r['spacing_m'], atol=2e-5)
    r.update(planned_xy_m=pred.tolist(), ate_m=ate, goal_error_m=goal, normalized_ate=ate/r['distance_m'], normalized_goal_error=goal/r['distance_m'], rollout_path=str(directory / 'navigation_rollouts' / f'{sample:06d}' / 'rollout.npz'))
    r['quality_pass'] = r['normalized_ate'] <= .25 and r['normalized_goal_error'] <= .30
    r['selection_score'] = max(r['normalized_ate'], r['normalized_goal_error'])
    return r


def render_case(r, collection='gallery'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import misc
    from PIL import ImageDraw, ImageFont
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    out = ROOT / collection / f"{r['dataset']}_{r['sample_id']:03d}"
    out.mkdir(parents=True, exist_ok=True)
    transform = misc.get_transform(224, [.5]*3, [.5]*3)
    paths = [Path(r['source_dir']) / f'{i}.jpg' for i in range(r['start_frame'], r['goal_frame']+1)]
    gt_images = [np.uint8(np.clip((transform(Image.open(p).convert('RGB')).permute(1,2,0).numpy()*.5+.5)*255,0,255)) for p in paths]
    data = np.load(r['rollout_path'])
    assert data['predicted_rgb'].shape == (8,3,224,224)
    assert np.isfinite(data['predicted_rgb']).all()
    predictions = [gt_images[0]] + [np.uint8(np.clip(x.transpose(1,2,0)*255,0,255)) for x in data['predicted_rgb']]
    r['prediction_temporal_rgb_mae_255'] = [float(np.abs(a.astype(float)-b.astype(float)).mean()) for a,b in zip(predictions[:-1],predictions[1:])]
    r['gt_temporal_rgb_mae_255'] = [float(np.abs(a.astype(float)-b.astype(float)).mean()) for a,b in zip(gt_images[:-1],gt_images[1:])]
    r['identical_predicted_pairs'] = [i for i in range(1,8) if np.array_equal(predictions[i],predictions[i+1])]
    for name, images in [('GT',gt_images), ('OpenNWM',predictions)]:
        folder=out/name;folder.mkdir(exist_ok=True)
        for t,img in enumerate(images):Image.fromarray(img).save(folder/f'{t:03d}.png')
    indices=[0,2,4,6,8]
    fig=plt.figure(figsize=(12,8),facecolor='white')
    top=fig.add_gridspec(1,3,width_ratios=[1,1.3,1],left=.11,right=.985,top=.925,bottom=.69,wspace=.40)
    grid=fig.add_gridspec(2,5,left=.11,right=.985,top=.61,bottom=.025,hspace=.12,wspace=.035)
    fig.suptitle(LABELS[r['dataset']],fontsize=15,y=.98)
    for sl,img,title in [(top[0,0],gt_images[0],'Start'),(top[0,2],gt_images[-1],'Goal')]:
        ax=fig.add_subplot(sl);ax.imshow(img);ax.axis('off');ax.set_title(title,pad=4)
    ax=fig.add_subplot(top[0,1]);gt=np.array(r['gt_xy_m']);pred=np.array(r['planned_xy_m'])
    ax.plot(-gt[:,1],gt[:,0],'--',color='#333333',lw=1.8,label='GT',zorder=4)
    ax.plot(-pred[:,1],pred[:,0],color='#168373',lw=2,marker='o',ms=2.7,label='OpenNWM')
    ax.scatter(0,0,s=30,facecolor='white',edgecolor='#333333',zorder=6)
    ax.scatter(-gt[-1,1],gt[-1,0],marker='*',s=100,color='#333333',zorder=6)
    ax.set_aspect('equal',adjustable='datalim');ax.margins(.18);ax.grid(color='#eeeeee',lw=.6);ax.set_axisbelow(True)
    ax.set_xlabel('Lateral (m)',fontsize=9,labelpad=1);ax.set_ylabel('Forward (m)',fontsize=9,labelpad=2);ax.tick_params(labelsize=8)
    ax.set_title(f"ATE {r['ate_m']:.3f} m   ·   Goal error {r['goal_error_m']:.3f} m",fontsize=10,pad=5)
    ax.legend(frameon=False,loc='upper right',fontsize=9)
    for row,(label,images) in enumerate([('GT',gt_images),('OpenNWM',predictions)],1):
        for col,t in enumerate(indices):
            a=fig.add_subplot(grid[row-1,col]);a.imshow(images[t]);a.axis('off')
            if row==1:a.set_title(f't = {t/4:g} s',fontsize=10,pad=4)
            if col==0:a.text(-.06,.5,label,transform=a.transAxes,ha='right',va='center',fontsize=10)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'navigation.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
    preview=Image.open(out/'navigation.png');preview.thumbnail((1500,1000));preview.save(out/'preview.png')
    with (out/'trajectories_m.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','time_s','GT_x','GT_y','OpenNWM_x','OpenNWM_y'])
        writer.writerows([[i,i/4,*gt[i],*pred[i]] for i in range(9)])
    inputs=paths+[Path(r['source_dir'])/f"{r['start_frame']-i}.jpg" for i in [3,2,1]]+[Path(r['source_dir'])/'traj_data.pkl',Path(r['split'])]
    if r['dataset']=='tum_rgbd':inputs.append(Path(r['source_dir'])/'frame_metadata.jsonl')
    r['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    r['rollout_sha256']=hashlib.sha256(Path(r['rollout_path']).read_bytes()).hexdigest()
    write(out/'sample.json',r)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    video_frames=[]
    for t in range(9):
        frame=Image.new('RGB',(672,256),'white');draw=ImageDraw.Draw(frame)
        for col,(label,img) in enumerate([('GT',gt_images[t]),('OpenNWM',predictions[t])]):
            frame.paste(Image.fromarray(img),(224*col,32));draw.text((224*col+9,7),label,fill='#333333' if col==0 else '#168373',font=font)
        draw.text((457,7),f't = {t/4:g} s',fill='#333333',font=font)
        xy=np.concatenate([gt,pred]);display=np.column_stack([-xy[:,1],xy[:,0]]);low=display.min(0);high=display.max(0);span=max(np.max(high-low),.1);center=(low+high)/2
        def points(path):
            z=(np.column_stack([-path[:,1],path[:,0]])-center)/span*180
            return [(int(560+x),int(145-y)) for x,y in z]
        for path,color in [(gt,'#333333'),(pred,'#168373')]:
            q=points(path[:t+1])
            if len(q)>1:draw.line(q,fill=color,width=3)
            x,y=q[-1];draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        x,y=points(gt[-1:])[0];draw.ellipse((x-4,y-4,x+4,y+4),outline='#333333',width=2)
        video_frames.append(np.asarray(frame).tobytes())
    subprocess.run(['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-y',
                    '-f','rawvideo','-pix_fmt','rgb24','-s','672x256','-r','4','-i','pipe:0',
                    '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'comparison.mp4')],
                   input=b''.join(video_frames),check=True)
    return out


def render():
    all_cases=[];selected=[]
    reviews=json.loads((ROOT/'visual_review.json').read_text()) if (ROOT/'visual_review.json').exists() else {}
    for ds in LABELS:
        cases=[]
        for p in (ROOT/'planning'/MODEL/ds/STEM/'sample_metrics').glob('*.json'):
            r=metrics(ds,int(p.stem))
            review=reviews.get(f"{ds}/{r['sample_id']}",{})
            r['visual_review']=review
            if Path(r['rollout_path']).exists():cases.append(r)
        cases.sort(key=lambda r:r['selection_score']);all_cases.extend(cases)
        good=[r for r in cases if r['quality_pass'] and not r['visual_review'].get('reject',False)]
        chosen=[]
        for diverse in [True,False]:
            for r in good:
                if len(chosen)==2:break
                if r not in chosen and (not diverse or r['trajectory'] not in {x['trajectory'] for x in chosen}):chosen.append(r)
        selected.extend(chosen)
        print(ds,[(r['sample_id'],round(r['normalized_ate'],3),round(r['normalized_goal_error'],3),r['quality_pass']) for r in cases],flush=True)
    write(ROOT/'selection_audit.json',all_cases)
    selected_keys={(r['dataset'],r['sample_id']) for r in selected}
    for r in all_cases:
        collection='gallery' if (r['dataset'],r['sample_id']) in selected_keys else 'all_candidates'
        name=f"{r['dataset']}_{r['sample_id']:03d}"
        out=ROOT/collection/name
        other=ROOT/('all_candidates' if collection=='gallery' else 'gallery')/name
        if other.exists() and not out.exists():
            out.parent.mkdir(exist_ok=True)
            shutil.move(str(other),str(out))
        if not all((out/name).exists() for name in ['navigation.pdf','navigation.png','comparison.mp4','sample.json']):render_case(r,collection)
    write(ROOT/'selected_samples.json',selected)
    lines=['# OpenNWM goal-conditioned navigation','', 'Decoded predicted images are reencoded for every rollout step. GT images follow the recorded trajectory; predicted images follow the planned trajectory.','']
    for r in selected:
        folder=f"{r['dataset']}_{r['sample_id']:03d}";lines += [f"## {LABELS[r['dataset']]} · {r['sample_id']}",'',f"![Navigation](gallery/{folder}/preview.png)",'',f"[PDF](gallery/{folder}/navigation.pdf) · [Video](gallery/{folder}/comparison.mp4)",'']
    (ROOT/'INDEX.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('prepare');sub.add_parser('render')
    p=sub.add_parser('run');p.add_argument('--gpu',required=True);p.add_argument('--datasets',nargs='+',required=True,choices=list(LABELS));p.add_argument('--ids',nargs='+',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args)
    else:render()
