"""Render an existing release navigation sample without rerunning inference."""
import argparse
import base64
import io
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAS = Path('/file_system/nas/algorithm/dujun.nie/nwm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-id', type=int, default=0)
    ap.add_argument('--output-dir', type=Path, default=NAS / 'demo_outputs/recon_navigation_example_20260910')
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = NAS / f'results/release_eval_20260820/planning_compact80/nwm_cdit_b/recon/CEM_N80_K5_RS1_rep3_OPT1/id_{args.sample_id}/preds_0.pth'
    d = torch.load(source, map_location='cpu', weights_only=False)
    with open(ROOT / 'data_splits/recon/test/navigation_eval.pkl', 'rb') as f:
        name, t, lo, hi = pickle.load(f)[args.sample_id]
    assert lo == hi == 8, 'This renderer requires the fixed eight-step goal protocol.'
    folder = NAS / 'data/recon' / name
    with open(folder / 'traj_data.pkl', 'rb') as f:
        raw = pickle.load(f)
    yaw = float(np.asarray(raw['yaw'][t]).item())
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    gt = (raw['position'][t:t+9, :2] - raw['position'][t, :2]) @ rot
    # Validate meter conversion against the original odometry, not just config.
    np.testing.assert_allclose(d['gt_actions'][:, :2].float().numpy() * .25, gt[1:], atol=1e-5)
    delta = d['deltas'][:, :2].float().numpy()
    delta = (delta + 1) / 2 * np.array([7.5, 8.0]) + np.array([-2.5, -4.0])
    pred = np.vstack([np.zeros(2), np.cumsum(delta, axis=0) * .25])
    ate = float(np.sqrt(np.mean(np.sum((pred[1:] - gt[1:])**2, axis=1))))
    endpoint = float(np.linalg.norm(pred[-1] - gt[-1]))
    def tensor_image(x):
        a = x.float().numpy().transpose(1, 2, 0)
        return Image.fromarray(np.uint8(np.clip((a + 1) / 2, 0, 1) * 255))
    images = [tensor_image(d['obs_image'][-1]), Image.open(folder / f'{t+8}.jpg').convert('RGB'),
              tensor_image(d['goal_image'][0]), tensor_image(d['nwm_preds'])]
    titles = [f'Current observation | frame {t}', f'Goal photo | frame {t+8}',
              'Saved goal used by evaluation', 'NWM predicted final view']
    fig = plt.figure(figsize=(16, 10), facecolor='#f5f7fb')
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1.35, .68], hspace=.34)
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(grid[0, i]); ax.imshow(im); ax.set_title(title, fontsize=11); ax.axis('off')
    ax = fig.add_subplot(grid[1:, :2])
    ax.plot(gt[:, 0], gt[:, 1], 'o-', color='#008c78', label='Recorded reference trajectory')
    ax.plot(pred[:, 0], pred[:, 1], 'o-', color='#e67e22', label='NWM planned trajectory')
    ax.scatter(0, 0, s=130, c='#25304b', label='Start', zorder=4)
    ax.scatter(*gt[-1], marker='*', s=350, c='#7c3aed', label='Goal location (scoring only)', zorder=5)
    ax.plot([gt[-1, 0], pred[-1, 0]], [gt[-1, 1], pred[-1, 1]], '--', c='#7c3aed')
    for k in [1, 4, 8]:
        ax.annotate(str(k), pred[k], xytext=(3, 7), textcoords='offset points')
    ax.set(xlabel='Forward x (m)', ylabel='Left y (m)', title='Start-relative coordinates | 8 planned steps')
    ax.axis('equal'); ax.grid(alpha=.2); ax.legend(fontsize=9, loc='upper left')
    ax = fig.add_subplot(grid[1, 2:]); ax.axis('off')
    ax.text(0, 1, 'HOW THE GOAL IS GIVEN', fontsize=15, weight='bold', va='top')
    ax.text(0, .84, f'1. Observation history: frames {t-3}, {t-2}, {t-1}, {t}.\n'
            f'2. Goal: a future camera image, frame {t+8} (t + 8).\n'
            '3. Sample 80 candidate action sequences; predict final views.\n'
            '4. Compare predicted views with the goal using LPIPS.\n'
            '5. Fit the top 5 candidates; output the 8-step plan.\n\n'
            f'ATE: {ate:.3f} m    |    Final position error: {endpoint:.3f} m\n'
            f'Saved final image cost: {float(d["loss"]):.3f}', fontsize=12, va='top', linespacing=1.6)
    ax = fig.add_subplot(grid[2, 2:]); ax.axis('off')
    ax.text(0, 1, 'Offline planning result, not a closed-loop robot execution.\n'
            'Goal coordinates are used for scoring, not supplied to the planner.\n'
            'Only the final generated view was saved in this evaluation.\n'
            'Meters = benchmark waypoint coordinates x 0.25.', va='top', fontsize=11, linespacing=1.6)
    fig.suptitle(f'NWM release / RECON navigation / sample {args.sample_id}\n{name}', fontsize=17, y=.99)
    fig.savefig(out / 'overview.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    def uri(im):
        b = io.BytesIO(); im.save(b, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
    payload = dict(gt=gt.tolist(), pred=pred.tolist(), frames=[uri(Image.open(folder / f'{k}.jpg')) for k in range(t, t+9)])
    cards = ''.join(f'<div><img src="{uri(im)}"><p>{title}</p></div>' for im, title in zip(images, titles))
    html = '''<!doctype html><meta charset="utf-8"><title>NWM RECON 导航评测示例</title>
<style>body{font:16px system-ui;background:#f5f7fb;color:#25304b;max-width:1200px;margin:30px auto;padding:20px}h1{font-size:28px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}img{width:100%;border-radius:8px}.panel{background:white;border-radius:14px;padding:22px;margin:20px 0}.row{display:grid;grid-template-columns:1fr 1fr;gap:30px}canvas{width:100%}input{width:75%}p{line-height:1.7}</style>
<h1>NWM · RECON 导航评测的一条真实样本</h1>'''
    html += f'<p>样本 {args.sample_id} · {name}</p><div class="cards">{cards}</div>'
    html += f'''<div class="panel"><b>Goal 如何给？</b><p>观测历史为第 {t-3}–{t} 帧；goal 是同一段录制中第 {t+8} 帧的相机图像（固定向后 8 步）。输入的是目标画面。目标坐标只用于误差计算。当前配置对 goal 做 VAE 重建，再以 LPIPS 比较预测终点图像和 goal。</p>
<p>规划过程：采样 80 条候选动作序列 → NWM 预测终点图像 → 每个候选重复评估 3 次 → 按图像代价选择前 5 条 → 拟合并输出 8 步动作。此实现将同一个平移增量重复 8 步，因此计划位置呈直线。</p>
<p>单样本 ATE：<b>{ate:.3f} 米</b>；终点位置误差：<b>{endpoint:.3f} 米</b>。原评测坐标按 0.25 米的 waypoint spacing 还原，并与原始里程计核对。</p></div>
<div class="panel"><input id="step" type="range" min="0" max="8" value="8"><b id="label"></b><div class="row"><canvas id="plot" width="560" height="430"></canvas><div><img id="frame"><p id="caption"></p></div></div></div>
<p>绿色：真实录制轨迹；橙色：规划轨迹；紫色：goal。拖动滑块展示各步位置及对应真实相机帧。该评测只保存预测终点图，右侧播放的是录制帧，不是 NWM 生成的中间画面。没有模拟器或机器人闭环执行，不能据此宣称导航成功率。</p>'''
    html += '<script>const D=' + json.dumps(payload) + ';const start=' + str(t) + ';'
    html += '''const c=document.getElementById('plot'),ctx=c.getContext('2d'),s=document.getElementById('step');
const pts=D.gt.concat(D.pred),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
const xmin=Math.min(...xs)-.3,xmax=Math.max(...xs)+.3,ymin=Math.min(...ys)-.35,ymax=Math.max(...ys)+.35;
const scale=Math.min(460/(xmax-xmin),310/(ymax-ymin));
function xy(p){return [65+(p[0]-xmin)*scale,355-(p[1]-ymin)*scale]}
function dot(p,color,r){let [x,y]=xy(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill()}
function line(a,color,k){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();a.slice(0,k+1).forEach((p,i)=>{let q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();a.slice(0,k+1).forEach(p=>dot(p,color,4))}
function draw(){let k=+s.value;ctx.clearRect(0,0,560,430);ctx.font='14px system-ui';ctx.fillStyle='#25304b';ctx.fillText('起点局部坐标（米）· x 向前，y 向左',30,25);ctx.strokeStyle='#ddd';ctx.strokeRect(45,45,490,330);line(D.gt,'#008c78',k);line(D.pred,'#e67e22',k);dot(D.gt[8],'#7c3aed',9);dot([0,0],'#25304b',7);dot(D.pred[k],'#e67e22',7);ctx.fillStyle='#25304b';ctx.fillText('规划位置: '+D.pred[k].map(x=>x.toFixed(2)).join(', ')+' m',30,410);document.getElementById('label').textContent=' 第 '+k+' / 8 步';document.getElementById('frame').src=D.frames[k];document.getElementById('caption').textContent='真实录制画面 · 第 '+(start+k)+' 帧'+(k===8?' · Goal':'');}s.oninput=draw;draw();</script>'''
    (out / 'index.html').write_text(html)
    meta = dict(sample_id=args.sample_id, trajectory=name, observation_frames=list(range(t-3,t+1)), goal_frame=t+8,
                source=str(source), goal_image_path=str(folder / f'{t+8}.jpg'),
                gt_xy_m=gt.tolist(), planned_xy_m=pred.tolist(), ate_m=ate, final_position_error_m=endpoint,
                image_cost=float(d['loss']), waypoint_spacing_m=.25, mode='offline planning; existing saved result')
    (out / 'sample.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print('Artifacts:', out)


if __name__ == '__main__':
    main()
