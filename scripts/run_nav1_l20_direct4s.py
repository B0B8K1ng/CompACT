"""Evaluate all three verified Nav1 L20 release checkpoints on five full splits."""
import concurrent.futures, csv, hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path
import torch, yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_a800_nwm_eval as bench
OUT=Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nav1_l20_direct4s_20260924')
WEIGHTS=Path('/file_system/nas/algorithm/dujun.nie/nwm/weights/modelscope/LittleBoss/NWM-Nav1-LatentPT-L20')
DATASETS=['recon','scand','tartan_drive','huron','go_stanford']
PY=Path(sys.executable); TORCHRUN=PY.parent/'torchrun'
(OUT/'logs').mkdir(exist_ok=True); (OUT/'jobs').mkdir(exist_ok=True)
def write(p,obj):
 tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)
def run(name,cmd,gpu):
 env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),NWM_DATA_ROOT=str(bench.DATA),NWM_INDEX_ROOT='/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices',TORCH_HOME='/file_system/vepfs/algorithm/dujun.nie/models',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 for k in list(env):
  if k.lower().endswith('_proxy'): env.pop(k)
 log=OUT/'logs'/f'{name}.log'; start=time.time()
 with log.open('a') as f:
  f.write(shlex.join(list(map(str,cmd)))+'\n'); f.flush()
  p=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  rec=dict(command=list(map(str,cmd)),cwd=str(ROOT),pid=p.pid,gpu=gpu,log=str(log),status='running'); write(OUT/'jobs'/f'{name}.json',rec)
  print('START',name,'PID',p.pid,'GPU',gpu,'LOG',log,flush=True)
  code=p.wait()
 rec.update(exit_code=code,seconds=time.time()-start,status='complete' if code==0 else 'failed');write(OUT/'jobs'/f'{name}.json',rec)
 print('EXIT',name,code,'seconds',round(time.time()-start,1),flush=True)
 if code: raise RuntimeError(f'{name} failed: {log}')
def infer_cmd(view,pred,name,gt=False,batch=32):
 return [TORCHRUN,'--standalone','--nproc-per-node=1',ROOT/'isolated_nwm_infer.py',f'exp_dir={view}','ckp=final',f'output_dir={OUT}',f'prediction_dir={pred}',f'hydra.run.dir={OUT}/hydra/{name}',f'datasets_to_eval=[{",".join(DATASETS)}]','eval_type=time','eval_len_traj_pred=16','time_horizons_seconds=[4]','eval_diffusion_steps=250',f'batch_size={batch}','num_workers=4','pin_memory=false','seed=0',f'gt={int(gt)}']
def prepare(label):
 p=WEIGHTS/f'nav1_latentpt_{label}_l20_final.pth.tar'
 while not p.exists(): time.sleep(5)
 state=torch.load(p,map_location='cpu',weights_only=False,mmap=True); cfg=state['config']
 print('CHECKPOINT',label,'steps',state.get('train_steps'),'metadata',state.get('two_stage_metadata'),'generator',cfg['model']['generator'],flush=True)
 view=OUT/'views'/label; (view/'.hydra').mkdir(parents=True,exist_ok=True);(view/'checkpoints').mkdir(exist_ok=True)
 cfg['model']['tokenizer']['model_path']=str(bench.LOCAL_VAE)
 (view/'.hydra/config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 link=view/'checkpoints/final.pth.tar'
 if not link.exists(): link.symlink_to(p)
 remote=json.loads((WEIGHTS/'remote_manifest.json').read_text())
 meta=next(f for f in remote['Data']['Files'] if f['Path']==p.name)
 write(view/'provenance.json',dict(checkpoint=str(p),sha256=meta['Sha256'],train_steps=state.get('train_steps'),metadata=state.get('two_stage_metadata'),config=cfg))
 return view,meta['Sha256']
def validate_images(folder,dataset):
 expected=bench.count(dataset,'time'); paths=list(folder.glob('id_*/4.png'))
 assert len(paths)==expected,(folder,len(paths),expected)
 assert {p.parent.name for p in paths}=={f'id_{i}' for i in range(expected)}
 assert all(bench.good_png(p) for p in paths)
def worker(label,gpu):
 view,digest=prepare(label);pred=OUT/'predictions'/label
 run(label+'_inference',infer_cmd(view,pred,label),gpu)
 gt_future.result()
 for d in DATASETS:
  validate_images(pred/d/'time',d); validate_images(OUT/'gt'/d/'time',d)
  target=OUT/'metrics'/f'{label}_{d}.json'; target.parent.mkdir(exist_ok=True)
  cmd=[PY,ROOT/'scripts/evaluate_nwm_predictions.py','--gt-dir',OUT/'gt'/d/'time','--pred-dir',pred/d/'time','--output',target,'--frames','4s:4','--dataset',d,'--eval-type','time','--eval-name','direct_4s_v1','--batch-size','32','--device','cuda','--dreamsim-cache','/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models','--fid','--inference-backend','nwm','--sampler','ddpm','--sampling-steps','250','--seed','0','--checkpoint-sha256',digest,'--split-sha256',bench.sha(bench.expected_split(d,'time')[0])]
  if not target.exists():run(label+'_'+d+'_metrics',cmd,gpu)
 return label
def gt():
 run('ground_truth',infer_cmd(bench.MODEL['nwm-release']['exp_dir'],OUT/'unused','ground_truth',gt=True,batch=32),5)
 for d in DATASETS: validate_images(OUT/'gt'/d/'time',d)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 gt_future=pool.submit(gt)
 futures=[pool.submit(worker,label,gpu) for label,gpu in zip(['u25l100','u50l100','u75l100'],[2,6,7])]
 for f in concurrent.futures.as_completed(futures):print('COMPLETE',f.result(),flush=True)
rows=[]
for label in ['u25l100','u50l100','u75l100']:
 for d in DATASETS:
  r=json.loads((OUT/'metrics'/f'{label}_{d}.json').read_text());rows.append(dict(checkpoint=label,dataset=d,**r['metrics']['4s']))
write(OUT/'results.json',rows)
with (OUT/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines=['| 权重 | 数据集 | N | LPIPS ↓ | DreamSim ↓ | PSNR ↑ | FID ↓ |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['checkpoint']} | {r['dataset']} | {r['sample_count']} | {r['lpips_alex']:.5f} | {r['dreamsim']:.5f} | {r['psnr']:.3f} | {r['fid']:.3f} |")
(OUT/'results.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines),flush=True)
print('ALL 15 EVALUATIONS COMPLETE',flush=True)
