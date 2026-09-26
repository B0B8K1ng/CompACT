import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nav1_l20_direct4s_20260924')
sys.path.insert(0,str(root/'scripts'));import run_a800_nwm_eval as bench
for label in ['u25l100','u50l100','u75l100']:
 d=sys.argv[1] if len(sys.argv)>1 else 'recon'; target=out/'metrics'/f'{label}_{d}.json';target.parent.mkdir(exist_ok=True)
 digest=json.loads((out/'views'/label/'provenance.json').read_text())['sha256']
 cmd=[sys.executable,str(root/'scripts/evaluate_nwm_predictions.py'),'--gt-dir',str(out/'gt'/d/'time'),'--pred-dir',str(out/'predictions'/label/d/'time'),'--output',str(target),'--frames','4s:4','--dataset',d,'--eval-type','time','--eval-name','direct_4s_v1','--batch-size','32','--device','cuda','--dreamsim-cache','/file_system/nas/algorithm/dujun.nie/nwm/compact/cache/models','--fid','--inference-backend','nwm','--sampler','ddpm','--sampling-steps','250','--seed','0','--checkpoint-sha256',digest,'--split-sha256',bench.sha(bench.expected_split(d,'time')[0])]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='5',TORCH_HOME='/file_system/vepfs/algorithm/dujun.nie/models',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4')
 for k in list(env):
  if k.lower().endswith('_proxy'):env.pop(k)
 log=out/'logs'/f'{label}_{d}_early_metrics.log'
 with log.open('w') as f:
  p=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  rec=dict(command=cmd,cwd=str(root),pid=p.pid,gpu=5,log=str(log));print(json.dumps(rec),flush=True)
  code=p.wait();rec['exit_code']=code;(out/'jobs'/f'{label}_{d}_early_metrics.json').write_text(json.dumps(rec,indent=2))
  print('EXIT',label,code,flush=True)
  if code:raise RuntimeError(log)
