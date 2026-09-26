#!/usr/bin/env python3
"""Diagnostic-only action amplitude sweep; never replaces benchmark predictions."""
import argparse,json,os,subprocess
from pathlib import Path
from run_corrected_rollout_search import ROOT
from run_opennwm_rollout_showcase import REPO
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--scales',nargs='+',required=True,type=float);a=p.parse_args()
source=ROOT/'logs/tum_rgbd_opennwm-finalLAM-100k_ids-124-81-20-127-73-92-34-86.status.json'
base=json.loads(source.read_text())['command']
for scale in a.scales:
 out=ROOT/'diagnostics/corrected_scale_sweep'/f'scale_{scale:g}';out.mkdir(parents=True,exist_ok=True)
 cmd=[f'output_dir={out}' if x.startswith('output_dir=') else f'prediction_dir={out}/predictions/opennwm-finalLAM-100k' if x.startswith('prediction_dir=') else 'eval_sample_indices=[124,81,20,127]' if x.startswith('eval_sample_indices=') else x for x in base]
 cmd.append(f'+diagnostic_rollout_action_scale={scale}')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,NWM_DATA_ROOT='/file_system/nas/algorithm/dujun.nie/nwm/data',NWM_INDEX_ROOT='/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices',TORCH_HOME='/file_system/vepfs/algorithm/dujun.nie/models',PYTHONUNBUFFERED='1')
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
  meta=dict(cwd=str(REPO),command=cmd,pid=proc.pid,log=str(out/'run.log'),gpu=a.gpu)
  print(json.dumps(meta),flush=True);(out/'command.json').write_text(json.dumps(meta,indent=2));status=proc.wait()
 meta['exit_status']=status;(out/'command.json').write_text(json.dumps(meta,indent=2));print('EXIT',status,scale,flush=True)
 if status:raise SystemExit(status)
