#!/usr/bin/env python3
import argparse,json,subprocess,shlex
from pathlib import Path
from run_a800_nwm_eval import command_for,env_for,ROOT,NAS
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--gpu',required=True);p.add_argument('--go2-ids',type=int,nargs='+',default=[0,1,2,3]);p.add_argument('--skip-recon',action='store_true');p.add_argument('--image-feedback',action='store_true');p.add_argument('--recon-ids',type=int,nargs='+',default=[3]);p.add_argument('--output-root',type=Path);a=p.parse_args()
out=a.output_root or NAS/('goal_navigation_pixel_paper_20260924' if a.image_feedback else 'goal_navigation_paper_20260924');(out/'logs').mkdir(parents=True,exist_ok=True)
for ds,ids in ([('recon',a.recon_ids)] if not a.skip_recon else [])+[('unitree_go2',a.go2_ids)]:
 job=dict(kind='navigation',model=a.model,dataset=ds,evaluation='navigation',ids=ids)
 cmd=command_for(job,out,80)
 cmd += ['--save-planned-trajectories'] if a.model=='rae-nwm' else ['+save_planned_trajectories=true']
 if a.image_feedback:cmd += ['--image-feedback'] if a.model=='rae-nwm' else ['+planning_image_feedback=true']
 env=env_for(a.model,a.gpu);env.update(TORCH_HOME='/file_system/vepfs/algorithm/dujun.nie/models',NWM_INDEX_ROOT='/file_system/nas/algorithm/dujun.nie/nwm/cache/dataset_indices')
 name=f'{a.model}_{ds}_'+ '-'.join(map(str,ids));log=out/'logs'/f'{name}.log'
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  meta=dict(command=cmd,shell_command=shlex.join(cmd),cwd=str(ROOT),pid=proc.pid,gpu=a.gpu,log=str(log));(out/'logs'/f'{name}.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
  status=proc.wait();meta['exit_status']=status;(out/'logs'/f'{name}.json').write_text(json.dumps(meta,indent=2));print('EXIT',status,flush=True)
  if status:raise SystemExit(status)
