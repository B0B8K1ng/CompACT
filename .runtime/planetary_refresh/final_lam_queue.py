import json, os, shlex, subprocess, time
from pathlib import Path
ROOT=Path('/file_system/vepfs/algorithm/dujun.nie/code/CompACT')
EVAL_ROOT=Path('/file_system/vepfs/algorithm/dujun.nie/code/CompACT-eval-huron-fix')
OUT=Path('/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/planetary_refresh_20260922')
NAV=OUT.parent/'navigation_largebatch_20260921'
PYTHON='/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/python'
MODEL='nwm-latentpt-pixel-action-finalLAM-ft-reset'
command=[PYTHON,'scripts/run_nwm_benchmark.py','--models',MODEL,'--metrics','direct','--datasets','planetary_rover','--gpus','6','--batch-size','10','--metric-batch-size','10','--benchmark-root',str(OUT),'--shared-benchmark-root',str(OUT)]
log=OUT/'logs/finalLAM_planetary_direct.log'
job_path=OUT/'finalLAM_planetary_job.json'
env=os.environ.copy(); env.update(PATH=str(Path(PYTHON).parent)+':'+env.get('PATH',''),PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
state={'state':'running','cwd':str(ROOT),'command':command,'shell_command':shlex.join(command),'gpu':6,'batch_size':10,'metric_batch_size':10,'queue_pid':os.getpid(),'log':str(log),'started_at':time.time(),'resume_navigation_on_success':True}
def save():
    tmp=job_path.with_name(job_path.name+f'.tmp.{os.getpid()}'); tmp.write_text(json.dumps(state,indent=2)+'\n'); tmp.replace(job_path)
save()
with log.open('a') as stream:
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    state['pid']=process.pid; save()
    print('START',shlex.join(command),'PID',process.pid,'LOG',log,flush=True)
    rc=process.wait()
state.update(exit_code=rc,finished_at=time.time(),state='complete' if rc==0 else 'failed_navigation_paused'); save()
if rc:
    print('FAILED exit',rc,'navigation remains paused for diagnosis',flush=True)
    raise SystemExit(rc)
# Resume exactly the navigation job paused for this completion run.
jobs=json.loads((NAV/'jobs.json').read_text())
resumed=[]
for j in jobs:
    if j.get('state')=='paused' and j.get('pause_reason')=='user_requested_finalLAM_planetary_direct_4s_completion':
        j['state']='pending'; j['resumed_at']=time.time(); resumed.append(f"{j['model']}/{j['dataset']}")
tmp=(NAV/'jobs.json').with_suffix('.planetary_resume.tmp'); tmp.write_text(json.dumps(jobs,indent=2)); tmp.replace(NAV/'jobs.json')
with (NAV/'logs/scheduler.log').open('a') as stream:
    scheduler=subprocess.Popen([PYTHON,'.runtime/navlarge/schedule.py'],cwd=EVAL_ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
(NAV/'scheduler.pid').write_text(str(scheduler.pid)+'\n')
state.update(state='complete_navigation_resumed',navigation_scheduler_pid=scheduler.pid,resumed_jobs=resumed); save()
print('COMPLETE exit=0; navigation scheduler PID',scheduler.pid,'resumed',resumed,flush=True)
