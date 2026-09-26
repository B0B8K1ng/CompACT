#!/usr/bin/env python3
"""Frozen candidate search, subprocess logs and reproducible pixel rollouts."""
import argparse,json,os,pickle,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import run_opennwm_rollout_showcase as s
from raenwm_infer import filter_huron_rows
from render_rollout_academic import LAYOUT,DATA
ROOT=s.NAS/'results/nwm_benchmark/opennwm_pixel_corrected_20260924'
OLD=s.NAS/'results/nwm_benchmark/opennwm_single_image_pixel_rollout_20260923'


def plan():
    s.CASES['recon']={'group':'ID','ids':[],'seconds':16,'split_dir':'recon'}
    all_cases=[];rankings={}
    for dataset,case in s.CASES.items():
        rows=pickle.loads((s.REPO/'data_splits'/case['split_dir']/'test/rollout.pkl').read_bytes())
        valid=filter_huron_rows(rows,DATA/'sacson') if dataset=='huron' else rows
        rank,_=s.rank_direct_candidates(dataset);rankings[dataset]=rank
        candidates=[]
        for row in rank:
            entry=rows[row['sample_id']]
            if entry in valid:
                i=valid.index(entry)
                if i not in candidates:candidates.append(i)
        if not candidates:candidates=list(range(min(6,len(valid))))
        ids=list(dict.fromkeys(case['ids']+candidates[:4]))
        for i in ids:
            all_cases.append(dict(dataset=dataset,sample_id=i,seconds=case['seconds'],group=case['group'],fps=4,expected_count=len(valid),source_entry=list(valid[i])))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'search_plan.json').write_text(json.dumps(all_cases,indent=2))
    (ROOT/'candidate_rankings.json').write_text(json.dumps(rankings,indent=2))
    (ROOT/'protocol.json').write_text(json.dumps(dict(context_initialization='repeat_current_image_four_times',feedback='decoded_prediction_image_reencoded_each_step',action_frame='current_body_frame',seed=0,fps=4),indent=2))
    return all_cases


def gt(cases):
    sys.path.insert(0,str(s.REPO));import misc
    transform=misc.get_transform(224,[.5]*3,[.5]*3)
    for case in cases:
        ds,i=case['dataset'],case['sample_id'];name,start,*_=case['source_entry'];folder=LAYOUT.get(ds,(ds,ds))[1]
        out=s.paths(ds,i,None,ROOT);out.mkdir(parents=True,exist_ok=True)
        initial=ROOT/'initial'/ds/f'id_{i}.png';initial.parent.mkdir(parents=True,exist_ok=True)
        for step in range(case['seconds']*4+1):
            dest=initial if step==0 else out/f'{step-1}.png'
            if dest.exists():continue
            with Image.open(DATA/folder/name/f'{start+step}.jpg') as im:t=transform(im.convert('RGB'))
            array=((t*.5+.5).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
            Image.fromarray(array).save(dest)
        print('GT',ds,i,flush=True)


def run(cases,model,gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),NWM_DATA_ROOT=str(DATA),NWM_INDEX_ROOT=str(s.NAS/'cache/dataset_indices'),TORCH_HOME='/file_system/vepfs/algorithm/dujun.nie/models',HF_HOME=str(s.RAE_ASSETS/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    for ds in dict.fromkeys(c['dataset'] for c in cases):
        group=[c for c in cases if c['dataset']==ds]
        pending=[]
        for c in group:
            output=s.paths(ds,c['sample_id'],model,ROOT)
            if model=='rae-nwm' and ds!='tum_rgbd':
                old=s.paths(ds,c['sample_id'],model,OLD)
                if old.exists() and len(list(old.glob('*.png')))==c['seconds']*4:shutil.copytree(old,output,dirs_exist_ok=True)
            if all((output/f'{k}.png').exists() for k in range(c['seconds']*4)):continue
            pending.append(c)
        if not pending:continue
        c=pending[0];ids=[p['sample_id'] for p in pending]
        s.CASES[ds]=dict(seconds=c['seconds'],ids=ids,expected_count=c['expected_count'])
        cmd=s.command(ds,ids[0],model,ROOT,True)
        if ds=='planetary_rover':
            split=s.REPO/'data_splits/planetary_rover/test/time.pkl'
            cmd += ['--split-file',str(split)] if model=='rae-nwm' else [f'+evaluation_datasets.planetary_rover.predefined_index_path={split}']
        if model=='rae-nwm':
            index=cmd.index('--sample-indices');cmd[index+1:index+2]=list(map(str,ids))
            cmd[cmd.index('--expected-sample-count')+1]=str(c['expected_count'])
            cmd[cmd.index('--batch-size')+1]='4'
        else:
            cmd=[f'eval_sample_indices={ids}'.replace(' ','') if x.startswith('eval_sample_indices=') else 'batch_size=4' if x=='batch_size=1' else x for x in cmd]
        job=f"{ds}_{model}_ids-{'-'.join(map(str,ids))}"
        log=ROOT/'logs'/f'{job}.log'
        print('RUN',json.dumps(dict(cwd=str(s.REPO),command=cmd,log=str(log))),flush=True)
        with log.open('w') as f:
            p=subprocess.Popen(cmd,cwd=s.REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            print('PID',p.pid,flush=True);status=p.wait()
        (ROOT/'logs'/f'{job}.status.json').write_text(json.dumps(dict(command=cmd,pid=p.pid,exit_status=status)))
        print('EXIT',status,ds,model,flush=True)
        if status: print(log.read_text()[-4000:],flush=True)
    print('DONE',model,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','gt','run']);p.add_argument('--model',choices=list(s.MODELS)+['all']);p.add_argument('--gpu',type=int,default=2);p.add_argument('--datasets',nargs='+');a=p.parse_args()
    cases=plan() if a.stage=='plan' else json.loads((ROOT/'search_plan.json').read_text())
    if a.datasets:cases=[c for c in cases if c['dataset'] in a.datasets]
    if a.stage=='gt':gt(cases)
    if a.stage=='run':
        for model in s.MODELS if a.model=='all' else [a.model]:run(cases,model,a.gpu)
