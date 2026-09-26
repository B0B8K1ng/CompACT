#!/usr/bin/env python3
"""Verify image counts, source alignment, timeline and complete video decoding."""
import concurrent.futures,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import misc
from run_corrected_rollout_search import ROOT
from render_rollout_academic import trajectory,LAYOUT,DATA


def verify(case):
    ds,i=case['dataset'],case['sample_id'];out=ROOT/'examples'/ds/f'id_{i}';n=case['seconds']*4
    initial=ROOT/'initial'/ds/f'id_{i}.png'
    with Image.open(initial) as im:assert im.size==(224,224);im.verify()
    name,start,*_=case['source_entry'];folder=LAYOUT.get(ds,(ds,ds))[1]
    transform=misc.get_transform(224,[.5]*3,[.5]*3)
    for offset,dest in [(0,initial),(n,out/'frames/GT'/f'{n-1:03}.png')]:
        with Image.open(DATA/folder/name/f'{start+offset}.jpg') as im:t=transform(im.convert('RGB'))
        expected=((t*.5+.5).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
        np.testing.assert_array_equal(expected,np.asarray(Image.open(dest).convert('RGB')))
    for model in ['GT','nwm-release','rae-nwm','opennwm-finalLAM-100k']:
        frames=sorted((out/'frames'/model).glob('*.png'))
        assert [p.name for p in frames]==[f'{k:03}.png' for k in range(n)],(ds,i,model)
        for p in frames:
            with Image.open(p) as im:assert im.size==(224,224);im.verify()
    with Image.open(out/'sequence.png') as im:
        assert im.size==(2200 if case['group']=='ID' else 1272,1110)
    local,_=trajectory(case);saved=json.loads((out/'trajectory.json').read_text())
    np.testing.assert_allclose(np.asarray(local,dtype=np.float64),np.asarray(saved['positions_local'],dtype=np.float64))
    assert len(local)==n+1
    p=subprocess.run(['/usr/bin/ffmpeg','-v','error','-threads','1','-i',str(out/'comparison.mp4'),'-f','null','-'],capture_output=True,text=True)
    assert p.returncode==0,(ds,i,p.stderr)
    info=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(out/'comparison.mp4')]))['streams'][0]
    assert (info['width'],info['height'],int(info['nb_frames']),info['r_frame_rate'])==(896,288,n+1,'4/1'),info
    return dict(dataset=ds,sample_id=i,prediction_frames_per_model=n,video_frames=n+1,full_video_decode='passed',image_decode='passed',trajectory='passed')

if __name__=='__main__':
    cases=json.loads((ROOT/'search_plan.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(verify,cases))
    (ROOT/'reproducibility/artifact_verification.json').write_text(json.dumps(results,indent=2))
    print('PASS',len(results),'cases',flush=True)
