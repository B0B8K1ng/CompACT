"""Direct multi-connection ModelScope download, pinned revision and SHA256 verified."""
import concurrent.futures, hashlib, json, os, sys, time
from pathlib import Path
import requests
from modelscope.hub.api import ModelScopeConfig
ROOT=Path('/file_system/nas/algorithm/dujun.nie/nwm/weights/modelscope/LittleBoss/NWM-Nav1-LatentPT-L20')
BASE='https://modelscope.cn/api/v1/models/LittleBoss/NWM-Nav1-LatentPT-L20'
def session():
 s=requests.Session();s.trust_env=False;s.cookies.update(ModelScopeConfig.get_cookies());return s
name=sys.argv[1]; label=name.split('_')[2]; started=time.time()
r=session().get(BASE+'/repo/files',params={'Revision':'master','Recursive':'true'},timeout=30);r.raise_for_status();manifest=r.json()
(ROOT/f'remote_manifest_{label}.json').write_text(json.dumps(manifest,indent=2))
f=next(f for f in manifest['Data']['Files'] if f['Path']==name)
p=ROOT/name;tmp=p.with_suffix(p.suffix+'.part');size=f['Size'];fd=os.open(tmp,os.O_CREAT|os.O_RDWR|os.O_TRUNC,0o644);os.ftruncate(fd,size)
def chunk(i):
 start=size*i//8;end=size*(i+1)//8-1
 for attempt in range(4):
  try:
   with session().get(BASE+'/repo',params={'Revision':f['Revision'],'FilePath':name},headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90)) as r:
    r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{size}'
    pos=start
    for b in r.iter_content(1<<20):
     assert pos+len(b)<=end+1
     view=memoryview(b)
     while view:
      n=os.pwrite(fd,view,pos);assert n>0;pos+=n;view=view[n:]
    assert pos==end+1
   print('CHUNK',i,'DONE',round(time.time()-started,1),flush=True);return
  except Exception as e:
   print('RETRY',i,attempt,type(e).__name__,flush=True)
   if attempt==3:raise
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(chunk,range(8)))
finally:os.close(fd)
h=hashlib.sha256()
with tmp.open('rb') as inp:
 for b in iter(lambda:inp.read(8<<20),b''):h.update(b)
assert tmp.stat().st_size==size and h.hexdigest()==f['Sha256']
tmp.replace(p);print('VERIFIED',name,size,h.hexdigest(),'seconds',round(time.time()-started,1),flush=True)
