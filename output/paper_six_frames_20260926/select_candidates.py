from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import random, json
ROOT=Path('/file_system/vepfs/algorithm/dujun.nie/code/CompACT')
DATA=Path('/file_system/nas/algorithm/dujun.nie/nwm/data')
OUT=Path('/file_system/nas/algorithm/dujun.nie/nwm/paper_assets/six_frames_20260926')
OUT.mkdir(parents=True, exist_ok=True)
allrows={}
for ds,folder in [('recon','recon'),('scand','scand'),('sacson','sacson'),('tartan_drive','tartan')]:
    names=(ROOT/f'data_splits/{ds}/test/traj_names.txt').read_text().split()
    random.Random(926).shuffle(names)
    rows=[]
    for name in names:
        p=DATA/folder/name
        if not p.is_dir(): continue
        files=sorted([f for f in p.iterdir() if f.suffix=='.jpg' and f.stem.isdigit()], key=lambda f:int(f.stem))
        if len(files)<18: continue
        stride=2
        start=max(0,len(files)//2-6)
        chosen=files[start:start+stride*6:stride]
        if len(chosen)!=6: continue
        rows.append({'dataset':ds,'trajectory':name,'frames':[str(f) for f in chosen]})
        if len(rows)==16: break
    sheet=Image.new('RGB',(6*180+20,len(rows)*143+40),'white')
    d=ImageDraw.Draw(sheet);d.text((10,8),ds,fill='black')
    for i,row in enumerate(rows):
        y=36+i*143;d.text((10,y),f'{i:02d} | {row["trajectory"]}',fill='black')
        for j,p in enumerate(row['frames']):
            im=Image.open(p).convert('RGB'); im.thumbnail((176,110))
            sheet.paste(im,(10+180*j,y+17))
            d.text((10+180*j,y+127),Path(p).stem,fill='black')
    sheet.save(OUT/f'candidates_{ds}.jpg',quality=92)
    allrows[ds]=rows
    print(ds,len(rows),str(OUT/f'candidates_{ds}.jpg'),flush=True)
(OUT/'candidates.json').write_text(json.dumps(allrows,indent=2))
