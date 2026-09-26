from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, shutil, hashlib, zipfile
OUT=Path('/file_system/nas/algorithm/dujun.nie/nwm/paper_assets/six_frames_20260926')
rows=json.loads((OUT/'candidates.json').read_text())
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(fontpath,18)
small=ImageFont.truetype(fontpath,14)
choices=[('scand',14,'recommended_scand','SCAND - campus walkway'),('recon',15,'alternative_recon','RECON - outdoor building'),('sacson',2,'alternative_huron','HuRoN / SACSoN - corridor'),('tartan_drive',13,'alternative_tartan','TartanDrive - curved dirt road')]
manifest=[]
overview=Image.new('RGB',(1520,4*246+60),'white');od=ImageDraw.Draw(overview)
od.text((16,12),'Four selections | first 3: observation history | last 3: recorded future frames',font=font,fill='#16243a')
for k,(ds,idx,sub,title) in enumerate(choices):
 row=rows[ds][idx]; folder=OUT/sub;folder.mkdir(exist_ok=True)
 frames=[]
 sheet=Image.new('RGB',(1000,630),'white');d=ImageDraw.Draw(sheet)
 d.text((16,8),title,font=font,fill='#16243a')
 d.text((16,38),'Observation history (left to right)',font=small,fill='#00796b')
 d.text((16,338),'Recorded future (top to bottom in the paper figure)',font=small,fill='#006bb4')
 oy=60+k*246;od.text((16,oy),title,font=font,fill='#16243a')
 for i,src in enumerate(row['frames']):
  src=Path(src); im=Image.open(src).convert('RGB')
  role='history' if i<3 else 'future'; dest=folder/f'{i+1:02d}_{role}_{i%3+1}_frame_{src.stem}'
  shutil.copy2(src,dest.with_suffix('.jpg')); im.save(dest.with_suffix('.png'))
  thumb=im.copy();thumb.thumbnail((320,240)); x=16+(i%3)*330;y=62+(i//3)*300
  sheet.paste(thumb,(x,y));d.text((x,y+242),f'{i+1:02d} | source frame {src.stem}',font=small,fill='#374151')
  thumb=im.copy();thumb.thumbnail((240,180));ox=16+i*250;overview.paste(thumb,(ox,oy+30));od.text((ox,oy+214),f'{role} {i%3+1} | frame {src.stem}',font=small,fill='#374151')
  frames.append({'slot':i+1,'role':role,'source_frame':int(src.stem),'source_path':str(src),'jpg':str(dest.with_suffix('.jpg')),'png':str(dest.with_suffix('.png')),'width':im.width,'height':im.height,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
 sheet.save(folder/'preview.png')
 record={'dataset':ds,'split':'test','trajectory':row['trajectory'],'source_frame_stride':2,'future_type':'recorded ground-truth; not model predictions','frames':frames}
 (folder/'manifest.json').write_text(json.dumps(record,indent=2))
 manifest.append(record)
overview.save(OUT/'four_options.png')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'README.md').write_text('''# 论文结构图六帧替换素材

推荐 `recommended_scand/`：SCAND 校园步道，画面明亮，建筑、树木、行人提供清楚的时间变化。

- 左侧三张（左→右）：01、02、03，对应源帧 37、39、41。
- 右侧三张（上→下）：04、05、06，对应源帧 43、45、47。
- 六帧来自同一测试集轨迹，源帧索引间隔为 2；如以源帧 41 为 t，则为 t−4、t−2、t、t+2、t+4、t+6。间隔是处理后帧索引，不代表秒数。
- JPG 为原文件逐字节复制；PNG 为同尺寸解码导出，无裁剪、调色、超分或生成。推荐组原始尺寸 320×240。
- 右侧是真实后续帧，仅供方法结构示意，不是模型预测结果。若图中强调模型生成，应在图注说明为示意，或改用实际推理帧。不要保留旧图中的 gen. 标签。
- 另有 RECON（640×480）、HuRoN/SACSoN（160×120）、TartanDrive（320×240）各一组六帧备选。
- 每组 manifest.json 包含源路径、帧号、原始尺寸与 SHA-256，便于追溯。

建议英文图注补充：Images illustrate the temporal input/output arrangement using recorded dataset frames; future images are ground-truth examples, not model predictions.
''')
for archive,subs in [('recommended_six_frames.zip',['recommended_scand']),('all_four_options.zip',[x[2] for x in choices])]:
 with zipfile.ZipFile(OUT/archive,'w',zipfile.ZIP_DEFLATED) as z:
  z.write(OUT/'README.md','README.md')
  for sub in subs:
   for p in sorted((OUT/sub).iterdir()):z.write(p,str(p.relative_to(OUT)))
  if len(subs)>1:z.write(OUT/'four_options.png','four_options.png')
for rec in manifest:
 for f in rec['frames']:
  assert hashlib.sha256(Path(f['jpg']).read_bytes()).hexdigest()==f['sha256']
  with Image.open(f['jpg']) as a,Image.open(f['png']) as b: assert a.convert('RGB').tobytes()==b.tobytes()
print('Export and pixel/source checks passed for 24 images.')
print(OUT)
