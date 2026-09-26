from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, shutil, hashlib, zipfile
root=Path('/file_system/nas/algorithm/dujun.nie/nwm/paper_assets/six_frames_20260926')
row=json.loads((root/'candidates.json').read_text())['sacson'][7]
out=root/'recommended_indoor_huron';out.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
sheet=Image.new('RGB',(752,484),'white');d=ImageDraw.Draw(sheet)
d.text((10,8),'HuRoN / SACSoN | indoor corridor | original exposure',font=font,fill='#283448')
d.text((10,34),'Observation history (left to right)',font=small,fill='#42566c')
d.text((10,258),'Recorded future (top to bottom in the paper)',font=small,fill='#42566c')
records=[]
for i,p in enumerate(row['frames']):
 src=Path(p);role='history' if i<3 else 'future';stem=f'{i+1:02d}_{role}_{i%3+1}_frame_{src.stem}'
 im=Image.open(src).convert('RGB');shutil.copy2(src,out/f'{stem}.jpg');im.save(out/f'{stem}.png')
 x=10+247*(i%3);y=54+224*(i//3)
 sheet.paste(im.resize((240,180),Image.Resampling.LANCZOS),(x,y))
 d.text((x,y+184),f'{i+1:02d} | frame {src.stem}',font=small,fill='#283448')
 assert (out/f'{stem}.jpg').read_bytes()==src.read_bytes()
 records.append({'slot':i+1,'role':role,'source_path':str(src),'source_frame':int(src.stem),'image_size':im.size,'export_stem':stem,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
sheet.save(out/'preview.png')
(out/'manifest.json').write_text(json.dumps({'dataset':'HuRoN/SACSoN','split':'test','trajectory':row['trajectory'],'frame_stride':2,'future_type':'recorded ground truth, not model-generated','frames':records},indent=2))
(out/'README.md').write_text('''# 无天空室内六帧素材

数据集：HuRoN/SACSoN，测试集。
轨迹：Feb-09-2023-bww8-intloss_00000011_1。

左侧从左到右：01、02、03（源帧 7、9、11）。
右侧从上到下：04、05、06（源帧 13、15、17）。

六帧均为室内走廊，没有天空，保留原始亮度。JPG 逐字节复制；PNG 同尺寸无损解码导出；原图 160×120，无调色、裁剪、超分或生成。preview.png 仅排版预览时放大，各张独立素材保留原始尺寸。

帧号间隔为 2，代表处理后帧索引而非秒数。右侧是真实后续帧，仅供方法结构示意，不应标注为模型生成结果。
''')
with zipfile.ZipFile(root/'recommended_indoor_six_frames.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):z.write(p,p.name)
print('Export verified: 6 original JPGs, 6 PNGs, manifest, preview, ZIP.')
print(out)
