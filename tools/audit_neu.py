from pathlib import Path
import json, hashlib, collections, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.fftpack import dct

PROJECT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);args=ap.parse_args()
BASE=args.data_root.resolve();DATA=BASE/'neu_rsdds';OUT=PROJECT/'data/manifests'
OUT.mkdir(parents=True,exist_ok=True)
PREVIEW=PROJECT/'runs/data_audit';PREVIEW.mkdir(parents=True,exist_ok=True)
images={p.stem:p for p in (DATA/'extracted/2D_Image').rglob('*.bmp')}
masks={p.stem:p for p in (DATA/'extracted/Ground Truth').rglob('*.png')}
assert len(images)==113 and len(masks)==113, (len(images),len(masks))
assert images.keys()==masks.keys(), 'Unpaired images/masks'

records=[]; phashes={}; pixel_hashes=collections.defaultdict(list)
for sid in sorted(images,key=lambda x:int(x)):
    with Image.open(images[sid]) as im: rgb=np.array(im.convert('RGB'))
    with Image.open(masks[sid]) as im:
        raw_mode=im.mode
        raw_values=np.unique(np.array(im)).tolist()
        mask=np.array(im.convert('L'))
    assert rgb.shape[:2]==mask.shape, (sid,rgb.shape,mask.shape)
    vals=np.unique(mask).tolist()
    assert set(vals)<={0,255}, (sid,vals)
    binary=mask>0
    labels,n=ndimage.label(binary,np.ones((3,3),dtype=int))
    sizes=np.bincount(labels.ravel())[1:]
    h,w=mask.shape
    pxhash=hashlib.sha256(rgb.tobytes()).hexdigest()
    pixel_hashes[pxhash].append(sid)
    gray=np.array(Image.fromarray(rgb).convert('L').resize((32,32),getattr(Image, 'Resampling', Image).LANCZOS),dtype=float)
    coeff=dct(dct(gray,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8].ravel()[1:]
    phashes[sid]=coeff>np.median(coeff)
    records.append({'source_id':sid,'image_file':images[sid].relative_to(BASE).as_posix(),
       'mask_file':masks[sid].relative_to(BASE).as_posix(), 'width':w,'height':h,
       'mask_values':vals,'raw_mask_mode':raw_mode,'raw_mask_values':raw_values,
       'foreground_pixels':int(binary.sum()),'foreground_fraction':float(binary.mean()),
       'connected_components_8':n,'smallest_component_px':int(sizes.min()) if n else 0,
       'sha256_image_file':hashlib.sha256(images[sid].read_bytes()).hexdigest(),
       'sha256_mask_file':hashlib.sha256(masks[sid].read_bytes()).hexdigest(),
       'sha256_rgb_pixels':pxhash,'eligible_demo_256_5frames_step32_margin16':w>=288 and h>=416})

duplicate_groups=[ids for ids in pixel_hashes.values() if len(ids)>1]
conflicts=[]; quarantine=set()
for group in duplicate_groups:
    first=group[0]
    a=np.array(Image.open(masks[first]).convert('L'))>0
    for other in group[1:]:
        b=np.array(Image.open(masks[other]).convert('L'))>0
        differing=int(np.count_nonzero(a!=b))
        if differing:
            conflicts.append({'ids':[first,other],'differing_pixels':differing,
                 'mask_iou':float((a&b).sum()/(a|b).sum()),
                 'decision':'quarantine both pending independent annotation review'})
            quarantine.update(group)
for rec in records:
    rec['quarantine_annotation_conflict']=rec['source_id'] in quarantine

near=[]
for j,a in enumerate(records):
    for b in records[j+1:]:
        ar_a=a['width']/a['height'];ar_b=b['width']/b['height']
        if abs(ar_a/ar_b-1)>.05: continue
        distance=int(np.count_nonzero(phashes[a['source_id']]!=phashes[b['source_id']]))
        if distance<=6: near.append({'id_a':a['source_id'],'id_b':b['source_id'],'phash_hamming63':distance})

summary={'dataset':'NEU RSDDS-113 RGB and pixel masks','task':'binary segmentation; not temporal ground truth',
 'pairs':len(records),'unpaired':0,'shape_mismatches':0,
 'width_range':[min(r['width'] for r in records),max(r['width'] for r in records)],
 'height_range':[min(r['height'] for r in records),max(r['height'] for r in records)],
 'unique_dimensions':len(set((r['width'],r['height']) for r in records)),
 'mask_values_union':sorted(set(v for r in records for v in r['mask_values'])),
 'all_background_masks':sum(r['foreground_pixels']==0 for r in records),
 'all_foreground_masks':sum(r['foreground_fraction']==1 for r in records),
 'foreground_fraction_min_median_max':np.quantile([r['foreground_fraction'] for r in records],[0,.5,1]).tolist(),
 'raw_mask_modes':sorted(set(r['raw_mask_mode'] for r in records)),
 'exact_duplicate_rgb_groups':duplicate_groups,
 'conflicting_duplicate_annotations':conflicts,
 'quarantined_ids':sorted(quarantine,key=int),
 'remaining_structurally_valid_pairs':len(records)-len(quarantine),
 'near_duplicate_candidates':near,
 'demo_eligible_count':sum(r['eligible_demo_256_5frames_step32_margin16'] for r in records),
 'screening_caveat':'pHash checks only same-aspect appearance; does not establish physical-rail independence or eliminate cropped duplicates',
 'split_status':'NOT ASSIGNED; physical source IDs not supplied; review near duplicates and source relationships first'}
manifest={'summary':summary,'provenance':json.loads((DATA/'download_manifest.json').read_text()),'records':records}
(OUT/'钢轨数据验收清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

try: font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
except OSError: font=ImageFont.load_default()
selected=['1','10','102']
# Each example is resized only for this inspection sheet, never as a source of higher-resolution data.
sheet=Image.new('RGB',(1260,3*300+90),'white');draw=ImageDraw.Draw(sheet)
draw.text((20,10),'NEU RSDDS-113：真实图像 / 缺陷掩膜 / 叠加核查（非模型预测）',fill='black',font=font)
for j,sid in enumerate(selected):
    rgb=Image.open(images[sid]).convert('RGB');mask=Image.open(masks[sid]).convert('L')
    array=np.array(rgb);fg=np.array(mask)>0
    edge=fg^ndimage.binary_erosion(fg)
    overlay=array.copy();overlay[edge]=[255,30,30]
    for k,panel in enumerate([rgb,mask.convert('RGB'),Image.fromarray(overlay)]):
        panel.thumbnail((400,245))
        x=20+420*k+(400-panel.width)//2;y=80+300*j
        sheet.paste(panel,(x,y))
    rec=next(r for r in records if r['source_id']==sid)
    draw.text((25,80+300*j+250),f"ID {sid} | {rec['width']}×{rec['height']} | 缺陷像素 {rec['foreground_fraction']:.2%}",fill='black',font=font)
sheet.save(PREVIEW/'neu_image_mask_audit.png')
