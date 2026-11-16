"""Audit actual files and build a conservative reproducible group split.

No source file is changed. Filename families are potential shared sources,
not verified physical rail identities. Natural images stay held out.
"""
import argparse, collections, hashlib, json, pathlib, re
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.spatial.distance import pdist

SEED=20261114

class Union:
    def __init__(self,keys):self.p={k:k for k in keys}
    def find(self,k):
        if self.p[k]!=k:self.p[k]=self.find(self.p[k])
        return self.p[k]
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=pathlib.Path,required=True)
    ap.add_argument('--project-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);a=ap.parse_args()
    root=a.data_root.resolve();src=root/'mcnet/extracted';project=a.project_root.resolve()
    manifest_dir=project/'data/manifests';split_dir=project/'data/splits'
    manifest_dir.mkdir(parents=True,exist_ok=True);split_dir.mkdir(parents=True,exist_ok=True)
    folders=[('training',src/'training/training',src/'annotations/annotations/training'),
      ('validation',src/'validation/validation',src/'annotations/annotations/validation'),
      ('artificial_test',src/'test/test/craft/images',src/'test/test/craft/annotations'),
      ('natural_test',src/'test/test/real/images',src/'test/test/real/annotations')]
    records=[];hash_groups=collections.defaultdict(list);label_hashes={};phashes=[]
    histogram=np.zeros(256,dtype=np.int64);mismatches=[]
    for split,folder,labels in folders:
        images=sorted(folder.glob('*.png'))
        assert images,(split,folder)
        assert {p.stem for p in images}=={p.stem for p in labels.glob('*.jpg')},'Unpaired data in '+split
        for ip in images:
            mp=labels/(ip.stem+'.jpg')
            with Image.open(ip) as im:rgb=np.asarray(im.convert('RGB'))
            with Image.open(mp) as im:mask=np.asarray(im.convert('L'))
            rid=split+'/'+ip.stem
            if rgb.shape[:2]!=mask.shape:mismatches.append(rid);continue
            # Upper/lower-case suffixes are included conservatively in one numbered family.
            family=re.sub(r'[A-Za-z]+$','',ip.stem.split('_')[0])
            pxhash=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
            mh=hashlib.sha256((mask>=128).tobytes()).hexdigest()
            label_hashes[rid]=mh;hash_groups[pxhash].append(rid)
            histogram+=np.bincount(mask.ravel(),minlength=256)
            tiny=np.asarray(Image.fromarray(rgb).convert('L').resize((32,32),getattr(Image, 'Resampling', Image).LANCZOS),dtype=float)
            coeff=dct(dct(tiny,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8].ravel()[1:]
            phashes.append(coeff>np.median(coeff))
            records.append({'id':rid,'original_split':split,'source_stem':ip.stem,'filename_family':family,
               'image':ip.relative_to(root).as_posix(),'mask':mp.relative_to(root).as_posix(),
               'width':rgb.shape[1],'height':rgb.shape[0],
               'rgb_pixel_sha256':pxhash,'binary128_mask_sha256':mh,
               'jpeg_intermediate_fraction':float(np.mean((mask>0)&(mask<255))),
               'foreground_fraction':float(np.mean(mask>=128)),
               'foreground_pixels_t64':int(np.count_nonzero(mask>=64)),
               'foreground_pixels_t128':int(np.count_nonzero(mask>=128)),
               'foreground_pixels_t192':int(np.count_nonzero(mask>=192))})
        print('AUDITED '+split+' '+str(len(images)),flush=True)
    assert not mismatches,mismatches
    ids=[r['id'] for r in records];byid={r['id']:r for r in records};u=Union(ids)
    families=collections.defaultdict(list)
    for r in records:families[r['filename_family']].append(r['id'])
    for group in families.values():
        for item in group[1:]:u.join(group[0],item)
    duplicates=[g for g in hash_groups.values() if len(g)>1];quarantine={};conflicts=[]
    for group in duplicates:
        for item in group[1:]:u.join(group[0],item)
        if len({label_hashes[i] for i in group})>1:
            conflicts.append(group)
            for item in group:quarantine[item]='same_rgb_conflicting_binary128_masks'
        else:
            for item in sorted(group)[1:]:quarantine[item]='exact_rgb_and_mask_duplicate'
    # Very strict appearance similarity is used only to make grouping more conservative.
    distances=pdist(np.asarray(phashes,dtype=bool),'hamming');near=[];index=0
    for j in range(len(records)-1):
        length=len(records)-j-1;small=np.flatnonzero(distances[index:index+length]<=2/63+1e-12)
        for offset in small:
            k=j+1+int(offset);aa=records[j];bb=records[k]
            if abs((aa['width']/aa['height'])/(bb['width']/bb['height'])-1)>.02:continue
            if aa['rgb_pixel_sha256']==bb['rgb_pixel_sha256']:continue
            near.append({'a':aa['id'],'b':bb['id'],'hamming63':int(round(distances[index+offset]*63))})
            u.join(aa['id'],bb['id'])
        index+=length
    components=collections.defaultdict(list)
    for item in ids:components[u.find(item)].append(item)
    group_id={item:'g_'+hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:16]
              for members in components.values() for item in members}
    natural_groups={group_id[r['id']] for r in records if r['original_split']=='natural_test'}
    remaining=sorted(set(group_id.values())-natural_groups,
       key=lambda g:hashlib.sha256(f'{SEED}:{g}'.encode()).hexdigest())
    n=len(remaining);ntrain=int(.6*n);nval=int(.2*n)
    group_split={g:('training' if j<ntrain else 'validation' if j<ntrain+nval else 'artificial_test')
                 for j,g in enumerate(remaining)}
    for g in natural_groups:group_split[g]='reserved_test_group'
    for r in records:
        r['group_id']=group_id[r['id']]
        if r['id'] in quarantine:r['assigned_split']='quarantine';r['quarantine_reason']=quarantine[r['id']]
        elif r['group_id'] in natural_groups:
            r['assigned_split']='natural_test' if r['original_split']=='natural_test' else 'related_to_natural_test_holdout'
        else:r['assigned_split']=group_split[r['group_id']]
    active=[r for r in records if r['assigned_split'] in ['training','validation','artificial_test','natural_test']]
    sets={name:{r['group_id'] for r in active if r['assigned_split']==name}
          for name in ['training','validation','artificial_test','natural_test']}
    names=list(sets)
    for j,name in enumerate(names):
        for other in names[j+1:]:assert not sets[name]&sets[other],(name,other)
    cross=[{'family':family,'counts':dict(collections.Counter(byid[i]['original_split'] for i in members))}
       for family,members in families.items() if len({byid[i]['original_split'] for i in members})>1]
    summary={'repository_commit':'a87a4ca6710346f54870f4ec8dac0a058ffd31df',
      'actual_pairs':len(records),'original_counts':dict(collections.Counter(r['original_split'] for r in records)),
      'width_range':[min(r['width'] for r in records),max(r['width'] for r in records)],
      'height_range':[min(r['height'] for r in records),max(r['height'] for r in records)],
      'shape_mismatches':len(mismatches),'decoded_jpeg_gray_values':np.flatnonzero(histogram).tolist(),
      'jpeg_intermediate_pixel_fraction':float(histogram[1:255].sum()/histogram.sum()),
      'aggregate_foreground_pixels':{str(t):sum(r['foreground_pixels_t'+str(t)] for r in records) for t in [64,128,192]},
      'exact_duplicate_groups':duplicates,'conflicting_duplicate_groups':conflicts,
      'strict_phash_near_pairs':near,'filename_family_count':len(families),
      'cross_original_split_family_count':len(cross),'cross_original_split_families':cross,
      'merged_group_count':len(components),'assigned_counts':dict(collections.Counter(r['assigned_split'] for r in records)),
      'assigned_group_counts':{k:len(v) for k,v in sets.items()},
      'all_background_at128':sum(r['foreground_pixels_t128']==0 for r in records),
      'group_split_leakage_count':0,'random_seed':SEED,
      'grouping_basis':'numbered filename family (letters stripped), exact RGB and pHash<=2 with aspect tolerance2%; conservative proxy, not verified physical rail identity',
      'protocol_status':'conservative_group_v1; model not trained; group membership never selected using model performance',
      'original_protocol_comparability':'This revised split is not directly comparable with published official-split scores.'}
    content=json.dumps({'summary':summary,'records':records},ensure_ascii=False,indent=2)
    (manifest_dir/'mcnet_audit.json').write_text(content,encoding='utf-8')
    split_content=json.dumps({'protocol':'conservative_group_v1','seed':SEED,'grouping_basis':summary['grouping_basis'],
       'records':[{k:r.get(k) for k in ['id','image','mask','filename_family','group_id','original_split','assigned_split','quarantine_reason']} for r in records]},ensure_ascii=False,indent=2)
    (split_dir/'conservative_group_v1.json').write_bytes(split_content.encode('utf-8'))
    (split_dir/'conservative_group_v1.sha256').write_text(hashlib.sha256(split_content.encode()).hexdigest()+'\n',encoding='ascii')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cross_original_split_families','strict_phash_near_pairs','exact_duplicate_groups','conflicting_duplicate_groups','decoded_jpeg_gray_values']},ensure_ascii=False,indent=2),flush=True)
    print('MCNET_AUDIT_AND_SPLIT_OK',flush=True)

if __name__=='__main__':main()
