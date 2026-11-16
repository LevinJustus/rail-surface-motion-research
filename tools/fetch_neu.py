"""Read only author-published image archives at a pinned revision; never run data code."""
from pathlib import Path
import json, hashlib, urllib.request, concurrent.futures, time, zipfile, argparse

ROOT = Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);args=ap.parse_args()
DATA = args.data_root.resolve() / 'neu_rsdds'
ARCH = DATA / 'archives'
ARCH.mkdir(parents=True, exist_ok=True)
TREE = json.loads((ROOT / 'data/manifests/neu_tree.json').read_text(encoding='utf-8'))
SHA = TREE['sha']
FILES = [x for x in TREE['tree'] if x['type']=='blob' and
         (x['path'].startswith('2D_Image.zip.') or x['path'] in ['Ground Truth.zip','README.md'])]

def fetch(item):
    dest = ARCH / item['path']
    url = f"https://raw.githubusercontent.com/neu-rail-rsdds/rsdds/{SHA}/" + urllib.parse.quote(item['path'])
    if not dest.exists() or dest.stat().st_size != item['size']:
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'Rail-Research-Data-Audit'})
                with urllib.request.urlopen(req,timeout=60) as inp, dest.with_suffix(dest.suffix+'.partial').open('wb') as out:
                    while True:
                        chunk=inp.read(1024*1024)
                        if not chunk: break
                        out.write(chunk)
                temp=dest.with_suffix(dest.suffix+'.partial')
                if temp.stat().st_size != item['size']: raise ValueError('unexpected archive size')
                temp.replace(dest)
                break
            except Exception:
                if attempt==2: raise
                time.sleep(attempt+1)
    payload=dest.read_bytes()
    git_sha=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    if git_sha != item['sha']: raise ValueError(f"Git blob mismatch: {item['path']}")
    print('VERIFIED '+item['path']+' '+str(len(payload)),flush=True)
    return {'file':item['path'],'bytes':len(payload),'url':url,'git_blob_sha1':git_sha,
            'sha256':hashlib.sha256(payload).hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    records=list(ex.map(fetch, FILES))
(DATA/'download_manifest.json').write_text(json.dumps({'repo':'neu-rail-rsdds/rsdds','commit':SHA,
    'scope':'RGB images, masks, author README only; depth not downloaded','files':records},indent=2),encoding='utf-8')

joined=ARCH/'2D_Image.zip'
with joined.open('wb') as out:
    for item in sorted([i for i in FILES if i['path'].startswith('2D_Image.zip.')],key=lambda x:x['path']):
        with (ARCH/item['path']).open('rb') as inp:
            while True:
                data=inp.read(1024*1024)
                if not data: break
                out.write(data)

for archive in [joined,ARCH/'Ground Truth.zip']:
    folder=DATA/'extracted'/archive.stem
    folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        listing=[{'name':i.filename,'size':i.file_size,'encrypted':bool(i.flag_bits&1),
                  'compression':i.compress_type} for i in z.infolist()]
        (DATA/(archive.stem.replace(' ','_')+'_contents.json')).write_text(json.dumps(listing,ensure_ascii=False,indent=2),encoding='utf-8')
        for item in z.infolist():
            dest=(folder/item.filename).resolve()
            if dest != folder.resolve() and folder.resolve() not in dest.parents:
                raise ValueError('Unsafe archive path')
        z.extractall(folder,pwd=b'neurail')
    print('EXTRACTED '+str(archive.name)+' '+str(len(listing)),flush=True)
print('DOWNLOAD_AND_EXTRACT_DONE',flush=True)
