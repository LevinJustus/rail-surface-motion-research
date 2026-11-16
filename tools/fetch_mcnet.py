"""Download only author-published data at a pinned revision; do not run author code.

Python >=3.8, standard library only. Keep split volumes and checksums intact.
The original author publishes the archive password in the public README.
"""
import argparse, concurrent.futures, hashlib, json, os, pathlib, stat
import subprocess, time, urllib.request, urllib.parse, zipfile

REPO='zdfcvsn/MCnet'
COMMIT='a87a4ca6710346f54870f4ec8dac0a058ffd31df'
TREE_SHA='af9e4dccd569c0c4c55d0b70149df8a37cc190cf'

def download(item, archive_root):
    relative=item['path'].split('dataset/',1)[1]
    dest=archive_root/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/'+urllib.parse.quote(item['path'])
    if not dest.exists() or dest.stat().st_size!=item['size']:
        for attempt in range(4):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'Rail-Surface-Motion-Research'})
                tmp=dest.with_name(dest.name+'.partial')
                with urllib.request.urlopen(req,timeout=90) as inp,tmp.open('wb') as out:
                    while True:
                        chunk=inp.read(1024*1024)
                        if not chunk:break
                        out.write(chunk)
                if tmp.stat().st_size!=item['size']:raise ValueError('Download size mismatch')
                tmp.replace(dest);break
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    sha1=hashlib.sha1(f"blob {item['size']}\0".encode());sha256=hashlib.sha256()
    with dest.open('rb') as inp:
        while True:
            chunk=inp.read(1024*1024)
            if not chunk:break
            sha1.update(chunk);sha256.update(chunk)
    if sha1.hexdigest()!=item['sha']:raise ValueError('Git hash mismatch: '+relative)
    return {'archive_volume':relative,'bytes':item['size'],'url':url,
            'git_blob_sha1':sha1.hexdigest(),'sha256':sha256.hexdigest()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=pathlib.Path,required=True)
    ap.add_argument('--metadata-out',type=pathlib.Path,required=True)
    ap.add_argument('--extractor',type=pathlib.Path)
    ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args()
    root=(a.data_root/'mcnet').resolve();archive_root=root/'archives'
    root.mkdir(parents=True,exist_ok=True);a.metadata_out.mkdir(parents=True,exist_ok=True)
    cache=root/'tree.json'
    if cache.exists():tree=json.loads(cache.read_text(encoding='utf-8'))
    else:
        req=urllib.request.Request(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1',headers={'User-Agent':'Rail-Surface-Motion-Research'})
        with urllib.request.urlopen(req,timeout=60) as f:tree=json.load(f)
        cache.write_text(json.dumps(tree,indent=2),encoding='utf-8')
    assert tree['sha'] in (COMMIT,TREE_SHA) and not tree.get('truncated',False)
    items=[x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith('dataset/') and '.zip.' in x['path']]
    print(f'DOWNLOADING {len(items)} volumes, {sum(x["size"] for x in items):,} bytes',flush=True)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(download,x,archive_root) for x in items]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records)%10==0 or len(records)==len(items):print('VERIFIED '+str(len(records))+'/'+str(len(items)),flush=True)
    manifest={'repository':REPO,'commit':COMMIT,'records':sorted(records,key=lambda x:x['archive_volume'])}
    text=json.dumps(manifest,indent=2)
    (root/'download_manifest.json').write_text(text,encoding='utf-8')
    (a.metadata_out/'mcnet_download_manifest.json').write_text(text,encoding='utf-8')
    groups={}
    for item in items:
        relative=item['path'].split('dataset/',1)[1]
        groups.setdefault(relative.rsplit('.',1)[0],[]).append(relative)
    for group,volumes in groups.items():
        merged=archive_root/group
        with merged.open('wb') as out:
            for volume in sorted(volumes):
                with (archive_root/volume).open('rb') as inp:
                    while True:
                        chunk=inp.read(1024*1024)
                        if not chunk:break
                        out.write(chunk)
        target=root/'extracted'/merged.stem
        target.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(merged) as z:
            entries=[]
            for info in z.infolist():
                final=(target/info.filename).resolve()
                if final!=target.resolve() and target.resolve() not in final.parents:raise ValueError('Unsafe archive path')
                if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Archive symlink not accepted')
                entries.append({'path':info.filename,'bytes':info.file_size})
            (root/(merged.stem+'_contents.json')).write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
            print('EXTRACT '+merged.stem+' entries='+str(len(entries)),flush=True)
            if not a.extractor:
                z.extractall(target,pwd=b'IDQCrail')
        if a.extractor:
            cmd=[str(a.extractor),'x','-y','-o+','-inul','-ibck','-pIDQCrail',str(merged),str(target)+os.sep]
            kwargs={}
            if os.name=='nt':
                si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
                kwargs={'startupinfo':si,'creationflags':subprocess.CREATE_NO_WINDOW}
            subprocess.run(cmd,check=True,**kwargs)
        print('EXTRACTED '+merged.stem,flush=True)
    print('MCNET_DATA_READY',flush=True)

if __name__=='__main__':main()
