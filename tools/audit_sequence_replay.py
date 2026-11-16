"""Compare new runs without rewriting PNGs; preserve time-chunk byte differences."""
import argparse,hashlib,json,struct,zlib
from pathlib import Path
from PIL import Image

def digest(data):return hashlib.sha256(data).hexdigest()

def png_content(path):
    raw=path.read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
    pos=8;canonical=raw[:8];times=[]
    while pos<len(raw):
        n=struct.unpack('>I',raw[pos:pos+4])[0];chunk=raw[pos:pos+n+12]
        typ=chunk[4:8];payload=chunk[8:8+n]
        assert len(chunk)==n+12 and zlib.crc32(typ+payload)&0xffffffff==struct.unpack('>I',chunk[-4:])[0]
        if typ==b'tIME':times.append(list(payload))
        else:canonical+=chunk
        pos+=n+12
    assert pos==len(raw)
    with Image.open(path) as im:
        pixel_hash=digest(im.tobytes());shape=[im.width,im.height];mode=im.mode
    return {'raw_sha256':digest(raw),'png_without_time_sha256':digest(canonical),
            'pixel_sha256':pixel_hash,'mode':mode,'size_wh':shape,'time_chunks':times}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('first');ap.add_argument('second');ap.add_argument('--out',required=True);args=ap.parse_args()
    first=Path(args.first);second=Path(args.second);out=Path(args.out);assert not out.exists()
    ma=json.loads((first/'evaluation_only/payload_manifest.json').read_text(encoding='utf-8'))
    mb=json.loads((second/'evaluation_only/payload_manifest.json').read_text(encoding='utf-8'))
    aa={r['path']:r for r in ma};bb={r['path']:r for r in mb};assert set(aa)==set(bb)
    rows=[];png=0;json_count=0;raw_diffs=0
    for rel in sorted(aa):
        pa=first/rel;pb=second/rel
        assert digest(pa.read_bytes())==aa[rel]['sha256'] and digest(pb.read_bytes())==bb[rel]['sha256']
        if pa.suffix=='.png':
            a=png_content(pa);b=png_content(pb)
            for field in ('png_without_time_sha256','pixel_sha256','mode','size_wh'):assert a[field]==b[field],(rel,field)
            diff=a['raw_sha256']!=b['raw_sha256'];raw_diffs+=diff;png+=1
            rows.append({'path':rel,'comparison':'PIXEL_AND_NON_TIME_CHUNKS_EXACT',
                         'raw_bytes_equal':not diff,'first':a,'second':b})
        else:
            assert pa.suffix=='.json' and pa.read_bytes()==pb.read_bytes();json_count+=1
            rows.append({'path':rel,'comparison':'RAW_BYTES_EXACT','sha256':digest(pa.read_bytes())})
    report={'state':'PASS_PIXEL_EXACT_WITH_PNG_TIMESTAMP_DIFFERENCE' if raw_diffs else 'PASS_RAW_BYTES_EXACT',
            'first_run':first.name,'second_run':second.name,'files':len(rows),'png_files':png,
            'json_files':json_count,'raw_png_time_differences':raw_diffs,
            'png_decoded_pixels_equal':png,'png_all_chunks_except_tIME_equal':png,
            'output_files_modified':0,'raw_source_images_opened':0,
            'audit_source_sha256':digest(Path(__file__).read_bytes()),'records':rows}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':main()
