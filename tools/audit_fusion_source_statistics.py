"""Independent source/count and bootstrap audit using Python scalar arithmetic.

Reads the audited frame CSVs and independently rebuilds the C4 primary source
metrics. Percentiles use an explicit sorted linear interpolation formula.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import time
import numpy as np

METHODS=['S','U','V','P','GV','GVP','GC','R'];SEEDS=[20261114,20261115,20261116]
FIELDS=['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def average(v):
    finite=[float(x) for x in v if math.isfinite(float(x))]
    return math.fsum(finite)/len(finite) if finite else math.nan
def percentile(v,q):
    v=sorted(v);position=(len(v)-1)*q;lower=math.floor(position);fraction=position-lower
    return v[lower]*(1-fraction)+v[min(lower+1,len(v)-1)]*fraction
def metrics(c):
    tp,fp,fn,tn,_,_,newfn,newfp=c
    divide=lambda a,b: a/b if b else math.nan
    return dict(IoU=tp/(tp+fp+fn) if tp+fp+fn else 1.,
        introduced_error_rate=divide(newfn+newfp,tp+fp+fn+tn),new_FN_rate=divide(newfn,tp+fn),new_FP_rate=divide(newfp,fp+tn))


def audit(root,summary,out):
    assert not out.exists(),'No overwrite';out.mkdir(parents=True);started=time.perf_counter()
    (out/'audit_fusion_source_statistics.py').write_bytes(Path(__file__).read_bytes())
    index=read(summary/'input_index.json');groups=defaultdict(lambda:[0]*8);source_to_group={};rows=0
    for run in index['runs']:
        folder=root/'runs'/run['audit_run'];assert sha(folder/'per_frame_counts.csv')==run['counts_sha256']
        for r in csv_rows(folder/'per_frame_counts.csv'):
            if r['threshold_kind']!='primary' or int(r['frame'])<2:continue
            key=(int(r['seed']),r['source_id'],r['method']);source_to_group[r['source_id']]=r['group_id']
            for j,k in enumerate(FIELDS):groups[key][j]+=int(r[k])
            rows+=1
    assert rows==34560 and len(source_to_group)==len(set(source_to_group.values()))==15
    sources=sorted(source_to_group);values={k:metrics(v) for k,v in groups.items()}
    existing=[r for r in csv_rows(summary/'per_source_metrics.csv') if r['frame_set']=='primary_2_9' and r['threshold_kind']=='primary']
    assert len(existing)==3*15*8;max_source_error=0.
    for r in existing:
        key=(int(r['seed']),r['source_id'],r['method'])
        assert all(int(r[k])==groups[key][j] for j,k in enumerate(FIELDS))
        for metric,value in values[key].items():
            if math.isfinite(value):max_source_error=max(max_source_error,abs(float(r[metric])-value))
            else:assert r[metric]==''
    summaries=[r for r in csv_rows(summary/'method_summary.csv') if r['frame_set']=='primary_2_9' and r['threshold_kind']=='primary']
    for r in summaries:
        m=r['method'];by_seed=[average([values[seed,s,m]['IoU'] for s in sources]) for seed in SEEDS]
        mu=average(by_seed);sd=math.sqrt(math.fsum((v-mu)**2 for v in by_seed)/2)
        assert abs(float(r['source_macro_IoU_mean'])-mu)<1e-12
        assert abs(float(r['source_macro_IoU_sample_SD'])-sd)<1e-12
    supplied=np.load(summary/'paired_source_bootstrap_indices.npy');assert supplied.shape==(10000,15)
    regenerated=np.random.Generator(np.random.PCG64(20261118)).integers(0,15,size=(10000,15),dtype=np.int64)
    assert np.array_equal(supplied,regenerated)
    max_interval_error=0.;checked=0
    for row in csv_rows(summary/'paired_contrasts.csv'):
        a,b=row['contrast'].split('-');metric=row['metric']
        d=[average([values[seed,s,a][metric]-values[seed,s,b][metric] for seed in SEEDS]) for s in sources]
        point=average(d);boot=[]
        for indices in supplied:
            value=average([d[int(k)] for k in indices])
            if math.isfinite(value):boot.append(value)
        assert len(boot)==int(row['defined_bootstrap_replicates'])
        assert sum(math.isfinite(v) for v in d)==int(row['defined_sources'])
        if len(boot)>=9500:
            low=percentile(boot,.025);high=percentile(boot,.975)
            max_interval_error=max(max_interval_error,abs(low-float(row['CI95_low'])),abs(high-float(row['CI95_high'])),abs(point-float(row['point'])))
        else:assert row['CI95_low']==row['CI95_high']==''
        checked+=1
    assert max_source_error<1e-12 and max_interval_error<1e-12 and checked==24
    result=dict(status='PASS_C4_INDEPENDENT_SOURCE_STATISTICS',primary_method_frame_rows_reaggregated=rows,
        source_metric_rows_checked=len(existing),paired_metric_contrasts_checked=24,bootstrap_replicates=10000,
        source_count=15,seed_count=3,maximum_source_metric_error=max_source_error,maximum_interval_error=max_interval_error,
        bootstrap_indices_exact_match=True,bootstrap_indices_sha256=sha(summary/'paired_source_bootstrap_indices.npy'),
        images_or_labels_opened=0,heldout_payloads_opened=0,numpy_version=np.__version__,wall_seconds=time.perf_counter()-started)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--summary-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.project_root.resolve(),a.summary_dir.resolve(),a.out.resolve())
