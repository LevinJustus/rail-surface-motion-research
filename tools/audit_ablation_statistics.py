"""Second C5 statistics audit: integer aggregation and scalar percentile checks."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

FIELDS=['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']
SEEDS=[20261114,20261115,20261116]


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def average(a):
    v=[float(v) for v in a if math.isfinite(float(v))];return math.fsum(v)/len(v) if v else math.nan
def metric(c,k):
    tp,fp,fn,tn,_,_,nfn,nfp=c
    if k=='IoU':return tp/(tp+fp+fn) if tp+fp+fn else 1.
    a,b={'introduced_error_rate':(nfn+nfp,tp+fp+fn+tn),'new_FN_rate':(nfn,tp+fn),'new_FP_rate':(nfp,fp+tn)}[k]
    return a/b if b else math.nan
def percentile(a,q):
    p=(len(a)-1)*q;j=math.floor(p);f=p-j;return a[j]*(1-f)+a[min(j+1,len(a)-1)]*f


def audit(root,summary,out):
    assert not out.exists();out.mkdir(parents=True);timer=time.perf_counter();index=read(summary/'input_index.json');groups=defaultdict(lambda:[0]*8)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    pollution=defaultdict(lambda:[0,0,0.,0,0,0,0.]);clean=defaultdict(lambda:[0]*8);n=0
    def add(key,c):
        for j,v in enumerate(c):groups[key][j]+=v
    for entry in index['audits']:
        folder=root/'runs'/entry['audit_run'];assert sha(folder/'per_frame_counts.csv')==entry['counts_sha256']
        for r in rows(folder/'per_frame_counts.csv'):
            seed=int(r['seed']);frame=int(r['frame']);source=r['source_id'];kind=r['threshold_kind'];sc=r['scenario'];m=r['method'];c=[int(r[k]) for k in FIELDS]
            if frame>=2:add(('primary_2_9',kind,seed,source,sc,m),c)
            if sc=='pollution':
                if frame>=4:add(('pollution_4_9',kind,seed,source,sc,m),c)
                p=pollution[kind,frame,m];p[0]+=1;p[1]+=int(r['probability_changed_pixels']);p[2]+=float(r['mean_probability_delta']);p[3]+=int(r['prediction_flip_pixels']);p[4]+=int(r['delta_FP']);p[5]+=int(r['delta_FN']);p[6]=max(p[6],float(r['max_probability_delta']))
                if m=='R_unbounded' and frame>=4:
                    gt=c[0]+c[2];bg=c[1]+c[3];fp=int(r['clean_FP']);fn=int(r['clean_FN']);key=('pollution_4_9',kind,seed,source,'clean_unbounded','R')
                    for j,v in enumerate([gt-fn,fp,fn,bg-fp]):clean[key][j]+=v
            n+=1
    assert n==767880
    c4=root/'runs/fusion_validation_summary_v1_01';c4index=read(c4/'input_index.json')
    for entry in c4index['runs']:
        folder=root/'runs'/entry['audit_run'];assert sha(folder/'per_frame_counts.csv')==entry['counts_sha256']
        for r in rows(folder/'per_frame_counts.csv'):
            frame=int(r['frame']);args=(r['threshold_kind'],int(r['seed']),r['source_id'],'baseline',r['method']);c=[int(r[k]) for k in FIELDS]
            if frame>=2:add(('primary_2_9',)+args,c)
            if frame>=4:add(('pollution_4_9',)+args,c)
    source_error=0.;checked_source=0
    for r in rows(summary/'per_source_metrics.csv'):
        key=r['frame_set'],r['threshold_kind'],int(r['seed']),r['source_id'],r['scenario'],r['method'];c=groups[key]
        assert [int(r[k]) for k in FIELDS]==c
        for k in ['IoU','introduced_error_rate','new_FN_rate','new_FP_rate']:
            v=metric(c,k)
            if math.isfinite(v):source_error=max(source_error,abs(v-float(r[k])))
            else:assert r[k]==''
        checked_source+=1
    for r in rows(summary/'unbounded_clean_source_counts.csv'):
        key='pollution_4_9',r['threshold_kind'],int(r['seed']),r['source_id'],'clean_unbounded','R'
        assert [int(r[k]) for k in FIELDS[:4]]==clean[key][:4]
        assert abs(float(r['IoU'])-metric(clean[key],'IoU'))<=1e-12
    groups.update(clean);sources=index['bootstrap_source_order'];draws=np.load(c4/'paired_source_bootstrap_indices.npy')
    regenerated=np.random.Generator(np.random.PCG64(20261118)).integers(0,15,size=(10000,15),dtype=np.int64);assert np.array_equal(draws,regenerated)
    interval_error=0.;checked_intervals=0
    for r in rows(summary/'paired_contrasts.csv'):
        fs,kind,sa,ma,sb,mb,k=[r[x] for x in ['frame_set','threshold_kind','scenario_a','method_a','scenario_b','method_b','metric']]
        delta=[average(metric(groups[fs,kind,seed,s,sa,ma],k)-metric(groups[fs,kind,seed,s,sb,mb],k) for seed in SEEDS) for s in sources]
        point=average(delta);bootstrap=sorted(v for v in (average(delta[int(j)] for j in draw) for draw in draws) if math.isfinite(v))
        values=[point,percentile(bootstrap,.025),percentile(bootstrap,.975)]
        for field,v in zip(['point','CI95_low','CI95_high'],values):interval_error=max(interval_error,abs(v-float(r[field])))
        assert len(bootstrap)==int(r['defined_bootstrap_replicates']);checked_intervals+=1
        if checked_intervals%50==0:print('C5_STATISTICS_AUDIT',checked_intervals,flush=True)
    for r in rows(summary/'pollution_by_frame.csv'):
        p=pollution[r['threshold_kind'],int(r['frame']),r['method']]
        for field,j in [('method_frames',0),('probability_changed_pixels',1),('prediction_flip_pixels',3),('delta_FP',4),('delta_FN',5)]:assert int(r[field])==p[j]
        assert abs(float(r['mean_probability_delta'])-p[2]/p[0])<=2e-12 and float(r['max_probability_delta'])==p[6]
    assert source_error<=1e-12 and interval_error<=1e-12
    report=dict(status='PASS_C5_INDEPENDENT_SOURCE_STATISTICS',input_count_rows=n,source_metric_rows=checked_source,contrast_rows=checked_intervals,
        maximum_source_error=source_error,maximum_interval_error=interval_error,bootstrap_indices_exact=True,wall_seconds=time.perf_counter()-timer,
        summary_sha256=sha(summary/'summary.json'),pollution_by_frame_sha256=sha(summary/'pollution_by_frame.csv'),heldout_payloads_opened=0)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    audit(a.project_root.resolve(),a.summary.resolve(),a.out.resolve())
