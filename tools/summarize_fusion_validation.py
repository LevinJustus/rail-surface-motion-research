"""Frozen 13C4 source-paired summaries; requires NumPy with PCG64.

Reads audited counts only, never images, masks or probabilities. The sampling
unit is the 15 source proxy groups, retaining all seeds/conditions per source.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

METHODS=['S','U','V','P','GV','GVP','GC','R']
SEEDS=[20261114,20261115,20261116]
FIELDS=['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
def ratio(a,b,empty=np.nan):return float(a/b) if b else empty
def metrics(c):
    tp,fp,fn,tn,cfn,cfp,nfn,nfp=map(int,c);gt=tp+fn;bg=fp+tn;n=gt+bg
    return dict(IoU=ratio(tp,tp+fp+fn,1.),Dice=ratio(2*tp,2*tp+fp+fn,1.),
        precision=ratio(tp,tp+fp),recall=ratio(tp,gt),FN_over_GT=ratio(fn,gt),FP_over_background=ratio(fp,bg),
        introduced_error_rate=ratio(nfn+nfp,n),new_FN_rate=ratio(nfn,gt),new_FP_rate=ratio(nfp,bg),
        corrected_FN_rate=ratio(cfn,gt),corrected_FP_rate=ratio(cfp,bg),
        valid_pixels=n,target_pixels=gt,background_pixels=bg,empty_union=int(tp+fp+fn==0))
def finite_mean(a):
    a=np.asarray(a,dtype=float);return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else np.nan


def summarize(root,audit_runs,out,calibration_path):
    assert not out.exists(),'No overwrite';out.mkdir(parents=True);started=time.perf_counter()
    (out/'summarize_fusion_validation.py').write_bytes(Path(__file__).read_bytes())
    protocol=read(root/'configs/fusion_protocol_v1.json');assert protocol['evaluation']['bootstrap']['seed']==20261118
    assert sha(root/'configs/fusion_protocol_v1.json')=='20fd3a8839131d294d64053d2b9cde0b4adf6ef837482f2d4604cb22d37cb6c1'
    freeze=read(calibration_path);assert freeze['status']=='FROZEN_THREE_SEED_TRAINING_CALIBRATION'
    assert len(freeze['choices'])==24
    source_rows=defaultdict(lambda:np.zeros(8,np.int64));condition_rows=defaultdict(lambda:np.zeros(8,np.int64))
    component_rows=defaultdict(lambda:np.zeros(12,np.int64));seen=set();source_groups={};profiles=set();input_indices=[];row_total=0
    for audit_run in audit_runs:
        folder=root/'runs'/audit_run;audit=read(folder/'audit.json')
        assert audit['status']=='PASS_C4_FROZEN_INDEPENDENT_AUDIT' and audit['mode']=='validation'
        seed=audit['seed'];assert seed in SEEDS and seed not in seen;seen.add(seed)
        input_indices.append(dict(audit_run=audit_run,audit_sha256=sha(folder/'audit.json'),
            counts_sha256=sha(folder/'per_frame_counts.csv'),components_sha256=sha(folder/'per_frame_components.csv'),
            prediction_manifest_sha256=audit['prediction_manifest_sha256']))
        count_rows=0
        with (folder/'per_frame_counts.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                assert int(r['seed'])==seed;source=r['source_id'];source_groups.setdefault(source,r['group_id']);assert source_groups[source]==r['group_id']
                profiles.add(r['profile']);frame=int(r['frame']);method=r['method'];kind=r['threshold_kind'];assert method in METHODS
                values=np.array([int(r[k]) for k in FIELDS],np.int64);assert values[:4].sum()==65536
                sets=['all_1_9']+(['primary_2_9'] if frame>=2 else [])+(['full_window_3_9'] if frame>=3 else [])
                for fs in sets:source_rows[fs,kind,seed,source,method]+=values
                if frame>=2:condition_rows[kind,seed,source,method,r['profile']]+=values
                count_rows+=1
        assert count_rows==25920;row_total+=count_rows
        with (folder/'per_frame_components.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if int(r['frame'])>=2:
                    vals=[int(r[k]) for k in ['gt_components','detected_gt_components','prediction_components','false_prediction_components']]+[int(r[f'bin_{b}']) for b in range(8)]
                    component_rows[r['threshold_kind'],seed,r['source_id'],r['method']]+=np.array(vals,np.int64)
    assert seen==set(SEEDS) and len(source_groups)==len(set(source_groups.values()))==15 and len(profiles)==12
    sources=sorted(source_groups);write(out/'input_index.json',dict(runs=input_indices,calibration_sha256=sha(calibration_path),count_rows=row_total))
    detailed=[];lookup={}
    for key,c in sorted(source_rows.items()):
        fs,kind,seed,source,method=key;v=metrics(c);lookup[key]=v
        expected_frames={'all_1_9':108,'primary_2_9':96,'full_window_3_9':84}[fs]
        assert v['valid_pixels']==expected_frames*65536
        detailed.append(dict(frame_set=fs,threshold_kind=kind,seed=seed,source_id=source,group_id=source_groups[source],method=method,
            **{k:int(c[j]) for j,k in enumerate(FIELDS)},**v))
    csv_write(out/'per_source_metrics.csv',detailed)
    condition_detail=[]
    for key,c in sorted(condition_rows.items()):
        kind,seed,source,method,profile=key
        condition_detail.append(dict(threshold_kind=kind,seed=seed,source_id=source,group_id=source_groups[source],method=method,profile=profile,**metrics(c)))
    csv_write(out/'per_condition_source_metrics.csv',condition_detail)
    seed_rows=[];summary_rows=[]
    for fs in ['primary_2_9','all_1_9','full_window_3_9']:
        for kind in ['primary','training_calibrated']:
            for m in METHODS:
                each=[]
                for seed in SEEDS:
                    v=[lookup[fs,kind,seed,s,m] for s in sources]
                    micro=metrics(sum((source_rows[fs,kind,seed,s,m] for s in sources),np.zeros(8,np.int64)))
                    nonempty=[a['IoU'] for a in v if a['target_pixels']>0]
                    row=dict(frame_set=fs,threshold_kind=kind,seed=seed,method=m,
                        source_macro_IoU=finite_mean([a['IoU'] for a in v]),foreground_source_macro_IoU=finite_mean(nonempty),
                        source_macro_Dice=finite_mean([a['Dice'] for a in v]),
                        background_only_sources=sum(a['target_pixels']==0 for a in v),empty_union_sources=sum(a['empty_union'] for a in v),
                        **{'micro_'+k:micro[k] for k in ['IoU','Dice','precision','recall','FN_over_GT','FP_over_background','introduced_error_rate','new_FN_rate','new_FP_rate']})
                    seed_rows.append(row);each.append(row)
                row=dict(frame_set=fs,threshold_kind=kind,method=m)
                for k in [k for k in each[0] if k not in ['frame_set','threshold_kind','seed','method']]:
                    a=np.array([v[k] for v in each],float);row[k+'_mean']=finite_mean(a)
                    row[k+'_sample_SD']=float(np.std(a[np.isfinite(a)],ddof=1)) if np.isfinite(a).sum()>1 else np.nan
                summary_rows.append(row)
    csv_write(out/'per_seed_summary.csv',seed_rows);csv_write(out/'method_summary.csv',summary_rows)
    conditions=[]
    for kind in ['primary','training_calibrated']:
        for profile in sorted(profiles):
            for m in METHODS:
                vals=[metrics(condition_rows[kind,seed,s,m,profile]) for seed in SEEDS for s in sources]
                conditions.append(dict(threshold_kind=kind,profile=profile,method=m,source_seed_macro_IoU=finite_mean([v['IoU'] for v in vals]),
                    source_seed_macro_Dice=finite_mean([v['Dice'] for v in vals]),new_FN_rate=finite_mean([v['new_FN_rate'] for v in vals]),
                    new_FP_rate=finite_mean([v['new_FP_rate'] for v in vals]),introduced_error_rate=finite_mean([v['introduced_error_rate'] for v in vals])))
    csv_write(out/'per_condition_summary.csv',conditions)
    regional=[]
    for kind in ['primary','training_calibrated']:
        for m in METHODS:
            c=sum((component_rows[kind,seed,s,m] for seed in SEEDS for s in sources),np.zeros(12,np.int64))
            frames=3*15*12*8
            row=dict(threshold_kind=kind,method=m,gt_components=int(c[0]),detected_gt_components=int(c[1]),
                per_frame_component_recall=ratio(c[1],c[0]),prediction_components=int(c[2]),false_prediction_components=int(c[3]),
                false_prediction_components_per_frame=ratio(c[3],frames),method_frames=frames)
            for b,label in enumerate(['1_15','16_63','64_255','ge_256']):
                row['GT_components_'+label]=int(c[4+2*b]);row['detected_'+label]=int(c[5+2*b]);row['recall_'+label]=ratio(c[5+2*b],c[4+2*b])
            regional.append(row)
    csv_write(out/'component_summary.csv',regional)
    # One source bootstrap draw table, shared by every metric and contrast.
    bc=protocol['evaluation']['bootstrap'];rng=np.random.Generator(np.random.PCG64(bc['seed']))
    draws=rng.integers(0,15,size=(bc['replicates'],15),dtype=np.int64)
    np.save(out/'paired_source_bootstrap_indices.npy',draws)
    contrasts=[('V','U'),('V','S'),('P','V'),('GVP','GV'),('GVP','GC'),('GV','V')]
    contrast_rows=[];contrast_lookup={}
    for a,b in contrasts:
        for metric in ['IoU','introduced_error_rate','new_FN_rate','new_FP_rate']:
            by_seed=np.array([[lookup['primary_2_9','primary',seed,s,a][metric]-lookup['primary_2_9','primary',seed,s,b][metric] for s in sources] for seed in SEEDS])
            # Missing denominators are identical by source across seeds/methods.
            assert np.all(np.isfinite(by_seed)==np.isfinite(by_seed[0:1]))
            d=np.array([finite_mean(by_seed[:,j]) for j in range(15)])
            sampled=d[draws];n=np.isfinite(sampled).sum(1)
            boot=np.divide(np.nansum(sampled,axis=1),n,out=np.full(len(n),np.nan),where=n>0)
            good=boot[np.isfinite(boot)];lo=hi=np.nan
            if len(good)>=9500:lo,hi=np.quantile(good,[.025,.975],method='linear')
            per_seed=[finite_mean(by_seed[j]) for j in range(3)]
            mixed=any(v>1e-12 for v in per_seed) and any(v < -1e-12 for v in per_seed)
            row=dict(contrast=a+'-'+b,metric=metric,point=finite_mean(d),CI95_low=float(lo),CI95_high=float(hi),
                defined_sources=int(np.isfinite(d).sum()),defined_bootstrap_replicates=len(good),
                source_wins=int((d>1e-12).sum()),source_ties=int((np.abs(d)<=1e-12).sum()),source_losses=int((d < -1e-12).sum()),
                seed20261114=per_seed[0],seed20261115=per_seed[1],seed20261116=per_seed[2],mixed_seed_direction=mixed,
                scope='paired source descriptive percentile interval; no multiple-comparison correction')
            contrast_rows.append(row);contrast_lookup[a+'-'+b,metric]=row
    csv_write(out/'paired_contrasts.csv',contrast_rows)
    def positive(name):
        r=contrast_lookup[name,'IoU'];return np.isfinite(r['CI95_low']) and r['CI95_low']>0 and not r['mixed_seed_direction']
    def harm_rule(name):
        total=contrast_lookup[name,'introduced_error_rate'];fn=contrast_lookup[name,'new_FN_rate'];fp=contrast_lookup[name,'new_FP_rate']
        return all(np.isfinite(v['CI95_high']) for v in [total,fn,fp]) and total['CI95_high']<0 and fn['CI95_high']<=0 and fp['CI95_high']<=0 and not any(v['mixed_seed_direction'] for v in [total,fn,fp])
    decisions=dict(H1=dict(current_validation_rule_supported=bool(positive('V-U') and positive('V-S'))),
        H2=dict(current_validation_rule_supported=bool(positive('P-V'))),
        H3=dict(current_validation_rule_supported=bool(all(positive(c) and harm_rule(c) for c in ['GVP-GV','GVP-GC']))),
        scope='C4 descriptive rule check only; C5 ablations and C6 conclusions remain pending; not novelty or final-test proof')
    write(out/'current_validation_rule_check.json',decisions)
    result=dict(status='PASS_13C4_PAIRED_SOURCE_SUMMARIES',sources=15,seeds=SEEDS,methods=METHODS,conditions=12,
        full_frame_count_rows=77760,primary_threshold_method_frames=34560,optimizer_updates=0,heldout_payloads_opened=0,
        bootstrap_generator='PCG64',bootstrap_seed=20261118,bootstrap_replicates=10000,bootstrap_indices_sha256=sha(out/'paired_source_bootstrap_indices.npy'),
        bootstrap_source_order=sources,numpy_version=np.__version__,python_version=platform.python_version(),
        background_only_sources=sum(lookup['primary_2_9','primary',SEEDS[0],s,'S']['target_pixels']==0 for s in sources),
        calibration_sha256=sha(calibration_path),wall_seconds=time.perf_counter()-started)
    write(out/'summary.json',result)
    print(json.dumps(clean(dict(summary=result,primary_methods=[r for r in summary_rows if r['frame_set']=='primary_2_9' and r['threshold_kind']=='primary'],decisions=decisions)),ensure_ascii=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--audit-runs',nargs=3,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--calibration-freeze',type=Path,required=True)
    a=p.parse_args();summarize(a.project_root.resolve(),a.audit_runs,a.out.resolve(),a.calibration_freeze.resolve())
