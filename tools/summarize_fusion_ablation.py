"""C5 descriptive source-paired statistics, fixed C4 draws and all planned arms.

Reads audited CSV counts only; no image, mask, model or probability access.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np
import summarize_fusion_validation as stats

read,sha,write,csv_write,metrics,mean=stats.read,stats.sha,stats.write,stats.csv_write,stats.metrics,stats.finite_mean
FIELDS=stats.FIELDS;SEEDS=stats.SEEDS


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        yield from csv.DictReader(f)


def summarize(root,audits,out):
    assert not out.exists();out.mkdir(parents=True);timer=time.perf_counter()
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    c4=root/'runs/fusion_validation_summary_v1_01';s4=read(c4/'summary.json');sources=s4['bootstrap_source_order']
    draws=np.load(c4/'paired_source_bootstrap_indices.npy');assert sha(c4/'paired_source_bootstrap_indices.npy')==s4['bootstrap_indices_sha256']=='3a3649a14b3d8c5544d2753d98730ed4717938a4ac7238ea08a731e5b2f09a07'
    groups=defaultdict(lambda:np.zeros(8,np.int64));conditions=defaultdict(lambda:np.zeros(8,np.int64));poll=defaultdict(lambda:np.zeros(7,float));seen=set();indices=[];rows=0
    for audit_id in audits:
        folder=root/'runs'/audit_id;report=read(folder/'audit.json');assert report['status']=='PASS_C5_INDEPENDENT_AUDIT'
        seed=report['seed'];assert seed in SEEDS and seed not in seen;seen.add(seed)
        assert sha(folder/'per_frame_counts.csv')==report['counts_sha256']
        indices.append(dict(audit_run=audit_id,audit_sha256=sha(folder/'audit.json'),counts_sha256=report['counts_sha256']))
        for r in csv_rows(folder/'per_frame_counts.csv'):
            source=r['source_id'];assert source in sources;frame=int(r['frame']);scenario=r['scenario'];m=r['method'];kind=r['threshold_kind'];vals=np.array([int(r[f]) for f in FIELDS],np.int64)
            assert vals[:4].sum()==65536
            if frame>=2:
                groups['primary_2_9',kind,seed,source,scenario,m]+=vals
                conditions[kind,seed,source,r['profile'],scenario,m]+=vals
            if scenario=='pollution' and frame>=4:groups['pollution_4_9',kind,seed,source,scenario,m]+=vals
            if scenario=='pollution':
                key=kind,seed,source,frame,m
                v=np.array([1,int(r['probability_changed_pixels']),float(r['mean_probability_delta']),int(r['prediction_flip_pixels']),int(r['delta_FP']),int(r['delta_FN']),float(r['max_probability_delta'])],float)
                poll[key][:6]+=v[:6];poll[key][6]=max(poll[key][6],v[6])
            rows+=1
    assert seen==set(SEEDS) and rows==767880
    # Add C4 same-input parents, separately reconstructing frames 4-9 for poison controls.
    c4index=read(c4/'input_index.json')
    for entry in c4index['runs']:
        folder=root/'runs'/entry['audit_run'];assert sha(folder/'per_frame_counts.csv')==entry['counts_sha256']
        for r in csv_rows(folder/'per_frame_counts.csv'):
            seed=int(r['seed']);source=r['source_id'];frame=int(r['frame']);kind=r['threshold_kind'];m=r['method'];vals=np.array([int(r[f]) for f in FIELDS],np.int64)
            if frame>=2:groups['primary_2_9',kind,seed,source,'baseline',m]+=vals
            if frame>=4:groups['pollution_4_9',kind,seed,source,'baseline',m]+=vals
    lookup={};detail=[]
    for key,c in sorted(groups.items()):
        fs,kind,seed,source,scenario,m=key;v=metrics(c);lookup[key]=v
        assert v['valid_pixels']==12*(8 if fs=='primary_2_9' else 6)*65536
        detail.append(dict(frame_set=fs,threshold_kind=kind,seed=seed,source_id=source,scenario=scenario,method=m,**{f:int(c[j]) for j,f in enumerate(FIELDS)},**v))
    csv_write(out/'per_source_metrics.csv',detail)
    arms=[(r['scenario'],r['method']) for r in read(root/'configs/fusion_ablation_v1.json')['arms']]+[('baseline',m) for m in stats.METHODS]
    summaries=[];seed_summaries=[]
    for fs in ['primary_2_9','pollution_4_9']:
        for kind in ['primary','training_calibrated']:
            for scenario,m in arms:
                if (fs,kind,SEEDS[0],sources[0],scenario,m) not in lookup:continue
                values=[]
                for seed in SEEDS:
                    entries=[lookup[fs,kind,seed,s,scenario,m] for s in sources]
                    micro=metrics(sum((groups[fs,kind,seed,s,scenario,m] for s in sources),np.zeros(8,np.int64)))
                    row=dict(frame_set=fs,threshold_kind=kind,seed=seed,scenario=scenario,method=m,
                        source_macro_IoU=mean([r['IoU'] for r in entries]),source_macro_Dice=mean([r['Dice'] for r in entries]),
                        foreground_source_macro_IoU=mean([r['IoU'] for r in entries if r['target_pixels']>0]),
                        **{'micro_'+k:micro[k] for k in ['IoU','precision','recall','introduced_error_rate','new_FN_rate','new_FP_rate']})
                    seed_summaries.append(row);values.append(row)
                summaries.append(dict(frame_set=fs,threshold_kind=kind,scenario=scenario,method=m,
                    **{k+'_mean':mean([v[k] for v in values]) for k in values[0] if k not in ['frame_set','threshold_kind','seed','scenario','method']},
                    source_macro_IoU_sample_SD=float(np.std([v['source_macro_IoU'] for v in values],ddof=1))))
    csv_write(out/'per_seed_summary.csv',seed_summaries);csv_write(out/'method_summary.csv',summaries)
    condition_detail=[]
    for key,c in sorted(conditions.items()):
        kind,seed,source,profile,scenario,m=key;condition_detail.append(dict(threshold_kind=kind,seed=seed,source_id=source,profile=profile,scenario=scenario,method=m,**metrics(c)))
    csv_write(out/'per_condition_source_metrics.csv',condition_detail)
    condition_summary=[]
    c_lookup={tuple(r[k] for k in ['threshold_kind','seed','source_id','profile','scenario','method']):r for r in condition_detail}
    profiles=sorted({r['profile'] for r in condition_detail})
    for kind in ['primary','training_calibrated']:
        for profile in profiles:
            for scenario,m in arms:
                if scenario=='baseline':continue
                v=[c_lookup[kind,seed,s,profile,scenario,m] for seed in SEEDS for s in sources]
                condition_summary.append(dict(threshold_kind=kind,profile=profile,scenario=scenario,method=m,source_macro_IoU=mean([r['IoU'] for r in v]),
                    new_FN_rate=mean([r['new_FN_rate'] for r in v]),new_FP_rate=mean([r['new_FP_rate'] for r in v])))
    csv_write(out/'per_condition_summary.csv',condition_summary)
    pairs=[]
    for scenario in ['bias_m6','bias_m2','bias_p2','bias_p6','alternating_1','alternating_3','all_missing']:
        for a,b in [('P','V'),('GVP','GV'),('GVP','GC')]:pairs.append(('primary_2_9',scenario,a,scenario,b))
    for scenario in ['window_1','window_5']:
        for m in ['V','GV','GVP']:pairs.append(('primary_2_9',scenario,m,'baseline',m))
    for scenario in ['remove_photo','remove_confidence','remove_cycle','remove_peak_margin']:pairs.append(('primary_2_9',scenario,'GV','baseline','GV'))
    pairs.extend([('primary_2_9','prior_alignment','GVP','baseline','GVP'),('primary_2_9','unbounded','R','baseline','R')])
    for m in stats.METHODS:pairs.append(('pollution_4_9','pollution',m,'baseline',m))
    # For the unbounded clean control, recover counts directly from stored clean FP/FN.
    clean_groups=defaultdict(lambda:np.zeros(4,np.int64))
    for entry in indices:
        for r in csv_rows(root/'runs'/entry['audit_run']/'per_frame_counts.csv'):
            if r['scenario']!='pollution' or r['method']!='R_unbounded' or int(r['frame'])<4:continue
            gt=int(r['TP'])+int(r['FN']);bg=int(r['FP'])+int(r['TN']);fp=int(r['clean_FP']);fn=int(r['clean_FN'])
            clean_groups[r['threshold_kind'],int(r['seed']),r['source_id']]+=np.array([gt-fn,fp,fn,bg-fp],np.int64)
    clean_detail=[]
    for (kind,seed,source),c in sorted(clean_groups.items()):
        iou=float(c[0]/(c[0]+c[1]+c[2])) if c[0]+c[1]+c[2] else 1.
        lookup['pollution_4_9',kind,seed,source,'clean_unbounded','R']=dict(IoU=iou)
        clean_detail.append(dict(threshold_kind=kind,seed=seed,source_id=source,TP=int(c[0]),FP=int(c[1]),FN=int(c[2]),TN=int(c[3]),IoU=iou))
    csv_write(out/'unbounded_clean_source_counts.csv',clean_detail)
    pairs.append(('pollution_4_9','pollution','R_unbounded','clean_unbounded','R'))
    contrasts=[]
    for fs,sa,ma,sb,mb in pairs:
        for kind in ['primary','training_calibrated']:
            for metric in ['IoU','introduced_error_rate','new_FN_rate','new_FP_rate']:
                if sb=='clean_unbounded' and metric!='IoU':continue
                v=np.array([[lookup[fs,kind,seed,s,sa,ma][metric]-lookup[fs,kind,seed,s,sb,mb][metric] for s in sources] for seed in SEEDS])
                d=np.array([mean(v[:,j]) for j in range(15)]);sample=d[draws];n=np.isfinite(sample).sum(1)
                boot=np.divide(np.nansum(sample,axis=1),n,out=np.full(len(n),np.nan),where=n>0);good=boot[np.isfinite(boot)]
                lo,hi=np.quantile(good,[.025,.975],method='linear');per_seed=[mean(a) for a in v]
                contrasts.append(dict(frame_set=fs,threshold_kind=kind,scenario_a=sa,method_a=ma,scenario_b=sb,method_b=mb,metric=metric,
                    point=mean(d),CI95_low=float(lo),CI95_high=float(hi),defined_sources=int(np.isfinite(d).sum()),defined_bootstrap_replicates=len(good),
                    seed20261114=per_seed[0],seed20261115=per_seed[1],seed20261116=per_seed[2],mixed_seed_direction=any(x>1e-12 for x in per_seed) and any(x< -1e-12 for x in per_seed),
                    scope='descriptive paired-source percentile interval; no multiple-comparison correction'))
    csv_write(out/'paired_contrasts.csv',contrasts)
    pollution=[]
    for kind in ['primary','training_calibrated']:
        for frame in range(1,10):
            for m in stats.METHODS+['R_unbounded']:
                vals=[poll[kind,seed,s,frame,m] for seed in SEEDS for s in sources];v=np.sum(vals,axis=0);n=v[0]
                pollution.append(dict(threshold_kind=kind,frame=frame,method=m,method_frames=int(n),probability_changed_pixels=int(v[1]),
                    mean_probability_delta=float(v[2]/n),prediction_flip_pixels=int(v[3]),delta_FP=int(v[4]),delta_FN=int(v[5]),
                    max_probability_delta=max(float(x[6]) for x in vals)))
    csv_write(out/'pollution_by_frame.csv',pollution)
    for r in pollution:
        if r['frame']>=6 and r['method']!='R_unbounded':assert r['probability_changed_pixels']==r['prediction_flip_pixels']==r['delta_FP']==r['delta_FN']==0
    write(out/'input_index.json',dict(audits=indices,C4_input_index_sha256=sha(c4/'input_index.json'),bootstrap_source_order=sources,bootstrap_indices_sha256=sha(c4/'paired_source_bootstrap_indices.npy')))
    result=dict(status='PASS_C5_SOURCE_SUMMARIES_PENDING_STATISTICS_AUDIT',count_rows=rows,logical_outputs=383940,source_count=15,seeds=SEEDS,profiles=12,
        arms_per_frame=79,paired_contrast_rows=len(contrasts),bootstrap_replicates=10000,bootstrap_source_order=sources,
        bootstrap_indices_sha256=sha(c4/'paired_source_bootstrap_indices.npy'),numpy_version=np.__version__,wall_seconds=time.perf_counter()-timer,
        scope='controlled developer-validation ablation; no new independent sources, real video, final test or FPS claim')
    write(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--audit-runs',nargs=3,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();summarize(a.project_root.resolve(),a.audit_runs,a.out.resolve())
