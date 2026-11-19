"""Independent E4 reconstruction and training-only evaluation after output freeze."""
import argparse,csv,json,math,time,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import evaluate_v2_screening as e3
import audit_fusion_training as old

FIELDS=e3.FIELDS
sha,read,write,csv_write=e3.sha,e3.read,e3.write,e3.csv_write

def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def audit_scalar_tables(folder):
    totals=defaultdict(lambda:[0]*8);sources=defaultdict(lambda:[0]*8)
    rows=csv_rows(folder/'per_frame_counts.csv');assert len(rows)==864
    for r in rows:
        c=[int(r[f]) for f in FIELDS];assert sum(c[:4])==65536
        if int(r['frame'])==1:continue
        for i,v in enumerate(c):totals[r['method']][i]+=v;sources[r['source_id'],r['method']][i]+=v
    for r in csv_rows(folder/'method_summary.csv'):
        c=totals[r['method']];assert c==[int(r[f]) for f in FIELDS]
        assert abs(float(r['IoU'])-c[0]/sum(c[:3]))<1e-12
        macro=math.fsum(c[0]/sum(c[:3]) for (s,m),c in sources.items() if m==r['method'])/3
        assert abs(macro-float(r['source_macro_IoU']))<1e-12
    for r in csv_rows(folder/'per_source_metrics.csv'):
        assert sources[r['source_id'],r['method']]==[int(r[f]) for f in FIELDS]
    return dict(status='PASS_E4_SCALAR_AGGREGATES',count_rows=864,source_rows=24)

def evaluate(root,run_id,count_id,out):
    assert not out.exists();out.mkdir(parents=True);timer=time.perf_counter()
    run=root/'runs'/run_id;plan=read(run/'frozen_plan.json');summary=read(run/'summary.json')
    assert summary['status']=='PASS_E4_PREDICTION_PENDING_EVALUATION'
    assert sha(run/'frozen_plan.json')==summary['plan_sha256'] and sha(run/'predictions.json')==summary['predictions_sha256']
    for item in plan['frozen_files']:assert sha(root/item['path'])==item['sha256']
    cfg=read(root/plan['design_config']);rows=read(run/'predictions.json');assert len(rows)==864
    index={(r['sequence_id'],r['frame'],r['method']):r for r in rows};assert len(index)==864
    for r in rows:assert sha(root/r['path'])==r['sha256']
    write(out/'prediction_freeze_before_masks.json',dict(index_sha256=summary['predictions_sha256'],outputs_verified=864,labels_opened=0))
    count_folder=root/'runs'/count_id;cs=read(count_folder/'summary.json');assert cs['status']=='PASS_E4_MATLAB_COUNTS'
    assert sha(count_folder/'per_frame_counts.csv')==cs['counts_sha256'] and cs['predictions_sha256']==summary['predictions_sha256']
    matlab={(r['sequence_id'],int(r['frame']),r['method']):r for r in csv_rows(count_folder/'per_frame_counts.csv')}
    e3counts={(r['sequence_id'],int(r['frame']),r['method']):r for r in csv_rows(root/'runs/v2_screening_evaluation_v1_01/per_frame_counts.csv')}
    payload={r['path']:r['sha256'] for r in read(root/'runs/step12_full_v2_01/evaluation_only/payload_manifest.json')}
    diagnostics={(r['sequence_id'],r['frame']):r for r in read(run/'diagnostics.json')}
    all_counts=[];opportunities=[];bins=[];pair_rows=[];risk_rows=[];groups=defaultdict(lambda:np.zeros(8,np.int64));profiles=defaultdict(lambda:np.zeros(8,np.int64))
    max_error=0.;max_change=0.;protected_total=0;bound_exceed_pixels=0
    for selected in plan['selection']:
        assert selected['assigned_split']=='training';seq=selected['sequence_id'];source=selected['source_id'];profile=selected['profile']
        P=[old.load_probability(root/index[seq,k,'S']['path']) for k in range(1,10)]
        registration=read(root/'runs'/plan['base_prediction_run']/'registration'/(seq+'.json'))
        edges=[[tuple(r) for r in np.atleast_2d(a['candidates'])] if a['available'] else [] for a in registration['main']]
        masks=[]
        for k in range(1,10):
            relative='evaluation_only/sequences/{}/mask_{:03d}.png'.format(seq,k);path=root/'runs/step12_full_v2_01'/relative
            assert sha(path)==payload[relative];masks.append(np.asarray(Image.open(str(path)))>0)
        relative='evaluation_only/sequences/'+seq+'/geometry_truth.json';path=root/'runs/step12_full_v2_01'/relative;assert sha(path)==payload[relative]
        transforms=[np.asarray(f['frame_to_source_3x3']) for f in read(path)['frames']]
        for k in range(9):
            p=P[k].astype(float);target=masks[k];start=max(0,k-2);n=k-start+1;value=p.copy();history=[]
            for j in range(start,k):
                e=e3.evidence(P,edges,j,k,cfg);risk=e['variance']+e['unknown'];gate=e['q']*np.exp(-risk/(2*cfg['risk_sigma']**2))
                assert np.all(gate+1e-12>=e['g']) and np.all(gate<=1+1e-12)
                value+=gate*(e['h']-p)/n;history.append((e,risk,gate))
            delta=value-p;protection=p>=cfg['strong_current_foreground'];unprotected=value.copy()
            value[protection]=np.maximum(value[protection],cfg['protected_decision_threshold']);protected=int(np.count_nonzero(value!=unprotected))
            assert protected==diagnostics[seq,k+1]['protected_pixels'] and np.max(abs(delta))<=(n-1)/n+1e-12
            assert np.min(unprotected)>=-1e-12 and np.max(unprotected)<=1+1e-12
            stored={m:old.load_probability(root/index[seq,k+1,m]['path']).astype(float) for m in plan['methods']}
            error=float(np.max(abs(value.astype(np.float32).astype(float)-stored['V2_no_conflict'])));max_error=max(max_error,error);assert error<=2e-6
            if k==0:assert np.array_equal(stored['V2_no_conflict'],p)
            bound=(n-1)/n*cfg['risk_sigma']/math.sqrt(math.e)
            assert np.max(abs(stored['V2']-p))<=bound+1e-7
            fn=target&(p<.5);tn=~target&(p<.5);opportunity=np.zeros_like(target);wrong=np.zeros_like(target)
            for j in range(start,k):
                h,t,visible,_,_=e3.oracle(P,masks,transforms,j,k)
                opportunity|=fn&visible&t&(h>=.5);wrong|=tn&visible&~t&(h>=.5)
            for method,a in stored.items():
                c,tr=old.binary_counts(a,target,p,.5);values=list(c)+list(tr);r=matlab[seq,k+1,method]
                assert all(int(r[f])==int(v) for f,v in zip(FIELDS,values))
                if method!='V2_no_conflict':assert all(int(e3counts[seq,k+1,method][f])==int(v) for f,v in zip(FIELDS,values))
                all_counts.append(dict(sequence_id=seq,source_id=source,profile=profile,frame=k+1,method=method,**{f:int(v) for f,v in zip(FIELDS,values)},oracle_opportunity=int(opportunity.sum()),opportunity_recovered=int((opportunity&(a>=.5)).sum())))
                if k>=1:groups[source,method]+=np.asarray(values,np.int64);profiles[profile,method]+=np.asarray(values,np.int64)
            if k==0:continue
            new=stored['V2_no_conflict'];previous=stored['V2'];protected_total+=protected;max_change=max(max_change,float(np.max(abs(delta))));bound_exceed_pixels+=int(np.count_nonzero(abs(delta)>bound+1e-7))
            _,transition=old.binary_counts(new,target,previous,.5)
            pair_rows.append(dict(sequence_id=seq,source_id=source,profile=profile,frame=k+1,**{f:int(v) for f,v in zip(FIELDS[4:],transition)}))
            opportunities.append(dict(sequence_id=seq,source_id=source,profile=profile,frame=k+1,current_FN=int(fn.sum()),oracle_opportunity=int(opportunity.sum()),v2_recovered=int((opportunity&(previous>=.5)).sum()),no_conflict_recovered=int((opportunity&(new>=.5)).sum()),historical_FP_risk=int(wrong.sum()),v2_FP_on_risk=int((wrong&(previous>=.5)).sum()),no_conflict_FP_on_risk=int((wrong&(new>=.5)).sum()),protected_pixels=protected))
            for low,high in zip([0,.1,.3,.3989115567145611,.5],[.1,.3,.3989115567145611,.5,1.00001]):
                sel=fn&(p>=low)&(p<high)
                bins.append(dict(sequence_id=seq,source_id=source,frame=k+1,lower=low,upper=high,current_FN=int(sel.sum()),oracle_opportunity=int((sel&opportunity).sum()),v2_FN_corrected=int((sel&(previous>=.5)).sum()),no_conflict_FN_corrected=int((sel&(new>=.5)).sum()),no_conflict_opportunity_recovered=int((sel&opportunity&(new>=.5)).sum())))
            for scope,sel in [('all',np.ones_like(target)),('opportunity',opportunity),('normal_TN',tn)]:
                for age,(e,risk,gate) in enumerate(history):
                    risk_rows.append(dict(sequence_id=seq,source_id=source,frame=k+1,history_index=age,scope=scope,pixels=int(sel.sum()),old_gate_sum=float(e['g'][sel].sum()),new_gate_sum=float(gate[sel].sum()),old_risk_sum=float(e['risk'][sel].sum()),new_risk_sum=float(risk[sel].sum())))
        print('E4_AUDIT',seq,'max_float32_error',max_error,flush=True)
    source_rows=[dict(source_id=s,method=m,**{f:int(v) for f,v in zip(FIELDS,c)},**e3.metrics(c)) for (s,m),c in sorted(groups.items())]
    profile_rows=[dict(profile=s,method=m,**{f:int(v) for f,v in zip(FIELDS,c)},**e3.metrics(c)) for (s,m),c in sorted(profiles.items())]
    method_rows=[]
    for m in plan['methods']:
        c=sum((v for (s,method),v in groups.items() if method==m),np.zeros(8,np.int64));sr=[r for r in source_rows if r['method']==m]
        method_rows.append(dict(method=m,**{f:int(v) for f,v in zip(FIELDS,c)},**e3.metrics(c),source_macro_IoU=sum(r['IoU'] for r in sr)/3))
    for name,table in [('per_frame_counts',all_counts),('per_source_metrics',source_rows),('per_profile_metrics',profile_rows),('method_summary',method_rows),('historical_opportunity',opportunities),('current_probability_bins',bins),('changes_relative_to_v2',pair_rows),('risk_diagnostics',risk_rows)]:csv_write(out/(name+'.csv'),table)
    assert len(all_counts)==864 and len(opportunities)==96 and sum(r['oracle_opportunity'] for r in opportunities)==26284 and sum(r['v2_recovered'] for r in opportunities)==2650
    scalar=audit_scalar_tables(out);write(out/'scalar_audit.json',scalar)
    summary=dict(status='PASS_E4_SINGLE_FACTOR_DIAGNOSTIC',sequences=12,sources=3,frames=108,primary_frames=96,count_rows=864,independently_rebuilt_outputs=108,maximum_float32_error=max_error,mask_files=108,geometry_files=12,
        oracle_opportunity=26284,v2_recovered=2650,no_conflict_recovered=sum(r['no_conflict_recovered'] for r in opportunities),protected_pixels=protected_total,maximum_absolute_update=max_change,pixels_exceeding_old_bound=bound_exceed_pixels,
        changes_relative_to_v2={f:sum(r[f] for r in pair_rows) for f in FIELDS[4:]},network_forward_calls=0,optimizer_updates=0,validation_and_heldout_payloads_opened=0,old_core_unchanged=True,
        predictions_sha256=summary['predictions_sha256'],plan_sha256=summary['plan_sha256'],wall_seconds=time.perf_counter()-timer,scope=plan['scope'])
    for item in plan['frozen_files']:assert sha(root/item['path'])==item['sha256']
    write(out/'summary.json',summary);shutil.copy2(Path(__file__),out/Path(__file__).name);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--prediction-run',required=True);p.add_argument('--count-run',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:evaluate(a.root.resolve(),a.prediction_run,a.count_run,a.out.resolve())
    except Exception as error:
        if a.out.exists():write(a.out/'failure.json',dict(error=repr(error)))
        raise
