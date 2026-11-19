"""E3 frozen, training-only diagnostic; oracle geometry is evaluation-only.

Independently rebuilds v2/matched integer outputs, compares all MATLAB counts,
and separates historical opportunity, candidate coverage and risk contributions.
"""
import argparse,csv,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import audit_fusion_training as old
import audit_v2_minimal as independent

FIELDS=['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']
sha,read,write,csv_write=old.sha,old.read,old.write,old.write_csv
Y,X=np.mgrid[1:257,1:257];GRID=np.array([X.ravel(),Y.ravel(),np.ones(65536)])


def safe(v):
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,list):return [safe(x) for x in v]
    if isinstance(v,np.generic):return safe(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def sample(a,qx,qy,nearest=False):
    valid=(qx>=1)&(qx<=256)&(qy>=1)&(qy<=256);x=np.clip(qx-1,0,255);y=np.clip(qy-1,0,255)
    if nearest:result=a[np.floor(y+.5).astype(int),np.floor(x+.5).astype(int)].astype(float)
    else:
        x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,255);y1=np.minimum(y0+1,255);fx=x-x0;fy=y-y0
        result=(1-fy)*((1-fx)*a[y0,x0]+fx*a[y0,x1])+fy*((1-fx)*a[y1,x0]+fx*a[y1,x1])
    result[~valid]=0;return result,valid


def oracle(P,masks,transforms,j,k):
    valid=np.ones((256,256),bool);q=None
    for m in range(j,k):
        B=np.linalg.solve(transforms[m],transforms[k]);q=B@GRID;valid&=(q[0].reshape(256,256)>=1)&(q[0].reshape(256,256)<=256)&(q[1].reshape(256,256)>=1)&(q[1].reshape(256,256)<=256)
    B=np.linalg.solve(transforms[j],transforms[k]);q=B@GRID;qx=q[0].reshape(256,256);qy=q[1].reshape(256,256)
    h,_=sample(P[j].astype(float),qx,qy);t,_=sample(masks[j],qx,qy,True)
    return h,t>=.5,valid,qx,qy


def edge_candidates(C,delta,std,cfg,actual):
    ranking=sorted(zip(-C.ravel(),np.repeat(np.arange(-20,21),41),np.tile(np.arange(-20,21),41)))
    mode=[ranking[0][2],ranking[0][1]];available=std>=1e-6 and np.isfinite(C).all() and C.max()>=.2 and not any(abs(v)==20 for v in mode)
    assert bool(actual['available'])==bool(available)
    if not available:return []
    if delta is not None and (not np.isfinite(delta) or abs(delta)>20):delta=None
    w,candidates=independent.candidates(C,delta,cfg);assert np.max(np.abs(w-np.asarray(actual['full_weights'])))<2e-8
    expected=np.asarray(candidates);stored=np.atleast_2d(np.asarray(actual['candidates']));assert stored.shape==expected.shape and np.max(abs(stored-expected))<2e-8
    # Downstream independent arithmetic uses the saved exact-double candidate
    # weights after NCC selection is independently verified, avoiding rank noise.
    return [tuple(r) for r in stored]


def evidence(P,edges,j,k,cfg,qx=None,qy=None):
    p=P[k].astype(float);q=np.zeros_like(p);first=q.copy();second=q.copy();known=q.copy();minimum=np.full_like(p,np.inf);possible=np.zeros_like(p,bool)
    paths=[] if any(not e for e in edges[j:k]) else independent.compose(edges[j:k])
    for shifts,w in paths:
        h,v=old.warp(P[j],shifts);wv=w*v;q+=wv;first+=wv*h;second+=wv*h*h;known+=wv*(h-p)**2
        possible|=v&(h>=.5)
        if qx is not None:
            dx=sum(s[0] for s in shifts);dy=sum(s[1] for s in shifts);error=np.hypot(X-dx-qx,Y-dy-qy);error[~v]=np.inf;minimum=np.minimum(minimum,error)
    q=np.clip(q,0,1);h=p.copy();sel=q>0;h[sel]=first[sel]/q[sel];var=np.zeros_like(p);var[sel]=np.maximum(0,second[sel]/q[sel]-h[sel]**2)
    conflict=q*(h-p)**2;variance=q*var;unknown=(1-q)*np.maximum(p*p,(1-p)**2);risk=known+unknown
    assert np.max(abs(known-variance-conflict))<1e-10;g=q*np.exp(-risk/(2*cfg['risk_sigma']**2))
    return dict(q=q,h=h,g=g,variance=variance,conflict=conflict,unknown=unknown,risk=risk,minimum=minimum,possible=possible,retained_mass=sum(w for _,w in paths))


def metrics(c):
    tp,fp,fn,tn=map(int,c[:4]);return dict(IoU=tp/(tp+fp+fn) if tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)


def evaluate(root,run_id,count_id,out):
    assert not out.exists();out.mkdir(parents=True);started=time.perf_counter();run=root/'runs'/run_id;summary=read(run/'summary.json');plan=read(run/'frozen_plan.json');cfg=read(root/plan['design_config'])
    assert summary['status']=='PASS_E3_PREDICTION_PENDING_EVALUATION' and sha(run/'predictions.json')==summary['predictions_sha256']
    assert sha(root/plan['design_config'])==plan['design_sha256'] and sha(run/'frozen_plan.json')==summary['plan_sha256']
    rows=read(run/'predictions.json');assert len(rows)==756;index={(r['sequence_id'],r['frame'],r['method']):r for r in rows}
    ids=[r['sequence_id'] for r in plan['selection']];assert len(ids)==12 and all(r['assigned_split']=='training' for r in plan['selection'])
    for r in rows:assert r['sequence_id'] in ids and sha(root/r['path'])==r['sha256']
    write(out/'prediction_freeze_before_masks.json',dict(predictions_sha256=summary['predictions_sha256'],logical_outputs_verified=756,labels_opened=0))
    count_folder=root/'runs'/count_id;cs=read(count_folder/'summary.json');assert cs['status']=='PASS_E3_MATLAB_COUNTS' and sha(count_folder/'per_frame_counts.csv')==cs['counts_sha256']
    matlab={(r['sequence_id'],int(r['frame']),r['method']):r for r in csv.DictReader((count_folder/'per_frame_counts.csv').open(encoding='utf-8-sig'))}
    payload={r['path']:r['sha256'] for r in read(root/'runs/step12_full_v2_01/evaluation_only/payload_manifest.json')}
    protocol=read(root/plan['protocol_config']);all_counts=[];opportunities=[];risks=[];motions=[];bins=[];groups=defaultdict(lambda:np.zeros(8,np.int64));maxerror=0.;nc=0;mask_reads=0;geo_reads=0
    for selected in plan['selection']:
        seq=selected['sequence_id'];input_folder=root/protocol['data']['input_root_relative']/seq;motion=read(input_folder/'motion_input.json')['frames'];RGB=[np.asarray(Image.open(str(input_folder/f['image']))) for f in motion]
        P=[old.load_probability(root/index[seq,k,'S']['path']) for k in range(1,10)];reg=read(run/'registration'/(seq+'.json'));main_edges=[];visual_edges=[]
        old_edges=read(root/'runs'/plan['base_run']/'registration'/(seq+'.json'))['edges']
        for k in range(1,9):
            C=independent.ncc(RGB[k-1],RGB[k]);std=float(old.gray(RGB[k-1])[64:192,64:192].std(ddof=1));a,b=motion[k-1:k+1];delta=None
            if a['odometry_available'] and b['odometry_available']:delta=b['simulated_odometry_px']-a['simulated_odometry_px']
            main_edges.append(edge_candidates(C,delta,std,cfg,reg['main'][k-1]));visual_edges.append(edge_candidates(C,None,std,cfg,reg['visual'][k-1]));nc+=2
        masks=[]
        for k in range(1,10):
            relative='evaluation_only/sequences/{}/mask_{:03d}.png'.format(seq,k);path=root/'runs/step12_full_v2_01'/relative;assert sha(path)==payload[relative];masks.append(np.asarray(Image.open(str(path)))>0);mask_reads+=1
        relative='evaluation_only/sequences/'+seq+'/geometry_truth.json';gp=root/'runs/step12_full_v2_01'/relative;assert sha(gp)==payload[relative];truth=read(gp);transforms=[np.asarray(f['frame_to_source_3x3']) for f in truth['frames']];geo_reads+=1
        for k in range(9):
            p=P[k].astype(float);target=masks[k];start=max(0,k-2);n=k-start+1;outmaps={m:old.load_probability(root/index[seq,k+1,m]['path']).astype(float) for m in plan['methods']};reconstructed={}
            for name,edges in [('V2',main_edges),('V2_visual',visual_edges)]:
                value=p.copy()
                for j in range(start,k):
                    e=evidence(P,edges,j,k,cfg);value+=(1/n)*e['g']*(e['h']-p)
                strong=p>=cfg['strong_current_foreground'];value[strong]=np.maximum(value[strong],.5);value=np.clip(value,0,1).astype(np.float32)
                error=float(np.max(abs(value.astype(float)-outmaps[name])));maxerror=max(maxerror,error);assert error<=2e-6,(seq,k,name,error)
            integer=p.copy()
            for j in range(start,k):
                if any(not e for e in main_edges[j:k]):continue
                h,v=old.warp(P[j],[(e[0][0],e[0][1]) for e in main_edges[j:k]]);d=h-p;d[~v]=0;integer+=(1/n)*d
            for name,value in [('MAP_integer',integer),('DECAY05_integer',p+.5*(integer-p))]:
                error=float(np.max(abs(value.astype(np.float32).astype(float)-outmaps[name])));maxerror=max(maxerror,error);assert error<=2e-6
            fn=target&(p<.5);tn=~target&(p<.5);opportunity=np.zeros_like(target);wrong=np.zeros_like(target);observable=np.zeros_like(target);history=[]
            for j in range(start,k):
                oh,ot,visible,qx,qy=oracle(P,masks,transforms,j,k);correct=visible&ot&(oh>=.5);bad=visible&~ot&(oh>=.5);opportunity|=fn&correct;wrong|=tn&bad;observable|=visible;history.append((j,visible,qx,qy))
            for name,value in outmaps.items():
                c,tr=old.binary_counts(value,target,p,.5);r=matlab[seq,k+1,name];assert all(int(r[f])==int(v) for f,v in zip(FIELDS,list(c)+list(tr)))
                result=dict(sequence_id=seq,source_id=selected['source_id'],profile=selected['profile'],frame=k+1,method=name,**{f:int(v) for f,v in zip(FIELDS,list(c)+list(tr))},oracle_opportunity=int(opportunity.sum()),opportunity_recovered=int((opportunity&(value>=.5)).sum()),exact_probability_changed_pixels=int(np.count_nonzero(value!=p)))
                all_counts.append(result)
                if k>=1:groups[selected['source_id'],name]+=np.r_[c,tr]
            if k<1:continue
            opportunities.append(dict(sequence_id=seq,source_id=selected['source_id'],profile=selected['profile'],frame=k+1,current_FN=int(fn.sum()),current_FN_with_history_view=int((fn&observable).sum()),oracle_opportunity=int(opportunity.sum()),current_TN_historical_FP_risk=int(wrong.sum()),v2_opportunity_recovered=int((opportunity&(outmaps['V2']>=.5)).sum()),below_update_recovery_bound_opportunity=int((opportunity&(p<.5-2/3*.25/math.sqrt(math.e))).sum())))
            boundaries=plan['current_probability_bins']
            for low,high in zip(boundaries[:-1],boundaries[1:]):
                sel=fn&(p>=low)&(p<high);bins.append(dict(sequence_id=seq,source_id=selected['source_id'],frame=k+1,lower=low,upper=high,current_FN=int(sel.sum()),oracle_opportunity=int((sel&opportunity).sum()),v2_FN_corrected=int((sel&(outmaps['V2']>=.5)).sum())))
            for j,visible,qx,qy in history:
                e=evidence(P,main_edges,j,k,cfg,qx,qy);covered1=e['minimum']<=1;covered2=e['minimum']<=2
                alignment={}
                for label,shifts in [('integer_MAP',[(a[0][0],a[0][1]) for a in main_edges[j:k]] if all(main_edges[j:k]) else []),('v1_subpixel',[(a['visual']['dx'],a['visual']['dy']) for a in old_edges[j:k]] if all(a['visual']['available'] for a in old_edges[j:k]) else [])]:
                    if shifts:
                        dx=sum(s[0] for s in shifts);dy=sum(s[1] for s in shifts);error=np.hypot(X-dx-qx,Y-dy-qy);_,v=old.warp(P[j],shifts);valid=visible&v
                        alignment[label+'_sampling_EPE_mean']=float(error[valid].mean()) if valid.any() else None
                        alignment[label+'_valid_pixels']=int(valid.sum())
                    else:alignment[label+'_sampling_EPE_mean']=None;alignment[label+'_valid_pixels']=0
                motions.append(dict(sequence_id=seq,source_id=selected['source_id'],profile=selected['profile'],frame=k+1,age=k-j,retained_path_mass=e['retained_mass'],oracle_visible_pixels=int(visible.sum()),candidate_within_1px=int((visible&covered1).sum()),candidate_within_2px=int((visible&covered2).sum()),opportunity_pixels=int((opportunity&visible).sum()),opportunity_candidate_within_1px=int((opportunity&visible&covered1).sum()),**alignment))
                for scope,sel in [('all',np.ones_like(target)),('current_FN',fn),('opportunity',opportunity),('opportunity_unrecovered',opportunity&(outmaps['V2']<.5)),('normal_TN',tn)]:
                    count=int(sel.sum());row=dict(sequence_id=seq,source_id=selected['source_id'],profile=selected['profile'],frame=k+1,age=k-j,scope=scope,pixels=count)
                    for key in ['q','g','variance','conflict','unknown','risk']:row[key+'_sum']=float(e[key][sel].sum())
                    row['q_lt_0_1']=int((sel&(e['q']<.1)).sum());row['g_lt_0_01']=int((sel&(e['g']<.01)).sum());risks.append(row)
        print('E3_AUDIT',seq,'max_error',maxerror,flush=True)
    assert len(all_counts)==756 and mask_reads==108 and geo_reads==12
    source_rows=[]
    for (source,m),c in sorted(groups.items()):source_rows.append(dict(source_id=source,method=m,**{f:int(v) for f,v in zip(FIELDS,c)},**metrics(c)))
    method_rows=[]
    for m in plan['methods']:
        c=sum((v for (s,method),v in groups.items() if method==m),np.zeros(8,np.int64));sr=[r for r in source_rows if r['method']==m]
        method_rows.append(dict(method=m,**{f:int(v) for f,v in zip(FIELDS,c)},**metrics(c),source_macro_IoU=float(np.mean([r['IoU'] for r in sr]))))
    for name,data in [('per_frame_counts',all_counts),('per_source_metrics',source_rows),('method_summary',method_rows),('historical_opportunity',opportunities),('current_probability_bins',bins),('candidate_geometry_coverage',motions),('risk_diagnostics',risks)]:csv_write(out/(name+'.csv'),safe(data))
    risk_summary=[]
    for scope in sorted({r['scope'] for r in risks}):
        rr=[r for r in risks if r['scope']==scope];n=sum(r['pixels'] for r in rr);row=dict(scope=scope,history_pixel_observations=n)
        for key in ['q','g','variance','conflict','unknown','risk']:row[key+'_mean']=sum(r[key+'_sum'] for r in rr)/n if n else None
        for key in ['q_lt_0_1','g_lt_0_01']:row[key+'_fraction']=sum(r[key] for r in rr)/n if n else None
        risk_summary.append(row)
    csv_write(out/'risk_summary.csv',risk_summary)
    result=dict(status='PASS_E3_TRAINING_DIAGNOSTIC',sequences=12,sources=3,profiles=4,frames=108,primary_frames=96,count_rows=756,
        independently_verified_NCC_selections=nc,independently_rebuilt_probability_outputs=432,maximum_probability_error=maxerror,mask_files=mask_reads,geometry_files=geo_reads,
        current_FN=sum(r['current_FN'] for r in opportunities),oracle_opportunity=sum(r['oracle_opportunity'] for r in opportunities),v2_opportunity_recovered=sum(r['v2_opportunity_recovered'] for r in opportunities),
        below_update_recovery_bound_opportunity=sum(r['below_update_recovery_bound_opportunity'] for r in opportunities),
        predictions_sha256=summary['predictions_sha256'],plan_sha256=summary['plan_sha256'],matlab_count_rows_verified=756,
        network_forward_calls=0,optimizer_updates=0,validation_and_heldout_payloads_opened=0,wall_seconds=time.perf_counter()-started,
        scope='developer training-source same-image sequences; oracle only diagnoses historical opportunity, not a deployable method or precision ceiling')
    write(out/'summary.json',safe(result));(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(safe(dict(summary=result,methods=method_rows,risk=risk_summary))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--run-id',required=True);p.add_argument('--count-run',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();evaluate(a.project_root.resolve(),a.run_id,a.count_run,a.out.resolve())
