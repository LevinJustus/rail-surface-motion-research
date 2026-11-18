"""Independent C5 arithmetic, content hashes and optional complete count audit.

Training contracts open no labels. Validation counts require a completed
separate MATLAB evaluation. Uses the historical independent C3 operators,
without modifying them; probability threshold comparisons are explicit float64.
"""
import argparse
from collections import defaultdict
import copy
import csv
import itertools
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
from PIL import Image
from scipy import signal
import audit_fusion_training as old

sha, read, write, csv_write = old.sha, old.read, old.write, old.write_csv
METHODS=old.METHODS
CONSTANT=.9597680584784836
FIELDS=['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']


def probability(root,row,prefix='probability'):
    path=Path(row[prefix+'_path']);assert not path.is_absolute() and '..' not in path.parts
    return old.load_probability(root/path)


def factors(e,cfg):
    a,b=e['visual'],e['reverse'];f=e['forward_diagnostic']['second_score'];r=e['reverse_diagnostic']['second_score']
    if not a['available'] or not b['available'] or a['boundary_peak'] or b['boundary_peak'] or f is None or r is None:
        return 0.,0.,0.
    score=min(np.clip((a['score']-.2)/cfg['score_scale'],0,1),np.clip((b['score']-.2)/cfg['score_scale'],0,1))
    margin=min(np.clip((a['score']-f)/cfg['peak_margin_scale'],0,1),np.clip((b['score']-r)/cfg['peak_margin_scale'],0,1))
    cycle=np.exp(-((a['dx']+b['dx'])**2+(a['dy']+b['dy'])**2)/(2*cfg['cycle_sigma_px']**2))
    return float(score*margin*cycle),float(score*margin),float(score*cycle)


def context(h,hg,p,cg,path,cfg,alignment='visual'):
    if not all(e[alignment]['available'] for e in path):return None
    hp,valid=old.warp(h,[(e[alignment]['dx'],e[alignment]['dy']) for e in path])
    history_gray,_=old.warp(hg,[(e[alignment]['dx'],e[alignment]['dy']) for e in path])
    kernel=np.ones(tuple(cfg['photo_window_hw']));den=signal.convolve2d(valid.astype(float),kernel,mode='same')
    residual=np.abs(cg-history_gray);residual[~valid]=0
    residual=signal.convolve2d(residual,kernel,mode='same')/np.maximum(den,1)
    photo=np.exp(-residual**2/(2*cfg['photo_sigma']**2));photo[den==0]=0
    eps=cfg['confidence_epsilon'];conf=np.clip(2*(eps+old.confidence(hp))/(2*eps+old.confidence(hp)+old.confidence(p)),0,1)
    qs=np.prod(np.array([factors(e,cfg) for e in path]),axis=0)
    z=float(np.prod([e['observed_factor'] for e in path]));d=hp-p;d[~valid]=0
    return d,photo,conf,qs,z


def recurrent(P,grays,edges,cfg):
    acc=P[0].astype(float)
    for k in range(1,len(P)):
        current=P[k].astype(float);e=edges[k-1]
        if e['visual']['available']:
            shift=[(e['visual']['dx'],e['visual']['dy'])];hp,v=old.warp(acc,shift);hg,_=old.warp(grays[k-1],shift)
            g=old.gate(current,hp,grays[k],hg,v,old.edge_quality(e,cfg['visual_gate']),cfg['visual_gate'])
            d=hp-current;d[~v]=0;acc=np.clip(current+.5*g*d,0,1)
        else:acc=current
    return acc.astype(np.float32)


def variants(images,P,edges,stresses,cfg):
    k=len(P);start=max(0,k-3);p=P[-1].astype(float);grays=[old.gray(i) for i in images]
    base=old.fuse(images[start:],P[start:],edges[start:],cfg,CONSTANT);out={}
    scenarios=['bias_m6','bias_m2','bias_p2','bias_p6','alternating_1','alternating_3','all_missing']
    for name,stress in zip(scenarios,stresses):
        f=old.fuse(images[start:],P[start:],stress[start:k-1],cfg,CONSTANT)
        for m in METHODS:out[name,m]=f[m]
    for window in [1,5]:
        startw=max(0,k-window);f=old.fuse(images[startw:],P[startw:],edges[startw:],cfg,CONSTANT)
        for m in ['S','V','GV','GVP']:out['window_'+str(window),m]=f[m]
    for removal in ['photo','confidence','cycle','peak_margin']:
        value=p.copy()
        for j in range(start,k-1):
            c=context(P[j],grays[j],p,grays[-1],edges[j:],cfg['visual_gate'])
            if c is None:continue
            d,photo,conf,qs,z=c;q=qs[0]
            if removal=='photo':photo=np.ones_like(photo)
            elif removal=='confidence':conf=np.ones_like(conf)
            elif removal=='cycle':q=qs[1]
            else:q=qs[2]
            value+=(1/min(k,3))*(q*photo*conf)*d
        out['remove_'+removal,'GV']=np.clip(value,0,1).astype(np.float32)
    value=p.copy()
    for j in range(start,k-1):
        c=context(P[j],grays[j],p,grays[-1],edges[j:],cfg['visual_gate'],'prior')
        if c is not None:
            d,photo,conf,qs,z=c;value+=(1/min(k,3))*z*(qs[0]*photo*conf)*d
    out['prior_alignment','GVP']=np.clip(value,0,1).astype(np.float32)
    out['unbounded','R']=recurrent(P,grays,edges,cfg)
    poisoned=list(P)
    if k>=3:poisoned[2]=(1-P[2].astype(float)).astype(np.float32)
    bad=old.fuse(images[start:],poisoned[start:],edges[start:],cfg,CONSTANT)
    for m in METHODS:out['pollution',m]=bad[m]
    out['pollution','R_unbounded']=recurrent(poisoned,grays,edges,cfg)
    assert len(out)==79
    return out


def observation_deltas(motion):
    frames=motion['frames'];result=[]
    for scenario in range(7):
        obs=[]
        for k,f in enumerate(frames):
            v=f['simulated_odometry_px'];has=f['odometry_available'] and v is not None
            if scenario==6 or not has:obs.append(None);continue
            offset=([-6,-2,2,6][scenario]*k if scenario<4 else [1,3][scenario-4]*(-1)**k)
            obs.append(v+offset)
        result.append([None if obs[k] is None or obs[k-1] is None else obs[k]-obs[k-1] for k in range(1,9)])
    return result


def check_edges(images,motion,reg,cfg):
    checks=0;maximum=0.;deltas=observation_deltas(motion)
    # Fixed adjacent pairs ending in frames 2 and 6, every frozen condition.
    for k in [1,5]:
        e=reg['base_edges'][k-1]
        if not e['visual']['available']:continue
        vf,_,second=old.ncc_estimates(images[k-1],images[k],None,cfg['registration'])
        vb,_,_=old.ncc_estimates(images[k],images[k-1],None,cfg['registration'])
        for actual,expected in [(e['visual'],vf),(e['reverse'],vb)]:
            for key in ['dx','dy','score']:
                err=abs(actual[key]-expected[key]);maximum=max(maximum,err);assert err<=2e-7,(key,err)
        for s in range(7):
            delta=deltas[s][k-1];edge=reg['stress_edges'][s][k-1]
            _,prior,_=old.ncc_estimates(images[k-1],images[k],delta,cfg['registration'])
            for key in ['dx','dy','score']:
                err=abs(edge['prior'][key]-prior[key]);maximum=max(maximum,err);assert err<=2e-7,(s,key,err)
            z=1.
            if delta is not None and abs(delta)<=20:z=float(np.exp(-(vf['dy']+delta)**2/(2*cfg['observed_gate']['sigma_px']**2)))
            assert abs(z-edge['observed_factor'])<=2e-7;checks+=1
    return checks,maximum


def audit(root,run_id,out,evaluation=None,synthetic=None):
    assert not out.exists();out.mkdir(parents=True);timer=time.perf_counter();run=root/'runs'/run_id
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    summary=read(run/'summary.json');assert summary['status']=='PASS_C5_PREDICTION_PENDING_AUDIT'
    assert sha(run/'predictions.jsonl')==summary['predictions_sha256']
    assert sha(run/'payloads.jsonl')==summary['payloads_index_sha256']
    assert sha(run/'registration_index.json')==summary['registration_index_sha256']
    assert sha(root/'configs/fusion_ablation_v1.json')==summary['config_sha256']=='82e4badd60122876c9aca97d05b36cfe443a38caa55fd95f1ccb4a47b8bb3994'
    protocol=read(root/'configs/fusion_protocol_v1.json');cfg=protocol['inference'];mode=summary['mode'];seed=summary['seed']
    ids=protocol['data']['contract_and_calibration_sequence_ids' if mode=='training' else 'validation_sequence_ids']
    selection={r['sequence_id']:r for r in read(root/protocol['data']['selection_file'])}
    assert all(selection[i]['assigned_split']==mode for i in ids);sample_source=sorted({selection[i]['source_id'] for i in ids})[0]
    plan=read(root/'configs/fusion_ablation_v1.json');arms=[(a['scenario'],a['method']) for a in plan['arms']]
    rows=[json.loads(line) for line in (run/'predictions.jsonl').open(encoding='utf-8-sig')];assert len(rows)==len(ids)*9*79
    unique={}
    for ri,r in enumerate(rows):
        arm=plan['arms'][ri%79]
        assert r['seed']==seed and r['sequence_id']==ids[ri//(9*79)] and r['frame']==(ri//79)%9+1
        assert all(r[field]==arm[field] for field in ['scenario','method','parent_method'])
        for prefix in ['probability','single_reference','clean_reference']:
            path=r[prefix+'_path'];h=r[prefix+'_sha256'];assert '..' not in Path(path).parts and not Path(path).is_absolute()
            if path in unique:assert unique[path]==h
            else:unique[path]=h
    for i,(p,h) in enumerate(unique.items()):
        assert sha(root/p)==h,p
        if (i+1)%10000==0:print('C5_HASH',seed,i+1,len(unique),flush=True)
    write(out/'prediction_freeze_before_masks.json',dict(manifest_sha256=summary['predictions_sha256'],unique_paths_verified=len(unique),mode=mode,labels_opened=0))
    reg_index={r['sequence_id']:r for r in read(run/'registration_index.json')}
    base=root/'runs'/summary['base_prediction_run'];bm=read(base/'prediction_manifest.json');assert sha(base/'prediction_manifest.json')==summary['base_manifest_sha256']
    raw={(r['sequence_id'],r['frame']):r for r in bm['raw']}
    maxerror=0.;comparisons=0;ncc_checks=0;ncc_error=0.;identities=defaultdict(int)
    for seq,group in itertools.groupby(rows,key=lambda r:r['sequence_id']):
        g=list(group);assert seq in ids and len(g)==9*79
        for k in range(1,10):
            frame=g[(k-1)*79:k*79];assert [(r['scenario'],r['method']) for r in frame]==arms
            by={(r['scenario'],r['method']):r for r in frame}
            for scenario in ['bias_m6','bias_m2','bias_p2','bias_p6','alternating_1','alternating_3','all_missing']:
                for m in ['S','U','V','GV','GC','R']:
                    r=by[scenario,m];assert r['probability_sha256']==r['clean_reference_sha256'];identities['unchanged_visual_outputs']+=1
            assert by['all_missing','P']['probability_sha256']==by['all_missing','V']['probability_sha256'];identities['missing_prior_fallback']+=1
            assert by['all_missing','GVP']['probability_sha256']==by['all_missing','GV']['probability_sha256'];identities['missing_gate_fallback']+=1
            for m in ['S','V','GV','GVP']:
                assert by['window_1',m]['probability_sha256']==by['window_1','S']['probability_sha256'];identities['W1_identity']+=1
            for m in METHODS:
                r=by['pollution',m]
                if k>=6 or k<=2:assert r['probability_sha256']==r['clean_reference_sha256'];identities['bounded_pollution_exact_recovery_or_pre_fault']+=1
        if selection[seq]['source_id']!=sample_source:continue
        regrow=reg_index[seq];assert sha(root/regrow['json_path'])==regrow['json_sha256'];reg=read(root/regrow['json_path'])
        folder=root/protocol['data']['input_root_relative']/seq;motion=read(folder/'motion_input.json');assert sha(folder/'motion_input.json')==reg['motion_sha256']
        for k,f in enumerate(motion['frames'],1):assert sha(folder/f['image'])==raw[seq,k]['image_sha256']
        images=[np.asarray(Image.open(str(folder/f['image']))) for f in motion['frames']]
        P=[old.load_probability(base/raw[seq,k]['path']) for k in range(1,10)]
        checks,err=check_edges(images,motion,reg,cfg);ncc_checks+=checks;ncc_error=max(ncc_error,err)
        for k in [2,6]:
            expected=variants(images[:k],P[:k],reg['base_edges'][:k-1],[e[:k-1] for e in reg['stress_edges']],cfg)
            for r in g[(k-1)*79:k*79]:
                actual=probability(root,r);error=float(np.max(np.abs(actual.astype(float)-expected[r['scenario'],r['method']].astype(float))))
                assert error<=2e-6,(seq,k,r['scenario'],r['method'],error);maxerror=max(maxerror,error);comparisons+=1
        print('C5_ARITHMETIC',seed,seq,comparisons,'max',maxerror,flush=True)
    assert comparisons==12*2*79
    report=dict(status='PASS_C5_TRAINING_CONTRACTS' if mode=='training' else 'PASS_C5_INDEPENDENT_AUDIT',mode=mode,seed=seed,
        prediction_run=run_id,predictions_sha256=summary['predictions_sha256'],logical_outputs=len(rows),unique_paths_verified=len(unique),
        independent_algorithm_outputs=comparisons,maximum_probability_error=maxerror,independent_prior_NCC_checks=ncc_checks,
        maximum_NCC_error=ncc_error,exact_identity_checks=dict(identities),training_masks_opened=0,validation_masks_opened=0,
        heldout_payloads_opened=0,model_forward_calls=0,optimizer_updates=0,python_version=platform.python_version(),numpy_version=np.__version__)
    if mode=='training':
        syn=read(root/'runs'/synthetic/'contracts.json');assert syn['status']=='PASS_C5_SYNTHETIC' and syn['maximum_probability_error']==0
        report['synthetic_report_sha256']=sha(root/'runs'/synthetic/'contracts.json')
        write(out/'audit.json',dict(report,wall_seconds=time.perf_counter()-timer))
        write(run/'contracts.json',dict(report,independent_audit_sha256=sha(out/'audit.json')))
    else:
        ev=root/'runs'/evaluation;es=read(ev/'summary.json');assert es['status']=='PASS_MATLAB_C5_COUNTS'
        assert es['predictions_sha256']==summary['predictions_sha256'] and sha(ev/'per_frame_counts.csv')==es['counts_sha256']
        report.update(audit_counts(root,rows,selection,seed,ev,out))
        report.update(evaluation_run=evaluation,matlab_counts_sha256=es['counts_sha256'],wall_seconds=time.perf_counter()-timer)
        write(out/'audit.json',report)
    print(json.dumps(report,ensure_ascii=True),flush=True)


def audit_counts(root,rows,selection,seed,ev,out):
    freeze=read(root/'configs/fusion_calibration_v1.json');thresholds={(r['seed'],r['method']):r['threshold'] for r in freeze['choices']}
    expected={r['path']:r['sha256'] for r in read(root/'runs/step12_full_v2_01/evaluation_only/payload_manifest.json')}
    csv_in=(ev/'per_frame_counts.csv').open(encoding='utf-8-sig',newline='');reader=csv.DictReader(csv_in)
    csv_out=(out/'per_frame_counts.csv').open('w',encoding='utf-8',newline='');writer=csv.DictWriter(csv_out,fieldnames=reader.fieldnames);writer.writeheader()
    n=0;mask_n=0;maxfloat=0.
    for (seq,k),group in itertools.groupby(rows,key=lambda r:(r['sequence_id'],r['frame'])):
        rel='evaluation_only/sequences/{}/mask_{:03d}.png'.format(seq,k);path=root/'runs/step12_full_v2_01'/rel;assert sha(path)==expected[rel]
        a=np.asarray(Image.open(str(path)));target=a>0;assert target.shape==(256,256);mask_n+=1
        pcache={};ccache={};poll_cache={};class_cache={};delta_cache={}
        def get(r,prefix):
            h=r[prefix+'_sha256']
            if h not in pcache:pcache[h]=probability(root,r,prefix).astype(np.float64)
            return pcache[h]
        for r in group:
            p=get(r,'probability');s=get(r,'single_reference');clean=get(r,'clean_reference')
            pair=(r['probability_sha256'],r['clean_reference_sha256'])
            if pair not in delta_cache:
                d=np.abs(p-clean);delta_cache[pair]=(int(np.count_nonzero(d)),float(d.max()),float(d.mean()))
            changes,maxdelta,meandelta=delta_cache[pair]
            for kind,threshold in [('primary',.5),('training_calibrated',thresholds[seed,r['parent_method']])]:
                key=(r['probability_sha256'],r['single_reference_sha256'],threshold)
                if key not in ccache:ccache[key]=old.binary_counts(p,target,s,threshold)
                c,tr=ccache[key]
                for h,a in [(r['single_reference_sha256'],s),(r['clean_reference_sha256'],clean)]:
                    if (h,threshold) not in class_cache:class_cache[h,threshold]=old.binary_counts(a,target,a,threshold)[0]
                base=class_cache[r['single_reference_sha256'],threshold];cl=class_cache[r['clean_reference_sha256'],threshold]
                assert c[2]==base[2]-tr[0]+tr[2] and c[1]==base[1]-tr[1]+tr[3]
                values=dict(zip(FIELDS,map(int,list(c)+list(tr))))
                if (pair,threshold) not in poll_cache:poll_cache[pair,threshold]=int(np.count_nonzero((p>=threshold)!=(clean>=threshold)))
                values.update(probability_changed_pixels=changes,prediction_flip_pixels=poll_cache[pair,threshold],
                    clean_FP=int(cl[1]),clean_FN=int(cl[2]),delta_FP=int(c[1]-cl[1]),delta_FN=int(c[2]-cl[2]))
                m=next(reader);assert m['sequence_id']==seq and int(m['frame'])==k and m['scenario']==r['scenario'] and m['method']==r['method'] and m['threshold_kind']==kind
                assert int(m['seed'])==seed and m['parent_method']==r['parent_method']
                assert all(m[field]==selection[seq][field] for field in ['source_id','group_id','profile'])
                assert float(m['threshold'])==threshold
                for field,value in values.items():assert int(m[field])==value,(seq,k,r['scenario'],r['method'],field,m[field],value)
                for field,value in [('max_probability_delta',maxdelta),('mean_probability_delta',meandelta)]:
                    error=abs(float(m[field])-value);assert error<=2e-12,(seq,k,field,error);maxfloat=max(maxfloat,error)
                writer.writerow(m);n+=1
        if mask_n%108==0:print('C5_COUNTS_AUDIT',seed,mask_n,'rows',n,flush=True)
    assert next(reader,None) is None;n_expected=180*9*79*2;assert n==n_expected
    csv_in.close();csv_out.close()
    return dict(count_rows=n,validation_masks_opened=mask_n,maximum_delta_stat_error=maxfloat,
        counts_sha256=sha(out/'per_frame_counts.csv'),counts_scope='all logical rows independently counted; float64 threshold comparisons')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--run-id',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--evaluation-run');p.add_argument('--synthetic-run');a=p.parse_args()
    try:audit(a.project_root.resolve(),a.run_id,a.out.resolve(),a.evaluation_run,a.synthetic_run)
    except Exception:
        if a.out.exists():(a.out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
