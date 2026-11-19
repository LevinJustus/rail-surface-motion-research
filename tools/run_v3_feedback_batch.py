"""One bounded v3 prototype batch: contracts, label-free production, then evaluation.

--contracts-only runs without any rail images, model weights or prior run caches.
The historical 12-sequence batch requires the unshared local E3/E4 input caches.
"""
import argparse,csv,hashlib,json,os,shutil,sys,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import rail_motion_v3 as v3

FIELDS=['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
    assert not p.exists(),p;p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def readp(p):return np.fromfile(str(p),dtype='<f4').reshape((256,256),order='F').astype(float)
def counts(a,t,p):
    b=a>=.5;s=p>=.5
    return [int(np.count_nonzero(x)) for x in [b&t,b&~t,~b&t,~b&~t,~s&t&b,s&~t&~b,s&t&~b,~s&~t&b]]
def metrics(c):
    tp,fp,fn=c[:3];return dict(IoU=tp/(tp+fp+fn) if tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)

def contracts(out):
    names=[];cfg=dict(v3.DEFAULTS);a=np.arange(65536,dtype=float).reshape(256,256)/65535
    h,valid=v3.warp(a,[(3,-4)]);assert np.array_equal(h[:-4,3:],a[4:,:-3]) and not valid[:,:3].any();names.append('forward_sign_and_valid_support')
    _,valid=v3.warp(a,[(300,0),(-300,0)]);assert not valid.any();names.append('intermediate_view_rejected')
    p=np.full((256,256),.1);history=np.full((256,256),.8)
    strong=v3.fuse([history,history,p],[[0,0],[0,0]],False,cfg);weak=v3.fuse([history,history,p],[[0,0],[0,0]],True,cfg)
    assert np.all(strong>=.5) and np.all(weak<.5);names.append('strong_correction_and_conservative_reference')
    assert counts(strong,np.zeros((256,256),bool),p)[7]==65536;names.append('wrong_history_false_positive_negative_case')
    random=np.random.RandomState(20261119);rgb=random.randint(0,256,(256,256,3)).astype(np.uint8);b=np.roll(np.roll(rgb,-5,axis=0),7,axis=1)
    hypotheses=v3.motion_hypotheses(v3.gray(rgb),v3.gray(b),None,cfg);assert np.linalg.norm(np.array(hypotheses[0]['xy'])-[7,-5])<.1;names.append('image_motion_sign_and_subpixel_peak')
    for i,h1 in enumerate(hypotheses):
        for h2 in hypotheses[i+1:]:assert max(abs(np.array(h1['integer_xy'])-h2['integer_xy']))>=cfg['peak_separation']
    names.append('separated_modes_not_neighbouring_pixels')
    # Deliberately ambiguous base scores, while actual image evidence favours +6.
    yy,xx=np.mgrid[0:256,0:256];texture=.5+.12*np.sin(2*np.pi*xx/12)+.12*np.cos(2*np.pi*yy/16)
    texture[40:180,20:23]-=.3;current=np.roll(texture,6,axis=1)
    raw=np.zeros((256,256));raw[40:180,20:23]=.9;weights=v3.historical_roi(raw,raw,[0,0],cfg)
    options=[dict(xy=[-6.,0.],base_score=.995),dict(xy=[6.,0.],base_score=.990)]
    selected,scores=v3.select_motion(texture,current,options,weights,cfg);assert selected==[6.,0.];names.append('historical_roi_image_check_can_disambiguate')
    assert v3.historical_roi(np.zeros((256,256)),raw,[0,0],cfg) is None;names.append('no_persistent_evidence_falls_back')
    first=np.zeros((256,256),np.float32);first[100:130,100:105]=.9;streams=[v3.MotionFeedback(),v3.MotionFeedback()];records=[]
    for stream,last in zip(streams,[np.full((256,256),.1),np.full((256,256),.9)]):
        for k in range(3):
            image=np.roll(rgb,k*3,axis=1);prob=np.roll(first,k*3,axis=1) if k<2 else last
            result,diagnostic=stream.step(image,prob,float(k));assert all(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 for a in result.values())
        assert np.array_equal(stream.raw[-1],np.asarray(last,np.float32));records.append(diagnostic)
    assert records[0]['feedback']==records[1]['feedback'] and records[0]['feedback_scores']==records[1]['feedback_scores'];names.append('current_prediction_excluded_from_motion_feedback')
    names.append('raw_cache_and_convex_probability_range')
    try:streams[0].step(rgb,first,2.)
    except ValueError:names.append('nonincreasing_timestamp_rejected')
    else:raise AssertionError('timestamp accepted')
    write(out/'contracts.json',dict(status='PASS_V3_CONTRACTS',contract_count=len(names),contracts=names,analytic_motion_feedback_scores=scores,scope='synthetic mechanism checks, not rail performance'))

def produce(root,out):
    cfg=read(root/'configs/algorithm_design_v3.json');assert cfg['parameters']==v3.DEFAULTS
    oldrun=root/'runs/v2_conflict_prediction_v1_01';oldrows=read(oldrun/'predictions.json');assert len(oldrows)==864
    oldindex={(r['sequence_id'],r['frame'],r['method']):r for r in oldrows};plan=read(oldrun/'frozen_plan.json');selection=plan['selection'];assert len(selection)==12
    expected={r['path']:r['sha256'] for r in read(root/'docs/evidence/step13c2/expected_input_manifest.json')['files']}
    sources=['python/rail_motion_v3.py','tools/run_v3_feedback_batch.py','configs/algorithm_design_v3.json']
    frozen={p:sha(root/p) for p in sources};write(out/'frozen_plan.json',dict(design=cfg,source_sha256=frozen,selection=selection,parent_index_sha256=sha(oldrun/'predictions.json'),seed=plan['seed'],checkpoint=plan['checkpoint'],parameters_searched=0,threshold=.5,authorization='User authorized autonomous v3 prototype, bounded controls and this batch GitHub update.'))
    (out/'source_snapshot').mkdir()
    for p in sources:shutil.copy2(root/p,out/'source_snapshot'/Path(p).name)
    for r in oldrows:assert sha(root/r['path'])==r['sha256']
    rows=list(oldrows);all_diagnostics=[];access=[];timer=time.perf_counter()
    for selected in selection:
        assert selected['assigned_split']=='training';seq=selected['sequence_id'];folder=root/'runs/step12_full_v2_01/algorithm_inputs'/seq;mp=folder/'motion_input.json'
        assert sha(mp)==expected['algorithm_inputs/'+seq+'/motion_input.json'];motion=read(mp)['frames'];stream=v3.MotionFeedback(cfg['parameters']);(out/'predictions'/seq).mkdir(parents=True)
        for k,f in enumerate(motion,1):
            ip=folder/f['image'];image_sha=sha(ip);assert image_sha==expected['algorithm_inputs/'+seq+'/'+f['image']]
            r=oldindex[seq,k,'S'];assert sha(root/r['path'])==r['sha256'];p=readp(root/r['path']);rgb=np.asarray(Image.open(str(ip)))
            odometry=f['simulated_odometry_px'] if f['odometry_available'] else None
            result,d=stream.step(rgb,p,f['timestamp_s'],odometry);d.update(sequence_id=seq,frame=k);all_diagnostics.append(d)
            for method,a in result.items():
                path=out/'predictions'/seq/('frame_%03d_%s.f32'%(k,method));path.write_bytes(a.astype('<f4').tobytes(order='F'))
                rows.append(dict(sequence_id=seq,frame=k,method=method,path=path.relative_to(root).as_posix(),sha256=sha(path)))
            access.append(dict(path=ip.relative_to(root).as_posix(),sha256=image_sha))
        print('V3_PRODUCTION',seq,flush=True)
    assert len(rows)==1404
    for p,h in frozen.items():assert sha(root/p)==h
    write(out/'predictions.json',rows);write(out/'motion_diagnostics.json',all_diagnostics);write(out/'input_access.json',access)
    write(out/'production.json',dict(status='PASS_V3_PREDICTIONS_FROZEN',new_outputs=540,total_outputs=1404,predictions_sha256=sha(out/'predictions.json'),frozen_plan_sha256=sha(out/'frozen_plan.json'),network_forward_calls=0,optimizer_updates=0,mask_and_geometry_payloads_opened=0,validation_and_test_payloads_opened=0,wall_seconds=time.perf_counter()-timer,scope='cached segmentation, Python prototype including registration and disk I/O; not full-system latency'))

def evaluate(root,out):
    # Evaluation-only dependencies and all mask/geometry payloads open here.
    import evaluate_v2_screening as evaluation
    timer=time.perf_counter();production=read(out/'production.json');assert production['status']=='PASS_V3_PREDICTIONS_FROZEN'
    assert sha(out/'predictions.json')==production['predictions_sha256'];plan=read(out/'frozen_plan.json');rows=read(out/'predictions.json')
    for r in rows:assert sha(root/r['path'])==r['sha256']
    write(out/'freeze_before_evaluation.json',dict(predictions_sha256=production['predictions_sha256'],verified_outputs=1404,labels_opened=0))
    idx={(r['sequence_id'],r['frame'],r['method']):r for r in rows};diagnostics={(r['sequence_id'],r['frame']):r for r in read(out/'motion_diagnostics.json')}
    payload={r['path']:r['sha256'] for r in read(root/'runs/step12_full_v2_01/evaluation_only/payload_manifest.json')}
    with (root/'runs/v2_conflict_evaluation_v1_01/per_frame_counts.csv').open(encoding='utf-8-sig') as f:oldcounts={(r['sequence_id'],int(r['frame']),r['method']):r for r in csv.DictReader(f)}
    methods=list(read(root/'runs/v2_conflict_prediction_v1_01/frozen_plan.json')['methods'])+list(v3.VARIANTS)
    allcounts=[];motionrows=[];pairrows=[];opportunityrows=[];groups=defaultdict(lambda:[0]*8);profiles=defaultdict(lambda:[0]*8)
    for selected in plan['selection']:
        seq=selected['sequence_id'];source=selected['source_id'];profile=selected['profile'];P=[readp(root/idx[seq,k,'S']['path']) for k in range(1,10)];masks=[]
        for k in range(1,10):
            rel='evaluation_only/sequences/%s/mask_%03d.png'%(seq,k);p=root/'runs/step12_full_v2_01'/rel;assert sha(p)==payload[rel];masks.append(np.asarray(Image.open(str(p)))>0)
        rel='evaluation_only/sequences/'+seq+'/geometry_truth.json';p=root/'runs/step12_full_v2_01'/rel;assert sha(p)==payload[rel];T=[np.asarray(f['frame_to_source_3x3']) for f in read(p)['frames']]
        for k in range(9):
            p=P[k];target=masks[k];values={m:readp(root/idx[seq,k+1,m]['path']) for m in methods};opportunity=np.zeros((256,256),bool)
            for j in range(max(0,k-2),k):
                h,t,valid,_,_=evaluation.oracle(P,masks,T,j,k);opportunity|=target&(p<.5)&valid&t&(h>=.5)
            for method,value in values.items():
                c=counts(value,target,p)
                if method not in v3.VARIANTS:assert c==[int(oldcounts[seq,k+1,method][f]) for f in FIELDS]
                allcounts.append(dict(sequence_id=seq,source_id=source,profile=profile,frame=k+1,method=method,**dict(zip(FIELDS,c))))
                if k:
                    for i,a in enumerate(c):groups[source,method][i]+=a;profiles[profile,method][i]+=a
                    opportunityrows.append(dict(sequence_id=seq,frame=k+1,method=method,opportunity=int(opportunity.sum()),recovered=int(np.count_nonzero(opportunity&(value>=.5)))))
            if not k:continue
            for name,proposed,baseline in [('feedback_conservative','C_feedback_conservative','B_base_conservative'),('feedback_strong','D_feedback_strong','E_base_strong'),('roi_vs_uniform','D_feedback_strong','F_uniform_strong')]:
                c=counts(values[proposed],target,values[baseline]);pairrows.append(dict(sequence_id=seq,source_id=source,profile=profile,frame=k+1,contrast=name,**dict(zip(FIELDS[4:],c[4:]))))
            d=diagnostics[seq,k+1];_,_,valid,qx,qy=evaluation.oracle(P,masks,T,k-1,k)
            roi=valid&target;central=np.zeros_like(valid);central[64:192,64:192]=True;central&=valid
            r=dict(sequence_id=seq,source_id=source,profile=profile,frame=k+1,feedback_available=d['feedback_available'],hypothesis_count=len(d['hypotheses']),selection_changed=d['base']!=d['feedback'],uniform_changed=d['base']!=d['uniform'],defect_pixels=int(roi.sum()))
            for name in ['base','feedback','uniform']:
                shift=d[name]
                if shift is None:r[name+'_central_EPE']=None;r[name+'_defect_EPE']=None
                else:
                    err=np.hypot(evaluation.X-shift[0]-qx,evaluation.Y-shift[1]-qy)
                    r[name+'_central_EPE']=float(err[central].mean());r[name+'_defect_EPE']=float(err[roi].mean()) if roi.any() else None
            r['best_candidate_central_EPE']=min(float(np.hypot(evaluation.X-h['xy'][0]-qx,evaluation.Y-h['xy'][1]-qy)[central].mean()) for h in d['hypotheses']) if d['hypotheses'] else None
            motionrows.append(r)
        print('V3_EVALUATION',seq,flush=True)
    sr=[dict(source_id=s,method=m,**dict(zip(FIELDS,c)),**metrics(c)) for (s,m),c in sorted(groups.items())];pr=[dict(profile=s,method=m,**dict(zip(FIELDS,c)),**metrics(c)) for (s,m),c in sorted(profiles.items())];mr=[]
    for m in methods:
        c=[sum(v[i] for (s,name),v in groups.items() if name==m) for i in range(8)];sources=[r for r in sr if r['method']==m]
        mr.append(dict(method=m,**dict(zip(FIELDS,c)),**metrics(c),source_macro_IoU=sum(r['IoU'] for r in sources)/3))
    for name,data in [('per_frame_counts',allcounts),('per_source_metrics',sr),('per_profile_metrics',pr),('method_summary',mr),('motion_metrics',motionrows),('paired_changes',pairrows),('historical_opportunity',opportunityrows)]:table(out/(name+'.csv'),data)
    assert len(allcounts)==1404 and len(motionrows)==96
    summary=dict(status='PASS_V3_BOUNDED_PROTOTYPE_BATCH',sequences=12,sources=3,frames=108,primary_frames=96,method_frames=1404,new_probability_outputs=540,
        feedback_available_edges=sum(r['feedback_available'] for r in motionrows),feedback_changed_edges=sum(r['selection_changed'] for r in motionrows),uniform_changed_edges=sum(r['uniform_changed'] for r in motionrows),
        motion={name:sum(r[name+'_central_EPE'] for r in motionrows if r[name+'_central_EPE'] is not None)/sum(r[name+'_central_EPE'] is not None for r in motionrows) for name in ['base','feedback','uniform']},
        paired_changes={name:{f:sum(r[f] for r in pairrows if r['contrast']==name) for f in FIELDS[4:]} for name in ['feedback_conservative','feedback_strong','roi_vs_uniform']},
        historical_opportunity={m:dict(opportunities=sum(r['opportunity'] for r in opportunityrows if r['method']==m),recovered=sum(r['recovered'] for r in opportunityrows if r['method']==m)) for m in methods},
        old_control_count_rows_exact=864,network_forward_calls=0,optimizer_updates=0,validation_and_test_payloads_opened=0,evaluation_wall_seconds=time.perf_counter()-timer,
        scope='development only, 3 training-source proxies; correlated same-image sequences; no final-test, real-video, novelty or full-system-FPS proof')
    for p,h in plan['source_sha256'].items():assert sha(root/p)==h
    write(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--contracts-only',action='store_true');args=parser.parse_args();out=Path(os.path.abspath(str(args.out)));out.relative_to(ROOT/'runs');assert not out.exists();out.mkdir(parents=True)
    try:
        contracts(out)
        if not args.contracts_only:produce(ROOT,out);evaluate(ROOT,out)
        else:print((out/'contracts.json').read_text(),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
