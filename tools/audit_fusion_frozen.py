"""C4 independent count/region audit at frozen inputs and thresholds.

Training mode calibrates the two remaining seeds. Validation mode never selects
thresholds. Reuses the immutable independent NumPy arithmetic accepted in C3.
"""
import argparse
import csv
import json
import platform
from pathlib import Path
import time
import traceback
import numpy as np
import scipy
import PIL
from PIL import Image

from audit_fusion_training import (METHODS, sha, read, write, write_csv,
    load_probability, fuse, ncc_estimates, binary_counts as legacy_binary_counts, region_counts, edge_quality)


def binary_counts(p,target,single,threshold=.5):
    # MATLAB compares single arrays to the actual double scalar threshold.
    # Older NumPy otherwise rounds that scalar to float32 before comparison.
    return legacy_binary_counts(p.astype(np.float64),target,single.astype(np.float64),float(threshold))


def run_audit(root,run_id,evaluation_id,out,calibration_path):
    assert not out.exists(),'No overwrite';out.mkdir(parents=True);started=time.perf_counter()
    (out/'audit_fusion_frozen.py').write_bytes(Path(__file__).read_bytes())
    helper=Path(__file__).with_name('audit_fusion_training.py')
    assert sha(helper)=='1b8f1c29d816e35d9d6f39ac5527e5efee94724892684b9bcf87abbc088ed8f1'
    protocol_path=root/'configs/fusion_protocol_v1.json'
    assert sha(protocol_path)=='20fd3a8839131d294d64053d2b9cde0b4adf6ef837482f2d4604cb22d37cb6c1'
    protocol=read(protocol_path);cfg=protocol['inference'];run=root/'runs'/run_id
    summary=read(run/'summary.json');mode=summary['mode'];seed=int(summary['seed'])
    assert summary['status']=='PASS_FROZEN_INFERENCE_PENDING_EVALUATION'
    assert mode in ['training','validation'] and seed in [20261114,20261115,20261116]
    ids=protocol['data']['contract_and_calibration_sequence_ids' if mode=='training' else 'validation_sequence_ids']
    selection={r['sequence_id']:r for r in read(root/protocol['data']['selection_file'])}
    assert all(selection[i]['assigned_split']==mode for i in ids)
    sources=sorted({selection[i]['source_id'] for i in ids});groups={selection[i]['group_id'] for i in ids}
    assert len(sources)==len(groups)==(3 if mode=='training' else 15)
    assert len(ids)==len(sources)*12
    thresholds=None
    if mode=='validation':
        calibration=read(calibration_path)
        assert calibration['status']=='FROZEN_THREE_SEED_TRAINING_CALIBRATION'
        assert sha(calibration_path)==sha(run/'frozen_calibration.json')
        thresholds={r['method']:r['threshold'] for r in calibration['choices'] if r['seed']==seed}
        assert set(thresholds)==set(METHODS)
    manifest_path=run/'prediction_manifest.json';manifest=read(manifest_path)
    assert sha(manifest_path)==summary['prediction_manifest_sha256']
    assert manifest['constant_sha256']==sha(run/'constant_control.json')
    raw={(r['sequence_id'],r['frame']):r for r in manifest['raw']}
    fused={(r['sequence_id'],r['frame'],r['method']):r for r in manifest['fused']}
    assert len(raw)==len(ids)*9 and len(fused)==len(ids)*72
    for entry in manifest['raw']+manifest['fused']:
        assert entry['sequence_id'] in ids and '..' not in Path(entry['path']).parts
        assert sha(run/entry['path'])==entry['sha256']
    evaluation=root/'runs'/evaluation_id;ev=read(evaluation/'summary.json')
    assert ev['status']=='PASS_MATLAB_FROZEN_COUNTS' and ev['mode']==mode and int(ev['seed'])==seed
    assert ev['prediction_manifest_sha256']==sha(manifest_path)
    assert ev['counts_sha256']==sha(evaluation/'per_frame_counts.csv')
    with (evaluation/'per_frame_counts.csv').open(encoding='utf-8-sig',newline='') as f:
        reference={(r['sequence_id'],int(r['frame']),r['method'],r['threshold_kind']):r for r in csv.DictReader(f)}
    expected_rows=len(ids)*9*8*(1 if mode=='training' else 2)
    assert len(reference)==expected_rows
    write(out/'prediction_freeze_before_masks.json',dict(prediction_manifest_sha256=sha(manifest_path),
        probability_files_sha256_verified=len(raw)+len(fused),mode=mode,seed=seed,
        calibration_sha256=sha(calibration_path) if mode=='validation' else None))
    expected={r['path']:r for r in read(root/'runs/step12_full_v2_01/evaluation_only/payload_manifest.json')}
    constant=read(run/'constant_control.json')['value'];assert abs(constant-.9597680584784836)<1e-15
    primary=[];components=[];mask_access=[];rgb_access=[];curve_counts=np.zeros((8,len(sources),19,4),np.int64)
    errors=0.;different=0;algorithm_frames=0;ncc_error=0.;ncc_pairs=0;quality_error=0.;vp_equal=0
    first_source=sources[0];registration=[]
    old13b={}
    if mode=='validation':
        with (root/'runs/registration_validation_v1_01/predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            old13b={(r['sequence_id'],int(r['current_frame']),r['method']):r for r in csv.DictReader(f)}
    for ii,i in enumerate(ids):
        sr=selection[i];si=sources.index(sr['source_id']);edges=read(run/'registration'/f'{i}.json')['edges']
        for k,e in enumerate(edges):
            quality_error=max(quality_error,abs(edge_quality(e,cfg['visual_gate'])-e['quality']))
            registration.append(dict(sequence_id=i,current_frame=k+2,visual_dx=e['visual']['dx'],visual_dy=e['visual']['dy'],
                prior_dx=e['prior']['dx'],prior_dy=e['prior']['dy'],quality=e['quality'],observed_factor=e['observed_factor']))
            if mode=='validation':
                for key,label in [('visual','visual_ncc'),('prior','prior_ncc')]:
                    old=old13b[i,k+2,label];a=e[key]
                    assert bool(int(old['available']))==a['available']
                    if a['available']:
                        assert max(abs(a[z]-float(old[z])) for z in ['dx','dy','score'])<1e-12
        images={};probabilities={}
        if sr['source_id']==first_source:
            for k in [1,2,4,5,6]:
                relative=f'algorithm_inputs/{i}/frame_{k:03d}.png';p=root/'runs/step12_full_v2_01'/relative
                assert sha(p)==expected[relative]['sha256']==raw[i,k]['image_sha256']
                with Image.open(p) as im:images[k]=np.array(im)
                assert images[k].shape==(256,256,3);rgb_access.append(relative)
                probabilities[k]=load_probability(run/raw[i,k]['path'])
            for k in [2,6]:
                e=edges[k-2];v,p,s2=ncc_estimates(images[k-1],images[k],e['observed_delta'],cfg)
                b,_,s2b=ncc_estimates(images[k],images[k-1],None,cfg)
                for a,ref in [(v,e['visual']),(p,e['prior']),(b,e['reverse'])]:
                    assert a['available']==ref['available'] and a['boundary_peak']==ref['boundary_peak']
                    if a['available']:ncc_error=max(ncc_error,max(abs(a[z]-ref[z]) for z in ['dx','dy','score']))
                ncc_error=max(ncc_error,abs(s2-e['forward_diagnostic']['second_score']),abs(s2b-e['reverse_diagnostic']['second_score']));ncc_pairs+=1
        for k in range(1,10):
            mp=f'evaluation_only/sequences/{i}/mask_{k:03d}.png';p=root/'runs/step12_full_v2_01'/mp
            assert sha(p)==expected[mp]['sha256']
            with Image.open(p) as im:t=np.array(im.convert('L'))
            assert t.shape==(256,256) and set(np.unique(t))<={0,255};target=t>=128;mask_access.append(dict(path=mp,sha256=expected[mp]['sha256']))
            single=load_probability(run/raw[i,k]['path']);independent=None
            if sr['source_id']==first_source and k in [2,6]:
                start=max(1,k-2)
                independent=fuse([images[j] for j in range(start,k+1)],
                    [probabilities[j] for j in range(start,k+1)],edges[start-1:k-1],cfg,constant)
            saved_v=None
            for mi,m in enumerate(METHODS):
                p=load_probability(run/fused[i,k,m]['path'])
                if m=='V':saved_v=p
                if m=='P' and np.array_equal(p,saved_v):vp_equal+=1
                if independent is not None:
                    other=independent[m];errors=max(errors,float(np.max(np.abs(p.astype(float)-other.astype(float)))))
                    different+=int(np.count_nonzero(p!=other));algorithm_frames+=1
                variants=[('primary',.5)]
                if mode=='validation':variants.append(('training_calibrated',thresholds[m]))
                for kind,threshold in variants:
                    c,tr=binary_counts(p,target,single,threshold);base,_=binary_counts(single,target,single,threshold)
                    assert c[2]==base[2]-tr[0]+tr[2] and c[1]==base[1]-tr[1]+tr[3]
                    rr=dict(seed=seed,sequence_id=i,source_id=sr['source_id'],group_id=sr['group_id'],profile=sr['profile'],
                        frame=k,method=m,threshold_kind=kind,threshold=threshold,
                        TP=int(c[0]),FP=int(c[1]),FN=int(c[2]),TN=int(c[3]),FN_to_TP=int(tr[0]),FP_to_TN=int(tr[1]),TP_to_FN=int(tr[2]),TN_to_FP=int(tr[3]))
                    ref=reference[i,k,m,kind]
                    for name in ['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']:
                        assert rr[name]==int(ref[name]),(i,k,m,kind,name)
                    primary.append(rr)
                    rc=region_counts(p.astype(np.float64)>=float(threshold),target);bins=rc.pop('size_bin_counts')
                    components.append(dict(seed=seed,sequence_id=i,source_id=sr['source_id'],group_id=sr['group_id'],profile=sr['profile'],
                        frame=k,method=m,threshold_kind=kind,**rc,**{f'bin_{b}':v for b,v in enumerate(bins)}))
                if mode=='training' and k>=2:
                    for ti,threshold in enumerate(protocol['calibration']['thresholds']):
                        c,_=binary_counts(p,target,single,threshold);curve_counts[mi,si,ti]+=c
        if (ii+1)%12==0:print(f'FROZEN_AUDIT {mode} seed={seed} {ii+1}/{len(ids)} count_rows={len(primary)}',flush=True)
    assert len(primary)==expected_rows and algorithm_frames==192 and ncc_pairs==24
    assert errors<=2e-6 and ncc_error<=2e-7 and quality_error<=1e-12
    write_csv(out/'per_frame_counts.csv',primary);write_csv(out/'per_frame_components.csv',components)
    write_csv(out/'registration_audit.csv',registration)
    if mode=='training':
        choices=[];curves=[];ts=protocol['calibration']['thresholds']
        for mi,m in enumerate(METHODS):
            tp,fp,fn=[curve_counts[mi,:,:,j].astype(float) for j in range(3)];union=tp+fp+fn
            macro=np.divide(tp,union,out=np.ones_like(tp),where=union>0).mean(0)
            candidates=np.flatnonzero(abs(macro-macro.max())<=1e-12);pick=sorted(candidates,key=lambda j:(abs(ts[j]-.5),-ts[j]))[0]
            choices.append(dict(seed=seed,method=m,threshold=ts[pick],training_source_macro_IoU=float(macro[pick]),scope='training-in-sample threshold reference only'))
            for j,threshold in enumerate(ts):
                c=curve_counts[mi,:,j].sum(0);curves.append(dict(method=m,threshold=threshold,source_macro_IoU=float(macro[j]),TP=int(c[0]),FP=int(c[1]),FN=int(c[2]),TN=int(c[3])))
        write(out/f'thresholds_seed{seed}.json',dict(status='FROZEN_TRAINING_IN_SAMPLE_REFERENCE',seed=seed,choices=choices,
            protocol_sha256=sha(protocol_path),prediction_manifest_sha256=sha(manifest_path)))
        write_csv(out/'training_threshold_curves.csv',curves)
    write(out/'mask_access.json',dict(mode=mode,files=mask_access,heldout_payloads_opened=0,accurate_geometry_payloads_opened=0))
    for entry in manifest['raw']+manifest['fused']:assert sha(run/entry['path'])==entry['sha256']
    assert sha(manifest_path)==summary['prediction_manifest_sha256']
    result=dict(status='PASS_C4_FROZEN_INDEPENDENT_AUDIT',mode=mode,seed=seed,sources=len(sources),sequences=len(ids),
        probability_files_sha_verified=len(raw)+len(fused),MATLAB_count_and_transition_rows_exactly_reproduced=expected_rows,
        independent_fusion_outputs_recomputed=algorithm_frames,maximum_independent_probability_error=errors,
        different_float32_pixels_in_fixed_sample=different,ncc_pairs_both_directions=24,maximum_independent_ncc_error=ncc_error,
        maximum_edge_quality_error=quality_error,V_P_exactly_equal_method_frames=vp_equal,
        mask_files_sha_verified=len(mask_access),RGB_sample_files_sha_verified=len(rgb_access),
        threshold_selection_on_validation=False,heldout_payloads_opened=0,accurate_geometry_payloads_opened=0,
        optimizer_updates=0,model_forward_calls_by_auditor=0,prediction_manifest_sha256=sha(manifest_path),
        threshold_comparison='float32 values promoted to float64 versus unchanged float64 threshold',
        wall_seconds=time.perf_counter()-started,
        runtime_versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,Pillow=PIL.__version__))
    write(out/'audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--run-id',required=True);p.add_argument('--evaluation-run-id',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibration-freeze',type=Path)
    a=p.parse_args()
    try:run_audit(a.project_root.resolve(),a.run_id,a.evaluation_run_id,a.out.resolve(),a.calibration_freeze.resolve() if a.calibration_freeze else None)
    except Exception:
        if a.out.is_dir() and not (a.out/'failure.txt').exists():(a.out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
