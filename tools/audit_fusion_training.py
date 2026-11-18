"""Independent 13C3 training-only audit and first-seed threshold calibration.

No model is loaded or optimized. MATLAB probability/registration outputs are
frozen and hashed before any training mask is opened. Validation and held-out
payloads are rejected. NumPy/SciPy/Pillow implement the independent arithmetic.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from PIL import Image
from scipy import ndimage, signal
import scipy
import PIL

METHODS = ['S', 'U', 'V', 'P', 'GV', 'GVP', 'GC', 'R']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def require_training(selection, ids):
    by_id = {row['sequence_id']: row for row in selection}
    if len(ids) != len(set(ids)) or any(i not in by_id or by_id[i]['assigned_split'] != 'training' for i in ids):
        raise ValueError('Only frozen training sequences are permitted')
    return {i: by_id[i] for i in ids}


def load_probability(path):
    p = np.fromfile(str(path), dtype='<f4')
    assert p.size == 65536 and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    return p.reshape((256,256), order='F')


def gray(rgb):
    rgb = rgb.astype(np.float64)
    return .2989*rgb[:,:,0]/255 + .5870*rgb[:,:,1]/255 + .1140*rgb[:,:,2]/255


def warp(p, shifts):
    yy, xx = np.mgrid[1:257,1:257]
    qx, qy = xx.astype(float), yy.astype(float)
    valid = np.ones((256,256), bool)
    for dx, dy in reversed(shifts):
        qx -= dx; qy -= dy
        valid &= (qx >= 1) & (qx <= 256) & (qy >= 1) & (qy <= 256)
    # Explicit four-neighbour sampling, independently of MATLAB interp2.
    qx = np.clip(qx-1, 0, 255); qy = np.clip(qy-1, 0, 255)
    x0 = np.floor(qx).astype(int); y0 = np.floor(qy).astype(int)
    x1 = np.minimum(x0+1,255); y1 = np.minimum(y0+1,255)
    fx = qx-x0; fy = qy-y0; p = p.astype(float)
    a = (1-fx)*p[y0,x0] + fx*p[y0,x1]
    b = (1-fx)*p[y1,x0] + fx*p[y1,x1]
    out = (1-fy)*a + fy*b; out[~valid] = 0
    return out, valid


def confidence(p):
    p = p.astype(float); h = np.zeros_like(p); inside = (p>0) & (p<1)
    h[inside] = -p[inside]*np.log2(p[inside])-(1-p[inside])*np.log2(1-p[inside])
    return 1-h


def gate(current, historical, cg, hg, valid, quality, cfg):
    kernel = np.ones(tuple(cfg['photo_window_hw']))
    den = signal.convolve2d(valid.astype(float), kernel, mode='same')
    residual = np.abs(cg-hg); residual[~valid] = 0
    residual = signal.convolve2d(residual, kernel, mode='same')/np.maximum(den,1)
    photo = np.exp(-residual**2/(2*cfg['photo_sigma']**2)); photo[den==0] = 0
    eps = cfg['confidence_epsilon']
    conf = np.clip(2*(eps+confidence(historical))/(2*eps+confidence(historical)+confidence(current)),0,1)
    g = quality*photo*conf; g[~valid] = 0
    return g


def edge_quality(e, cfg):
    f, b = e['visual'], e['reverse']
    if not f['available'] or not b['available'] or f['boundary_peak'] or b['boundary_peak']:
        return 0.
    s2f = e['forward_diagnostic']['second_score']; s2b = e['reverse_diagnostic']['second_score']
    if s2f is None or s2b is None:
        return 0.
    a = min(np.clip((f['score']-.2)/cfg['score_scale'],0,1),np.clip((b['score']-.2)/cfg['score_scale'],0,1))
    m = min(np.clip((f['score']-s2f)/cfg['peak_margin_scale'],0,1),np.clip((b['score']-s2b)/cfg['peak_margin_scale'],0,1))
    cycle = (f['dx']+b['dx'])**2+(f['dy']+b['dy'])**2
    return float(a*m*np.exp(-cycle/(2*cfg['cycle_sigma_px']**2)))


def fuse(images, probabilities, edges, cfg, constant):
    p = probabilities[-1].astype(float); n = len(probabilities)
    out = {m:p.copy() for m in METHODS}
    if n == 1:
        return {m:a.astype(np.float32) for m,a in out.items()}
    grays = [gray(a) for a in images]; w = 1/n
    for j in range(n-1):
        out['U'] += w*(probabilities[j].astype(float)-p)
        path = edges[j:]
        if all(e['visual']['available'] for e in path):
            shifts = [(e['visual']['dx'],e['visual']['dy']) for e in path]
            hp, valid = warp(probabilities[j],shifts); hg,_ = warp(grays[j],shifts)
            q = float(np.prod([edge_quality(e,cfg['visual_gate']) for e in path]))
            z = float(np.prod([e['observed_factor'] for e in path]))
            g = gate(p,hp,grays[-1],hg,valid,q,cfg['visual_gate'])
            d = hp-p; d[~valid] = 0
            out['V'] += w*d; out['GV'] += w*g*d; out['GVP'] += w*z*g*d; out['GC'] += w*constant*g*d
        if all(e['prior']['available'] for e in path):
            shifts = [(e['prior']['dx'],e['prior']['dy']) for e in path]
            hp,valid = warp(probabilities[j],shifts); d = hp-p; d[~valid] = 0; out['P'] += w*d
    acc = probabilities[0].astype(float)
    for k in range(1,n):
        current = probabilities[k].astype(float); e = edges[k-1]
        if e['visual']['available']:
            shifts = [(e['visual']['dx'],e['visual']['dy'])]
            hp,valid = warp(acc,shifts); hg,_ = warp(grays[k-1],shifts)
            g = gate(current,hp,grays[k],hg,valid,edge_quality(e,cfg['visual_gate']),cfg['visual_gate'])
            d = hp-current; d[~valid] = 0; acc = current+.5*g*d
        else:
            acc = current
    out['R'] = acc
    return {m:np.clip(a,0,1).astype(np.float32) for m,a in out.items()}


def ncc_estimates(previous, current, delta, cfg):
    a,b = gray(previous),gray(current); patch=a[64:192,64:192]
    centered=patch-patch.mean(); n=patch.size
    numerator=signal.correlate(b,centered,mode='valid',method='fft')
    integral=np.pad(b,((1,0),(1,0)),mode='constant').cumsum(0).cumsum(1)
    integral2=np.pad(b*b,((1,0),(1,0)),mode='constant').cumsum(0).cumsum(1)
    sums=integral[128:,128:]-integral[:-128,128:]-integral[128:,:-128]+integral[:-128,:-128]
    squares=integral2[128:,128:]-integral2[:-128,128:]-integral2[128:,:-128]+integral2[:-128,:-128]
    denominator=np.sqrt(np.maximum(squares-sums*sums/n,0)*np.sum(centered*centered))
    corr=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    def select(xs,ys):
        sub=corr[np.ix_(64+ys,64+xs)]
        k=int(np.argmax(sub.ravel(order='F'))); row,col=np.unravel_index(k,sub.shape,order='F'); score=float(sub[row,col])
        available=np.isfinite(score) and score>=.2
        def quadratic(a,b,c):
            den=a-2*b+c
            return float(np.clip(.5*(a-c)/den,-.5,.5)) if den < -1e-12 else 0.
        ox=quadratic(sub[row,col-1],score,sub[row,col+1]) if 0<col<len(xs)-1 else 0.
        oy=quadratic(sub[row-1,col],score,sub[row+1,col]) if 0<row<len(ys)-1 else 0.
        return dict(available=bool(available),dx=float(xs[col]+ox),dy=float(ys[row]+oy),score=score,
                    integer_dx=int(xs[col]),integer_dy=int(ys[row]),
                    boundary_peak=bool(row==0 or col==0 or row==len(ys)-1 or col==len(xs)-1))
    xs=np.arange(-20,21); visual=select(xs,xs)
    prior=visual.copy()
    if delta is not None and np.isfinite(delta) and abs(delta)<=20:
        ys=np.arange(max(-20,int(np.ceil(-delta-4))),min(20,int(np.floor(-delta+4)))+1)
        local=select(np.arange(-4,5),ys)
        if local['available']: prior=local
    xx,yy=np.meshgrid(xs,xs)
    other=np.maximum(abs(xx-visual['integer_dx']),abs(yy-visual['integer_dy']))>2
    second=float(corr[np.ix_(64+xs,64+xs)][other].max())
    return visual,prior,second


def binary_counts(p,t,s,threshold=.5):
    b=p>=threshold; base=s>=threshold
    c=np.array([np.count_nonzero(b&t),np.count_nonzero(b&~t),np.count_nonzero(~b&t),np.count_nonzero(~b&~t)],np.int64)
    tr=np.array([np.count_nonzero(~base&t&b),np.count_nonzero(base&~t&~b),np.count_nonzero(base&t&~b),np.count_nonzero(~base&~t&b)],np.int64)
    return c,tr


def region_counts(pred,target):
    g,ng=ndimage.label(target,structure=np.ones((3,3),int)); h,nh=ndimage.label(pred,structure=np.ones((3,3),int))
    gs=np.bincount(g.ravel(),minlength=ng+1); hs=np.bincount(h.ravel(),minlength=nh+1)
    cover=np.bincount(g[pred].ravel(),minlength=ng+1)
    detected=cover[1:]>=.25*gs[1:]; best=np.zeros(nh+1)
    both=(g>0)&(h>0)
    if both.any():
        pairs,intersections=np.unique(h[both].astype(np.int64)*(ng+1)+g[both],return_counts=True)
        hp,gp=np.divmod(pairs,ng+1)
        iou=intersections/(hs[hp]+gs[gp]-intersections)
        np.maximum.at(best,hp,iou)
    bins=[]
    for lower,upper in [(1,15),(16,63),(64,255),(256,np.inf)]:
        sel=(gs[1:]>=lower)&(gs[1:]<=upper)
        bins.extend([int(sel.sum()),int((sel&detected).sum())])
    return dict(gt_components=int(ng),detected_gt_components=int(detected.sum()),
                prediction_components=int(nh),false_prediction_components=int((best[1:]<.1).sum()),size_bin_counts=bins)


def region_contract():
    z=np.zeros((8,8),bool)
    assert region_counts(z,z)['gt_components']==0 and region_counts(z,z)['false_prediction_components']==0
    p=z.copy();p[0,0]=p[6,6]=True
    assert region_counts(p,z)['false_prediction_components']==2
    target=z.copy();target[0:2,0:2]=True;target[6:8,6:8]=True
    pred=target.copy();np.fill_diagonal(pred,True)
    result=region_counts(pred,target)
    assert result['gt_components']==2 and result['prediction_components']==1 and result['detected_gt_components']==2 and result['false_prediction_components']==0
    target=z.copy();target[1:5,1:5]=True;pred=z.copy();pred[1,1]=pred[4,4]=True
    result=region_counts(pred,target)
    assert result['gt_components']==1 and result['prediction_components']==2 and result['detected_gt_components']==0 and result['false_prediction_components']==2
    return dict(status='PASS',cases=['empty','no_GT_two_false_components','merge_8_connectivity','fragment_under_coverage_and_IoU_threshold'])


def audit(root,run_id,matlab_evaluation_run_id,out):
    assert not out.exists(),'No overwrite';out.mkdir(parents=True)
    started=time.perf_counter();run=root/'runs'/run_id
    (out/'audit_fusion_training.py').write_bytes(Path(__file__).read_bytes())
    protocol=read(root/'configs/fusion_protocol_v1.json');cfg=protocol['inference']
    assert sha(root/'configs/fusion_protocol_v1.json')=='20fd3a8839131d294d64053d2b9cde0b4adf6ef837482f2d4604cb22d37cb6c1'
    summary=read(run/'summary.json')
    assert summary['status']=='PASS_TRAINING_INPUT_PIPELINE_PENDING_INDEPENDENT_AUDIT' and summary['seed']==20261114
    manifest=read(run/'prediction_manifest.json')
    assert sha(run/'prediction_manifest.json')==summary['prediction_manifest_sha256']
    assert manifest['protocol_sha256']==sha(root/'configs/fusion_protocol_v1.json')
    assert manifest['constant_sha256']==sha(run/'constant_control.json')
    matlab_evaluation=root/'runs'/matlab_evaluation_run_id
    matlab_summary=read(matlab_evaluation/'summary.json')
    assert matlab_summary['status']=='PASS_MATLAB_TRAINING_COUNTS'
    assert matlab_summary['prediction_manifest_sha256']==summary['prediction_manifest_sha256']
    assert sha(matlab_evaluation/'per_frame_counts.csv')==matlab_summary['counts_sha256']
    with (matlab_evaluation/'per_frame_counts.csv').open(encoding='utf-8-sig',newline='') as f:
        matlab_counts={(r['sequence_id'],int(r['frame']),r['method']):r for r in csv.DictReader(f)}
    assert len(matlab_counts)==2592
    selection=read(root/protocol['data']['selection_file']);ids=protocol['data']['contract_and_calibration_sequence_ids']
    selected=require_training(selection,ids)
    # Executable rejection path; never opens a validation payload.
    rejected=False
    try: require_training(selection,ids+[protocol['data']['validation_sequence_ids'][0]])
    except ValueError: rejected=True
    assert rejected
    expected={r['path']:r for r in read(root/'runs/step12_full_v2_01/evaluation_only/payload_manifest.json')}
    for entry in manifest['raw']+manifest['fused']:
        assert entry['sequence_id'] in selected
        assert not Path(entry['path']).is_absolute() and '..' not in Path(entry['path']).parts
        assert sha(run/entry['path'])==entry['sha256']
    raw={(r['sequence_id'],r['frame']):r for r in manifest['raw']}
    fused={(r['sequence_id'],r['frame'],r['method']):r for r in manifest['fused']}
    assert len(raw)==324 and len(fused)==2592
    # This guard is recorded before any mask read.
    write(out/'prediction_freeze_before_masks.json',dict(prediction_manifest_sha256=sha(run/'prediction_manifest.json'),
          probability_files_sha256_verified=2916,validation_payloads_opened=0))
    const=read(run/'constant_control.json');constant=const['value']
    reg_rows={i:read(run/'registration'/f'{i}.json')['edges'] for i in ids}
    factors=[];qmax=0.;zmax=0.
    for i in ids:
        edges=reg_rows[i]
        for e in edges:
            qmax=max(qmax,abs(edge_quality(e,cfg['visual_gate'])-e['quality']))
            z=1.
            if e['observed_available'] and e['visual']['available']:
                z=float(np.exp(-(e['visual']['dy']+e['observed_delta'])**2/(2*cfg['observed_gate']['sigma_px']**2)))
            zmax=max(zmax,abs(z-e['observed_factor']))
        for k in range(1,9):
            for j in range(max(0,k-2),k):
                path=edges[j:k]
                if all(e['visual']['available'] for e in path) and any(e['observed_available'] for e in path):
                    factors.append(float(np.prod([e['observed_factor'] for e in path])))
    recalculated=float(np.mean(factors)) if factors else 1.
    assert abs(recalculated-constant)<1e-12 and len(factors)==const['eligible_paths'] and qmax<1e-12 and zmax<1e-12
    sources=sorted({r['source_id'] for r in selected.values()});assert len(sources)==3
    counts_grid=np.zeros((8,3,19,4),np.int64);primary=[];components=[];algorithm_error=0.;different_pixels=0;compared=0
    ncc_error=0.;ncc_checks=0;mask_access=[];rgb_access=[]
    first_source=sources[0]
    for ix,i in enumerate(ids):
        images=[];probabilities=[];targets=[];source_index=sources.index(selected[i]['source_id'])
        for k in range(1,10):
            ip=f'algorithm_inputs/{i}/frame_{k:03d}.png';path=root/'runs/step12_full_v2_01'/ip
            assert sha(path)==expected[ip]['sha256']==raw[i,k]['image_sha256']
            with Image.open(path) as im: image=np.array(im)
            assert image.shape==(256,256,3) and image.dtype==np.uint8
            images.append(image);probabilities.append(load_probability(run/raw[i,k]['path']));rgb_access.append(ip)
            mp=f'evaluation_only/sequences/{i}/mask_{k:03d}.png';path=root/'runs/step12_full_v2_01'/mp
            assert sha(path)==expected[mp]['sha256']
            with Image.open(path) as im: target=np.array(im.convert('L'))
            assert target.shape==(256,256) and set(np.unique(target))<={0,255}
            targets.append(target>=128);mask_access.append(dict(path=mp,sha256=expected[mp]['sha256']))
        edges=reg_rows[i]
        # Predeclared first training source, adjacent pairs ending at frames 2/6.
        if selected[i]['source_id']==first_source:
            for k in [1,5]:
                e=edges[k-1];v,p,s2=ncc_estimates(images[k-1],images[k],e['observed_delta'],cfg)
                b,_,s2b=ncc_estimates(images[k],images[k-1],None,cfg)
                for actual,reference in [(v,e['visual']),(p,e['prior']),(b,e['reverse'])]:
                    assert actual['available']==reference['available'] and actual['boundary_peak']==reference['boundary_peak']
                    for key in ['dx','dy','score']:
                        if actual['available']: ncc_error=max(ncc_error,abs(actual[key]-reference[key]))
                ncc_error=max(ncc_error,abs(s2-e['forward_diagnostic']['second_score']),abs(s2b-e['reverse_diagnostic']['second_score']))
                ncc_checks+=1
        for k in range(9):
            start=max(0,k-2);independent=fuse(images[start:k+1],probabilities[start:k+1],edges[start:k],cfg,constant)
            target=targets[k];single=probabilities[k]
            base,_=binary_counts(single,target,single)
            for mi,m in enumerate(METHODS):
                saved=load_probability(run/fused[i,k+1,m]['path']);other=independent[m]
                algorithm_error=max(algorithm_error,float(np.max(np.abs(saved.astype(float)-other.astype(float)))))
                different_pixels+=int(np.count_nonzero(saved!=other));compared+=1
                c,tr=binary_counts(saved,target,single)
                assert c[2]==base[2]-tr[0]+tr[2] and c[1]==base[1]-tr[1]+tr[3]
                row=dict(sequence_id=i,source_id=selected[i]['source_id'],frame=k+1,method=m,
                    TP=int(c[0]),FP=int(c[1]),FN=int(c[2]),TN=int(c[3]),
                    FN_to_TP=int(tr[0]),FP_to_TN=int(tr[1]),TP_to_FN=int(tr[2]),TN_to_FP=int(tr[3]))
                primary.append(row)
                reference=matlab_counts[i,k+1,m]
                for name in ['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP']:
                    assert row[name]==int(reference[name]),(i,k+1,m,name)
                rc=region_counts(saved>=.5,target);bins=rc.pop('size_bin_counts')
                components.append(dict(sequence_id=i,frame=k+1,method=m,**rc,**{f'bin_{b}':v for b,v in enumerate(bins)}))
                if k>0:
                    for ti,threshold in enumerate(protocol['calibration']['thresholds']):
                        cc,_=binary_counts(saved,target,single,threshold);counts_grid[mi,source_index,ti]+=cc
        print(f'INDEPENDENT_FUSION {ix+1}/36 outputs_checked={compared}',flush=True)
    assert compared==2592 and algorithm_error<=2e-6 and ncc_checks==24 and ncc_error<=2e-7
    assert len(mask_access)==324 and len(rgb_access)==324
    fixture=read(run/'analytic_contracts.json')
    assert len(fixture['checks'])==20 and all(r['status']=='PASS' for r in fixture['checks'])
    fixture_source=root/'runs/fusion_analytic_v1_01'
    sample=read(fixture_source/'count_fixture.json')
    pp=np.fromfile(str(fixture_source/'fixture_probability.f32'),dtype='<f4')
    cc,tt=binary_counts(pp,np.array(sample['target'],bool),np.array(sample['single_probability'],np.float32))
    assert np.array_equal(cc,sample['counts']) and np.array_equal(tt,sample['transitions'])
    region_report=region_contract()
    choices=[];curves=[]
    for mi,m in enumerate(METHODS):
        tp,fp,fn=[counts_grid[mi,:,:,k].astype(float) for k in range(3)]
        union=tp+fp+fn;iou=np.divide(tp,union,out=np.ones_like(tp),where=union>0);macro=iou.mean(axis=0)
        candidates=np.flatnonzero(np.abs(macro-macro.max())<=1e-12)
        thresholds=protocol['calibration']['thresholds'];chosen=sorted(candidates,key=lambda k:(abs(thresholds[k]-.5),-thresholds[k]))[0]
        choices.append(dict(method=m,seed=20261114,threshold=thresholds[chosen],
            training_source_macro_IoU=float(macro[chosen]),scope='training-in-sample threshold reference only'))
        for ti,threshold in enumerate(thresholds):
            agg=counts_grid[mi,:,ti].sum(0)
            curves.append(dict(method=m,threshold=threshold,source_macro_IoU=float(macro[ti]),
                TP=int(agg[0]),FP=int(agg[1]),FN=int(agg[2]),TN=int(agg[3])))
    write_csv(out/'per_frame_counts.csv',primary);write_csv(out/'per_frame_components.csv',components)
    write_csv(out/'training_threshold_curves.csv',curves)
    write(out/'thresholds_seed20261114.json',dict(status='FROZEN_TRAINING_IN_SAMPLE_REFERENCE',seed=20261114,
        choices=choices,protocol_sha256=manifest['protocol_sha256'],prediction_manifest_sha256=sha(run/'prediction_manifest.json'),
        second_and_third_seed_calibration='pending C4; not selected here'))
    write(out/'mask_access.json',dict(training_only=True,files=mask_access,validation_or_heldout_payloads_opened=0,accurate_geometry_payloads_opened=0))
    checks=fixture['checks']+[
        dict(id='training_calibration_excludes_validation_gt',status='PASS',scope='training metadata membership, validation-ID rejection and actual mask access ledger'),
        dict(id='region_metrics_empty_and_split_merge_cases',status='PASS',scope=region_report['cases'])]
    assert {r['id'] for r in checks}==set(protocol['required_c3_contract_ids'])
    for entry in manifest['raw']+manifest['fused']:assert sha(run/entry['path'])==entry['sha256']
    assert sha(run/'prediction_manifest.json')==summary['prediction_manifest_sha256']
    result=dict(status='PASS_13C3_TRAINING_IMPLEMENTATION_AUDIT',contracts_passed=22,checks=checks,
        seed=20261114,training_sources=3,training_sequences=36,raw_probability_files_verified=324,
        fused_probability_files_verified=2592,primary_count_identities_verified=2592,
        MATLAB_primary_and_transition_rows_exactly_reproduced=2592,
        independent_fusion_outputs_recomputed=2592,maximum_independent_probability_error=algorithm_error,
        numerically_different_float32_pixels=different_pixels,probability_tolerance=2e-6,
        ncc_pairs_recomputed_both_directions=ncc_checks,maximum_independent_ncc_error=ncc_error,ncc_tolerance=2e-7,
        edge_quality_max_error=qmax,observed_factor_max_error=zmax,constant_value=constant,constant_paths=len(factors),
        constant_recomputed_error=abs(recalculated-constant),training_mask_files_verified=324,
        validation_or_heldout_payloads_opened=0,accurate_geometry_payloads_opened=0,optimizer_updates=0,
        model_forward_calls_by_python=0,calibration_scope='first seed, three in-sample training sources, not validation benefit',
        runtime_versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,Pillow=PIL.__version__),
        prediction_manifest_sha256=summary['prediction_manifest_sha256'],wall_seconds=time.perf_counter()-started,
        scope='Implementation acceptance only; H1/H2/H3 performance not evaluated on validation')
    write(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--matlab-evaluation-run-id',required=True)
    args=p.parse_args()
    try:
        audit(args.project_root.resolve(),args.run_id,args.matlab_evaluation_run_id,args.out.resolve())
    except Exception:
        target=args.out.resolve()/'failure.txt'
        if args.out.resolve().is_dir() and not target.exists():
            target.write_text(traceback.format_exc(),encoding='utf-8')
        raise
