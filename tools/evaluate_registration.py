"""Offline geometry evaluation and independent NumPy NCC checks; no masks.

Run only after run_registration_inputs has frozen its predictions and summary.
Requires NumPy and Pillow. Paths are explicit; no training/test enumeration.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(p, rows):
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def yes(x):
    return str(x).lower() in ('1', 'true')


def gray(p):
    a = np.asarray(Image.open(p), dtype=np.float64)
    return .2989*a[:, :, 0]/255 + .5870*a[:, :, 1]/255 + .1140*a[:, :, 2]/255


def independent_ncc(a, b):
    t = a[64:192, 64:192]
    t = t-t.mean()
    conv = np.fft.irfftn(np.fft.rfftn(b, s=(512, 512), axes=(0, 1))*np.fft.rfftn(t[::-1, ::-1], s=(512, 512), axes=(0, 1)), s=(512, 512), axes=(0, 1))
    def integral(x):
        return np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums, squares = integral(b), integral(b*b)
    rc = np.arange(44, 85)
    r, c = np.meshgrid(rc, rc, indexing='ij')
    def box(s):
        return s[r+128, c+128]-s[r, c+128]-s[r+128, c]+s[r, c]
    den = np.sqrt(np.maximum(0, box(squares)-box(sums)**2/(128*128))*np.sum(t*t))
    return np.divide(conv[r+127, c+127], den, out=np.zeros_like(den), where=den>0)


def peak(C, xs, ys, cfg):
    sub = C[np.ix_(ys+20, xs+20)]
    row, col = np.unravel_index(np.argmax(sub.ravel(order='F')), sub.shape, order='F')
    score = float(sub[row, col])
    if score < cfg['minimum_ncc']:
        return None
    def q(a, b, c):
        den = a-2*b+c
        return float(np.clip(.5*(a-c)/den, -.5, .5)) if den < -1e-12 else 0.
    ox = q(sub[row, col-1], score, sub[row, col+1]) if 0 < col < len(xs)-1 else 0.
    oy = q(sub[row-1, col], score, sub[row+1, col]) if 0 < row < len(ys)-1 else 0.
    return float(xs[col]+ox), float(ys[row]+oy), score


def main(args):
    run, seqroot = Path(args.run), Path(args.sequences)
    out = run/'evaluation'
    out.mkdir(exist_ok=False)
    summary = read(run/'summary.json')
    assert summary['status'] == 'PASS'
    prediction_hash = sha(run/'predictions.csv')
    assert prediction_hash == summary['predictions_sha256']
    cfg = read(run/'frozen_registration_config.json')
    assert sha(run/'frozen_registration_config.json') == summary['config_sha256']
    ids = cfg[summary['mode']+'_sequence_ids']
    source_index = {x['sequence_id']: x for x in read(seqroot/'evaluation_only/sequence_index.json')}
    assert all(source_index[x]['assigned_split'] == ('validation' if summary['mode']=='validation' else 'training') for x in ids)
    predictions = list(csv.DictReader((run/'predictions.csv').open(encoding='utf-8-sig', newline='')))
    assert len(predictions) == len(ids)*8*3
    expected = {(s, k, m) for s in ids for k in range(2, 10) for m in ('odometry', 'visual_ncc', 'prior_ncc')}
    assert {(r['sequence_id'], int(r['current_frame']), r['method']) for r in predictions} == expected
    access = read(run/'input_access.json')
    assert len(access) == len(ids)*10
    assert len({x['path'] for x in access}) == len(access)
    for x in access:
        p = Path(x['path'])
        assert len(p.parts)==2 and p.parts[0] in ids and p.name in ['motion_input.json']+[f'frame_{k:03}.png' for k in range(1, 10)]
        assert sha(seqroot/'algorithm_inputs'/p) == x['sha256']
    xx, yy = np.meshgrid(np.linspace(1, 256, 9), np.linspace(1, 256, 9))
    grid = np.vstack([xx.ravel(), yy.ravel(), np.ones(81)])
    center = np.array([128.5, 128.5, 1.])
    truths, metadata, motions, metrics = {}, {}, {}, []
    for seq in ids:
        p = seqroot/'evaluation_only/sequences'/seq/'geometry_truth.json'
        truths[seq] = read(p)
        metadata[seq] = sha(p)
        motions[seq] = read(seqroot/'algorithm_inputs'/seq/'motion_input.json')['frames']
    lookup = {(r['sequence_id'], int(r['current_frame']), r['method']): r for r in predictions}
    for r in predictions:
        seq, k, method = r['sequence_id'], int(r['current_frame']), r['method']
        assert int(r['previous_frame']) == k-1
        frames = truths[seq]['frames']
        previous = np.array(frames[k-2]['frame_to_source_3x3'])
        current = np.array(frames[k-1]['frame_to_source_3x3'])
        A = np.linalg.solve(current, previous)
        np.testing.assert_allclose(A, frames[k-1]['previous_to_current_point_3x3'], atol=1e-10)
        mapped = (A@grid)[:2]
        valid = np.all((mapped>=1)&(mapped<=256), axis=0)
        assert valid.sum() > 0
        available = yes(r['available'])
        avg = medcenter = maximum = overlap = None
        if available:
            shift = np.array([float(r['dx']), float(r['dy'])])
            assert np.all(np.isfinite(shift))
            estimate = grid[:2]+shift[:, None]
            errors = np.linalg.norm(estimate[:, valid]-mapped[:, valid], axis=0)
            avg, maximum = float(errors.mean()), float(errors.max())
            medcenter = float(np.linalg.norm(center[:2]+shift-(A@center)[:2]))
            overlap = float(np.all((estimate[:, valid]>=1)&(estimate[:, valid]<=256), axis=0).mean())
            # Independent affine residual construction, without mapped-point subtraction.
            residual = (np.eye(2)-A[:2, :2])@grid[:2, valid]+(shift-A[:2, 2])[:, None]
            np.testing.assert_allclose(np.linalg.norm(residual, axis=0), errors, atol=1e-10)
        metrics.append(dict(sequence_id=seq, source_id=source_index[seq]['source_id'], group_id=source_index[seq]['group_id'], profile=source_index[seq]['profile'], current_frame=k, method=method, available=available, mean_epe_px=avg, center_error_px=medcenter, max_epe_px=maximum, gt_valid_points=int(valid.sum()), estimated_overlap_fraction=overlap, failure_1px=(not available or avg>1), failure_2px=(not available or avg>2), failure_4px=(not available or avg>4), fallback=r['fallback'], used_prior=yes(r['used_prior'])))
        delta = float(r['observed_delta_odometry'])
        before, after = motions[seq][k-2:k]
        expected_delta = (after['simulated_odometry_px']-before['simulated_odometry_px']) if before['odometry_available'] and after['odometry_available'] else float('nan')
        assert (np.isnan(delta) and np.isnan(expected_delta)) or abs(delta-expected_delta)<1e-12
        if method=='odometry':
            assert available == (np.isfinite(delta) and abs(delta)<=20)
            if available:
                assert float(r['dx'])==0 and float(r['dy'])==-delta
        if method=='prior_ncc' and r['fallback']=='missing_prior_to_visual':
            v = lookup[seq, k, 'visual_ncc']
            assert r['dx']==v['dx'] and r['dy']==v['dy'] and r['available']==v['available']
    def aggregate(rows, label, dimension):
        good = [r for r in rows if r['available']]
        vals = np.array([r['mean_epe_px'] for r in good])
        sources = sorted({r['source_id'] for r in rows})
        return dict(dimension=dimension, subset=label, method=rows[0]['method'], pairs=len(rows), independent_sources=len(sources), available=len(good), coverage=len(good)/len(rows), mean_epe_px=float(vals.mean()) if len(vals) else None, median_epe_px=float(np.median(vals)) if len(vals) else None, p95_epe_px=float(np.percentile(vals, 95)) if len(vals) else None, mean_center_error_px=float(np.mean([r['center_error_px'] for r in good])) if good else None, failure_1px=sum(r['failure_1px'] for r in rows)/len(rows), failure_2px=sum(r['failure_2px'] for r in rows)/len(rows), failure_4px=sum(r['failure_4px'] for r in rows)/len(rows), source_macro_mean_epe_px=float(np.mean([np.mean([r['mean_epe_px'] for r in good if r['source_id']==s]) for s in sources if any(r['source_id']==s for r in good)])) if good else None, prior_used=sum(r['used_prior'] for r in rows), fallbacks=sum(r['fallback']!='none' for r in rows))
    aggregates=[]
    for method in ('odometry','visual_ncc','prior_ncc'):
        rows=[r for r in metrics if r['method']==method]
        aggregates.append(aggregate(rows,'all','overall'))
        for profile in sorted({r['profile'] for r in rows}):
            aggregates.append(aggregate([r for r in rows if r['profile']==profile],profile,'profile'))
        for source in sorted({r['source_id'] for r in rows}):
            aggregates.append(aggregate([r for r in rows if r['source_id']==source],source,'source'))
    # Preselected first validation/training source, all profiles, adjacent pairs 1 and 5.
    first_source = sorted({source_index[s]['source_id'] for s in ids})[0]
    audited = []
    for seq in ids:
        if source_index[seq]['source_id'] != first_source:
            continue
        for k in (2, 6):
            folder = seqroot/'algorithm_inputs'/seq
            C = independent_ncc(gray(folder/f'frame_{k-1:03}.png'), gray(folder/f'frame_{k:03}.png'))
            visual = peak(C, np.arange(-20,21), np.arange(-20,21), cfg)
            delta = float(lookup[seq,k,'odometry']['observed_delta_odometry'])
            prior = None
            if np.isfinite(delta) and abs(delta)<=20:
                prior = peak(C, np.arange(-4,5), np.arange(max(-20,int(np.ceil(-delta-4))),min(20,int(np.floor(-delta+4)))+1),cfg)
            if prior is None:
                prior=visual
            for method, result in (('visual_ncc',visual),('prior_ncc',prior)):
                r = lookup[seq,k,method]
                assert yes(r['available']) == (result is not None)
                if result:
                    np.testing.assert_allclose([float(r['dx']),float(r['dy']),float(r['score'])],result,atol=2e-7,rtol=0)
            audited.append(dict(sequence_id=seq,current_frame=k))
    assert len(audited)==24
    unavailable=sum(not r['available'] for r in metrics if r['method']=='odometry')
    if summary['mode']=='validation':
        assert unavailable==75
    assert sha(run/'predictions.csv')==prediction_hash
    write_csv(out/'pair_metrics.csv',metrics)
    write_csv(out/'aggregate_metrics.csv',aggregates)
    save(out/'audit.json',dict(status='PASS',prediction_sha256=prediction_hash,config_sha256=summary['config_sha256'],input_hashes_verified=len(access),geometry_files_read=len(truths),segmentation_masks_read=0,test_sequences_read=0,prediction_rows=len(metrics),all_geometry_recomputed=True,numpy_ncc_pairs=len(audited),ncc_audited_pairs=audited,odometry_unavailable_pairs=unavailable,geometry_sha256=metadata,scope='Same-image synthetic crops; pixels only; no fusion or detector evaluation. Valid-error means exclude unavailable; failure denominators include them.'))
    print(json.dumps({'status':'PASS','overall':[r for r in aggregates if r['dimension']=='overall'],'numpy_ncc_pairs':len(audited)},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--sequences',required=True)
    main(parser.parse_args())
