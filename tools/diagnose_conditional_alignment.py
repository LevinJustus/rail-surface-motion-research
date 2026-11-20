"""Evaluation-only comparison of estimated and exact geometric alignment."""

import argparse
from collections import Counter, defaultdict

import numpy as np
from scipy import ndimage

import run_conditional_v2_pilot as pilot


PILOT = pilot.ROOT / 'runs' / 'conditional_v2_pilot_nov20_01'
PILOT_INDEX_SHA = '55800e25cfb9caccd69c306fe781f5694bebc071ebf162a73bc8bd35bba9ef7f'
Y, X = np.mgrid[1:257, 1:257].astype(float)
METHODS = ('B_estimated', 'E_estimated', 'B_exact', 'E_exact')


def cached_frames():
    parent = {name: pilot.read(pilot.checked(pilot.PARENT / name, digest))
              for name, digest in pilot.PARENT_HASHES.items()}
    selection = parent['frozen_plan.json']['selection']
    assert len(selection) == 12 and all(r['assigned_split'] == 'training' for r in selection)
    inputs = {r['path']: r['sha256'] for r in parent['input_access.json']}
    motion = {(r['sequence_id'], r['frame']): r['base'] for r in parent['motion_diagnostics.json']}
    rows = pilot.read(pilot.checked(PILOT / 'predictions.json', PILOT_INDEX_SHA))
    assert [(r['sequence_id'], r['frame']) for r in rows] == [
        (s['sequence_id'], k) for s in selection for k in range(1, 10)]
    for row in rows:
        seq, k = row['sequence_id'], row['frame']
        if k == 1:
            images, raw, edges = [], [], []
        rel = '{}/algorithm_inputs/{}/frame_{:03d}.png'.format(pilot.DATA.relative_to(pilot.ROOT).as_posix(), seq, k)
        images = (images + [pilot.v3.gray(pilot.image_array(pilot.checked(pilot.ROOT / rel, inputs[rel])))])[-3:]
        p, b, e = [pilot.readp(pilot.checked(pilot.ROOT / r['path'], r['sha256'])).astype(np.float32)
                   for r in row['controls']]
        raw = (raw + [p])[-3:]
        if k > 1:
            edges = (edges + [motion[seq, k]])[-2:]
        yield row, images, raw, edges, b, e


def evaluation_files():
    path = pilot.DATA / 'evaluation_only' / 'payload_manifest.json'
    return {r['path']: r['sha256'] for r in pilot.read(pilot.checked(path, pilot.MASK_MANIFEST_SHA))}


def evaluation_path(files, seq, filename):
    rel = 'evaluation_only/sequences/{}/{}'.format(seq, filename)
    return pilot.checked(pilot.DATA / rel, files[rel])


def warp_exact(probability, transforms, start):
    valid = np.ones(X.shape, dtype=bool)
    qx, qy = X, Y
    for index in range(len(transforms) - 2, start - 1, -1):
        mapping = np.linalg.solve(transforms[index], transforms[-1])
        assert np.allclose(mapping[2], [0, 0, 1])
        qx = mapping[0, 0] * X + mapping[0, 1] * Y + mapping[0, 2]
        qy = mapping[1, 0] * X + mapping[1, 1] * Y + mapping[1, 2]
        valid &= (qx >= 1) & (qx <= 256) & (qy >= 1) & (qy <= 256)
    aligned = ndimage.map_coordinates(probability.astype(float), [qy - 1, qx - 1],
                                      order=1, mode='constant', cval=0., prefilter=False)
    return aligned, valid


def fuse_exact(raw, transforms):
    p = raw[-1].astype(float)
    conservative, strong = p.copy(), p.copy()
    cfg = pilot.v3.DEFAULTS
    for index in range(len(raw) - 1):
        aligned, valid = warp_exact(raw[index], transforms, index)
        difference = aligned - p
        strong += valid * difference / len(raw)
        conservative += valid * np.exp(-difference ** 2 / (2 * cfg['risk_sigma'] ** 2)) * difference / len(raw)
    protected = p >= cfg['strong_current']
    for value in (conservative, strong):
        value[protected] = np.maximum(value[protected], cfg['decision'])
    return tuple(np.clip(value, 0., 1.).astype(np.float32) for value in (conservative, strong))


def check_warp():
    image = np.arange(256 ** 2, dtype=np.float32).reshape(256, 256) / (256 ** 2 - 1)
    transforms = [np.eye(3), np.eye(3), np.eye(3)]
    transforms[1][:2, 2] = [5.5, -3.25]
    transforms[2][:2, 2] = [-2.25, 4.5]
    edges = [[-5.5, 3.25], [7.75, -7.75]]
    aligned, valid = warp_exact(image, transforms, 0)
    expected, expected_valid = pilot.v3.warp(image, edges)
    assert np.array_equal(valid, expected_valid) and np.array_equal(aligned[valid], expected[valid])
    for value, flag in zip(fuse_exact([image] * 3, transforms), (True, False)):
        assert np.array_equal(value, pilot.v3.fuse([image] * 3, edges, flag, pilot.v3.DEFAULTS))


def summarize(rows, keys, value_keys):
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].update({key: row[key] for key in value_keys})
    return [dict(zip(keys, key), **dict(values)) for key, values in sorted(grouped.items())]


def run(out):
    check_warp()
    files = evaluation_files()
    rows, funnel = [], []
    for receipt, images, raw, edges, b, e in cached_frames():
        seq, k = receipt['sequence_id'], receipt['frame']
        if k == 1:
            geometry = pilot.read(evaluation_path(files, seq, 'geometry_truth.json'))
            transforms = [np.asarray(frame['frame_to_source_3x3']) for frame in geometry['frames']]
            continue
        target = pilot.image_array(evaluation_path(files, seq, 'mask_{:03d}.png'.format(k))) > 0
        p = raw[-1]
        exact_b, exact_e = fuse_exact(raw, transforms[k-len(raw):k])
        info = {key: receipt[key] for key in ('source_id', 'sequence_id', 'frame', 'profile')}
        for name, value in zip(METHODS, (b, e, exact_b, exact_e)):
            rows.append(dict(info, method=name, **dict(zip(pilot.FIELDS, pilot.counts(value, target, p)))))
        _, masks = pilot.v1.fuse_conditional(images, raw, edges)
        with np.load(str(pilot.checked(PILOT / receipt['path'], receipt['sha256'])), allow_pickle=False) as saved:
            assert np.array_equal(masks['strong_correction'], saved['v1_trigger'])
            contrast = saved['v2_trigger']
        flips = (b < .5) & (e >= .5)
        persistent = flips & (p < .5) & masks['persistent_history']
        aligned = persistent & masks['correspondence']
        for kind, region in (('beneficial', target), ('harmful', ~target)):
            stages = (flips, persistent, aligned, masks['strong_correction'] & flips, contrast & flips)
            keys = ('strong_additions', 'eligible_persistent', 'passes_correspondence', 'passes_quality_v1', 'passes_contrast_v2')
            funnel.append(dict(info, kind=kind, **{key: int((stage & region).sum()) for key, stage in zip(keys, stages)}))
    assert len(rows) == 384 and len(funnel) == 192
    totals = summarize(rows, ('method',), pilot.FIELDS)
    filters = summarize(funnel, ('kind',), keys)
    for row in filters:
        row['v1_retained_fraction'] = row['passes_quality_v1'] / row['strong_additions']
        row['v2_retained_fraction'] = row['passes_contrast_v2'] / row['strong_additions']
    pilot.table(out / 'counts.csv', rows)
    pilot.table(out / 'by_source.csv', summarize(rows, ('source_id', 'method'), pilot.FIELDS))
    pilot.table(out / 'gate_funnel.csv', funnel)
    pilot.table(out / 'gate_funnel_by_source.csv', summarize(funnel, ('source_id', 'kind'), keys))
    result = dict(status='PASS_ALIGNMENT_DIAGNOSTIC', frames=96, totals=totals, filters=filters,
                  pilot_index_sha256=PILOT_INDEX_SHA, evaluation_manifest_sha256=pilot.MASK_MANIFEST_SHA,
                  source_sha256=pilot.sha(pilot.ROOT / 'tools' / 'diagnose_conditional_alignment.py'))
    pilot.write(out / 'summary.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('conditional_alignment_') or not args.run_id.replace('_', '').isalnum():
        parser.error('use a new conditional_alignment_* run ID')
    out = pilot.ROOT / 'runs' / args.run_id
    out.mkdir()
    run(out)
