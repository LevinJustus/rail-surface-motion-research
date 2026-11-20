"""Compare fixed fusion gates on the existing v3 training-source pilot."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from run_v3_feedback_batch import ROOT, counts, metrics, read, readp, sha, table, write, FIELDS
import numpy as np
from PIL import Image

from check_conditional_v2 import run as check_contracts
import rail_motion_v3 as v3
import rail_motion_conditional as v1
import rail_motion_conditional_v2 as v2


PARENT = ROOT / 'runs' / 'v3_feedback_batch_v1_01'
DATA = ROOT / 'runs' / 'step12_full_v2_01'
METHODS = ('S', 'B_base_conservative', 'E_base_strong', 'conditional_v1', 'conditional_v2')
SOURCE_FILES = (
    'python/rail_motion_v3.py', 'python/rail_motion_conditional.py', 'python/rail_motion_conditional_v2.py',
    'configs/algorithm_design_v3.json', 'configs/algorithm_design_conditional_v1.json',
    'configs/algorithm_design_conditional_v2.json', 'tools/check_conditional_v2.py',
    'tools/run_conditional_v2_pilot.py', 'tools/run_v3_feedback_batch.py',
)
PARENT_HASHES = {
    'frozen_plan.json': '2168b98542b50029a94b9bbee781fc5ed7ca50d9dacb2603d93ee6cc53ae8c59',
    'predictions.json': 'f2640e9ef5aad19b63360ab1e4e94644a6827483680d9e13fdc8b81c7a4ba926',
    'input_access.json': '38a5218adf879b9e0ec83f00b716226d80c2120b1e6fa00b6c0dfad97707056c',
    'motion_diagnostics.json': '2c1dbe059c34cb2825b0adda2c6d601c5fa7dd150f6bc5daf81f7ddf20aaaa11',
}
MASK_MANIFEST_SHA = 'edcb965d2e3d04095b68439f4c790f062a21b0448f891fd66d9271c038b85d9d'


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError('SHA-256 mismatch: {}'.format(path))
    return path


def image_array(path):
    with Image.open(str(path)) as image:
        return np.asarray(image)


def produce(out):
    parent = {name: read(checked(PARENT / name, digest)) for name, digest in PARENT_HASHES.items()}
    plan = parent['frozen_plan.json']
    selection = plan['selection']
    assert len(selection) == 12 and all(s['assigned_split'] == 'training' and s['frame_count'] == 9 for s in selection)
    inputs = {r['path']: r['sha256'] for r in parent['input_access.json']}
    motion = {(r['sequence_id'], r['frame']): r['base'] for r in parent['motion_diagnostics.json']}
    cached = {(r['sequence_id'], r['frame'], r['method']): r for r in parent['predictions.json']}
    v2_config = read(ROOT / 'configs' / 'algorithm_design_conditional_v2.json')
    params = read(checked(ROOT / 'configs' / v2_config['parent_config'], v2_config['parent_sha256']))['parameters']
    minimum = v2_config['minimum_ring_deviations']
    assert params == v1.DEFAULTS and minimum == v2.MINIMUM_RING_DEVIATIONS
    assert read(ROOT / 'configs' / 'algorithm_design_v3.json')['parameters'] == v3.DEFAULTS
    hashes = {name: sha(ROOT / name) for name in SOURCE_FILES}
    for name, digest in hashes.items():
        destination = out / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked(ROOT / name, digest), destination)
        checked(destination, digest)
    write(out / 'frozen_plan.json', dict(source_sha256=hashes, parent_sha256=PARENT_HASHES,
          label_manifest_sha256=MASK_MANIFEST_SHA, config=v2_config, seed=plan['seed'],
          checkpoint=plan['checkpoint'], selection=selection, threshold=.5, warmup_excluded=1))
    write(out / 'synthetic_checks.json', check_contracts())
    (out / 'predictions').mkdir()
    receipts = []
    for selected in selection:
        seq = selected['sequence_id']
        images, raw, edges = [], [], []
        for frame in range(1, 10):
            rel = '{}/algorithm_inputs/{}/frame_{:03d}.png'.format(DATA.relative_to(ROOT).as_posix(), seq, frame)
            images = (images + [v3.gray(image_array(checked(ROOT / rel, inputs[rel])))])[-3:]
            parents = [cached[seq, frame, method] for method in METHODS[:3]]
            p, b, e = [readp(checked(ROOT / record['path'], record['sha256'])).astype(np.float32) for record in parents]
            raw = (raw + [p])[-3:]
            if frame > 1:
                edges = (edges + [motion[seq, frame]])[-2:]
            first, first_masks = v1.fuse_conditional(images, raw, edges, params)
            second, second_masks = v2.fuse_conditional_v2(images, raw, edges, params, minimum_ring_deviations=minimum)
            trigger1, trigger2 = first_masks['strong_correction'], second_masks['strong_correction']
            assert np.array_equal(first, np.where(trigger1, e, b))
            assert np.array_equal(second, np.where(trigger2, e, b))
            assert np.all(~trigger2 | trigger1)
            path = out / 'predictions' / '{}_{:03d}.npz'.format(seq, frame)
            np.savez_compressed(str(path), conditional_v1=first, conditional_v2=second,
                                v1_trigger=trigger1, v2_trigger=trigger2)
            receipts.append(dict(sequence_id=seq, frame=frame, source_id=selected['source_id'],
                                 profile=selected['profile'], path=path.relative_to(out).as_posix(),
                                 sha256=sha(path), controls=parents))
    assert len(receipts) == 108
    for name, digest in hashes.items():
        checked(ROOT / name, digest)
    write(out / 'predictions.json', receipts)
    digest = sha(out / 'predictions.json')
    write(out / 'prediction_freeze.json', dict(predictions_sha256=digest, frames=len(receipts), labels_read=0))
    return digest


def evaluate(out, digest):
    receipts = read(checked(out / 'predictions.json', digest))
    for row in receipts:
        checked(out / row['path'], row['sha256'])
    payload_path = checked(DATA / 'evaluation_only' / 'payload_manifest.json', MASK_MANIFEST_SHA)
    payload = {r['path']: r['sha256'] for r in read(payload_path)}
    frame_rows, paired_rows, triggers = [], [], []
    totals = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    fields = FIELDS[4:]
    for row in receipts:
        info = {k: row[k] for k in ('sequence_id', 'source_id', 'frame', 'profile')}
        label = 'evaluation_only/sequences/{}/mask_{:03d}.png'.format(row['sequence_id'], row['frame'])
        target = image_array(checked(DATA / label, payload[label])) > 0
        control = [readp(checked(ROOT / r['path'], r['sha256'])) for r in row['controls']]
        with np.load(str(out / row['path']), allow_pickle=False) as prediction:
            values = dict(zip(METHODS, control + [prediction[m] for m in METHODS[3:]]))
            t1, t2 = prediction['v1_trigger'], prediction['v2_trigger']
        for method, value in values.items():
            result = counts(value, target, control[0])
            frame_rows.append(dict(info, method=method, **dict(zip(FIELDS, result))))
            if row['frame'] > 1:
                totals[row['source_id'], method] += result
        if row['frame'] == 1:
            continue
        for baseline in METHODS[:-1]:
            flips = counts(values['conditional_v2'], target, values[baseline])[4:]
            paired_rows.append(dict(info, baseline=baseline, **dict(zip(fields, flips))))
        rejected = t1 & ~t2
        triggers.append(dict(info, v1=int(t1.sum()), v2=int(t2.sum()),
                             rejected_foreground=int((rejected & target).sum()),
                             rejected_background=int((rejected & ~target).sum())))

    def metric_row(method, count, **info):
        count = [int(n) for n in count]
        return dict(info, method=method, **dict(zip(FIELDS, count)), **metrics(count))

    sources = sorted({r['source_id'] for r in receipts})
    source_metrics = [metric_row(method, totals[source, method], source_id=source)
                      for source in sources for method in METHODS]
    aggregate = [metric_row(method, sum(totals[source, method] for source in sources),
                            source_macro_IoU=float(np.mean([r['IoU'] for r in source_metrics if r['method'] == method])))
                 for method in METHODS]
    contrasts = [dict(baseline=baseline, **{k: sum(r[k] for r in paired_rows if r['baseline'] == baseline)
                                          for k in fields}) for baseline in METHODS[:-1]]
    for name, rows in [('per_frame_counts', frame_rows), ('paired_changes', paired_rows),
                       ('triggers', triggers), ('per_source_metrics', source_metrics), ('method_summary', aggregate)]:
        table(out / (name + '.csv'), rows)
    summary = dict(status='PASS_PILOT', sequences=12, frames=len(receipts), primary_frames=len(triggers),
                   prediction_sha256=digest, aggregate=aggregate, paired_v2=contrasts,
                   trigger_counts={key: sum(r[key] for r in triggers) for key in
                                   ('v1', 'v2', 'rejected_foreground', 'rejected_background')})
    write(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('conditional_v2_pilot_') or not args.run_id.replace('_', '').isalnum():
        parser.error('use a new conditional_v2_pilot_* run ID with letters, digits and underscores')
    out = ROOT / 'runs' / args.run_id
    out.mkdir()
    evaluate(out, produce(out))
