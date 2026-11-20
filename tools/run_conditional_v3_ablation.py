"""Run the frozen conditional v3 comparison on the local development cache."""

import argparse

import numpy as np

import diagnose_conditional_alignment as alignment
from diagnose_conditional_alignment import pilot
from check_conditional_v3 import run as check_contracts
import rail_motion_conditional_v3 as candidate


METHODS = pilot.METHODS + ('conditional_v3',)
CONFIG = 'configs/algorithm_design_conditional_v3.json'
CONFIG_SHA = '6456decb3b7d915101507d0e42a784eed80036c47aa830bf7cea2f75a6e6da0d'
SOURCE_FILES = pilot.SOURCE_FILES + (CONFIG, 'python/rail_motion_conditional_v3.py',
    'tools/check_conditional_v3.py', 'tools/diagnose_conditional_alignment.py', 'tools/run_conditional_v3_ablation.py')


def produce(out):
    config = pilot.read(pilot.checked(pilot.ROOT / CONFIG, CONFIG_SHA))
    parent = pilot.read(pilot.checked(pilot.ROOT / 'configs' / config['parent_config'], config['parent_sha256']))
    params = pilot.read(pilot.checked(pilot.ROOT / 'configs' / parent['parent_config'], parent['parent_sha256']))['parameters']
    assert config['minimum_ring_deviations'] == parent['minimum_ring_deviations'] == pilot.v2.MINIMUM_RING_DEVIATIONS
    hashes = {name: pilot.sha(pilot.ROOT / name) for name in SOURCE_FILES}
    frozen = dict(config=config, config_sha256=CONFIG_SHA, source_sha256=hashes,
                  pilot_index_sha256=alignment.PILOT_INDEX_SHA, evaluation_manifest_sha256=pilot.MASK_MANIFEST_SHA)
    pilot.write(out / 'frozen_plan.json', frozen)
    pilot.write(out / 'synthetic_checks.json', check_contracts())
    (out / 'predictions').mkdir()
    rows = []
    for row, images, raw, edges, b, e in alignment.cached_frames():
        probability, masks = candidate.fuse_conditional_v3(images, raw, edges, params,
                          minimum_ring_deviations=config['minimum_ring_deviations'])
        trigger = masks['strong_correction']
        with np.load(str(pilot.checked(alignment.PILOT / row['path'], row['sha256'])), allow_pickle=False) as prior:
            assert np.array_equal(prior['v2_trigger'], trigger & masks['quality_drop'])
        assert np.array_equal(probability, np.where(trigger, e, b))
        path = out / 'predictions' / '{}_{:03d}.npz'.format(row['sequence_id'], row['frame'])
        np.savez_compressed(str(path), probability=probability, trigger=trigger)
        rows.append(dict(parent=row, path=path.relative_to(out).as_posix(), sha256=pilot.sha(path)))
    assert len(rows) == 108
    for name, digest in hashes.items():
        pilot.checked(pilot.ROOT / name, digest)
    pilot.write(out / 'predictions.json', rows)
    digest = pilot.sha(out / 'predictions.json')
    pilot.write(out / 'prediction_freeze.json', dict(predictions_sha256=digest, labels_read=0, geometry_read=0))
    return digest


def evaluate(out, digest):
    rows = pilot.read(pilot.checked(out / 'predictions.json', digest))
    for row in rows:
        pilot.checked(out / row['path'], row['sha256'])
    files = alignment.evaluation_files()
    fields = pilot.FIELDS[4:]
    counts, paired, triggers = [], [], []
    for row in rows:
        parent = row['parent']
        if parent['frame'] == 1:
            continue
        info = {key: parent[key] for key in ('source_id', 'sequence_id', 'frame', 'profile')}
        mask_name = 'mask_{:03d}.png'.format(info['frame'])
        target = pilot.image_array(alignment.evaluation_path(files, info['sequence_id'], mask_name)) > 0
        controls = [pilot.readp(pilot.checked(pilot.ROOT / r['path'], r['sha256'])) for r in parent['controls']]
        with np.load(str(pilot.checked(alignment.PILOT / parent['path'], parent['sha256'])), allow_pickle=False) as prior:
            controls.extend(prior[key] for key in ('conditional_v1', 'conditional_v2'))
        with np.load(str(out / row['path']), allow_pickle=False) as prediction:
            probability, trigger = prediction['probability'], prediction['trigger']
        for method, value in zip(METHODS, controls + [probability]):
            counts.append(dict(info, method=method, **dict(zip(pilot.FIELDS, pilot.counts(value, target, controls[0])))))
        for method, value in zip(METHODS, controls):
            paired.append(dict(info, baseline=method,
                               **dict(zip(fields, pilot.counts(probability, target, value)[4:]))))
        b, e = controls[1:3]
        possible = (b < .5) & (e >= .5)
        flips = possible & trigger
        triggers.append(dict(info, total=int(trigger.sum()), effective=int(flips.sum()),
                       possible_beneficial=int((possible & target).sum()), kept_beneficial=int((flips & target).sum()),
                       possible_harmful=int((possible & ~target).sum()), kept_harmful=int((flips & ~target).sum())))
    assert len(triggers) == 96
    totals = alignment.summarize(counts, ('method',), pilot.FIELDS)
    per_source = alignment.summarize(counts, ('source_id', 'method'), pilot.FIELDS)
    for row in totals + per_source:
        row.update(pilot.metrics([row[key] for key in pilot.FIELDS]))
    contrasts = alignment.summarize(paired, ('baseline',), fields)
    source_contrasts = alignment.summarize(paired, ('source_id', 'baseline'), fields)
    trigger_fields = ('total', 'effective', 'possible_beneficial', 'kept_beneficial', 'possible_harmful', 'kept_harmful')
    trigger_stats = alignment.summarize(triggers, (), trigger_fields)[0]
    b_to_candidate = next(r for r in contrasts if r['baseline'] == 'B_base_conservative')
    fewer_misses = b_to_candidate['FN_to_TP'] > b_to_candidate['TP_to_FN']
    no_more_false_positives = b_to_candidate['TN_to_FP'] <= b_to_candidate['FP_to_TN']
    for name, values in [('per_frame_counts', counts), ('method_summary', totals), ('per_source', per_source),
                         ('paired_changes', paired), ('paired_by_source', source_contrasts), ('triggers', triggers)]:
        pilot.table(out / (name + '.csv'), values)
    result = dict(status='PASS_EXECUTION', passed_development_screen=fewer_misses and no_more_false_positives,
                  prediction_sha256=digest, primary_frames=96, totals=totals, contrasts=contrasts,
                  trigger_counts=trigger_stats)
    pilot.write(out / 'summary.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('conditional_v3_ablation_') or not args.run_id.replace('_', '').isalnum():
        parser.error('use a new conditional_v3_ablation_* run ID')
    out = pilot.ROOT / 'runs' / args.run_id
    out.mkdir()
    evaluate(out, produce(out))
