"""Freeze Step10 learning-rate diagnostics and full-baseline metadata plans.

Standard library only. This tool never opens images/masks or starts training.
Existing files must match byte-for-byte; a changed plan needs a new version.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE1_CONFIG = 'configs/training_stage1_v1.json'
DIAGNOSTICS = [
    ('3e4', 3e-4, 'configs/training_stage2_lr3e4_v1.json'),
    ('1e4', 1e-4, 'configs/training_stage2_lr1e4_v1.json'),
]
SUPERSEDED_FULL_CONFIG = 'configs/training_stage2_full_v1.json'
FULL_CONFIG = 'configs/training_stage2_full_v2.json'
FULL_SCHEDULE = 'data/splits/full_stage2_schedule_v1.json'
REFERENCE_SUMMARY = 'docs/evidence/step09/summary.json'
ORDER_SEED = 20261114
INITIALIZATION_SEEDS = [20261114, 20261115, 20261116]
MAX_EPOCHS = 20


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def read_record(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    payload = path.read_bytes()
    value = json.loads(payload)
    try:
        label = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        label = str(path.resolve())
    return value, {'file': label, 'sha256': digest(payload)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def frozen_inputs():
    cfg, origin = read_record(STAGE1_CONFIG)
    evidence, _ = read_record('docs/evidence/stage1_plan_check.json')
    require(origin['sha256'] == evidence['training_config_sha256'], 'Frozen stage1 config hash mismatch')
    references = [
        ('model_config', 'model_config_sha256', 'model_config_sha256'),
        ('source_split', 'source_split_sha256', 'source_split_sha256'),
        ('subsets', 'subsets_sha256', 'subset_sha256'),
        ('batch_schedule', 'batch_schedule_sha256', 'schedule_sha256'),
    ]
    loaded = {}
    for path_key, config_key, evidence_key in references:
        value, record = read_record(cfg[path_key])
        require(record['sha256'] == cfg[config_key] == evidence[evidence_key],
                'Frozen input hash mismatch: ' + cfg[path_key])
        loaded[path_key] = value
    return cfg, origin, loaded


def diagnostic_payloads():
    stage1, origin, _ = frozen_inputs()
    outputs = {}
    for label, rate, path in DIAGNOSTICS:
        cfg = copy.deepcopy(stage1)
        cfg['plan_id'] = 'stage2_lr_diagnostic_' + label + '_v1'
        cfg['mode'] = 'lr_diagnostic'
        cfg['scope'] = ('Step10 finite learning-rate diagnostic only; same fixed128/64, '
                        '192 updates, initialization, sample order and dropout as stage1 pilot')
        cfg['status'] = 'planned_not_started'
        cfg['optimizer']['learning_rate'] = rate
        cfg['parent_stage1_config'] = origin['file']
        cfg['parent_stage1_config_sha256'] = origin['sha256']
        cfg['diagnostic_profile'] = 'pilot'
        cfg['learning_rate_only_change'] = True
        payload = canonical(cfg)
        outputs[path] = payload
        outputs[str(Path(path).with_suffix('.sha256'))] = (digest(payload) + '\n').encode()
    return outputs


def freeze(outputs, check=False):
    # Validate the entire output set before creating any missing file.
    for name, payload in outputs.items():
        path = ROOT / name
        if path.exists():
            require(path.read_bytes() == payload, 'Refusing to overwrite different frozen file: ' + name)
        elif check:
            raise ValueError('Missing frozen file: ' + name)
    if not check:
        for name, payload in outputs.items():
            path = ROOT / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)


def diagnostic_command(check):
    outputs = diagnostic_payloads()
    freeze(outputs, check)
    records = [{'file': path, 'sha256': digest(outputs[path]), 'learning_rate': rate}
               for _, rate, path in DIAGNOSTICS]
    print(json.dumps({'status': 'PASS', 'mode': 'lr_diagnostic',
                      'check_only': check, 'training_started': False,
                      'raw_images_opened': 0, 'configs': records}, indent=2))


def checked_candidate(summary_path, rate, expected_config_sha, stage1):
    summary, source = read_record(summary_path)
    require(summary.get('status') == 'PASS', 'Candidate run is not PASS: ' + source['file'])
    require(summary.get('fresh_initialization') is True, 'Candidate must use fresh initialization')
    required_counts = {'main_optimizer_updates': 192, 'unique_training_images': 128,
                       'training_proxy_groups': 45, 'unique_validation_images': 64,
                       'validation_proxy_groups': 15, 'heldout_test_images_opened': 0}
    for key, count in required_counts.items():
        require(summary.get(key) == count, 'Candidate count mismatch: ' + key)
    hashes = summary.get('hashes', {})
    require(hashes.get('training_config_sha256') == expected_config_sha,
            'Candidate training-config hash mismatch: ' + source['file'])
    expected_hashes = {'model_config_sha256': stage1['model_config_sha256'],
                       'source_split_sha256': stage1['source_split_sha256'],
                       'subset_sha256': stage1['subsets_sha256'],
                       'schedule_sha256': stage1['batch_schedule_sha256']}
    for key, expected in expected_hashes.items():
        require(hashes.get(key) == expected, 'Candidate frozen-input mismatch: ' + key)
    final = summary.get('final_validation', {})
    require(final.get('global_update') == 192 and final.get('sample_count') == 64 and
            final.get('proxy_group_count') == 15, 'Selection must use complete last192 validation')
    require(final.get('foreground_threshold') == 0.5, 'Candidate prediction threshold changed')
    score = final.get('micro_foreground_IoU')
    loss = final.get('total_loss')
    require(finite_number(score) and 0 <= score <= 1, 'Undefined/invalid last192 candidate IoU')
    require(finite_number(loss), 'Nonfinite last192 validation loss')
    return {'learning_rate': rate, 'last_update': 192, 'last_validation_micro_IoU': score,
            'last_validation_loss': loss, 'run_id': summary.get('run_id'),
            'training_config_sha256': expected_config_sha, 'summary': source}


def select_learning_rate(summary_paths, stage1, stage1_origin, diagnostic_outputs):
    expected = {digest(diagnostic_outputs[path]): (rate, path)
                for _, rate, path in DIAGNOSTICS}
    candidates = []
    observed = set()
    for path in summary_paths:
        summary, source = read_record(path)
        key = summary.get('hashes', {}).get('training_config_sha256')
        require(key in expected, 'Summary does not match either frozen diagnostic: ' + source['file'])
        require(key not in observed, 'The same diagnostic candidate was supplied twice')
        observed.add(key)
        rate, _ = expected[key]
        candidates.append(checked_candidate(path, rate, key, stage1))
    require(observed == set(expected), 'Both learning-rate diagnostic runs are required')
    candidates.append(checked_candidate(REFERENCE_SUMMARY, 1e-3, stage1_origin['sha256'], stage1))
    order = [3e-4, 1e-4, 1e-3]
    candidates.sort(key=lambda row: order.index(row['learning_rate']))
    selected = min(candidates, key=lambda row: (-row['last_validation_micro_IoU'],
                                              row['last_validation_loss'],
                                              order.index(row['learning_rate'])))
    return selected, candidates


def select_batch(benchmark_path, stage1):
    benchmark, source = read_record(benchmark_path)
    rows = benchmark.get('candidates')
    require(isinstance(rows, list), 'Benchmark must contain a candidates array')
    require(len(rows) == 3 and {r.get('batch_size') for r in rows} == {2, 4, 8},
            'Benchmark must contain exactly one record for each batch size 2,4,8')
    for key in ['model_config_sha256', 'source_split_sha256']:
        if key in benchmark:
            require(benchmark[key] == stage1[key], 'Benchmark context hash mismatch: ' + key)
    passed = []
    for row in rows:
        require(isinstance(row.get('status'), str), 'Benchmark record must include status')
        if row['status'] != 'PASS':
            continue
        require(finite_number(row.get('median_images_per_s')) and row['median_images_per_s'] > 0,
                'PASS benchmark must contain positive finite median_images_per_s')
        require(finite_number(row.get('median_update_time_s')) and row['median_update_time_s'] > 0,
                'PASS benchmark must contain positive finite median_update_time_s')
        passed.append(row)
    require(passed, 'No PASS computational benchmark candidate')
    selected = min(passed, key=lambda row: (-row['median_images_per_s'], row['batch_size']))
    return selected, rows, source


def full_records(parent):
    all_rows = parent['records']
    require(len({r['id'] for r in all_rows}) == len(all_rows), 'Parent IDs must be unique')
    train = [r for r in all_rows if r['assigned_split'] == 'training']
    validation = [r for r in all_rows if r['assigned_split'] == 'validation']
    require(len(train) == 1396 and len(validation) == 400, 'Full1396/400 count mismatch')
    train_groups = {r['group_id'] for r in train}; val_groups = {r['group_id'] for r in validation}
    require(len(train_groups) == 45 and len(val_groups) == 15, 'Full45/15 proxy-group count mismatch')
    require(not train_groups & val_groups, 'Training/validation proxy-group overlap')
    for key in ['id', 'image', 'mask']:
        require(not {r[key] for r in train} & {r[key] for r in validation}, 'Full split overlap: ' + key)
    return sorted(train, key=lambda r: r['id']), sorted(validation, key=lambda r: r['id'])


def make_full_schedule(train, validation, batch_size, split_sha):
    updates = []
    expected = {r['id'] for r in train}
    for epoch in range(1, MAX_EPOCHS + 1):
        ids = sorted(expected, key=lambda rid: (
            digest(f'{ORDER_SEED}|full_stage2_v1|{epoch}|{rid}'.encode('utf-8')), rid))
        visited = []
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            updates.append({'update': len(updates) + 1, 'epoch': epoch,
                            'sample_count': len(batch), 'sample_ids': batch})
            visited.extend(batch)
        require(collections.Counter(visited) == collections.Counter(expected), 'Epoch coverage mismatch')
    steps = math.ceil(len(train) / batch_size)
    require(len(updates) == steps * MAX_EPOCHS, 'Full update budget mismatch')
    return {'schema_version': 1, 'protocol': 'full_stage2_v1', 'order_seed': ORDER_SEED,
            'source_split_sha256': split_sha, 'ordering_key': 'SHA256(order_seed|full_stage2_v1|epoch|id)',
            'replacement': False, 'shared_across_initialization_seeds': True,
            'batch_size': batch_size, 'allow_smaller_last_batch': True,
            'max_epochs': MAX_EPOCHS, 'steps_per_epoch': steps,
            'maximum_optimizer_updates_per_seed': steps * MAX_EPOCHS,
            'training_images': len(train), 'validation_images': len(validation),
            'full': updates, 'validation_order': [r['id'] for r in validation]}


def full_command(summary_paths, benchmark_path, check):
    stage1, origin, loaded = frozen_inputs()
    _, superseded_config = read_record(SUPERSEDED_FULL_CONFIG)
    diagnostics = diagnostic_payloads()
    # Diagnostics have to have been frozen before their candidate runs.
    freeze(diagnostics, check=True)
    chosen_lr, candidates = select_learning_rate(summary_paths, stage1, origin, diagnostics)
    chosen_batch, bench_rows, bench_source = select_batch(benchmark_path, stage1)
    train, validation = full_records(loaded['source_split'])
    batch_size = chosen_batch['batch_size']
    schedule = make_full_schedule(train, validation, batch_size, stage1['source_split_sha256'])
    schedule_bytes = canonical(schedule)
    train_ids = {r['id'] for r in train}
    fixed_train = stage1['visualization']['fixed_train_ids']
    require(len(fixed_train) == 8 and len(set(fixed_train)) == 8 and set(fixed_train) <= train_ids,
            'Frozen training8 is not contained in full training')
    optimizer = copy.deepcopy(stage1['optimizer']); optimizer['learning_rate'] = chosen_lr['learning_rate']
    optimizer['learning_rate_schedule'] = 'piecewise_epoch_factors'
    cfg = {
        'schema_version': 1, 'plan_id': 'full_stage2_recipe_v2', 'mode': 'full_baseline',
        'status': 'planned_not_started',
        'scope': 'Step10 full1396/400 baseline; three fresh seeds; validation-only model selection; no heldout test',
        'model_config': stage1['model_config'], 'model_config_sha256': stage1['model_config_sha256'],
        'source_split': stage1['source_split'], 'source_split_sha256': stage1['source_split_sha256'],
        'optimizer': optimizer, 'loss': copy.deepcopy(stage1['loss']),
        'jpeg_label_threshold': stage1['jpeg_label_threshold'],
        'output_channel_order': stage1['output_channel_order'],
        'execution_environment': 'cpu', 'precision': 'single', 'augmentation': 'none',
        'normalization': stage1['normalization'], 'image_preprocessing': stage1['image_preprocessing'],
        'batch_size': batch_size, 'allow_smaller_last_batch': True,
        'training_images': 1396, 'training_proxy_groups': 45,
        'validation_images': 400, 'validation_proxy_groups': 15,
        'max_epochs': MAX_EPOCHS, 'initialization_seeds': INITIALIZATION_SEEDS,
        'fresh_initialization_per_seed': True, 'order_seed': ORDER_SEED,
        'validation_epochs': list(range(MAX_EPOCHS + 1)),
        'learning_rate_factors': [1.0] * 3 + [0.3] * 3 + [0.1] * 14,
        'learning_rate_rule': 'base*factor[epoch-1]; no automatic batch-size LR scaling',
        'early_stopping': {
            'enabled': True, 'min_epochs': 8, 'patience': 4, 'min_delta': 1e-4,
            'monitor': 'validation_micro_foreground_IoU', 'mode': 'max',
            'rule': ('Initialize significant-improvement anchor from epoch0; track patience from epoch1; '
                     'reset only if current IoU > anchor+min_delta; allow stopping only after '
                     'epoch>=min_epochs and patience consecutive non-improving epochs. '
                     'A valid NaN IoU does not improve. Numerical prediction/loss errors stop as failures.'),
            'separate_from_best_checkpoint': True,
        },
        'batch_schedule': FULL_SCHEDULE, 'batch_schedule_sha256': digest(schedule_bytes),
        'fixed_train_ids': fixed_train,
        'fixed_validation_ids': [r['id'] for r in validation[:8]],
        'fixed_validation_rule': 'first8 complete-validation IDs sorted lexicographically before full training',
        'foreground_threshold': 0.5, 'postprocessing': 'none',
        'prediction_reload_max_abs_error': 1e-6,
        'checkpoint_rule': 'initial, every completed epoch, actual last, validation best, consistent interrupted state',
        'best_rule': 'highest validation micro foreground IoU; tie lower valid loss; then earlier epoch; includes epoch0',
        'undefined_best': 'skip legal NaN IoU; if all undefined record best_unavailable and keep last',
        'primary_report': 'all three seeds, per-seed best and actual last plus complete curves; no best-seed cherry-picking',
        'repeat_rule': 'execute all three predeclared seeds sequentially; no performance-triggered seed replacement',
        'metadata': {
            'supersedes_unexecuted_config': superseded_config,
            'reason': 'pre-run metadata correction only; no training executed under v1',
            'parent_stage1_config': origin,
            'parent_stage1_subsets_sha256': stage1['subsets_sha256'],
            'parent_stage1_schedule_sha256': stage1['batch_schedule_sha256'],
            'diagnostic_candidates': candidates,
            'learning_rate_selection_rule': 'last192 validation micro IoU descending, valid loss ascending, fixed order [3e-4,1e-4,1e-3]',
            'selected_learning_rate': chosen_lr['learning_rate'],
            'batch_benchmark': bench_source, 'batch_benchmark_candidates': bench_rows,
            'batch_selection_rule': 'PASS candidate highest median_images_per_s; tie smaller batch; no model metric used',
            'selected_batch_size': batch_size,
            'maximum_training_images_per_seed': 1396 * MAX_EPOCHS,
            'maximum_optimizer_updates_per_seed': schedule['maximum_optimizer_updates_per_seed'],
            'estimated_max_training_seconds_per_seed': 1396 * MAX_EPOCHS / chosen_batch['median_images_per_s'],
            'estimate_scope': 'computational benchmark extrapolation for training only; validation/loading/checkpoints/plots extra; not a guarantee',
            'raw_images_opened_by_plan': 0, 'training_started_by_plan': False,
        },
    }
    config_bytes = canonical(cfg)
    outputs = {FULL_SCHEDULE: schedule_bytes,
               str(Path(FULL_SCHEDULE).with_suffix('.sha256')): (digest(schedule_bytes) + '\n').encode(),
               FULL_CONFIG: config_bytes,
               str(Path(FULL_CONFIG).with_suffix('.sha256')): (digest(config_bytes) + '\n').encode()}
    freeze(outputs, check)
    print(json.dumps({'status': 'PASS', 'check_only': check, 'training_started': False,
                      'raw_images_opened': 0, 'selected_learning_rate': chosen_lr['learning_rate'],
                      'selected_batch_size': batch_size, 'training_images': 1396,
                      'validation_images': 400, 'initialization_seeds': INITIALIZATION_SEEDS,
                      'maximum_epochs': MAX_EPOCHS,
                      'maximum_updates_per_seed': schedule['maximum_optimizer_updates_per_seed'],
                      'training_config': FULL_CONFIG, 'training_config_sha256': digest(config_bytes),
                      'schedule': FULL_SCHEDULE, 'schedule_sha256': digest(schedule_bytes)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    diagnostic = commands.add_parser('diagnostic', help='Freeze the two learning-rate-only diagnostic configs')
    diagnostic.add_argument('--check', action='store_true', help='Read-only byte-exact verification')
    full = commands.add_parser('full', help='Select from completed diagnostic summaries and freeze full1396/400 plan')
    full.add_argument('--summaries', nargs=2, required=True, metavar=('LR3E4_SUMMARY', 'LR1E4_SUMMARY'))
    full.add_argument('--benchmark', required=True, help='JSON with candidates for batch sizes2/4/8')
    full.add_argument('--check', action='store_true', help='Read-only byte-exact verification')
    args = parser.parse_args()
    if args.command == 'diagnostic':
        diagnostic_command(args.check)
    else:
        full_command(args.summaries, args.benchmark, args.check)


if __name__ == '__main__':
    main()
