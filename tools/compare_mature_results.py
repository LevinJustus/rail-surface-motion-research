"""Read-only, descriptive comparison of completed Stage10 and Stage11 runs.

Python 3.7+ standard library. Reads JSON/config metadata only, never images or MAT.
Writes exclusively to a new directory below runs/step11_preflight.
Exit codes: 0 complete comparison or explicit baseline-only; 2 pending; 1 error.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
SAFE_OUTPUT_ROOT = ROOT / 'runs/step11_preflight'
METRICS = ('micro_foreground_IoU', 'micro_foreground_Dice',
           'micro_precision', 'micro_recall')
ALLOWED_RECIPE_DIFFERENCES = {'plan_id', 'mode', 'scope', 'model_config',
                              'model_config_sha256', 'metadata'}


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def label(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read(path):
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('ascii')
    return hashlib.sha256(payload).hexdigest()


def integral(value, description):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value == int(value), description)
    return int(value)


def score(value, description):
    if value is None:
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1,
            'Invalid recorded metric: ' + description)
    return float(value)


def parameter_evidence(method, config):
    if method == 'small_unet':
        path = ROOT / 'docs/evidence/baseline_environment.json'
        evidence = read(path)
        model = read(resolve(config['model_config']))
        require(evidence['baseline_id'] == model['baseline_id'] == 'small_unet_v1',
                'Baseline parameter evidence belongs to another model')
        require(evidence['input_size_hwc'] == model['input_size_hwc'],
                'Baseline parameter evidence input mismatch')
        count = integral(evidence['parameters'], 'Invalid baseline parameter count')
    else:
        ref = config['metadata']['engineering_preflight']
        path = resolve(ref['file'])
        require(sha(path) == ref['sha256'], 'Mature parameter evidence hash mismatch')
        evidence = read(path)
        require(evidence['status'] == 'PASS' and
                evidence['parent_model_sha256'] == config['model_config_sha256'],
                'Mature parameter evidence belongs to another model')
        count = integral(evidence['architecture']['parameter_count'],
                         'Invalid mature parameter count')
        require(count == config['metadata']['parameter_count'],
                'Mature declared/observed parameter count mismatch')
    require(count > 0, 'Parameter count must be positive')
    return {'count': count, 'source': label(path), 'source_sha256': sha(path),
            'scope': 'Recorded MATLAB count, not independently decoded from MAT'}


def config_evidence(path):
    config = read(path)
    for name in ('model_config', 'source_split', 'batch_schedule'):
        require(sha(resolve(config[name])) == config[name + '_sha256'],
                'Referenced file hash mismatch: ' + config[name])
    seeds = [integral(x, 'Noninteger initialization seed')
             for x in config['initialization_seeds']]
    require(seeds == [20261114, 20261115, 20261116],
            'Unexpected predeclared seed list')
    return config, {'training_config_sha256': sha(path),
                    'model_config_sha256': config['model_config_sha256'],
                    'source_split_sha256': config['source_split_sha256'],
                    'batch_schedule_sha256': config['batch_schedule_sha256']}


def input_evidence(path, config, hashes):
    access = read(path)
    require(access['config'] == config, 'Run input configuration differs from frozen file')
    require(access['model_config'] == read(resolve(config['model_config'])),
            'Run input model configuration differs from frozen file')
    for key in ('training_config_sha256', 'model_config_sha256', 'source_split_sha256'):
        require(access['hashes'][key] == hashes[key], 'Run input hash mismatch: ' + key)
    require(access['unique_training_images_opened'] == 1396 and
            access['validation_images_opened'] == 400 and
            access['heldout_test_images_opened'] == 0, 'Unexpected data-access counts')
    signatures = {}
    for partition, field, size, groups in (
            ('training', 'train_access', 1396, 45),
            ('validation', 'validation_access', 400, 15)):
        records = access[field]
        ids = [x['id'] for x in records]
        require(len(records) == len(set(ids)) == size and ids == sorted(ids),
                'Unexpected ordered IDs in ' + partition)
        require(len(set(x['group_id'] for x in records)) == groups,
                'Unexpected proxy-group count in ' + partition)
        require(all(x['assigned_split'] == partition for x in records),
                'Revised partition mismatch')
        signatures[partition] = {
            'images': size, 'proxy_groups': groups,
            'valid_pixels': sum(x['valid_pixels'] for x in records),
            'target_foreground_pixels': sum(
                x['preprocessing']['foreground_pixels_after'] for x in records),
            'ordered_complete_access_records_sha256': canonical_sha(records)}
    for key in ('id', 'group_id', 'image', 'mask'):
        require(not (set(x[key] for x in access['train_access']) &
                     set(x[key] for x in access['validation_access'])),
                'Train/validation overlap: ' + key)
    return signatures


def load_method(name, run, config_path):
    config, hashes = config_evidence(config_path)
    result = {'method': name, 'run': label(run), 'config_file': label(config_path),
              'hashes': hashes, 'parameters': parameter_evidence(name, config),
              'expected_seeds': [int(x) for x in config['initialization_seeds']],
              'completed_seeds': [], 'pending_seeds': [], 'pending_evidence': [],
              'data_access': None, 'seed_summaries': {}, 'source_commit': None}
    input_path = run / 'input_access.json'
    if input_path.is_file():
        result['data_access'] = input_evidence(input_path, config, hashes)
        result['input_access_sha256'] = sha(input_path)
    else:
        result['pending_evidence'].append('input_access.json absent')
    group = None
    group_path = run / 'summary.json'
    if group_path.is_file():
        try:
            group = read(group_path)
        except json.JSONDecodeError:
            result['pending_evidence'].append('group summary is not yet complete JSON')
        if group is not None:
            require(group.get('status') == 'PASS', 'Group summary is not PASS: ' + name)
            require(group['heldout_test_images_opened'] == 0,
                    'Group reports heldout-image access')
            require([int(x) for x in group['initialization_seeds']] == result['expected_seeds'],
                    'Group seed order differs from configuration')
            require(len(group['seed_summaries']) == 3 and
                    sorted(int(x['initialization_seed']) for x in group['seed_summaries'])
                    == result['expected_seeds'], 'Group has missing or extra seed summaries')
            require(group['training_images'] == 1396 and group['validation_images'] == 400,
                    'Group population differs from the shared protocol')
            require(group['source']['git_dirty'] is False, 'Run did not start from clean source')
            if name == 'mature_mobilenet_deeplab':
                require(group['parameter_count'] == result['parameters']['count'] and
                        group['bn_count'] == config['metadata']['bn_layer_count'],
                        'Formal mature group model counts differ from tested architecture')
            for key in ('training_config_sha256', 'model_config_sha256', 'source_split_sha256'):
                require(group['hashes'][key] == hashes[key], 'Group hash mismatch: ' + key)
            result['source_commit'] = group['source']['git_commit']
            result['group_summary_sha256'] = sha(group_path)
    else:
        result['pending_evidence'].append('group summary.json absent; run not accepted complete')
    parent_model = read(resolve(config['model_config']))
    for seed in result['expected_seeds']:
        directory = run / 'seeds' / ('seed_' + str(seed))
        summary_path = directory / 'summary.json'
        if not summary_path.is_file():
            result['pending_seeds'].append(seed)
            continue
        try:
            summary = read(summary_path)
        except json.JSONDecodeError:
            result['pending_seeds'].append(seed)
            result['pending_evidence'].append('seed summary JSON incomplete: ' + str(seed))
            continue
        require(summary.get('status') == 'PASS', 'Seed summary is not PASS: ' + str(seed))
        require(integral(summary['initialization_seed'], 'Invalid summary seed') == seed,
                'Seed summary stored under another seed directory')
        require(summary['heldout_test_images_opened'] == 0, 'Seed reports heldout-image access')
        if name == 'mature_mobilenet_deeplab':
            require(summary['parameter_count'] == result['parameters']['count'] and
                    summary['bn_count'] == config['metadata']['bn_layer_count'],
                    'Formal mature seed model counts differ from tested architecture')
        for key, expected in hashes.items():
            require(summary['hashes'][key] == expected, 'Seed hash mismatch: ' + key)
        seed_path = directory / 'model_config_seed.json'
        require(sha(seed_path) == summary['hashes']['seed_model_config_sha256'],
                'Seed model configuration byte hash mismatch')
        expected_model = dict(parent_model)
        expected_model['random_seed'] = seed
        require(read(seed_path) == expected_model, 'Seed model changed more than random_seed')
        epochs = integral(summary['completed_epochs'], 'Invalid completed epoch count')
        updates = integral(summary['main_optimizer_updates'], 'Invalid update count')
        require(config['early_stopping']['min_epochs'] <= epochs <= config['max_epochs']
                and updates == epochs * 175,
                'Completed epoch/update count mismatch')
        selected_epoch = integral(summary['best_validation']['epoch'], 'Invalid selected epoch')
        require(0 <= selected_epoch <= epochs and
                summary['best_validation']['global_update'] == selected_epoch * 175,
                'Selected checkpoint is beyond actual last')
        if group is not None:
            candidates = [x for x in group['seed_summaries']
                          if int(x['initialization_seed']) == seed]
            require(candidates == [summary], 'Group/per-seed summary mismatch')
            require(summary['source'] == group['source'], 'Group/per-seed source mismatch')
        if result['data_access'] is not None:
            validation = result['data_access']['validation']
            last = summary['final_validation']
            require(last['sample_count'] == 400 and
                    last['valid_pixels'] == validation['valid_pixels'] and
                    last['TP'] + last['FN'] == validation['target_foreground_pixels'],
                    'Last metrics use another validation population')
        result['completed_seeds'].append(seed)
        result['seed_summaries'][seed] = {
            'summary_sha256': sha(summary_path), 'summary': summary}
    require(len(set(x['summary']['initial_parameter_sha256']
                    for x in result['seed_summaries'].values()))
            == len(result['completed_seeds']), 'Completed seeds reuse initial parameter fingerprints')
    result['complete'] = (group is not None and result['data_access'] is not None and
                          result['completed_seeds'] == result['expected_seeds'])
    return result, config


def per_seed_rows(method):
    rows = []
    for seed in method['completed_seeds']:
        summary = method['seed_summaries'][seed]['summary']
        for role, key in (('selected', 'best_validation'), ('last', 'final_validation')):
            selected = summary[key]
            epoch = (selected['epoch'] if role == 'selected'
                     else summary['completed_epochs'])
            update = (selected['global_update'] if role == 'selected'
                      else summary['main_optimizer_updates'])
            row = {'method': method['method'], 'seed': seed, 'checkpoint_role': role,
                   'parameters': method['parameters']['count'],
                   'completed_epochs': summary['completed_epochs'],
                   'completed_updates': summary['main_optimizer_updates'],
                   'checkpoint_epoch': epoch, 'checkpoint_update': update,
                   'is_untrained_reference': epoch == 0}
            for metric in METRICS:
                row[metric] = score(selected[metric], method['method'] + '/' + metric)
            rows.append(row)
    return rows


def describe(values):
    defined = [x for x in values if x is not None]
    return {'expected_count': 3, 'record_count': len(values), 'defined_count': len(defined),
            'values': values, 'mean': statistics.mean(defined) if defined else None,
            'sample_std': statistics.stdev(defined) if len(defined) >= 2 else None}


def statistics_rows(rows, group_key):
    output = []
    groups = sorted(set((x[group_key], x['checkpoint_role']) for x in rows))
    for name, role in groups:
        selected = [x for x in rows if x[group_key] == name and x['checkpoint_role'] == role]
        for metric in METRICS:
            record = {group_key: name, 'checkpoint_role': role, 'metric': metric}
            record.update(describe([x[metric] for x in selected]))
            output.append(record)
    return output


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', default='runs/full_stage2_v2_01')
    parser.add_argument('--mature-root', default='runs/mature_stage3_v1_01')
    parser.add_argument('--baseline-config', default='configs/training_stage2_full_v2.json')
    parser.add_argument('--mature-config', default='configs/training_stage3_mature_v1.json')
    parser.add_argument('--output-dir', default='runs/step11_preflight/comparison_final')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--partial', action='store_true', help='Allow explicitly incomplete output; exit2')
    flags.add_argument('--baseline-only', action='store_true', help='Do not read mature-run results')
    args = parser.parse_args()
    output = resolve(args.output_dir)
    try:
        output.relative_to(SAFE_OUTPUT_ROOT.resolve())
    except ValueError:
        raise AuditError('Output must remain below runs/step11_preflight')
    require(output != SAFE_OUTPUT_ROOT.resolve(), 'Use a separate output subdirectory')
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            'Refuse to overwrite existing comparison evidence; use a new output directory')
    baseline, baseline_config = load_method('small_unet', resolve(args.baseline_root),
                                             resolve(args.baseline_config))
    methods = [baseline]
    shared = {'status': 'NOT_COMPARED_BASELINE_ONLY'}
    if not args.baseline_only:
        mature, mature_config = load_method('mature_mobilenet_deeplab',
                                            resolve(args.mature_root),
                                            resolve(args.mature_config))
        methods.append(mature)
        strip = lambda d: {k: v for k, v in d.items() if k not in ALLOWED_RECIPE_DIFFERENCES}
        require(strip(baseline_config) == strip(mature_config),
                'Shared recipe differs outside the explicitly allowed model/identity fields')
        shared = {'recipe_fields_equal_except': sorted(ALLOWED_RECIPE_DIFFERENCES),
                  'shared_recipe_verified': True, 'data_access_records_verified': False,
                  'dataset_and_schedule_sha256': {
                      'source_split': baseline['hashes']['source_split_sha256'],
                      'schedule': baseline['hashes']['batch_schedule_sha256']}}
        if baseline['data_access'] is not None and mature['data_access'] is not None:
            require(baseline['data_access'] == mature['data_access'],
                    'Ordered data access/path/raw-file-hash/preprocessing records differ')
            shared['data_access_records_verified'] = True
    complete = all(x['complete'] for x in methods)
    if not complete and not args.partial:
        print('PENDING: both complete group summaries and all three seeds are required; '
              'use --partial for a clearly incomplete snapshot.', file=sys.stderr)
        return 2
    status = 'BASELINE_ONLY' if args.baseline_only else ('PASS' if complete else 'INCOMPLETE')
    rows = [row for method in methods for row in per_seed_rows(method)]
    stats = statistics_rows(rows, 'method')
    pairs = []
    indexed = {(x['method'], x['seed'], x['checkpoint_role']): x for x in rows}
    if not args.baseline_only:
        for seed in baseline['expected_seeds']:
            for role in ('selected', 'last'):
                left = indexed.get(('small_unet', seed, role))
                right = indexed.get(('mature_mobilenet_deeplab', seed, role))
                if left is None or right is None:
                    continue
                pair = {'comparison': 'mature_minus_small_unet', 'seed': seed,
                        'checkpoint_role': role}
                for metric in METRICS:
                    pair[metric] = (right[metric] - left[metric]
                                    if left[metric] is not None and right[metric] is not None else None)
                pairs.append(pair)
    paired_stats = statistics_rows(pairs, 'comparison')
    metadata = []
    for method in methods:
        item = {k: v for k, v in method.items() if k != 'seed_summaries'}
        item['per_seed_summary_sha256'] = {
            str(k): v['summary_sha256'] for k, v in method['seed_summaries'].items()}
        metadata.append(item)
    report = {'schema_version': 1, 'status': status,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'shared_conditions': shared, 'methods': metadata, 'per_seed': rows,
              'statistics': stats, 'paired_seed_deltas': pairs,
              'paired_delta_statistics': paired_stats,
              'delta_definition': 'Mature minus small U-Net; absolute metric units, not percent or p-values',
              'selected_definition': 'Each run own validation-selected best, includes epoch0; actual last separate',
              'statistics_definition': 'Arithmetic mean and sample SD of defined recorded values; null remains undefined',
              'limitations': [
                  'Same seeds are run labels, not identical weights or identical stochastic trajectories across architectures.',
                  'The shared recipe was previously selected using the U-Net pilot; no equal per-model tuning claim.',
                  'Three repeats on the same400 validation images are descriptive; no significance or confidence-interval claim.',
                  'Validation-selected scores are not heldout test or real-world generalization estimates.',
                  'Different early-stop epochs are allowed under the same rule; equal cap is not equal compute.',
                  'Parameter counts are recorded MATLAB evidence; this helper does not decode MAT or rerun networks.',
                  'This metadata comparison supplements the per-step final auditor; it does not independently reconstruct training.',
                  'No inference timing, realtime, K230, real-video or multi-frame-fusion conclusion.',
                  'INCOMPLETE or BASELINE_ONLY outputs do not certify the mature-model comparison complete.'],
              'raw_images_or_test_data_opened_by_helper': 0,
              'formal_run_files_modified': 0}
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / 'comparison.json'
    with json_path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    write_csv(output / 'per_seed.csv', rows,
              ['method', 'seed', 'checkpoint_role', 'parameters', 'completed_epochs',
               'completed_updates', 'checkpoint_epoch', 'checkpoint_update',
               'is_untrained_reference'] + list(METRICS))
    write_csv(output / 'statistics.csv', stats,
              ['method', 'checkpoint_role', 'metric', 'expected_count', 'record_count',
               'defined_count', 'values', 'mean', 'sample_std'])
    write_csv(output / 'paired_seed_deltas.csv', pairs,
              ['comparison', 'seed', 'checkpoint_role'] + list(METRICS))
    write_csv(output / 'paired_delta_statistics.csv', paired_stats,
              ['comparison', 'checkpoint_role', 'metric', 'expected_count', 'record_count',
               'defined_count', 'values', 'mean', 'sample_std'])
    print(json.dumps({'status': status, 'output': label(output),
                      'completed_seeds': {x['method']: x['completed_seeds'] for x in methods},
                      'paired_seed_role_rows': len(pairs)}, ensure_ascii=True))
    return 0 if status in ('PASS', 'BASELINE_ONLY') else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (AuditError, KeyError, ValueError, OSError) as error:
        print('COMPARISON_ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
