"""Read-only Step10 full-baseline audit (Python 3.7+, standard library only).

Exit codes: 0 final PASS, 1 final FAIL, 2 INCOMPLETE. --partial always reports
INCOMPLETE and checks only the common, fully logged epoch prefix. JSON files
that are missing, changing, or temporarily unparsable are pending evidence,
not evidence that the experiment failed. No raw image/mask or MAT is decoded.
Use --relative-paths for a portable report without physical workspace paths.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from datetime import datetime, timezone


CONFIG = 'configs/training_stage2_full_v2.json'
METRICS = ['micro_foreground_IoU', 'micro_foreground_Dice',
           'micro_precision', 'micro_recall']
COUNTS = ['TP', 'FP', 'FN', 'TN']
LOSSES = ['total_loss', 'BCE_component', 'Dice_component']
SHA_RE = re.compile(r'^[0-9a-f]{64}$')


class Mismatch(Exception):
    pass


class Unavailable(Exception):
    pass


def need(condition, message):
    if not condition:
        raise Mismatch(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return finite(value) and value == int(value)


def same_number(actual, expected, label, tolerance=1e-10):
    if expected is None:
        need(actual is None, label + ': expected undefined/null')
    else:
        need(finite(actual) and finite(expected) and
             math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
             '{}: {} != {}'.format(label, actual, expected))


def records(value, label):
    if isinstance(value, dict):
        return [value]
    need(isinstance(value, list) and all(isinstance(r, dict) for r in value),
         label + ': expected MATLAB struct object/array')
    return value


def vector(value):
    return value if isinstance(value, list) else [value]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def defined_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise Unavailable(str(path) + ': changed during SHA256 read')
    return digest.hexdigest()


class Audit:
    def __init__(self, root, run_id, partial):
        self.root = root
        self.run = root / 'runs' / run_id
        self.partial = partial
        self.errors = []
        self.pending = []
        self.files = {}
        self.seed_reports = []

    def label(self, path):
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def read(self, path, required=True):
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise Unavailable(self.label(path) + ': changed during read')
            value = json.loads(payload.decode('utf-8-sig'),
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            self.files[self.label(path)] = sha_bytes(payload)
            return value
        except (OSError, UnicodeError, ValueError, Unavailable) as error:
            if required:
                self.pending.append('{}: temporarily unavailable ({})'.format(self.label(path), error))
            return None

    def read_jsonl(self, path):
        try:
            payload = path.read_bytes()
        except OSError as error:
            self.pending.append('{}: unavailable ({})'.format(self.label(path), error))
            return []
        self.files[self.label(path)] = sha_bytes(payload)
        result = []
        for number, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode('utf-8'),
                                   parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                need(isinstance(value, dict), 'JSONL row is not an object')
                result.append(value)
            except (UnicodeError, ValueError, Mismatch):
                self.pending.append('{}: row {} not yet parseable; use preceding complete rows'.format(
                    self.label(path), number))
                break
        return result

    def capture(self, scope, function):
        try:
            return function()
        except Unavailable as error:
            self.pending.append(scope + ': ' + str(error))
        except (Mismatch, KeyError, TypeError, IndexError, ZeroDivisionError, OSError) as error:
            self.errors.append(scope + ': ' + str(error))
        return None

    def hash_reference(self, name, expected, sidecar=False):
        path = self.root / name
        try:
            actual = sha_file(path)
            self.files[self.label(path)] = actual
            need(actual == expected, name + ': SHA256 mismatch')
            if sidecar:
                token = path.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]
                need(token == actual, name + ': .sha256 mismatch')
        except OSError as error:
            raise Unavailable(name + ': ' + str(error))


def count_metrics(row, label):
    for key in COUNTS:
        need(integer(row[key]) and row[key] >= 0, label + ': invalid ' + key)
    tp, fp, fn, tn = [row[key] for key in COUNTS]
    valid = tp + fp + fn + tn
    need(valid > 0 and row['valid_pixels'] == valid, label + ': invalid valid-pixel total')
    need(row['target_foreground_pixels'] == tp + fn and
         row['predicted_foreground_pixels'] == tp + fp, label + ': foreground counts')
    expected = {'foreground_IoU': ratio(tp, tp + fp + fn),
                'foreground_Dice': ratio(2 * tp, 2 * tp + fp + fn),
                'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
                'target_foreground_fraction': (tp + fn) / valid,
                'predicted_foreground_fraction': (tp + fp) / valid}
    for key, value in expected.items():
        same_number(row[key], value, label + '/' + key)
    need(row['all_background_prediction'] is (tp + fp == 0), label + ': all-background flag')
    need(finite(row['mean_defect_probability']) and -1e-6 <= row['mean_defect_probability'] <= 1 + 1e-6,
         label + ': mean probability is not finite in [0,1]')
    return expected


def check_evaluation(value, expected_ids, access, label):
    images = records(value['per_image'], label + '/per_image')
    n = len(expected_ids)
    need(value['sample_count'] == n and [r['id'] for r in images] == expected_ids,
         label + ': wrong, missing, duplicated, or reordered validation IDs')
    need(value['output_channel_order'] == ['background', 'defect'] and
         value['foreground_threshold'] == 0.5 and value['threshold_comparison'] == '>=',
         label + ': changed inference contract')
    group_order = []
    grouped = collections.OrderedDict()
    for image in images:
        rid = image['id']; source = access[rid]
        need(image['group_id'] == source['group_id'] and image['image_count'] == 1,
             label + '/' + rid + ': group/image count')
        need(image['valid_pixels'] == source['valid_pixels'] and
             image['target_foreground_pixels'] == source['preprocessing']['foreground_pixels_after'],
             label + '/' + rid + ': target/valid counts differ from frozen preprocessing')
        count_metrics(image, label + '/' + rid)
        if image['group_id'] not in grouped:
            grouped[image['group_id']] = []
            group_order.append(image['group_id'])
        grouped[image['group_id']].append(image)
    groups = records(value['per_group'], label + '/per_group')
    need([g['id'] for g in groups] == group_order and len(groups) == value['proxy_group_count'],
         label + ': proxy-group order/count')
    group_ious = []
    for group in groups:
        gid = group['id']; rows = grouped[gid]
        need(group['group_id'] == gid and group['image_count'] == len(rows), label + ': group identity')
        for key in COUNTS + ['valid_pixels', 'target_foreground_pixels', 'predicted_foreground_pixels']:
            need(group[key] == sum(r[key] for r in rows), label + ': pooled group ' + key)
        derived = count_metrics(group, label + '/group/' + gid)
        group_ious.append(derived['foreground_IoU'])
        expected_mean = sum(r['mean_defect_probability'] * r['valid_pixels'] for r in rows) / group['valid_pixels']
        same_number(group['mean_defect_probability'], expected_mean, label + ': pooled mean probability')
    totals = {key: sum(r[key] for r in images) for key in COUNTS}
    for key, count in totals.items():
        need(value[key] == count, label + ': micro confusion ' + key)
    tp, fp, fn, tn = [totals[key] for key in COUNTS]
    valid = tp + fp + fn + tn
    need(value['valid_pixels'] == valid, label + ': total valid pixels')
    derived = {'micro_foreground_IoU': ratio(tp, tp + fp + fn),
               'micro_foreground_Dice': ratio(2 * tp, 2 * tp + fp + fn),
               'micro_precision': ratio(tp, tp + fp), 'micro_recall': ratio(tp, tp + fn),
               'macro_image_foreground_IoU': defined_mean([r['foreground_IoU'] for r in images]),
               'macro_group_foreground_IoU': defined_mean(group_ious),
               'target_foreground_fraction': (tp + fn) / valid,
               'predicted_foreground_fraction': (tp + fp) / valid}
    for key, expected in derived.items():
        same_number(value[key], expected, label + '/' + key)
    for key, denominator in {'IoU': tp + fp + fn, 'Dice': 2 * tp + fp + fn,
                             'precision': tp + fp, 'recall': tp + fn}.items():
        need(value['micro_metric_denominators'][key] == denominator, label + ': denominator ' + key)
    need(value['all_background_prediction'] is (tp + fp == 0) and
         value['all_background_image_count'] == sum(r['all_background_prediction'] for r in images) and
         value['zero_target_image_count'] == sum(r['target_foreground_pixels'] == 0 for r in images),
         label + ': all-background / empty-target counts')
    for key, rows in [('undefined_image_metrics', images), ('undefined_group_metrics', groups)]:
        for metric in ['foreground_IoU', 'foreground_Dice', 'precision', 'recall']:
            need(value[key][metric] == sum(r[metric] is None for r in rows), label + ': undefined count')
    batch_rows = records(value['batch_diagnostics'], label + '/batch_diagnostics')
    first = 1; loss_sums = dict.fromkeys(LOSSES, 0.0)
    batch_size = value['inference_batch_size']
    need(batch_size == 2 and value['evaluation_batch_count'] == int(math.ceil(n / batch_size)),
         label + ': validation batch contract')
    for row in batch_rows:
        last = min(first + batch_size - 1, n)
        need(row['first_index'] == first and row['last_index'] == last and
             row['sample_count'] == last - first + 1, label + ': batch coverage')
        for key in LOSSES:
            need(finite(row[key]) and row[key] >= 0, label + ': nonfinite/negative batch loss')
            loss_sums[key] += row['sample_count'] * row[key]
        need(finite(row['maximum_channel_sum_error']) and 0 <= row['maximum_channel_sum_error'] <= 1e-4,
             label + ': batch channel probability sum')
        first = last + 1
    need(first == n + 1 and len(batch_rows) == value['evaluation_batch_count'], label + ': incomplete batch diagnostics')
    for key in LOSSES:
        same_number(value[key], loss_sums[key] / n, label + '/weighted_' + key, tolerance=1e-8)
    checks = value['probability_checks']
    need(checks['all_finite'] is True and finite(checks['minimum']) and finite(checks['maximum']) and
         -1e-6 <= checks['minimum'] <= checks['maximum'] <= 1 + 1e-6, label + ': invalid probability range')
    for key, tolerance in [('maximum_channel_sum_error', 1e-4), ('loss_inference_maximum_difference', 1e-5)]:
        need(finite(checks[key]) and 0 <= checks[key] <= tolerance, label + ': inference/loss probability mismatch')
    return derived


def better(candidate, best):
    score = candidate['micro_foreground_IoU']
    need(finite(candidate['total_loss']), 'best candidate has nonfinite loss')
    if score is None:
        return False
    need(finite(score) and 0 <= score <= 1, 'best candidate has invalid IoU')
    if best is None:
        return True
    return (-score, candidate['total_loss'], candidate['global_update']) < (
        -best['micro_foreground_IoU'], best['total_loss'], best['global_update'])


def compare_candidate(actual, expected, label):
    for key in ['epoch', 'global_update', 'total_loss'] + METRICS:
        same_number(actual[key], expected[key], label + '/' + key)


def contiguous_epochs(rows, steps, label):
    for index, row in enumerate(rows):
        need(row['epoch'] == index and row['global_update'] == index * steps,
             label + ': missing/reordered epoch or update')
    return max(0, len(rows) - 1)


def check_early_stopping(evaluations, epoch_rows, history, cfg, steps):
    anchor = evaluations[0]['micro_foreground_IoU']
    anchor = anchor if anchor is not None else -math.inf
    wait = 0; first_stop = None
    rule = cfg['early_stopping']
    for epoch, row in enumerate(epoch_rows, 1):
        evaluation = evaluations[epoch]
        score = evaluation['micro_foreground_IoU']
        if score is not None and score > anchor + rule['min_delta']:
            anchor = score; wait = 0
        else:
            wait += 1
        need(row['epoch'] == epoch and row['global_update'] == epoch * steps,
             'epoch_history order/update mismatch')
        same_number(row['learning_rate'], cfg['optimizer']['learning_rate'] * cfg['learning_rate_factors'][epoch - 1],
                    'epoch_history learning_rate')
        same_number(row['validation_IoU'], score, 'epoch_history validation IoU')
        same_number(row['validation_loss'], evaluation['total_loss'], 'epoch_history validation loss')
        need(row['no_meaningful_improvement_epochs'] == wait, 'early-stopping patience counter mismatch')
        if math.isfinite(anchor):
            same_number(row['meaningful_best_IoU'], anchor, 'early-stopping anchor')
        else:
            need(row['meaningful_best_IoU'] is None, 'undefined early-stopping anchor must be null')
        batches = history[(epoch - 1) * steps:epoch * steps]
        mean = sum(r['total_loss'] * r['sample_count'] for r in batches) / sum(r['sample_count'] for r in batches)
        same_number(row['training_loss_sample_weighted'], mean, 'epoch weighted training loss', tolerance=1e-8)
        if epoch >= rule['min_epochs'] and wait >= rule['patience'] and epoch < cfg['max_epochs']:
            if first_stop is None:
                first_stop = epoch
        need(first_stop is None or epoch <= first_stop, 'training continued past first predeclared stopping epoch')
    return first_stop, wait, anchor


def audit_seed(audit, cfg, schedule, context, source, seed, group_finished=False):
    out = audit.run / 'seeds' / ('seed_%d' % seed)
    report = {'seed': seed, 'status': 'INCOMPLETE', 'complete_epochs_audited': 0}
    audit.seed_reports.append(report)
    if not out.is_dir():
        audit.pending.append('seed {} has not started'.format(seed))
        return None
    model = audit.read(out / 'model_config_seed.json')
    if model is not None:
        expected = copy.deepcopy(context['model_config']); expected['random_seed'] = seed
        need(model == expected, 'seed model config changed fields other than random_seed')
    status = audit.read(out / 'status.json')
    summary = audit.read(out / 'summary.json', required=False)
    if status is not None and status.get('status') == 'FAILED':
        raise Mismatch('runtime status explicitly reports FAILED')
    finished = status is not None and status.get('status') == 'PASS' and summary is not None
    # A seed writes PASS immediately before its history handle is closed.
    # Final group PASS is written only after all three seed handles close.
    final_audit = finished and group_finished and not audit.partial
    history_value = audit.read(out / 'history.json')
    val_value = audit.read(out / 'validation_evaluations.json')
    epoch_value = audit.read(out / 'epoch_history.json', required=finished)
    reload_value = audit.read(out / 'reload_checks.json')
    best_value = audit.read(out / 'best_selection.json')
    pending_before_jsonl = len(audit.pending)
    jsonl = audit.read_jsonl(out / 'history.jsonl')
    jsonl_unavailable = len(audit.pending) > pending_before_jsonl
    if any(v is None for v in [history_value, val_value, reload_value, best_value]) or model is None:
        audit.pending.append('seed {} lacks a stable complete evidence snapshot'.format(seed))
        return None
    if final_audit and (epoch_value is None or jsonl_unavailable):
        raise Unavailable('final history evidence is not yet fully parseable; retry a stable snapshot')
    history = records(history_value, 'history')
    validations = records(val_value, 'validation')
    epochs = records(epoch_value, 'epoch_history') if epoch_value is not None else []
    reloads = records(reload_value, 'reloads')
    candidates = records(best_value['candidates'], 'best candidates')
    steps = schedule['steps_per_epoch']
    v_epochs = contiguous_epochs(validations, steps, 'validation')
    r_epochs = contiguous_epochs(reloads, steps, 'reload')
    b_epochs = contiguous_epochs(candidates, steps, 'best candidates')
    for index, row in enumerate(epochs, 1):
        need(row['epoch'] == index and row['global_update'] == index * steps, 'epoch_history is not contiguous')
    completed = min(len(history) // steps, len(jsonl) // steps, v_epochs, r_epochs, b_epochs, len(epochs))
    if final_audit:
        need(summary['status'] == 'PASS' and summary['initialization_seed'] == seed,
             'summary status/seed mismatch')
        need(integer(summary['completed_epochs']) and 1 <= summary['completed_epochs'] <= cfg['max_epochs'],
             'invalid final completed epoch count')
        actual_epochs = int(summary['completed_epochs'])
        need(completed == actual_epochs and len(history) == len(jsonl) == actual_epochs * steps and
             len(validations) == len(reloads) == len(candidates) == actual_epochs + 1 and len(epochs) == actual_epochs,
             'final history/validation/reload/selection/epoch evidence lengths disagree')
        need(status['completed_updates'] == actual_epochs * steps, 'final seed status updates mismatch')
    report['complete_epochs_audited'] = completed
    report['completed_updates_audited'] = completed * steps
    report['logged_history_updates'] = len(history)
    report['jsonl_updates_visible'] = len(jsonl)
    report['runtime_status'] = status.get('status') if status else 'unavailable'
    report['seed_finished'] = finished
    history = history[:completed * steps]
    val_prefix = validations[:completed + 1]
    epoch_prefix = epochs[:completed]
    train_access = {r['id']: r for r in context['train_access']}
    val_access = {r['id']: r for r in context['validation_access']}
    for index, h in enumerate(history):
        batch = schedule['full'][index]
        for key in ['global_update', 'epoch', 'sample_count']:
            expected = batch['update'] if key == 'global_update' else batch[key]
            need(h[key] == expected, 'history schedule mismatch: update {} {}'.format(index + 1, key))
        need(vector(h['sample_ids']) == batch['sample_ids'], 'history sample ID order differs from frozen schedule')
        need(vector(h['valid_pixel_count']) == [train_access[rid]['valid_pixels'] for rid in batch['sample_ids']],
             'history valid-pixel counts differ from input metadata')
        same_number(h['learning_rate'], cfg['optimizer']['learning_rate'] * cfg['learning_rate_factors'][h['epoch'] - 1],
                    'history learning-rate factor')
        for key in LOSSES + ['unclipped_global_gradient_norm', 'clip_factor', 'parameter_delta_L2', 'update_time_s']:
            need(finite(h[key]) and h[key] >= 0, 'history nonfinite/negative ' + key)
        need(h['unclipped_global_gradient_norm'] > 0 and h['parameter_delta_L2'] > 0 and
             h['gradient_nonzero_elements'] > 0, 'zero gradient/update in logged training')
        same_number(h['clip_factor'], min(1, cfg['optimizer']['max_gradient_norm'] / h['unclipped_global_gradient_norm']),
                    'gradient clipping factor', tolerance=1e-8)
        need(h == jsonl[index], 'history.json and JSONL differ at update {}'.format(index + 1))
    for epoch in range(1, completed + 1):
        rows = history[(epoch - 1) * steps:epoch * steps]
        ids = [rid for row in rows for rid in vector(row['sample_ids'])]
        need(collections.Counter(ids) == collections.Counter(train_access.keys()), 'epoch does not visit all1396 exactly once')
        need([r['sample_count'] for r in rows] == [8] * (steps - 1) + [4], 'epoch batch sizes / tail4 mismatch')
    for evaluation in val_prefix:
        check_evaluation(evaluation, schedule['validation_order'], val_access,
                         'validation epoch{}'.format(evaluation['epoch']))
        need(evaluation['proxy_group_count'] == 15, 'full validation proxy-group count is not15')
    best = None
    for evaluation, candidate in zip(val_prefix, candidates[:completed + 1]):
        compare_candidate(candidate, evaluation, 'best candidate')
        selected = better(evaluation, best)
        need(candidate['selected_now'] is selected and
             candidate['is_untrained_reference'] is (evaluation['epoch'] == 0), 'best candidate flags/rule mismatch')
        if selected:
            best = candidate
    first_stop, wait, anchor = check_early_stopping(val_prefix, epoch_prefix, history, cfg, steps)
    for reload in reloads[:completed + 1]:
        need(reload['status'] == 'PASS' and reload['parameters_state_moments_exact'] is True,
             'reload did not report exact parameter/State/Adam agreement')
        need(reload['next_schedule_position'] == reload['global_update'] + 1, 'reload next position is not k+1')
        need(finite(reload['prediction_max_abs_error']) and
             0 <= reload['prediction_max_abs_error'] <= cfg['prediction_reload_max_abs_error'], 'reload probability error')
        path = out / 'checkpoints' / ('epoch_%03d.mat' % reload['epoch'])
        if not path.is_file() or path.stat().st_size == 0:
            raise Unavailable('checkpoint not yet stable/present: ' + str(path))
    report['validation_metrics_recomputed'] = len(val_prefix)
    report['early_stopping_wait_at_prefix'] = wait
    report['best_epoch_at_prefix'] = best['epoch'] if best else None
    if not final_audit:
        audit.pending.append('seed {}: final-summary/MAT-alias checks deferred until group PASS without --partial'.format(seed))
        return None
    need(summary['main_optimizer_updates'] == completed * steps and summary['max_epochs'] == cfg['max_epochs'],
         'summary final update/max-epochs mismatch')
    if first_stop is not None:
        need(completed == first_stop and summary['stopped_early'] is True and
             summary['stop_reason'] == 'predeclared_validation_patience', 'incorrect early-stop endpoint/reason')
    else:
        need(completed == cfg['max_epochs'] and summary['stopped_early'] is False and
             summary['stop_reason'] == 'max_epochs_reached', 'terminated without declared stopping condition')
    need(summary['initial_validation'] == validations[0] and summary['final_validation'] == validations[-1],
         'summary initial/final validation mismatch')
    need(summary['reload_checks'] == reload_value, 'summary reload checks mismatch')
    need(summary['source'] == source and summary['heldout_test_images_opened'] == 0, 'summary source/heldout mismatch')
    expected_hashes = copy.deepcopy(context['hashes'])
    expected_hashes['seed_model_config_sha256'] = sha_file(out / 'model_config_seed.json')
    expected_hashes['batch_schedule_sha256'] = cfg['batch_schedule_sha256']
    need(summary['hashes'] == expected_hashes, 'summary input/seed/schedule hashes mismatch')
    need(SHA_RE.match(summary['initial_parameter_sha256']) is not None, 'invalid initial-parameter fingerprint')
    if best is not None:
        compare_candidate(best_value['best'], best, 'best_selection best')
        compare_candidate(summary['best_validation'], best, 'summary best')
        need(summary['selected_checkpoint'] == 'checkpoints/best_validation.mat', 'incorrect selected-checkpoint alias path')
        best_path = out / 'checkpoints' / ('epoch_%03d.mat' % best['epoch'])
        alias = out / summary['selected_checkpoint']
        need(sha_file(alias) == sha_file(best_path), 'best MAT alias is not byte-identical to selected epoch')
        report['best_checkpoint_sha256'] = sha_file(alias)
    else:
        need(best_value['best'] == [] and summary['best_validation'].get('status') == 'best_unavailable',
             'all-undefined candidates did not preserve best_unavailable')
    last_epoch = out / 'checkpoints' / ('epoch_%03d.mat' % completed)
    last_path = out / summary['primary_last_checkpoint']
    need(summary['primary_last_checkpoint'] == 'checkpoints/last.mat' and
         sha_file(last_path) == sha_file(last_epoch), 'last MAT alias differs from actual final epoch')
    report['last_checkpoint_sha256'] = sha_file(last_path)
    same_number(summary['update_time_s'], sum(r['update_time_s'] for r in history), 'summary update time', tolerance=1e-8)
    same_number(summary['validation_time_s'], sum(r['evaluation_time_s'] for r in validations),
                'summary validation time', tolerance=1e-8)
    report['initial_parameter_sha256'] = summary['initial_parameter_sha256']
    report['status'] = 'PASS'
    return summary


def check_statistics(actual, expected_values, label):
    values = vector(actual['values'])
    need(len(values) == len(expected_values), label + ': value count')
    for a, b in zip(values, expected_values):
        same_number(a, b, label + '/values')
    defined = [v for v in expected_values if v is not None]
    need(actual['defined_count'] == len(defined), label + ': defined count')
    same_number(actual['mean'], defined_mean(defined), label + '/mean')
    deviation = statistics.stdev(defined) if len(defined) > 1 else (0.0 if defined else None)
    same_number(actual['sample_std'], deviation, label + '/sample_std')


def relative_report_paths(value, root, supplied_root=None):
    """Transform output text only; never use these labels for input access.

    Both the resolved root and its user-supplied spelling are recognized, so
    Windows junction/drive aliases and escaped paths in OS errors are covered.
    Sibling workspace paths use [workspace]; unrelated drive paths use a
    logical label rather than exposing an absolute machine-specific path.
    """
    roots = [Path(root)]
    if supplied_root is not None:
        roots.append(Path(supplied_root).absolute())

    def normalized(text):
        return text.replace('\\\\', '/').replace('\\', '/')

    prefixes = {}
    for project in roots:
        prefixes[normalized(str(project)).rstrip('/')] = '.'
        prefixes.setdefault(normalized(str(project.parent)).rstrip('/'), '[workspace]')
    prefixes = sorted(((p, label) for p, label in prefixes.items() if p),
                      key=lambda item: -len(item[0]))

    def transform(item):
        if isinstance(item, dict):
            return {transform(key): transform(child) for key, child in item.items()}
        if isinstance(item, list):
            return [transform(child) for child in item]
        if not isinstance(item, str):
            return item
        text = normalized(item)
        for prefix, label in prefixes:
            text = re.sub(re.escape(prefix) + r'(?=/|$|[\s\'":)\]])',
                          lambda match, label=label: label, text, flags=re.IGNORECASE)
        return re.sub(r'(?i)\b[A-Z]:/[^\s\'"<>]+', '[absolute-path]', text)

    return transform(value)


def audit_run(audit):
    cfg = audit.read(audit.root / CONFIG)
    context = audit.read(audit.run / 'input_access.json')
    source = audit.read(audit.run / 'source_version.json')
    status = audit.read(audit.run / 'status.json')
    if any(v is None for v in [cfg, context, source, status]):
        return False
    if status.get('status') == 'FAILED':
        raise Mismatch('group runtime status explicitly reports FAILED')
    audit.hash_reference(CONFIG, context['hashes']['training_config_sha256'], sidecar=True)
    need(cfg == context['config'], 'run configuration differs from frozen v2 config')
    for key in ['model_config_sha256', 'source_split_sha256']:
        need(context['hashes'][key] == cfg[key], 'input access hash differs from config: ' + key)
    need(cfg['plan_id'] == 'full_stage2_recipe_v2' and cfg['mode'] == 'full_baseline' and
         cfg['optimizer']['learning_rate_schedule'] == 'piecewise_epoch_factors', 'incorrect full recipe identity')
    need(cfg['batch_size'] == 8 and cfg['max_epochs'] == 20 and cfg['initialization_seeds'] == [20261114, 20261115, 20261116],
         'predeclared full budget/seeds changed')
    need(cfg['learning_rate_factors'] == [1.0] * 3 + [0.3] * 3 + [0.1] * 14, 'LR factor recipe changed')
    need(cfg['early_stopping']['enabled'] is True and cfg['early_stopping']['min_epochs'] == 8 and
         cfg['early_stopping']['patience'] == 4 and cfg['early_stopping']['min_delta'] == 1e-4,
         'predeclared early-stopping recipe changed')
    for key in ['model_config', 'source_split', 'batch_schedule']:
        audit.hash_reference(cfg[key], cfg[key + '_sha256'], sidecar=key == 'batch_schedule')
    parent = audit.read(audit.root / cfg['source_split'])
    schedule = audit.read(audit.root / cfg['batch_schedule'])
    if parent is None or schedule is None:
        return False
    parent_rows = records(parent['records'], 'parent records')
    training = sorted([r for r in parent_rows if r['assigned_split'] == 'training'], key=lambda r: r['id'])
    validation = sorted([r for r in parent_rows if r['assigned_split'] == 'validation'], key=lambda r: r['id'])
    need(len(training) == 1396 and len(validation) == 400 and
         len(set(r['group_id'] for r in training)) == 45 and len(set(r['group_id'] for r in validation)) == 15,
         'frozen split counts/groups mismatch')
    for key in ['id', 'group_id', 'image', 'mask']:
        need(not set(r[key] for r in training) & set(r[key] for r in validation), 'split overlap: ' + key)
    for key, expected in [('train_access', training), ('validation_access', validation)]:
        actual = records(context[key], key)
        need(len(actual) == len(expected), key + ': access count')
        for a, e in zip(actual, expected):
            for field in ['id', 'group_id', 'assigned_split', 'image', 'mask']:
                need(a[field] == e[field], key + ': frozen record membership/order mismatch')
            need(integer(a['valid_pixels']) and a['valid_pixels'] > 0, key + ': invalid valid pixels')
            for field in ['image_file_sha256', 'mask_file_sha256']:
                need(SHA_RE.match(a[field]) is not None, key + ': invalid recorded file hash')
    need(context['unique_training_images_opened'] == 1396 and context['validation_images_opened'] == 400 and
         context['heldout_test_images_opened'] == 0, 'input access counts/heldout mismatch')
    need(schedule['source_split_sha256'] == cfg['source_split_sha256'] and schedule['protocol'] == 'full_stage2_v1' and
         schedule['order_seed'] == cfg['order_seed'] == 20261114 and schedule['shared_across_initialization_seeds'] is True,
         'schedule source/ordering contract mismatch')
    need(schedule['validation_order'] == [r['id'] for r in validation] and
         cfg['fixed_validation_ids'] == schedule['validation_order'][:8], 'validation/fixed8 order mismatch')
    steps = 175
    need(schedule['steps_per_epoch'] == steps and len(schedule['full']) == steps * 20, 'schedule size mismatch')
    train_ids = set(r['id'] for r in training)
    for epoch in range(1, 21):
        rows = schedule['full'][(epoch - 1) * steps:epoch * steps]
        ordered = sorted(train_ids, key=lambda rid: (
            sha_bytes('{}|full_stage2_v1|{}|{}'.format(cfg['order_seed'], epoch, rid).encode('utf-8')), rid))
        for j, row in enumerate(rows):
            expected = ordered[j * 8:(j + 1) * 8]
            need(row['update'] == (epoch - 1) * steps + j + 1 and row['epoch'] == epoch and
                 row['sample_ids'] == expected and row['sample_count'] == len(expected),
                 'schedule differs from independently regenerated shared SHA ordering')
        need(collections.Counter(rid for row in rows for rid in row['sample_ids']) == collections.Counter(train_ids),
             'schedule epoch does not cover1396 once')
    need(source['git_dirty'] is False and re.match(r'^[0-9a-f]{40}$', source['git_commit']), 'source revision not clean/pinned')
    for row in records(source['files'], 'source files'):
        path = audit.run / 'source_snapshot' / row['name']
        need(sha_file(path) == row['sha256'], 'source snapshot SHA mismatch: ' + row['name'])
    finished = []
    for seed in cfg['initialization_seeds']:
        result = audit.capture('seed {}'.format(seed), lambda seed=seed: audit_seed(
            audit, cfg, schedule, context, source, seed, group_finished=status.get('status') == 'PASS'))
        if result is not None:
            finished.append(result)
    if len(finished) < 3 or status.get('status') != 'PASS':
        audit.pending.append('final aggregate audit requires all three finalized seeds and absence of --partial')
        return False
    need(status['completed_seeds'] == 3, 'group PASS completed-seed count')
    need(len(set(r['initial_parameter_sha256'] for r in finished)) == 3, 'different seeds have identical initial weights')
    aggregate = audit.read(audit.run / 'summary.json')
    completed_log = audit.read(audit.run / 'completed_seed_summaries.json')
    if aggregate is None or completed_log is None:
        return False
    need(records(completed_log, 'completed seeds') == finished and
         records(aggregate['seed_summaries'], 'aggregate seeds') == finished, 'aggregate/per-seed summaries differ')
    need(aggregate['status'] == 'PASS' and aggregate['initialization_seeds'] == cfg['initialization_seeds'] and
         aggregate['reference_seed'] == cfg['initialization_seeds'][0], 'aggregate status/seed/reference mismatch')
    need(aggregate['source'] == source and aggregate['hashes'] == context['hashes'], 'aggregate source/hashes mismatch')
    need(aggregate['training_images'] == 1396 and aggregate['validation_images'] == 400 and
         aggregate['heldout_test_images_opened'] == 0, 'aggregate input/heldout counts')
    for key in METRICS:
        check_statistics(aggregate['last_statistics'][key], [r['final_validation'][key] for r in finished], 'last/' + key)
        check_statistics(aggregate['selected_statistics'][key], [r['best_validation'][key] for r in finished], 'selected/' + key)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--relative-paths', action='store_true',
                        help='Use relative/logical path labels in the output report; checks are unchanged')
    args = parser.parse_args()
    if re.match(r'^[A-Za-z0-9_-]+$', args.run_id) is None:
        parser.error('run-id must contain only letters, digits, underscore, or hyphen')
    root = args.project_root.resolve()
    audit = Audit(root, args.run_id, args.partial)
    complete = audit.capture('run', lambda: audit_run(audit)) is True
    if args.partial:
        status = 'INCOMPLETE'
    elif audit.errors:
        status = 'FAIL'
    elif complete and not audit.pending:
        status = 'PASS'
    else:
        status = 'INCOMPLETE'
    report = {'schema_version': 1, 'status': status, 'partial': args.partial,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'project_root': str(root), 'run_id': args.run_id,
              'final_aggregate_audit_completed': complete,
              'checks_passed_on_available_complete_epochs': not audit.errors,
              'errors': audit.errors, 'pending_evidence': audit.pending,
              'seeds': audit.seed_reports, 'read_json_and_reference_sha256': audit.files,
              'raw_images_or_masks_opened': 0, 'training_started': False,
              'evidence_limits': [
                  'MAT files are byte-hashed, not decoded. State/Adam/RNG correctness relies on the recorded MATLAB reload assertions.',
                  'Loss is reaggregated from logged inference-batch losses; logits and the objective are not independently recomputed.',
                  'Source-group membership and preprocessing counts are verified from frozen metadata; raw image/mask data are not opened.',
                  'Initial-weight distinctness uses the actual initialized-parameter fingerprints recorded by each completed run.']}
    if args.relative_paths:
        report = relative_report_paths(report, root, args.project_root)
        report['project_root'] = '.'
        report['relative_paths'] = True
    output = args.out.resolve()
    need(output not in [root / name for name in audit.files] and output.suffix.lower() == '.json',
         'out must be a separate JSON audit file')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(output))
    output_label = relative_report_paths(str(output), root, args.project_root) if args.relative_paths else str(output)
    print(json.dumps({'status': status, 'out': output_label, 'errors': len(audit.errors),
                      'pending_evidence': len(audit.pending),
                      'complete_epochs_audited': {str(r['seed']): r['complete_epochs_audited'] for r in audit.seed_reports}}, indent=2))
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[status]


if __name__ == '__main__':
    sys.exit(main())
