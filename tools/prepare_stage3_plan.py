"""Freeze a mature-model comparison under the existing full baseline recipe."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def reference(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return {'file': path.relative_to(root).as_posix(), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--postcheck', required=True)
    parser.add_argument('--optimizer-probe', required=True)
    parser.add_argument('--output', default='configs/training_stage3_mature_v1.json')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).absolute().parent.parent
    parent_path = root / 'configs/training_stage2_full_v2.json'
    parent = read(parent_path)
    model_path = root / 'configs/mobilenet_deeplab_scratch_v1.json'
    model = read(model_path)
    refs = [reference(root, x) for x in (args.preflight, args.postcheck, args.optimizer_probe)]
    preflight, postcheck, optimizer_probe = [read(root / x['file']) for x in refs]
    assert all(x['status'] == 'PASS' for x in (preflight, postcheck, optimizer_probe))
    assert preflight['optimizer_updates'] == 5 and preflight['raw_training_images_opened'] == 8
    assert preflight['validation_images_opened'] == preflight['heldout_test_images_opened'] == 0
    assert preflight['formal_training_started'] is False
    assert preflight['parent_model_sha256'] == sha(model_path) == postcheck['model_config_sha256']
    assert preflight['architecture']['pretrained_weights_used'] is False
    assert preflight['architecture']['geometry']['status'] == 'PASS'
    assert postcheck['optimizer_updates'] == postcheck['raw_images_opened'] == 0
    assert model['backbone_weights'] == 'none' and model['initialization'] == 'random'
    assert parent['batch_size'] == 8 and parent['max_epochs'] == 20
    assert parent['initialization_seeds'] == [20261114, 20261115, 20261116]
    assert sha(root / parent['batch_schedule']) == parent['batch_schedule_sha256']
    config = copy.deepcopy(parent)
    config.update(plan_id='mature_stage3_recipe_v1', mode='mature_comparison', status='planned_not_started',
                  scope='Step11 project implementation of mature MobileNetV2/DeepLabv3+; full1396/400; random initialization; shared recipe; no heldout test',
                  model_config=model_path.relative_to(root).as_posix(), model_config_sha256=sha(model_path))
    config['metadata'] = {
        'inherits_full_baseline_recipe': reference(root, parent_path),
        'comparison_baseline_summary': reference(root, 'docs/evidence/step10/full/summary.json'),
        'engineering_preflight': refs[0], 'checkpoint_bn_postcheck': refs[1],
        'local_adam_factor_probe': refs[2],
        'optimization_scope': 'Shared recipe chosen previously using the U-Net pilot; not equal per-model hyperparameter optimization',
        'pretrained_weights_used': False,
        'implementation_scope': model['implementation'],
        'parameter_count': preflight['architecture']['parameter_count'],
        'bn_layer_count': preflight['architecture']['bn_count'],
        'maximum_optimizer_updates_per_seed': 3500,
        'maximum_training_images_per_seed': 27920,
        'median_preflight_batch8_update_s': preflight['timing']['median_update_time_s'],
        'maximum_pure_update_seconds_three_seeds': preflight['timing']['maximum_pure_updates_s_three_seeds'],
        'estimate_scope': 'Compute-only extrapolation from three timed disposable updates; loading, validation, initialization, saves, pauses and plotting excluded',
        'formal_networks_must_be_fresh': True,
        'raw_images_opened_by_plan': 0, 'training_started_by_plan': False,
        'original_full_baseline_files_unchanged': True,
    }
    encoded = (json.dumps(config, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    output = root / args.output
    sidecar = output.with_suffix('.sha256')
    checksum = hashlib.sha256(encoded).hexdigest()
    checksum_bytes = (checksum + '  ' + output.name + '\n').encode('ascii')
    if args.check:
        assert output.read_bytes() == encoded and sidecar.read_bytes() == checksum_bytes
    else:
        assert not output.exists() and not sidecar.exists(), 'Refuse to overwrite a frozen plan'
        output.write_bytes(encoded); sidecar.write_bytes(checksum_bytes)
    print(json.dumps({'status': 'PASS', 'mode': 'check' if args.check else 'freeze',
                      'file': output.relative_to(root).as_posix(), 'sha256': checksum,
                      'batch_schedule_unchanged': True, 'optimizer_updates_per_seed_max': 3500,
                      'raw_images_opened': 0}, ensure_ascii=True))


if __name__ == '__main__':
    main()
