"""Freeze a size/ID-only motion input plan; never opens any raw image or mask."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    split_path = ROOT / 'data/splits/conservative_group_v1.json'
    audit_path = ROOT / 'data/manifests/mcnet_audit.json'
    assert sha(split_path) == 'c65f8c439ac3f245541a6678f3fe937bd3dcf7136584f088fd90eae1609270a6'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    dims = {r['id']: {'height': r['height'], 'width': r['width']} for r in audit['records']}
    active = [r for r in split['records'] if r['assigned_split'] in ('training', 'validation')]
    assert len(active) == 1796
    selected, candidates, missing = [], [], []
    for partition, expected in [('training', 45), ('validation', 15)]:
        rows = sorted([r for r in active if r['assigned_split'] == partition], key=lambda r: r['id'])
        groups = sorted({r['group_id'] for r in rows})
        assert len(groups) == expected
        for group in groups:
            members = [dict(r, **dims[r['id']]) for r in rows if r['group_id'] == group]
            eligible = []
            for r in members:
                ok = min(r['height'], r['width']) >= 300
                candidates.append({'id': r['id'], 'group_id': group, 'assigned_split': partition,
                                   'height': r['height'], 'width': r['width'], 'eligible': ok,
                                   'reason': 'native_minimum_300' if ok else 'native_dimension_below_300'})
                if ok:
                    eligible.append(r)
            if eligible:
                r = eligible[0]
                r['longitudinal_span_px'] = min(64, r['height'] - 256 - 20 - 6)
                selected.append(r)
            else:
                missing.append({'assigned_split': partition, 'group_id': group,
                                'candidate_ids': [r['id'] for r in members]})
    plan = {'schema_version': 1, 'plan_id': 'step12_sources_v1',
            'selection_rule': 'One ascending-ID eligible source per frozen proxy group; dimensions only; no mask or prediction read',
            'parent_split_sha256': sha(split_path), 'size_audit_sha256': sha(audit_path),
            'active_count': len(active), 'candidate_records': candidates,
            'selected_records': selected, 'missing_groups': missing,
            'raw_images_opened': 0, 'raw_masks_opened': 0,
            'physical_independence_claim': False}
    plan_path = ROOT / 'data/splits/step12_sources_v1.json'
    config_path = ROOT / 'configs/motion_sequences_v1.json'
    assert not plan_path.exists() and not config_path.exists(), 'Refuse to overwrite frozen plan'
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    profiles = [
        {'name': 'clean', 'trajectory': 'constant', 'jitter': False, 'blur_px': 0, 'lighting': False, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'variable_speed', 'trajectory': 'variable', 'jitter': False, 'blur_px': 0, 'lighting': False, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'geometric_jitter', 'trajectory': 'constant', 'jitter': True, 'blur_px': 0, 'lighting': False, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'blur_1px', 'trajectory': 'constant', 'jitter': False, 'blur_px': 1, 'lighting': False, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'blur_3px', 'trajectory': 'constant', 'jitter': False, 'blur_px': 3, 'lighting': False, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'persistent_light', 'trajectory': 'constant', 'jitter': False, 'blur_px': 0, 'lighting': True, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'odom_scale_plus5', 'trajectory': 'constant', 'jitter': False, 'blur_px': 0, 'lighting': False, 'scale': 1.05, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'odom_scale_minus5', 'trajectory': 'constant', 'jitter': False, 'blur_px': 0, 'lighting': False, 'scale': 0.95, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'odom_quantization', 'trajectory': 'constant', 'jitter': False, 'blur_px': 0, 'lighting': False, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 4, 'drop_frames': []},
        {'name': 'stop_reverse', 'trajectory': 'stop_reverse', 'jitter': False, 'blur_px': 0, 'lighting': False, 'scale': 1.02, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': []},
        {'name': 'odom_dropout', 'trajectory': 'constant', 'jitter': False, 'blur_px': 0, 'lighting': False, 'scale': 1.0, 'drift_px': 0.6, 'quantization_px': 0.5, 'drop_frames': [4,5]},
        {'name': 'combined', 'trajectory': 'variable', 'jitter': True, 'blur_px': 3, 'lighting': True, 'scale': 0.98, 'drift_px': 0.6, 'quantization_px': 2, 'drop_frames': [5]},
    ]
    config = {'schema_version': 1, 'plan_id': 'controlled_motion_v1', 'scope': 'step12_generation_and_audit_only',
              'source_plan': str(plan_path.relative_to(ROOT)).replace('\\','/'), 'source_plan_sha256': sha(plan_path),
              'source_split': str(split_path.relative_to(ROOT)).replace('\\','/'), 'source_split_sha256': sha(split_path),
              'size_audit': str(audit_path.relative_to(ROOT)).replace('\\','/'), 'size_audit_sha256': sha(audit_path),
              'seed': 20261116, 'random_draws': False, 'frame_size_hw': [256,256], 'frames_per_sequence': 9,
              'sampling_interval_s': 0.05, 'exposure_subsamples': 9, 'jpeg_label_threshold': 128,
              'rotation_amplitude_deg': 0.7, 'lateral_amplitude_px': 1.5,
              'trajectory_knots': {'constant': [0,.125,.25,.375,.5,.625,.75,.875,1],
                                  'variable': [0,.03,.10,.22,.40,.60,.80,.94,1],
                                  'stop_reverse': [0,.2,.4,.4,.4,.3,.2,.1,0]},
              'drift_rule': '0.6*(frame_index_zero_based/8)^2 pixels',
              'lighting_rule': '(0.65+0.05*cos(2*pi*t/8))*(0.75+0.25*x/256)*image + 0.08*exp(-((x-0.65*256)/(0.18*256))^2)',
              'longitudinal_span_rule': 'min(64, source_height-256-20-6)',
              'center_crop_rule': 'geometric centre of full trajectory footprint; no foreground access',
              'contract_source_indices_one_based': [1,30,60],
              'algorithm_frame_fields': ['image','timestamp_s','simulated_odometry_px','odometry_available'],
              'units': 'source pixels and seconds; no physical calibration, measured encoder, or FP/m',
              'profiles': profiles}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for p in (plan_path, config_path):
        p.with_suffix('.sha256').write_text(sha(p)+'  '+p.name+'\n', encoding='ascii')
    print(json.dumps({'selected':len(selected),'missing_groups':len(missing),'config_sha256':sha(config_path),
                      'spans':dict(__import__('collections').Counter(r['longitudinal_span_px'] for r in selected))}))

if __name__ == '__main__':
    main()
