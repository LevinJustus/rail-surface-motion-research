"""Deterministic v3/v4 comparisons with synthetic images and probabilities."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rail_motion_conditional_v3 import fuse_conditional_v3
from rail_motion_conditional_v4 import (
    CURRENT_SEED_THRESHOLD, CURRENT_SEED_WINDOW, compare_conditional_v3_v4, fuse_conditional_v4,
)
import rail_motion_v3 as motion


def run():
    config = json.loads((ROOT / 'configs/algorithm_design_conditional_v4.json').read_text(encoding='utf-8'))
    for path, expected in (
        (ROOT / 'configs' / config['parent_config'], config['parent_config_sha256']),
        (ROOT / 'python/rail_motion_conditional_v3.py', config['parent_module_sha256']),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert config['current_seed_window'] == CURRENT_SEED_WINDOW == 5
    assert config['current_seed_threshold'] == CURRENT_SEED_THRESHOLD == motion.DEFAULTS['decision']

    site = np.s_[127:130, 127:130]
    texture = np.random.RandomState(20261124).uniform(.15, .85, (256, 256))
    dark = texture.copy()
    dark[site] = .05
    neutral = texture.copy()
    neutral[site] = .50
    p = np.full(texture.shape, .10, np.float32)
    h = p.copy()
    h[site] = .85
    edges = [[0, 0], [0, 0]]
    records = []

    def compare(name, images, raw, base_edges, expected, target=None):
        (before, old_masks), (after, masks) = compare_conditional_v3_v4(images, raw, base_edges)
        direct, direct_masks = fuse_conditional_v3(images, raw, base_edges)
        conservative = motion.fuse(raw, base_edges, True, motion.DEFAULTS)
        assert np.array_equal(before, direct), name
        assert all(np.array_equal(old_masks[key], direct_masks[key]) for key in old_masks), name
        assert all(np.array_equal(masks[key], old_masks[key]) for key in old_masks
                   if key != 'strong_correction'), name
        seed = raw[-1] >= .50
        padded = np.pad(seed, 2, mode='constant')
        local = np.logical_or.reduce([padded[y:y + 256, x:x + 256] for y in range(5) for x in range(5)])
        assert np.array_equal(masks['current_local_seed'], local), name
        retained = masks['strong_correction']
        assert np.array_equal(retained, old_masks['strong_correction'] & local), name
        assert np.array_equal(after, np.where(retained, before, conservative)), name
        assert np.array_equal(after[seed], before[seed]), name
        assert retained.sum() == expected, name
        assert np.all(np.isfinite(after)) and after.dtype == np.float32 and 0 <= after.min() <= after.max() <= 1
        records.append(dict(case=name, synthetic_target_frame=len(raw) - 1, target_foreground=target,
                            v3_promotions=int(old_masks['strong_correction'].sum()), v4_promotions=int(retained.sum()),
                            v3_foreground_in_site=int((before[site] >= .5).sum()),
                            v4_foreground_in_site=int((after[site] >= .5).sum())))
        return after, masks

    compare('complete_current_miss', [dark] * 3, [h, h, p], edges, 0, True)
    assert records[-1]['v3_foreground_in_site'] == 9 and records[-1]['v4_foreground_in_site'] == 0
    seeded = p.copy()
    seeded[128, 125] = .50
    after, masks = compare('threshold_seed_at_distance_two', [dark] * 3, [h, h, seeded], edges, 3, True)
    expected = np.zeros_like(p, dtype=bool)
    expected[127:130, 127] = True
    assert np.array_equal(masks['strong_correction'], expected)
    assert after[128, 125] < .50  # The raw seed still counts when B suppresses it.
    standalone, standalone_masks = fuse_conditional_v4([dark] * 3, [h, h, seeded], edges)
    assert np.array_equal(standalone, after) and all(np.array_equal(standalone_masks[k], v) for k, v in masks.items())
    compare('same_seed_can_support_wrong_history', [dark] * 3, [h, h, seeded], edges, 3, False)

    below = seeded.copy()
    below[128, 125] = np.nextafter(np.float32(.50), np.float32(0.))
    compare('one_float32_step_below_threshold', [dark] * 3, [h, h, below], edges, 0, True)
    far = p.copy()
    far[128, 124] = .95
    compare('seed_at_distance_three', [dark] * 3, [h, h, far], edges, 0, True)
    compare('seed_without_persistent_history', [dark] * 3, [h, p, seeded], edges, 0)
    compare('seed_with_image_mismatch', [dark, dark, 1. - dark], [h, h, seeded], edges, 0)
    compare('seed_with_no_historical_contrast', [neutral] * 3, [h, h, seeded], edges, 0)

    def translated(array, steps):
        return np.roll(array, (2 * steps, -3 * steps), axis=(0, 1))

    compare('seed_with_nonzero_base_motion', [translated(dark, 2), translated(dark, 1), dark],
            [translated(h, 2), translated(h, 1), seeded], [[3, -2], [3, -2]], 3, True)

    actual, _ = compare('missing_last_edge', [dark] * 3, [h, h, seeded], [edges[0], None], 0)
    assert np.array_equal(actual, seeded)
    compare('two_frames_with_history', [dark] * 2, [h, seeded], [edges[0]], 0)
    corner = p.copy()
    corner[0, 0] = .75
    actual, debug = compare('first_frame_corner_seed_no_wrap', [dark], [corner], [], 0)
    assert np.array_equal(actual, corner)
    assert debug['current_local_seed'].sum() == 9 and debug['current_local_seed'][:3, :3].all()
    return records


if __name__ == '__main__':
    print(json.dumps(dict(status='PASS', cases=run()), indent=2))
