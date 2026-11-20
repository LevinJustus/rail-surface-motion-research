"""Data-free checks of the conditional v3 ablation and its remaining failure."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import rail_motion_conditional as v1
import rail_motion_conditional_v2 as v2
import rail_motion_conditional_v3 as candidate
import rail_motion_v3 as v3


def run():
    site = np.s_[127:130, 127:130]
    texture = np.random.RandomState(20261120).uniform(.15, .85, (256, 256))
    dark = texture.copy()
    dark[site] = .05
    neutral = texture.copy()
    neutral[site] = .5
    bright = texture.copy()
    bright[site] = .95
    degraded = texture.copy()
    degraded[122:135, 122:135] = .5
    p = np.full(texture.shape, .1, dtype=np.float32)
    h = p.copy()
    h[site] = .85
    raw = [h, h, p]
    edges = [[0, 0], [0, 0]]
    weak = v3.fuse(raw, edges, True, v3.DEFAULTS)
    strong = v3.fuse(raw, edges, False, v3.DEFAULTS)
    results = []
    for name, images, target, expected_v2, expected_v3 in (
        ('clear_current_missed_dark_defect', [dark, dark, dark], True, 0, 9),
        ('degraded_current_missed_dark_defect', [dark, dark, degraded], True, 9, 9),
        ('stable_neutral_false_history', [neutral, neutral, neutral], False, 0, 0),
        ('aligned_normal_dark_mark_false_history', [dark, dark, dark], False, 0, 9),
        ('opposite_polarity_false_history', [bright, dark, dark], False, 0, 0),
        ('current_context_mismatch', [dark, dark, 1. - dark], False, 0, 0),
    ):
        first, _ = v1.fuse_conditional(images, raw, edges)
        second, second_masks = v2.fuse_conditional_v2(images, raw, edges)
        third, masks = candidate.fuse_conditional_v3(images, raw, edges)
        keep = masks['strong_correction']
        assert keep.sum() == expected_v3 and keep[site].sum() == expected_v3, name
        assert second_masks['strong_correction'].sum() == expected_v2, name
        assert np.array_equal(third, np.where(keep, strong, weak)), name
        assert np.array_equal(second, np.where(keep & masks['quality_drop'], strong, weak)), name
        results.append(dict(case=name, label_foreground=target, positive_pixels={
            method: int((value[site] >= .5).sum()) for method, value in
            (('single', p), ('B', weak), ('E', strong), ('conditional_v1', first),
             ('conditional_v2', second), ('conditional_v3', third))}))
    for images, probabilities, motion in (([dark], [p], []), ([dark, dark], [h, p], [None]),
                                          ([dark] * 3, raw, [edges[0], None])):
        value, masks = candidate.fuse_conditional_v3(images, probabilities, motion)
        assert np.array_equal(value, p) and not masks['strong_correction'].any()
    return results


if __name__ == '__main__':
    print(json.dumps(dict(status='PASS', cases=run()), indent=2))
