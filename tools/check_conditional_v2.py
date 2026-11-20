"""Checks for the contrast gate and the remaining ambiguity of image evidence."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import rail_motion_conditional as v1
import rail_motion_conditional_v2 as v2
import rail_motion_v3 as v3


def run():
    site = np.s_[127:130, 127:130]
    texture = np.random.RandomState(20261120).uniform(.15, .85, (256, 256))
    current = texture.copy()
    current[122:135, 122:135] = .5
    p = np.full(current.shape, .1, dtype=np.float32)
    h = p.copy()
    h[site] = .85
    edges = [[3, -2], [3, -2]]
    shift = lambda array, steps: np.roll(array, (2 * steps, -3 * steps), axis=(0, 1))
    raw = [shift(h, 2), shift(h, 1), p]
    weak = v3.fuse(raw, edges, True, v3.DEFAULTS)
    strong = v3.fuse(raw, edges, False, v3.DEFAULTS)
    results = []
    for name, older_level, previous_level, positive, expected in (
        ('dark_defect', .05, .05, True, 9),
        ('bright_defect', .95, .95, True, 9),
        ('neutral_stable_false_history', .5, .5, False, 0),
        ('one_neutral_history', .05, .5, False, 0),
        ('opposite_polarity', .05, .95, False, 0),
        ('dark_normal_mark_remaining_failure', .05, .05, False, 9),
    ):
        older = texture.copy()
        previous = texture.copy()
        older[site], previous[site] = older_level, previous_level
        images = [shift(older, 2), shift(previous, 1), current]
        first, first_masks = v1.fuse_conditional(images, raw, edges)
        second, second_masks = v2.fuse_conditional_v2(images, raw, edges)
        keep = second_masks['strong_correction']
        assert first_masks['strong_correction'][site].all(), name
        assert keep.sum() == expected and keep[site].sum() == expected, name
        assert np.array_equal(second, np.where(keep, first, weak)), name
        results.append(dict(case=name, true_foreground=positive, positives={
            method: int(np.count_nonzero(values[site] >= .5)) for method, values in
            (('single', p), ('B', weak), ('E', strong), ('v1', first), ('v2', second))}))
    fallback, masks = v2.fuse_conditional_v2([current], [p], [])
    assert np.array_equal(fallback, p) and not any(mask.any() for mask in masks.values())
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))
