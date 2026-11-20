"""Run deterministic, data-free mechanism checks for conditional history fusion."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import rail_motion_v3 as v3
import rail_motion_conditional as conditional


def main():
    with (ROOT / 'configs' / 'algorithm_design_conditional_v1.json').open(encoding='utf-8') as handle:
        frozen = json.load(handle)
    cfg = frozen['parameters']
    assert cfg == conditional.DEFAULTS
    assert (frozen['local_window'], frozen['ring_outer_window']) == (conditional.LOCAL_SIZE, conditional.RING_SIZE)
    for key in ('decision', 'history_threshold', 'strong_current'):
        assert frozen[key] == v3.DEFAULTS[key]
    decision = frozen['decision']

    texture = np.random.RandomState(20261120).uniform(.1, .9, (256, 256))
    degraded = texture.copy()
    degraded[122:135, 122:135] = .5
    site = np.s_[127:130, 127:130]
    p = np.full(texture.shape, .1, dtype=np.float32)
    h = p.copy()
    h[site] = .85
    edges = [[3, -2], [3, -2]]
    results = []

    def evaluate(name, images, raw, paths, expect_strong, foreground):
        raw = [array.astype(np.float32) for array in raw]
        weak = v3.fuse(raw, paths, True, v3.DEFAULTS)
        strong = v3.fuse(raw, paths, False, v3.DEFAULTS)
        result, masks = conditional.fuse_conditional(images, raw, paths, cfg)
        chosen = masks['strong_correction']
        expected = np.zeros(texture.shape, bool)
        if expect_strong:
            expected[site] = True
        assert np.array_equal(chosen, expected), name
        assert np.array_equal(result, np.where(chosen, strong, weak)), name
        assert result.dtype == np.float32 and np.isfinite(result).all(), name
        assert 0 <= result.min() <= result.max() <= 1, name
        counts = {method: int(np.count_nonzero(values[site] >= decision)) for method, values in
                  (('single', raw[-1]), ('conservative', weak), ('strong', strong), ('conditional', result))}
        results.append(dict(case=name, foreground_pixels_in_site=9 if foreground else 0,
                            positive_pixels_in_site=counts, strong_corrections=int(chosen.sum())))
        return result, masks

    def shifted(array, steps):
        return np.roll(array, (2 * steps, -3 * steps), axis=(0, 1))

    history_images = [shifted(texture, 2), shifted(texture, 1)]
    history_raw = [shifted(h, 2), shifted(h, 1)]
    raw = history_raw + [p]

    _, masks = evaluate('severe_current_miss_local_degradation', history_images + [degraded],
                        raw, edges, True, True)
    assert all(masks[key][site].all() for key in masks)
    _, masks = evaluate('stable_false_history_clear_current', history_images + [texture],
                        raw, edges, False, False)
    assert not masks['quality_drop'][site].any() and masks['correspondence'][site].all()
    evaluate('stable_false_history_degraded_current_limit', history_images + [degraded],
             raw, edges, True, False)

    _, masks = evaluate('mismatched_current_context', history_images + [1. - degraded],
                        raw, edges, False, True)
    assert masks['quality_drop'][site].all() and not masks['correspondence'][site].any()
    _, masks = evaluate('mismatched_older_context', [1. - history_images[0], history_images[1], degraded],
                        raw, edges, False, True)
    assert masks['quality_drop'][site].all() and not masks['correspondence'][site].any()
    _, masks = evaluate('low_detail_history', [shifted(degraded, 2), shifted(degraded, 1), degraded],
                        raw, edges, False, True)
    assert not masks['quality_drop'][site].any()
    _, masks = evaluate('nonpersistent_history', history_images + [degraded],
                        [p, raw[1], p], edges, False, True)
    assert masks['quality_drop'][site].all() and not masks['persistent_history'][site].any()
    weak_history = p.copy()
    weak_history[site] = .55
    evaluate('persistent_but_below_strong_decision', history_images + [degraded],
             [shifted(weak_history, 2), shifted(weak_history, 1), p], edges, False, True)

    evaluate('two_frames_conservative_fallback', [history_images[1], degraded],
             [raw[1], p], [edges[1]], False, True)
    evaluate('old_edge_missing_recent_history_valid', history_images + [degraded],
             raw, [None, edges[1]], False, True)
    no_history, _ = evaluate('no_history_path', history_images + [degraded],
                             raw, [edges[0], None], False, True)
    first_frame, _ = evaluate('first_frame', [degraded], [p], [], False, True)
    assert np.array_equal(no_history, p) and np.array_equal(first_frame, p)

    current = p.copy()
    current[:, :3] = .32
    new_view, masks = evaluate('new_region_outside_history', history_images + [texture],
                               history_raw + [current], edges, False, False)
    assert np.array_equal(new_view[:, :3], current[:, :3])
    assert all(not mask[:, :3].any() for mask in masks.values())
    _, masks = evaluate('intermediate_view_outside', [texture, texture, degraded], [h, h, p],
                        [[-140, 0], [140, 0]], False, True)
    assert not masks['correspondence'][site].any()

    foreground = p.copy()
    foreground[site] = .75
    protected, _ = evaluate('current_foreground_protection', history_images + [degraded],
                            [p, p, foreground], edges, False, True)
    assert (protected[site] >= decision).all()
    strong_protected = v3.fuse([p, p, foreground], edges, False, v3.DEFAULTS)
    assert (strong_protected[site] == decision).all()

    # Use the real v3 stream state as the integration entry point.
    stream = v3.MotionFeedback()
    for index in range(4):
        degraded_frame = index == 3
        image = shifted(degraded if degraded_frame else texture, -index)
        rgb = np.repeat(np.rint(image * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        outputs, _ = stream.step(rgb, shifted(p if degraded_frame else h, -index), float(index))
        integrated, masks = conditional.fuse_conditional(stream.images, stream.raw, stream.edges['base'])
        expected = np.zeros(texture.shape, bool)
        expected[site] = degraded_frame
        chosen = masks['strong_correction']
        assert np.array_equal(chosen, shifted(expected, -index))
        assert np.array_equal(integrated, np.where(chosen, outputs['E_base_strong'],
                                                  outputs['B_base_conservative']))
        assert np.array_equal(stream.raw[-1], shifted(p if degraded_frame else h, -index))

    print(json.dumps(dict(status='PASS', cases=results, stream_frames=4), indent=2))


if __name__ == '__main__':
    main()
