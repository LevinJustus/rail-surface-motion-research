"""Gate conditional v3 promotion on a local seed in the current raw prediction."""

import numpy as np
from scipy import ndimage

import rail_motion_conditional_v2 as v2
from rail_motion_conditional_v3 import fuse_conditional_v3
import rail_motion_v3 as motion


CURRENT_SEED_WINDOW = 5
CURRENT_SEED_THRESHOLD = 0.50


def compare_conditional_v3_v4(images, raw, base_edges, config=None, *,
                               minimum_ring_deviations=v2.MINIMUM_RING_DEVIATIONS):
    """Return ((v3_probability, v3_masks), (v4_probability, v4_masks)).

    Both branches use the same v3 evaluation of the grayscale frames, raw
    probabilities and adjacent forward motion edges, in oldest-first order.
    The v4 diagnostics preserve v3's evidence masks and add current_local_seed.
    Its strong_correction mask identifies only promotions retained by v4.
    """
    baseline, baseline_masks = fuse_conditional_v3(
        images, raw, base_edges, config, minimum_ring_deviations=minimum_ring_deviations)
    current = np.asarray(raw[-1], dtype=np.float32)
    local_seed = ndimage.maximum_filter(current >= CURRENT_SEED_THRESHOLD,
                                        size=CURRENT_SEED_WINDOW, mode='constant', cval=0)
    kept = baseline_masks['strong_correction'] & local_seed
    rejected = baseline_masks['strong_correction'] & ~local_seed
    output = baseline.copy()
    if rejected.any():
        conservative = motion.fuse([np.asarray(p, dtype=np.float32) for p in raw],
                                   base_edges, True, motion.DEFAULTS)
        output[rejected] = conservative[rejected]
    masks = dict(baseline_masks, current_local_seed=local_seed, strong_correction=kept)
    return (baseline, baseline_masks), (output, masks)


def fuse_conditional_v4(images, raw, base_edges, config=None, *,
                        minimum_ring_deviations=v2.MINIMUM_RING_DEVIATIONS):
    """Return the v4 probability map and diagnostic masks."""
    _, candidate = compare_conditional_v3_v4(
        images, raw, base_edges, config, minimum_ring_deviations=minimum_ring_deviations)
    return candidate
