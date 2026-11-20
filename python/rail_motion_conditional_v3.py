"""Single-factor ablation of the conditional quality veto."""

import numpy as np

import rail_motion_conditional as v1
import rail_motion_conditional_v2 as v2
import rail_motion_v3 as v3


def fuse_conditional_v3(images, raw, base_edges, config=None, *, minimum_ring_deviations=v2.MINIMUM_RING_DEVIATIONS):
    """Keep v2's motion, persistence and image evidence; log the v1 quality mask."""
    if not np.isfinite(minimum_ring_deviations) or minimum_ring_deviations <= 0:
        raise ValueError('minimum_ring_deviations must be finite and positive')
    output, masks = v1.fuse_conditional(images, raw, base_edges, config)
    if masks['strong_correction'].any():
        output = v3.fuse([np.asarray(p, dtype=np.float32) for p in raw], base_edges, True, v3.DEFAULTS)
    masks['strong_correction'] = np.zeros_like(masks['quality_drop'])
    masks['historical_contrast'] = np.zeros_like(masks['quality_drop'])
    raw = [np.asarray(p, dtype=np.float32) for p in raw]
    decision = v3.DEFAULTS['decision']
    eligible = (raw[-1] < decision) & masks['correspondence'] & masks['persistent_history']
    if not eligible.any():
        return output, masks

    strong = v3.fuse(raw, base_edges, False, v3.DEFAULTS)
    eligible &= strong >= decision
    if not eligible.any():
        return output, masks

    older, older_valid = v3.warp(images[0], base_edges)
    previous, previous_valid = v3.warp(images[1], base_edges[1:])
    shared_view = older_valid & previous_valid
    older_contrast = v2._signed_contrast(older, shared_view)
    previous_contrast = v2._signed_contrast(previous, shared_view)
    lower = np.minimum(older_contrast, previous_contrast)
    upper = np.maximum(older_contrast, previous_contrast)
    supported = eligible & ((lower >= minimum_ring_deviations) | (upper <= -minimum_ring_deviations))
    masks['historical_contrast'] = supported
    masks['strong_correction'] = supported
    output[supported] = strong[supported]
    return output, masks
