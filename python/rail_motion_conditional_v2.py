"""Add same-polarity historical contrast to conditional v1's strong correction."""

import numpy as np

import rail_motion_conditional as v1
import rail_motion_v3 as v3


MINIMUM_RING_DEVIATIONS = 1.0


def _signed_contrast(image, valid):
    count = np.maximum(np.rint(v1._ring_sum(valid.astype(float))), 1)
    values = np.where(valid, image, 0.)
    mean = v1._ring_sum(values) / count
    variance = np.maximum(v1._ring_sum(values ** 2) / count - mean ** 2, 0.)
    deviation = np.sqrt(variance)
    return np.divide(image - mean, deviation, out=np.zeros_like(image), where=valid & (deviation > 0))


def fuse_conditional_v2(images, raw, base_edges, config=None, *, minimum_ring_deviations=MINIMUM_RING_DEVIATIONS):
    """Reuse v1 inputs and return its masks plus v2's historical contrast check.

    The extra mask is evaluated only at v1's strong-correction pixels. The
    returned strong_correction mask identifies the corrections retained by v2.
    """
    if not np.isfinite(minimum_ring_deviations) or minimum_ring_deviations <= 0:
        raise ValueError('minimum_ring_deviations must be finite and positive')
    output, masks = v1.fuse_conditional(images, raw, base_edges, config)
    candidate = masks['strong_correction']
    masks['historical_contrast'] = np.zeros_like(candidate)
    if not candidate.any():
        return output, masks

    older, older_valid = v3.warp(images[0], base_edges)
    previous, previous_valid = v3.warp(images[1], base_edges[1:])
    shared_view = older_valid & previous_valid
    older_contrast = _signed_contrast(older, shared_view)
    previous_contrast = _signed_contrast(previous, shared_view)
    lower = np.minimum(older_contrast, previous_contrast)
    upper = np.maximum(older_contrast, previous_contrast)
    supported = candidate & ((lower >= minimum_ring_deviations) | (upper <= -minimum_ring_deviations))
    rejected = candidate & ~supported
    if rejected.any():
        conservative = v3.fuse([np.asarray(p, dtype=np.float32) for p in raw], base_edges, True, v3.DEFAULTS)
        output[rejected] = conservative[rejected]
    masks['historical_contrast'] = supported
    masks['strong_correction'] = supported
    return output, masks
