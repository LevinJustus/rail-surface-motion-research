"""Conditional foreground recovery using the frozen v3 base motion path."""

import numpy as np
from scipy import ndimage

import rail_motion_v3 as v3


DEFAULTS = dict(minimum_history_detail=0.02, maximum_detail_ratio=0.5,
                minimum_ring_overlap=0.8, minimum_ring_std=0.02,
                minimum_ring_ncc=0.8)
LOCAL_SIZE = 9
RING_SIZE = 21
LOCAL_AREA = LOCAL_SIZE ** 2
RING_AREA = RING_SIZE ** 2 - LOCAL_AREA


def _box_sum(values, size):
    return ndimage.uniform_filter(values, size=size, mode='constant') * size ** 2


def _ring_sum(values):
    return _box_sum(values, RING_SIZE) - _box_sum(values, LOCAL_SIZE)


def _ring_correspondence(first, second, valid, cfg):
    count = np.rint(_ring_sum(valid.astype(float)))
    divisor = np.maximum(count, 1)
    first = np.where(valid, first, 0.)
    second = np.where(valid, second, 0.)
    first_mean = _ring_sum(first) / divisor
    second_mean = _ring_sum(second) / divisor
    first_var = _ring_sum(first ** 2) / divisor - first_mean ** 2
    second_var = _ring_sum(second ** 2) / divisor - second_mean ** 2
    covariance = _ring_sum(first * second) / divisor - first_mean * second_mean
    min_variance = cfg['minimum_ring_std'] ** 2
    textured = (first_var >= min_variance) & (second_var >= min_variance)
    denominator = np.sqrt(np.maximum(first_var, 0.) * np.maximum(second_var, 0.))
    ncc = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=textured)
    return ((count >= cfg['minimum_ring_overlap'] * RING_AREA) & textured
            & (ncc >= cfg['minimum_ring_ncc']))


def _local_detail(image):
    dx = ndimage.sobel(image, axis=1) / 8.
    dy = ndimage.sobel(image, axis=0) / 8.
    return np.hypot(dx, dy)


def fuse_conditional(images, raw, base_edges, config=None):
    """Fuse one to three grayscale [0,1] images and raw probability maps.

    Inputs are oldest first and have v3's 256x256 shape. Edges are forward
    [dx, dy] displacements between adjacent frames, or None if unavailable.
    Returns a float32 map and four boolean masks in current-frame coordinates.
    """
    cfg = DEFAULTS.copy()
    if config is not None:
        if set(config) - set(DEFAULTS):
            raise ValueError('unknown conditional fusion parameter')
        cfg.update(config)
    if any(not np.isfinite(value) or not 0 < value <= 1 for value in cfg.values()):
        raise ValueError('conditional fusion parameters must be in (0, 1]')

    if not 1 <= len(raw) <= 3 or len(images) != len(raw) or len(base_edges) != len(raw) - 1:
        raise ValueError('expected one to three paired frames and one edge per adjacent pair')
    images = [np.asarray(image, dtype=float) for image in images]
    raw = [np.asarray(probability, dtype=float) for probability in raw]
    for array in images + raw:
        if array.shape != (256, 256) or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError('frames must be finite 256x256 arrays in [0, 1]')
    for edge in base_edges:
        if edge is not None and (np.shape(edge) != (2,) or not np.isfinite(edge).all()):
            raise ValueError('each motion edge must be a finite [dx, dy] pair or None')

    raw = [probability.astype(np.float32) for probability in raw]
    current = raw[-1]
    output = v3.fuse(raw, base_edges, True, v3.DEFAULTS)
    masks = {name: np.zeros_like(current, dtype=bool) for name in
             ('quality_drop', 'correspondence', 'persistent_history', 'strong_correction')}
    if len(raw) < 3 or any(edge is None for edge in base_edges):
        return output, masks

    paths = (base_edges, base_edges[1:])
    older, older_valid = v3.warp(images[0], paths[0])
    previous, previous_valid = v3.warp(images[1], paths[1])
    common = older_valid & previous_valid
    masks['correspondence'] = (common
        & _ring_correspondence(older, previous, common, cfg)
        & _ring_correspondence(previous, images[-1], previous_valid, cfg))

    # Compute gradients before warping so invalid borders do not create edges.
    detail = [_local_detail(image) for image in images]
    older_detail, _ = v3.warp(detail[0], paths[0])
    previous_detail, _ = v3.warp(detail[1], paths[1])
    history_detail = np.minimum(_box_sum(older_detail, LOCAL_SIZE),
                                _box_sum(previous_detail, LOCAL_SIZE)) / LOCAL_AREA
    current_detail = _box_sum(detail[-1], LOCAL_SIZE) / LOCAL_AREA
    full_local_view = np.rint(_box_sum(common.astype(float), LOCAL_SIZE)) == LOCAL_AREA
    masks['quality_drop'] = (full_local_view
        & (history_detail >= cfg['minimum_history_detail'])
        & (current_detail <= cfg['maximum_detail_ratio'] * history_detail))

    older_probability, _ = v3.warp(raw[0], paths[0])
    previous_probability, _ = v3.warp(raw[1], paths[1])
    history_threshold = v3.DEFAULTS['history_threshold']
    masks['persistent_history'] = (common & (older_probability >= history_threshold)
                                   & (previous_probability >= history_threshold))

    decision = v3.DEFAULTS['decision']
    eligible = ((current < decision) & masks['quality_drop']
                & masks['correspondence'] & masks['persistent_history'])
    if eligible.any():
        strong = v3.fuse(raw, base_edges, False, v3.DEFAULTS)
        masks['strong_correction'] = eligible & (strong >= decision)
        output[masks['strong_correction']] = strong[masks['strong_correction']]
    return output, masks
