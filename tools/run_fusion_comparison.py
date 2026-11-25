"""Run the five frozen holdout fusion methods on one ordered sequence."""

import argparse
from collections import deque
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import rail_motion_v3 as motion
from rail_motion_conditional_v4 import fuse_conditional_v4


METHODS = ("S", "U", "A", "B", "v4")


def compare(rgb, raw_probability):
    if (rgb.ndim != 4 or rgb.shape[1:] != (256, 256, 3)
            or rgb.dtype != np.uint8 or len(rgb) == 0):
        raise ValueError("rgb must be uint8 with shape (T,256,256,3), T >= 1")
    if (raw_probability.shape != rgb.shape[:3]
            or raw_probability.dtype.kind != "f"
            or not np.isfinite(raw_probability).all()
            or raw_probability.min() < 0 or raw_probability.max() > 1):
        raise ValueError("raw_probability must be floating point in [0,1], shape (T,256,256)")

    config = motion.DEFAULTS
    output = {name: np.empty(rgb.shape[:3], dtype=np.float32) for name in METHODS}
    displacement = np.full((len(rgb), 2), np.nan, dtype=np.float64)
    images, raw, edges = deque(maxlen=3), deque(maxlen=3), deque(maxlen=2)
    for t, (frame, probability) in enumerate(zip(rgb, raw_probability)):
        current = motion.gray(frame)
        if images:
            candidates = motion.motion_hypotheses(images[-1], current, None, config)
            edge = candidates[0]["xy"] if candidates else None
            edges.append(edge)
            if edge is not None:
                displacement[t] = edge
        images.append(current)
        raw.append(probability.astype(np.float32))
        frames, probabilities, estimated = list(images), list(raw), list(edges)
        zero = [(0., 0.)] * len(estimated)
        output["S"][t] = probabilities[-1]
        output["U"][t] = motion.fuse(probabilities, zero, False, config)
        output["A"][t] = motion.fuse(probabilities, estimated, False, config)
        output["B"][t] = motion.fuse(probabilities, estimated, True, config)
        output["v4"][t] = fuse_conditional_v4(frames, probabilities, estimated)[0]
    output["displacement_xy"] = displacement
    output["threshold"] = np.float32(config["decision"])
    return output


def demo_inputs():
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, size=(256, 256, 3)).astype(np.uint8)
    logits = np.full((256, 256), 0.1, dtype=np.float32)
    logits[114:142, 110:126] = 0.9
    rgb = np.stack([np.roll(image, 4 * k, axis=1) for k in range(3)])
    raw = np.stack([np.roll(logits, 4 * k, axis=1) for k in range(3)])
    raw[2] *= 0.25
    return rgb, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="NPZ containing rgb and raw_probability")
    source.add_argument("--demo", action="store_true", help="use a built-in synthetic sequence")
    parser.add_argument("--output", type=Path, required=True, help="new output NPZ file")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz" or args.output.exists():
        parser.error("--output must name a new .npz file")
    try:
        if args.demo:
            result = compare(*demo_inputs())
        else:
            with np.load(str(args.input), allow_pickle=False) as archive:
                result = compare(archive["rgb"], archive["raw_probability"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, **result)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    accepted = np.isfinite(result["displacement_xy"][:, 0]).sum()
    print("Wrote {} frames, {} accepted edges: {}".format(len(result["S"]), accepted, args.output))


if __name__ == "__main__":
    main()
