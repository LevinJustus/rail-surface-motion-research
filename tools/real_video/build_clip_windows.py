"""Build a shuffled clip order and training windows without crossing source boundaries."""
import argparse
import csv
from pathlib import Path
import random


SPLITS = ("train", "development", "test", "unassigned")


def write_rows(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def build_windows(assets, split, seed, frames_per_window, step, stride):
    if split not in SPLITS or min(frames_per_window, step, stride) < 1:
        raise ValueError("Expected a known split and positive window settings")
    ids = set()
    group_splits = {}
    selected = []
    for row in assets:
        asset_id, group, partition = (row[k].strip() for k in ("asset_id", "group_id", "split"))
        length = int(row["frame_count"])
        if not asset_id or asset_id in ids or length < 1 or partition not in SPLITS:
            raise ValueError("Each asset needs a unique ID, positive frame count and valid split")
        ids.add(asset_id)
        if partition != "unassigned":
            if not group:
                raise ValueError("Assigned assets need a verified source group")
            if group_splits.setdefault(group, partition) != partition:
                raise ValueError("A source group crosses data splits")
        if partition == split:
            selected.append((asset_id, length))
    if not selected:
        raise ValueError("No assets in the requested split")
    selected.sort()
    random.Random(seed).shuffle(selected)
    offset = 0
    clips = []
    windows = []
    covered = (frames_per_window - 1) * stride + 1
    for clip, (asset, length) in enumerate(selected):
        clips.append((clip, asset, offset, offset + length - 1, 1))
        if split != "unassigned":
            for start in range(0, length - covered + 1, step):
                windows.append((clip, asset, start, start + covered - 1, stride, 1))
        offset += length
    return clips, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path, help="CSV columns: asset_id,group_id,split,frame_count")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--frames-per-window", type=int, default=3)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    with args.assets.open(encoding="utf-8-sig", newline="") as stream:
        clips, windows = build_windows(csv.DictReader(stream), args.split, args.seed, args.frames_per_window,
                                       args.step, args.stride)
    args.output.mkdir()
    write_rows(args.output / "clip_order.csv", ("clip_id", "asset_id", "output_start_zero_based",
               "output_end_zero_based", "reset_before_clip"), clips)
    write_rows(args.output / "windows.csv", ("clip_id", "asset_id", "source_start_zero_based",
               "source_end_zero_based", "stride", "reset_before_window"), windows)


if __name__ == "__main__":
    main()
