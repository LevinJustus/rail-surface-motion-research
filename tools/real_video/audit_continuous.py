"""Recount saved masks and independently replay a prefix through v3's stream API."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PIXELS = 256 ** 2


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    result=np.fromfile(str(path),dtype="<f4").reshape((256,256),order="F")
    assert np.isfinite(result).all() and float(result.min())>=0 and float(result.max())<=1
    return result


def main(args):
    sys.path.insert(0,str(args.repo/"python"))
    from rail_motion_v3 import MotionFeedback
    root=args.run
    freeze=json.loads((root/"pre_fusion_freeze.json").read_text())
    prior_status=json.loads((args.inputs/"analysis_summary.json").read_text())
    assert freeze["motion_trace_sha256"]==prior_status["diagnostics_sha256"]
    input_file=root/"frozen_input_probability_manifest.csv"
    assert sha(input_file)==freeze["manifest_sha256"]
    inputs={}
    for r in rows(input_file):
        key=(r["source_video"],int(r["frame_index_zero_based"]))
        inputs[key]=r
        assert sha(args.inputs/r["rgb_path"])==r["rgb_sha256"]
        assert sha(root/r["deeplab_raw_path"])==r["deeplab_raw_sha256"]
    assert len(inputs)==540
    paired_checks=0
    for r in rows(args.paired/"prediction_manifest.csv"):
        if r["condition"] != "deeplab14_320":
            continue
        item=inputs[r["video"],int(r["frame"])]
        with Image.open(str(args.inputs/item["rgb_path"])) as a, Image.open(str(args.paired/r["input_path"])) as b:
            assert np.array_equal(np.asarray(a),np.asarray(b))
        assert sha(args.paired/r["probability_path"])==r["probability_sha256"]
        assert np.array_equal(load(args.paired/r["probability_path"]),load(root/item["deeplab_raw_path"]))
        paired_checks+=1
    assert paired_checks==12
    source_data={}
    prefix={}
    repeated,matches,exact_fallback=0,0,0
    diagnostics=rows(root/"per_frame_diagnostics.csv")
    for r in diagnostics:
        i=int(r["frame_index"])
        stride=int(r["stride"])
        clip=r["clip"]
        tag=(clip,stride,i)
        assert tag not in source_data
        source_data[tag]=r
        source=inputs[r["source_video"],i]
        assert r["raw_path"]==source["deeplab_raw_path"] and r["rgb_path"]==source["rgb_path"]
        rp,bp,cp=(root/r[k] for k in ("raw_path","B_path","conditional_v3_path"))
        trigger=root/r["trigger_path"]
        for p,name in ((bp,"B"),(cp,"conditional_v3"),(trigger,"trigger")):
            assert sha(p)==r[name+"_sha256"]
        a,b,c=(load(p) for p in (rp,bp,cp))
        ma,mb,mc=(p.astype(float)>=.5 for p in (a,b,c))
        mask=np.unpackbits(np.fromfile(str(trigger),dtype="uint8")).reshape((256,256)).astype(bool)
        assert int(mask.sum())==int(r["conditional_written_pixels"])
        assert np.all((b==c)|mask)
        assert np.all(~mask|((a<.5)&(c>=.5)))
        for name,d in (("single",ma),("B",mb),("conditional_v3",mc)):
            added=d.astype(int)-ma
            for suffix,value in (("foreground",d.sum()),("added_vs_single",(added==1).sum()),("removed_vs_single",(added==-1).sum())):
                assert int(value)==int(r[name+"_"+suffix])
        d=mc.astype(int)-mb
        assert (d==-1).sum()==int(r["conditional_removed_vs_B"])==0
        assert (d==1).sum()==int(r["conditional_added_vs_B"])
        assert int((b!=c).sum())==int(r["conditional_changed_probability_pixels"])
        if int(r["available_history_paths"])==0:
            assert np.array_equal(a,b) and np.array_equal(a,c)
            exact_fallback+=1
        if i<=60:
            group=(clip,stride)
            if group not in prefix:
                prefix[group]=MotionFeedback()
            with Image.open(str(args.inputs/r["rgb_path"])) as im:
                rgb=np.asarray(im.convert("RGB"))
            estimate,info=prefix[group].step(rgb,a,float(r["decoder_time_s"]))
            expected=None if not r["dx"] else [float(r["dx"]),float(r["dy"])]
            assert info["base"]==expected and np.array_equal(estimate["B_base_conservative"],b)
            repeated+=1
    assert len(source_data)==630
    for r in rows(root/"matched_even_frames.csv"):
        i=int(r["frame_index"])
        a=source_data[r["clip"],1,i]
        b=source_data[r["clip"],2,i]
        assert i%2==0 and a["raw_path"]==b["raw_path"]==r["raw_path"]
        assert a["single_foreground"]==b["single_foreground"]==r["deeplab_single_foreground"]
        matches+=1
    assert matches==90
    aggregate=rows(root/"aggregate_continuous.csv")
    assert len(aggregate)==30
    for row in aggregate:
        i=row["method"]
        stride=int(row["stride"])
        group=[r for (clip,step,frame),r in source_data.items() if clip==row["clip"] and step==stride and
               (row["scope"]=="full_group" or frame%2==0)]
        assert int(row["target_frames"])==len(group)
        assert int(row["target_pixels"])==PIXELS*len(group)
        assert math.isclose(float(row["foreground_fraction"]),int(row["foreground"])/(PIXELS*len(group)),abs_tol=1e-15)
        if row["model"]=="DeepLabV3+":
            for column in ("foreground","added_vs_single","removed_vs_single"):
                assert int(row[column])==sum(int(r[i+"_"+column]) for r in group)
    pictures={}
    for r in rows(root/"visual_manifest.csv"):
        path=root/r["figure"]
        if r["figure"] not in pictures:
            with Image.open(str(path)) as im:
                assert im.width==1280 and im.height in (584,876)
                im.verify()
            pictures[r["figure"]]=sha(path)
    report=dict(status="PASS",input_hashes=540,raw_hashes=540,paired_deeplab_replays=paired_checks,
                fused_probability_hashes=2*len(diagnostics),trigger_hashes=len(diagnostics),
                per_target_count_rechecks=len(diagnostics),exact_fallback_or_reset_targets=exact_fallback,
                v3_stream_replay_targets=repeated,matched_targets=matches,aggregate_rows=len(aggregate),figures=pictures,
                audit_script_sha256=sha(Path(__file__)))
    with (root/"audit.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(report)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("run","inputs","repo","paired"):
        ap.add_argument("--"+key,type=Path,required=True)
    main(ap.parse_args())
