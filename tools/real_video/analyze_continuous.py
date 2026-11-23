"""Use a frozen v3 motion trace and a shared raw cache for three causal comparisons."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import deque
from pathlib import Path

import numpy as np
import scipy
from PIL import Image, ImageDraw

SIZE = 256
PIXELS = SIZE**2
METHODS = ("single", "B", "conditional_v3")
PRED = ("foreground", "added_vs_single", "removed_vs_single")
DETAIL = ("conditional_added_vs_B", "conditional_removed_vs_B", "conditional_written_pixels",
          "conditional_changed_probability_pixels")
MASKS = ("correspondence", "persistent_history", "quality_drop", "historical_contrast")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as file:
        table = csv.DictWriter(file, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_prob(path):
    p = np.fromfile(str(path), dtype="<f4").reshape((SIZE,SIZE), order="F")
    assert np.isfinite(p).all() and float(p.min())>=0 and float(p.max())<=1
    return p


def read_rgb(path):
    with Image.open(str(path)) as image:
        rgb = np.asarray(image.convert("RGB"))
    assert rgb.shape == (SIZE,SIZE,3)
    return rgb


def count(mask):
    return int(np.count_nonzero(mask))


def pairs(a,b):
    return count(b&~a), count(a&~b)


def pred_stats(values):
    masks = [x.astype(float) >= .5 for x in values]
    records = {}
    for name, binary in zip(METHODS,masks):
        added, removed = pairs(masks[0],binary)
        records.update({name+"_foreground":count(binary),name+"_added_vs_single":added,
                        name+"_removed_vs_single":removed})
    records["conditional_added_vs_B"],records["conditional_removed_vs_B"]=pairs(masks[1],masks[2])
    return records


def write_prediction(path, p):
    path.write_bytes(p.astype("<f4").tobytes(order="F"))
    return sha(path)


def make_group(args, p, v3, cond, frames, edges, alias, stride):
    run = args.run
    directory = run / ("{}_stride{}".format(alias,stride))
    directory.mkdir()
    gray_history = deque(maxlen=3)
    raw_history = deque(maxlen=3)
    past_edges = deque(maxlen=2)
    output = []
    for image in frames[::stride]:
        video = image["source_video"]
        i = int(image["frame_index_zero_based"])
        edge = edges[(video,stride,i)]
        assert edge["decoder_time_s"] == image["decoder_time_s"]
        assert edge["previous_frame"] == (str(i-stride) if i else "")
        rgb = read_rgb(args.inputs/image["rgb_path"])
        raw = read_prob(run/image["deeplab_raw_path"])
        if i:
            step = [float(edge["dx"]),float(edge["dy"])] if edge["edge_status"]=="accepted" else None
            assert step is not None or (not edge["dx"] and not edge["dy"])
            past_edges.append(step)
        gray_history.append(v3.gray(rgb))
        raw_history.append(raw)
        a = list(raw_history)
        e = list(past_edges)
        base = v3.fuse(a,e,True,v3.DEFAULTS)
        corrected,masks = cond.fuse_conditional_v3(list(gray_history),a,e,
            minimum_ring_deviations=p["minimum_ring_deviations"])
        deduced_paths = sum(all(step is not None for step in e[j:]) for j in range(len(e)))
        assert deduced_paths == int(edge["available_history_paths"])
        changed = corrected!=base
        trigger = masks["strong_correction"]
        assert np.all(~changed|trigger) and np.all(~trigger|(raw<.5)) and np.all(~trigger|(corrected>=.5))
        if trigger.any():
            strong = v3.fuse(a,e,False,v3.DEFAULTS)
            assert np.array_equal(corrected[trigger],strong[trigger])
        else:
            assert not changed.any()
        if deduced_paths==0:
            assert np.array_equal(base,raw) and np.array_equal(corrected,raw)
        full_history = len(e)==2 and deduced_paths==2
        if not full_history:
            assert not trigger.any()
        record = dict(clip=alias, source_video=video, stride=stride, frame_index=i,
            previous_frame=edge["previous_frame"], decoder_time_s=edge["decoder_time_s"],
            edge_status=edge["edge_status"], peak_boundary=int(edge["peak_boundary"]), peak_ncc=edge["peak_ncc"],
            dx=edge["dx"], dy=edge["dy"], available_history_paths=deduced_paths,
            valid_any_history_pixels=int(edge["valid_any_history_pixels"]), full_history=int(full_history),
            rgb_path=image["rgb_path"], raw_path=image["deeplab_raw_path"])
        record.update(pred_stats((raw,base,corrected)))
        record.update(conditional_written_pixels=count(trigger),
                      conditional_changed_probability_pixels=count(changed))
        for name in MASKS:
            record[name+"_pixels"]=count(masks[name])
        assert record["conditional_removed_vs_B"]==0
        assert record["conditional_added_vs_B"]<=record["conditional_changed_probability_pixels"]<=record["conditional_written_pixels"]
        for name,values in zip(METHODS[1:],(base,corrected)):
            path=directory/("frame_{:06d}_{}.f32".format(i,name))
            record[name+"_path"]=path.relative_to(run).as_posix()
            record[name+"_sha256"]=write_prediction(path,values)
        path=directory/("frame_{:06d}_trigger.bits".format(i))
        path.write_bytes(np.packbits(trigger.ravel()).tobytes())
        record["trigger_path"]=path.relative_to(run).as_posix()
        record["trigger_sha256"]=sha(path)
        output.append(record)
    return output


def scope_info(rows):
    steps=rows[1:]
    return dict(target_frames=len(rows), target_pixels=len(rows)*PIXELS, tested_edges=len(steps),
                rejected_edges=sum(r["edge_status"]!="accepted" for r in steps),
                boundary_edges=sum(int(r["peak_boundary"]) for r in steps),
                fallback_targets=sum(int(r["available_history_paths"])==0 for r in steps),
                initial_resets=1)


def aggregate(rows, scope, model):
    core=scope_info(rows)
    out=[]
    for method in (METHODS if model=="DeepLabV3+" else METHODS[:2]):
        current={}
        for metric in PRED:
            field=method+"_"+metric
            current[metric]=sum(r[field] for r in rows)
        assert current["foreground"]==sum(r["single_foreground"] for r in rows)+current["added_vs_single"]-current["removed_vs_single"]
        extras={key:sum(r[key] for r in rows) if method=="conditional_v3" else "" for key in DETAIL}
        out.append(dict(clip=rows[0]["clip"], scope=scope, stride=rows[0]["stride"], model=model, method=method,
            **core,**current,foreground_fraction=current["foreground"]/core["target_pixels"],**extras,
            eligible_two_edge_targets=sum(r["full_history"] for r in rows) if method=="conditional_v3" else "",
            trigger_targets=sum(r["conditional_written_pixels"]>0 for r in rows) if method=="conditional_v3" else "",
            added_targets_vs_B=sum(r["conditional_added_vs_B"]>0 for r in rows) if method=="conditional_v3" else ""))
    return out


def unet_row(row, prior):
    output=dict(row)
    output["single_foreground"]=int(prior["raw_positive_pixels"])
    output["B_foreground"]=int(prior["fused_positive_pixels"])
    output["B_added_vs_single"]=int(prior["added_positive"])
    output["B_removed_vs_single"]=int(prior["removed_positive"])
    return output


def manual_panel(args, rows):
    reference = rows[0]
    max_change = max(rows, key=lambda row: row["conditional_added_vs_B"])
    max_write = max(rows, key=lambda row: row["conditional_written_pixels"])
    max_base = max(rows, key=lambda row: row["B_added_vs_single"]+row["B_removed_vs_single"])
    picks=[("Largest B change vs single",max_base)]
    picks.append(("Largest conditional addition vs B",max_change)
                 if max_change["conditional_added_vs_B"] else ("Most conditional writes",max_write))
    rejected=next((row for row in rows[1:] if row["edge_status"]!="accepted"),None)
    if rejected is not None:
        picks.append(("First rejected edge / fallback",rejected))
    sheet=Image.new("RGB",(5*256,292*len(picks)),"white")
    draw=ImageDraw.Draw(sheet)
    figure="{}_stride{}_examples.png".format(reference["clip"],reference["stride"])
    provenance=[]
    for n,(label,row) in enumerate(picks):
        y=n*292
        rgb=read_rgb(args.inputs/row["rgb_path"])
        arrays=[read_prob(args.run/row[name+"_path"]).astype(float)>=.5 for name in ("raw","B","conditional_v3")]
        masks=[[],[(arrays[0],[241,82,186])],[(arrays[1],[241,82,186])],
               [(arrays[2],[241,82,186])],[(arrays[2]&~arrays[1],[255,174,44])]]
        draw.text((5,y+2),"{} zero-based frame {} | stride {} | {}".format(
            row["clip"],row["frame_index"],row["stride"],label),fill="black")
        names=["Input RGB","Single >=0.50","B >=0.50","Conditional >=0.50","New foreground vs B (amber)"]
        for j,(overlay,caption) in enumerate(zip(masks,names)):
            tile=rgb.copy()
            for mask,color in overlay:
                tile[mask]=(tile[mask]*.38+.62*np.array(color)).astype("uint8")
            sheet.paste(Image.fromarray(tile),(256*j,y+17))
            draw.text((256*j+4,y+276),caption,fill="black")
        provenance.append(dict(figure=figure,panel_row=n+1,rule=label,clip=row["clip"],source_video=row["source_video"],
            frame_index=row["frame_index"],stride=row["stride"],rgb_path=row["rgb_path"],raw_path=row["raw_path"],
            B_path=row["B_path"],conditional_path=row["conditional_v3_path"],trigger_path=row["trigger_path"]))
    sheet.save(str(args.run/figure))
    return provenance


def main(args):
    p=read_json(args.run/"protocol.json")
    assert p["threshold"]==.5 and p["role"]=="unlabeled_real_video_development"
    for name,h in p["source_hashes"].items():
        assert sha(args.repo/"python"/name)==h
    settings=args.repo/"configs"/"algorithm_design_conditional_v3.json"
    assert sha(settings)==p["conditional_cfg_sha256"]
    p["minimum_ring_deviations"]=read_json(settings)["minimum_ring_deviations"]
    sys.path.insert(0,str(args.repo/"python"))
    import rail_motion_v3 as v3
    import rail_motion_conditional_v3 as cond
    assert v3.DEFAULTS["decision"]==p["threshold"]
    inference=read_json(args.run/"inference_status.json")
    assert inference["status"]=="PASS" and inference["frozen_state_and_parameters"]
    assert inference["model_sha256"]==p["checkpoint_sha256"] and inference["optimizer_updates"]==0
    raw_index=read_csv(args.run/"prediction_index.csv")
    raw_by_key={(r["source_video"],int(r["frame_index"])):r["raw_path"] for r in raw_index}
    sources=read_csv(args.inputs/"frame_input_manifest.csv")
    edge_rows=read_csv(args.inputs/"per_frame_diagnostics.csv")
    edges={(r["source_video"],int(r["stride"]),int(r["frame_index"])):r for r in edge_rows}
    assert len(raw_index)==len(raw_by_key)==len(sources)==int(inference["forward_calls"])==540
    assert len(edge_rows)==len(edges)==630
    data_rows=[]
    for item in sources:
        video,i=item["source_video"],int(item["frame_index_zero_based"])
        path=raw_by_key.pop((video,i))
        assert sha(args.inputs/item["rgb_path"])==item["rgb_sha256"]
        item.update(deeplab_raw_path=path,deeplab_raw_sha256=sha(args.run/path))
        data_rows.append(item)
    assert not raw_by_key
    write_csv(args.run/"frozen_input_probability_manifest.csv",data_rows)
    stage={"status":"PASS", "raw_maps":len(data_rows),
           "manifest_sha256":sha(args.run/"frozen_input_probability_manifest.csv"),
           "motion_trace_sha256":sha(args.inputs/"per_frame_diagnostics.csv")}
    (args.run/"pre_fusion_freeze.json").write_text(json.dumps(stage,indent=2),encoding="utf-8")
    aggregate_rows, frame_rows, pictures=[],[],[]
    matched={}
    for video,alias,strides in zip(p["clips"],p["clip_aliases"],p["strides"]):
        frames=[item for item in data_rows if item["source_video"]==video]
        assert [int(r["frame_index_zero_based"]) for r in frames]==list(range(180))
        assert np.all(np.diff([float(r["decoder_time_s"]) for r in frames])>0)
        for stride in strides:
            records=make_group(args,p,v3,cond,frames,edges,alias,stride)
            frame_rows.extend(records)
            pictures.extend(manual_panel(args,records))
            old=[unet_row(r,edges[(video,stride,r["frame_index"])]) for r in records]
            aggregate_rows.extend(aggregate(records,"full_group","DeepLabV3+"))
            aggregate_rows.extend(aggregate(old,"full_group","Small U-Net"))
            if len(strides)==2:
                matched[stride]=(records[::2] if stride==1 else records,
                                 old[::2] if stride==1 else old)
        print("DONE",alias,flush=True)
    assert len(frame_rows)==630
    matched_rows=[]
    for pairs_on_grid in zip(*matched[1],*matched[2]):
        a,ua,b,ub=pairs_on_grid
        assert a["frame_index"]==b["frame_index"] and a["raw_path"]==b["raw_path"]
        assert a["single_foreground"]==b["single_foreground"] and ua["single_foreground"]==ub["single_foreground"]
        line=dict(clip=a["clip"],frame_index=a["frame_index"],raw_path=a["raw_path"],
                  deeplab_single_foreground=a["single_foreground"],unet_single_foreground=ua["single_foreground"])
        for name,row in (("native",a),("sparse",b)):
            for key in ("previous_frame","edge_status","available_history_paths","B_foreground",
                        "conditional_v3_foreground","conditional_written_pixels","conditional_added_vs_B"):
                line[name+"_"+key]=row[key]
        matched_rows.append(line)
    assert len(matched_rows)==90
    for stride,(records,old) in sorted(matched.items()):
        aggregate_rows.extend(aggregate(records,"matched_even_targets","DeepLabV3+"))
        aggregate_rows.extend(aggregate(old,"matched_even_targets","Small U-Net"))
    write_csv(args.run/"per_frame_diagnostics.csv",frame_rows)
    write_csv(args.run/"matched_even_frames.csv",matched_rows)
    write_csv(args.run/"aggregate_continuous.csv",aggregate_rows)
    write_csv(args.run/"visual_manifest.csv",pictures)
    proof=dict(status="PASS", method_frames=3*len(frame_rows), input_maps=len(data_rows), comparisons=len(frame_rows),
               matched_targets=len(matched_rows), analysis_sha256=sha(Path(__file__)),
               protocol_sha256=sha(args.run/"protocol.json"), aggregate_sha256=sha(args.run/"aggregate_continuous.csv"),
               python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    (args.run/"analysis_status.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ("run","inputs","repo"):
        parser.add_argument("--"+option, type=Path, required=True)
    main(parser.parse_args())
