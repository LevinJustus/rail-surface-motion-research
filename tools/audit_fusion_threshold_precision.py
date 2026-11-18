"""Verify frozen training threshold choices under explicit float64 comparison.

Post-audit numerical verification only: never changes the 24 frozen thresholds,
never reads validation/test masks, and never loads or re-runs a network.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def audit(root,out):
    assert not out.exists(),'No overwrite';out.mkdir(parents=True);started=time.perf_counter()
    (out/'audit_fusion_threshold_precision.py').write_bytes(Path(__file__).read_bytes())
    protocol=read(root/'configs/fusion_protocol_v1.json');calibration=root/'configs/fusion_calibration_v1.json'
    original_hash=sha(calibration);assert original_hash=='66a21cb635398bf953a1b7e1b2256a7cf20a1f05ca08330ed4d087b47c5aa9a6'
    frozen=read(calibration);ts=np.array(protocol['calibration']['thresholds'],dtype=np.float64)
    methods=['S','U','V','P','GV','GVP','GC','R'];ids=protocol['data']['contract_and_calibration_sequence_ids']
    selected={r['sequence_id']:r for r in read(root/protocol['data']['selection_file'])}
    assert all(selected[i]['assigned_split']=='training' for i in ids)
    sources=sorted({selected[i]['source_id'] for i in ids});assert len(sources)==3
    expected={r['path']:r['sha256'] for r in read(root/'runs/step12_full_v2_01/evaluation_only/payload_manifest.json')}
    all_curves=[];choices=[];different_count_rows=0;maximum_count_difference=0;maximum_score_difference=0.
    setups=[(20261114,'fusion_training_v1_02','fusion_training_independent_audit_v1_03'),
            (20261115,'fusion_calibration_seed20261115_v1_01','fusion_calibration_audit_seed20261115_v1_01'),
            (20261116,'fusion_calibration_seed20261116_v1_01','fusion_calibration_audit_seed20261116_v1_01')]
    for seed,run_id,old_audit in setups:
        run=root/'runs'/run_id;manifest=read(run/'prediction_manifest.json')
        outputs={(r['sequence_id'],r['frame'],r['method']):r for r in manifest['fused']}
        counts=np.zeros((8,3,19,4),np.int64)
        for i in ids:
            si=sources.index(selected[i]['source_id'])
            for k in range(2,10):
                mp=f'evaluation_only/sequences/{i}/mask_{k:03d}.png';mask=root/'runs/step12_full_v2_01'/mp
                assert sha(mask)==expected[mp]
                with Image.open(mask) as im:t=np.array(im.convert('L'))>=128
                fg=int(t.sum());bg=t.size-fg
                for mi,m in enumerate(methods):
                    entry=outputs[i,k,m];p=run/entry['path'];assert sha(p)==entry['sha256']
                    values=np.fromfile(str(p),dtype='<f4').reshape((256,256),order='F').astype(np.float64)
                    bin_index=np.searchsorted(ts,values,side='right')
                    fgh=np.bincount(bin_index[t],minlength=20);bgh=np.bincount(bin_index[~t],minlength=20)
                    tp=np.cumsum(fgh[::-1])[::-1][1:];fp=np.cumsum(bgh[::-1])[::-1][1:]
                    counts[mi,si]+=np.stack([tp,fp,fg-tp,bg-fp],axis=1)
        with (root/'runs'/old_audit/'training_threshold_curves.csv').open(encoding='utf-8-sig',newline='') as f:
            legacy={(r['method'],float(r['threshold'])):r for r in csv.DictReader(f)}
        for mi,m in enumerate(methods):
            tp,fp,fn=[counts[mi,:,:,j].astype(float) for j in range(3)];den=tp+fp+fn
            macro=np.divide(tp,den,out=np.ones_like(tp),where=den>0).mean(0)
            candidates=np.flatnonzero(np.abs(macro-macro.max())<=1e-12)
            chosen=sorted(candidates,key=lambda j:(abs(ts[j]-.5),-ts[j]))[0]
            original=next(r for r in frozen['choices'] if r['seed']==seed and r['method']==m)
            choices.append(dict(seed=seed,method=m,frozen_threshold=original['threshold'],float64_optimum_threshold=float(ts[chosen]),
                unchanged=bool(original['threshold']==float(ts[chosen])),float64_training_macro_IoU=float(macro[chosen])))
            for j,threshold in enumerate(ts):
                c=counts[mi,:,j].sum(0);old=legacy[m,float(threshold)]
                delta=max(abs(int(c[k])-int(old[name])) for k,name in enumerate(['TP','FP','FN','TN']))
                maximum_count_difference=max(maximum_count_difference,delta);different_count_rows+=int(delta>0)
                maximum_score_difference=max(maximum_score_difference,abs(float(macro[j])-float(old['source_macro_IoU'])))
                all_curves.append(dict(seed=seed,method=m,threshold=float(threshold),source_macro_IoU=float(macro[j]),
                    TP=int(c[0]),FP=int(c[1]),FN=int(c[2]),TN=int(c[3]),maximum_legacy_count_difference=delta))
        print(f'THRESHOLD_PRECISION seed={seed} checked=8 methods x 19 thresholds',flush=True)
    assert sha(calibration)==original_hash
    for name,data in [('training_curves_float64.csv',all_curves),('threshold_choices_check.csv',choices)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result=dict(status='PASS_FROZEN_THRESHOLD_CHOICES_UNCHANGED' if all(r['unchanged'] for r in choices) else 'FROZEN_THRESHOLD_SELECTION_PRECISION_DEVIATION',
        choices_checked=24,choices_unchanged=sum(r['unchanged'] for r in choices),training_curve_rows_checked=456,
        legacy_curve_rows_with_count_changes=different_count_rows,maximum_legacy_count_difference=maximum_count_difference,
        maximum_legacy_source_macro_IoU_difference=maximum_score_difference,
        comparison='float32 stored values promoted to float64 compared with unchanged float64 thresholds',
        calibration_sha256=original_hash,thresholds_modified=False,validation_or_test_masks_opened=0,
        network_forward_calls=0,optimizer_updates=0,post_validation_numerical_audit=True,
        numpy_version=np.__version__,wall_seconds=time.perf_counter()-started)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.project_root.resolve(),a.out.resolve())
