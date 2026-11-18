"""Freeze the 24 training-only thresholds before C4 validation inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def freeze(root,out):
    assert not out.exists(),'No overwrite'
    protocol=root/'configs/fusion_protocol_v1.json'
    assert sha(protocol)=='20fd3a8839131d294d64053d2b9cde0b4adf6ef837482f2d4604cb22d37cb6c1'
    for seed in [20261114,20261115,20261116]:
        assert not (root/'runs'/f'fusion_validation_seed{seed}_v1_01').exists(),'Validation started before calibration freeze'
    c3=root/'runs/fusion_training_independent_audit_v1_03'
    assert read(c3/'audit.json')['status']=='PASS_13C3_TRAINING_IMPLEMENTATION_AUDIT'
    rows=[];inputs=[]
    for seed,folder in [(20261114,c3),(20261115,root/'runs/fusion_calibration_audit_seed20261115_v1_01'),
                        (20261116,root/'runs/fusion_calibration_audit_seed20261116_v1_01')]:
        if seed!=20261114:
            audit=read(folder/'audit.json')
            assert audit['status']=='PASS_C4_FROZEN_INDEPENDENT_AUDIT' and audit['mode']=='training' and audit['seed']==seed
        path=folder/f'thresholds_seed{seed}.json';record=read(path)
        assert record['status']=='FROZEN_TRAINING_IN_SAMPLE_REFERENCE' and record['seed']==seed
        assert {r['method'] for r in record['choices']}=={'S','U','V','P','GV','GVP','GC','R'}
        assert all(r['seed']==seed and r['threshold'] in read(protocol)['calibration']['thresholds'] for r in record['choices'])
        rows.extend(record['choices']);inputs.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path),audit_sha256=sha(folder/'audit.json')))
    constant=root/'runs/fusion_training_v1_02/constant_control.json'
    value=read(constant)['value'];assert abs(value-.9597680584784836)<1e-15
    execution_sources=['matlab/run_fusion_frozen_inference.m','matlab/evaluate_fusion_frozen_counts.m',
        'tools/audit_fusion_frozen.py','tools/summarize_fusion_validation.py','tools/freeze_fusion_calibration.py']
    core=read(root/'docs/evidence/step13c3/source_manifest.json')['sha256']
    for path,expected in core.items():assert sha(root/path)==expected,path
    record=dict(status='FROZEN_THREE_SEED_TRAINING_CALIBRATION',created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(protocol),constant=value,constant_record_sha256=sha(constant),choices=rows,
        input_records=inputs,source_groups=3,training_sequences=36,formal_validation_payloads_opened_before_freeze=0,
        threshold_rule='19-point training source macro IoU, frames 2-9; tie nearest .5 then higher threshold',
        validation_threshold_selection_allowed=False,algorithm_parameter_search_budget=0,
        inference_sources_sha256={p:sha(root/p) for p in execution_sources},
        immutable_core_sha256=core,
        scope='Training-in-sample calibration only; not validation or test performance')
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    out.with_suffix('.sha256').write_text(sha(out)+'  '+out.name+'\n',encoding='ascii')
    print(json.dumps(dict(status=record['status'],threshold_choices=len(rows),sha256=sha(out),created_utc=record['created_utc'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();freeze(a.project_root.resolve(),a.out.resolve())
