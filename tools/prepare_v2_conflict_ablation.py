"""Freeze the user-approved E4 single-factor ablation before generating outputs."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    original=read(ROOT/'runs/v2_screening_stage_20261119_01/artifact_manifest.json')['sha256']
    e3=ROOT/'runs/v2_screening_prediction_v1_01'
    assert read(e3/'summary.json')['status']=='PASS_E3_PREDICTION_PENDING_EVALUATION'
    selection=read(e3/'frozen_plan.json')
    core=['matlab/'+n for n in ['v2_select_motion.m','v2_motion_weights.m','v2_compose_motion.m','v2_history_evidence.m','v2_fuse_window.m','v2_compensation_step.m']]
    for p in core+['configs/algorithm_design_v2.json']:assert sha(ROOT/p)==original[p]
    rows=read(e3/'predictions.json')
    assert len(rows)==756
    for r in rows:assert sha(ROOT/r['path'])==r['sha256']
    sources=['tools/prepare_v2_conflict_ablation.py','tools/evaluate_v2_conflict_ablation.py','matlab/v2_no_conflict_fuse_window.m','matlab/check_v2_conflict_ablation.m','matlab/run_v2_conflict_ablation.m','matlab/count_v2_conflict_ablation.m']
    paths=core+sources+['matlab/warp_motion_history.m','matlab/fusion_binary_counts.m','configs/algorithm_design_v2.json','configs/v2_screening_v1.json','tools/evaluate_v2_screening.py','tools/audit_v2_minimal.py','tools/audit_fusion_training.py','runs/v2_screening_prediction_v1_01/predictions.json','runs/v2_screening_prediction_v1_01/frozen_plan.json','runs/v2_screening_evaluation_v1_01/per_frame_counts.csv']
    for s in selection['selection']:
        for ext in ['mat','json']:paths.append('runs/v2_screening_prediction_v1_01/registration/'+s['sequence_id']+'.'+ext)
    frozen=[dict(path=p,sha256=sha(ROOT/p)) for p in paths]
    plan=dict(version='E4-conflict-only-v1',date_local='2026-11-19',created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User confirmed the proposed single-factor ablation; no feedback, retraining, parameter tuning or publication.',
        base_prediction_run='v2_screening_prediction_v1_01',design_config=selection['design_config'],design_sha256=selection['design_sha256'],
        seed=selection['seed'],checkpoint=selection['checkpoint'],selection=selection['selection'],threshold=.5,primary_frames=[2,9],
        methods=selection['methods']+['V2_no_conflict'],risk_change='q*variance + unknown; omit only q*(historical_mean-current)^2',
        unchanged=['candidate sets and weights','unknown mass','risk_sigma','window','current-foreground protection','threshold','raw cache','source selection'],
        decision='Compare source and pooled IoU, correction/new-error counts, opportunity recovery; no significance claim from 3 sources. Do not select or tune with validation/test.',
        frozen_files=frozen,scope='12 correlated synthetic training-source sequences from 3 source proxies; not real video or final test')
    stage=ROOT/'runs/v2_conflict_ablation_stage_20261119_01';stage.mkdir(exist_ok=False);(stage/'before').mkdir()
    for p in ['README.md','WORK_PLAN.md','WORK_LOG.md','AGENTS.md']:shutil.copy2(ROOT/p,stage/'before'/p)
    for p in core+sources:
        dest=stage/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    target=ROOT/'configs/v2_conflict_ablation_v1.json';write(target,plan)
    (ROOT/'configs/v2_conflict_ablation_v1.sha256').write_text(sha(target)+'\n',encoding='ascii')
    write(stage/'freeze.json',dict(status='FROZEN_BEFORE_E4_PREDICTION',plan_sha256=sha(target),source_count=len(sources),frozen_files=len(frozen),old_core_sha256={p:sha(ROOT/p) for p in core},new_mask_or_geometry_payloads_opened=0))
    print(json.dumps(dict(status='E4_PLAN_FROZEN',sha256=sha(target),sequences=12,sources=3)))

if __name__=='__main__':main()
