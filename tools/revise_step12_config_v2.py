"""Explicit pre-execution correction; keep original frozen v1 unchanged."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=ROOT/'configs/motion_sequences_v1.json';new=ROOT/'configs/motion_sequences_v2.json'
    assert sha(old)=='03fe30642d2b01f97f1ea2c0a176b0c1c75a29cb3c64640be65b0b01ca6458bf'
    assert not new.exists()
    cfg=json.loads(old.read_text(encoding='utf-8'));cfg['plan_id']='controlled_motion_v2'
    for p in cfg['profiles']:
        if p['name']=='odom_quantization':p['quantization_px']=3
    cfg['lighting_rule']='t is zero-based FRAME INDEX (not seconds). '+cfg['lighting_rule']
    cfg['revision']={'supersedes_config':old.name,'sha256':sha(old),
                     'reason':'Independent pre-execution audit: 4px quantization resonates with 8px/frame and removes all quantization error; use 3px. Clarify lighting time unit. No raw images opened or sequence run under v1.'}
    new.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    new.with_suffix('.sha256').write_text(sha(new)+'  '+new.name+'\n',encoding='ascii')
    out=ROOT/'docs/evidence/step12';out.mkdir(exist_ok=True,parents=True)
    report={'state':'PRE_EXECUTION_CONFIG_CORRECTION','old_config_sha256':sha(old),'new_config_sha256':sha(new),
            'v1_odom_quantization_px':4,'v2_odom_quantization_px':3,
            'true_offsets_px':[0,8,16,24,32,40,48,56,64],
            'v1_observed_offsets_px':[0,8,16,24,32,40,48,56,64],
            'v2_observed_offsets_px':[0,9,15,24,33,39,48,57,66],
            'raw_sources_opened_during_correction':0,'v1_sequence_runs':0,
            'retained_original_config':str(old.relative_to(ROOT)).replace('\\','/')}
    dest=out/'config_revision_v1_to_v2.json';assert not dest.exists()
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(sha(new))
if __name__=='__main__':main()
