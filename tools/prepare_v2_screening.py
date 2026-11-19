"""Freeze the authorized E3 diagnostic budget before reading any mask payload."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    path=ROOT/'configs/v2_screening_v1.json';assert not path.exists()
    protocol=read(ROOT/'configs/fusion_protocol_v1.json');selection=read(ROOT/protocol['data']['selection_file'])
    profiles=['clean','geometric_jitter','blur_3px','combined'];rows=[r for r in selection if r['assigned_split']=='training' and r['profile'] in profiles]
    rows=sorted(rows,key=lambda r:r['sequence_id']);assert len(rows)==12 and len({r['source_id'] for r in rows})==3
    checkpoint=protocol['segmentation']['checkpoints'][0];assert sha(ROOT/checkpoint['checkpoint_relative'])==checkpoint['checkpoint_sha256']
    plan=dict(version='E3-diagnostic-v1',date_local='2026-11-19',created_utc=datetime.now(timezone.utc).isoformat(),seed=20261114,
        scope='three training-source proxy groups / four original profiles; not real video or independent final evaluation',
        design_config='configs/algorithm_design_v2.json',design_sha256=sha(ROOT/'configs/algorithm_design_v2.json'),
        protocol_config='configs/fusion_protocol_v1.json',protocol_sha256=sha(ROOT/'configs/fusion_protocol_v1.json'),
        base_run='fusion_training_v1_02',base_manifest_sha256=sha(ROOT/'runs/fusion_training_v1_02/prediction_manifest.json'),
        checkpoint=checkpoint,selection=rows,threshold=.5,primary_frames=list(range(2,10)),window_frames=3,
        methods=['S','MAP_integer','DECAY05_integer','V_subpixel','GV_subpixel','V2','V2_visual'],
        simple_decay=.5,additional_network_forward_calls=0,optimizer_updates=0,
        opportunity='current FN and at least one oracle-visible historical prediction>=.5 on historically positive GT',
        geometry_sampling='bilinear probability, nearest historical mask, all intermediate views valid; evaluation only',
        candidate_sampling_error_thresholds_px=[1,2],low_mass_diagnostic=.1,low_gate_diagnostic=.01,
        current_probability_bins=[0,.1,.3,.3989115567145611,.5],
        diagnostic_only=True,parameters_refitted=False,validation_and_heldout_payloads_allowed=False,
        note='Source/schema and prior development evidence already known; selection precedes new opportunity/count analysis. No feedback or new fusion variant.')
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path.with_suffix('.sha256').write_text(sha(path)+'  '+path.name+'\n',encoding='ascii')
    print(json.dumps(dict(status='FROZEN_E3_DIAGNOSTIC',sequences=[r['sequence_id'] for r in rows],sources=3,profiles=profiles,sha256=sha(path))))
if __name__=='__main__':main()
