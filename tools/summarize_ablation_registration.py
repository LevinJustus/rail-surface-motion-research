"""C5 metadata-only diagnostics: prior displacement changes and attenuation.

No images, probabilities or labels. Descriptive mechanics, not a new primary
endpoint, parameter search or independent geometric accuracy measurement.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(root,prediction_run,out):
    assert not out.exists();out.mkdir(parents=True);run=root/'runs'/prediction_run;s=read(run/'summary.json');index=read(run/'registration_index.json')
    assert s['mode']=='validation' and len(index)==180 and sha(run/'registration_index.json')==s['registration_index_sha256']
    selection={r['sequence_id']:r for r in read(root/'docs/evidence/step13c2/sequence_selection.json')}
    names=['baseline','bias_m6','bias_m2','bias_p2','bias_p6','alternating_1','alternating_3','all_missing'];rows=[];grouped=defaultdict(list)
    for item in index:
        assert sha(root/item['json_path'])==item['json_sha256'];reg=read(root/item['json_path']);sel=selection[item['sequence_id']]
        for scenario,edges in zip(names,[reg['base_edges']]+reg['stress_edges']):
            for k,e in enumerate(edges,2):
                v,p=e['visual'],e['prior'];both=v['available'] and p['available'];delta=float(np.hypot(p['dx']-v['dx'],p['dy']-v['dy'])) if both else None
                row=dict(sequence_id=item['sequence_id'],source_id=sel['source_id'],profile=sel['profile'],frame=k,scenario=scenario,
                    observed_available=int(e['observed_available']),observed_delta=e['observed_delta'],visual_available=int(v['available']),prior_available=int(p['available']),
                    prior_visual_displacement_difference_px=delta,prior_differs=int(both and delta>1e-12),observed_factor=e['observed_factor'],visual_quality=e['quality'],
                    prior_used=int(p['used_prior']),prior_fallback=p['fallback'])
                rows.append(row);grouped[scenario].append(row)
    with (out/'per_edge_registration.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for scenario in names:
        r=grouped[scenario];z=np.array([x['observed_factor'] for x in r]);ds=[x['prior_visual_displacement_difference_px'] for x in r if x['prior_visual_displacement_difference_px'] is not None]
        summary.append(dict(scenario=scenario,adjacent_pairs=len(r),observed_available=sum(x['observed_available'] for x in r),prior_differs=sum(x['prior_differs'] for x in r),
            mean_prior_visual_displacement_difference_px=float(np.mean(ds)) if ds else None,max_prior_visual_displacement_difference_px=max(ds) if ds else None,
            mean_observed_factor=float(z.mean()),observed_factor_lt_0_1=int((z<.1).sum()),observed_factor_exactly_1=int((z==1).sum())))
    with (out/'registration_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    report=dict(status='PASS_C5_METADATA_DIAGNOSTICS',registration_index_sha256=s['registration_index_sha256'],rows=len(rows),
        scope='cached motion estimates and gate mechanics only; not ground-truth displacement error',image_probability_mask_payloads_opened=0)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--prediction-run',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();summarize(a.project_root.resolve(),a.prediction_run,a.out.resolve())
