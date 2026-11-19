"""Fixed artificial repeated-texture challenge; never a real-rail benchmark.

Nine sequences (3 predetermined seeds x 3 cases), three frames each.
The last image loses nonperiodic background texture; an off-template local
structure remains visible. All probabilities are manufactured, not U-Net output.
"""
import argparse,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
import rail_motion_v3 as v3
from run_v3_feedback_batch import write,table,counts,metrics,sha

def run(out):
    assert not out.exists();out.mkdir(parents=True);timer=time.perf_counter()
    seeds=[20261119,20261120,20261121];cases=['ambiguous_texture','severe_current_miss','stable_false_history']
    write(out/'frozen_challenge.json',dict(seeds=seeds,cases=cases,frames_per_sequence=3,source_sha256=sha(Path(__file__)),algorithm_sha256=sha(ROOT/'python/rail_motion_v3.py'),parameters=v3.DEFAULTS,
        design='first two views have nonperiodic texture; third loses that background component while off-template structure remains; probabilities manufactured; no parameter sweep'))
    rows=[];motions=[];details=[]
    for seed in seeds:
        random=np.random.RandomState(seed);yy,xx=np.mgrid[0:400,0:400]
        background=.5+.13*np.sin(2*np.pi*xx/12)+.13*np.cos(2*np.pi*yy/16);noise=random.normal(0,.035,(400,400));marker=np.zeros((400,400),bool)
        for y in range(120,245):
            x=98+int(2*np.sin(y/17.));marker[y,x:x+3]=True
        background[marker]-=.3
        for case in cases:
            stream=v3.MotionFeedback();inputs=[];outputs=[]
            for k in range(3):
                y=70+3*k;x=70-3*k;image=np.clip(background+(noise if k<2 else 0),0,1)[y:y+256,x:x+256]
                rgb=np.repeat(np.rint(image[:,:,None]*255).astype(np.uint8),3,axis=2);site=marker[y:y+256,x:x+256]
                raw=np.full((256,256),.05,np.float32);raw[site]=.8 if k<2 else (.45 if case=='ambiguous_texture' else .1)
                target=site.copy() if case!='stable_false_history' else np.zeros_like(site)
                values,d=stream.step(rgb,raw,float(k));inputs.append((raw,target));outputs.append((values,d))
            raw,target=inputs[-1];values,d=outputs[-1];values=dict(values,S=raw)
            for method,p in values.items():
                c=counts(p.astype(float),target,raw.astype(float));rows.append(dict(seed=seed,case=case,method=method,**dict(zip(['TP','FP','FN','TN','FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP'],c)),**metrics(c)))
            record=dict(seed=seed,case=case,feedback_available=d['feedback_available'],hypotheses=len(d['hypotheses']),selection_changed=d['base']!=d['feedback'])
            for name in ['base','feedback','uniform']:record[name+'_EPE']=float(np.linalg.norm(np.array(d[name])-[3.,-3.])) if d[name] is not None else None
            record['best_candidate_EPE']=min(float(np.linalg.norm(np.array(h['xy'])-[3.,-3.])) for h in d['hypotheses']) if d['hypotheses'] else None
            motions.append(record);details.append(dict(seed=seed,case=case,diagnostic=d));print('V3_ARTIFICIAL',seed,case,record,flush=True)
    table(out/'metrics.csv',rows);table(out/'motion_metrics.csv',motions);write(out/'motion_diagnostics.json',details)
    summary=dict(status='PASS_V3_ARTIFICIAL_CHALLENGE_EXECUTED',sequences=9,frames=27,seeds=seeds,feedback_changed_sequences=sum(r['selection_changed'] for r in motions),
        motion={name:sum(r[name+'_EPE'] for r in motions if r[name+'_EPE'] is not None)/sum(r[name+'_EPE'] is not None for r in motions) for name in ['base','feedback','uniform']},wall_seconds=time.perf_counter()-timer,
        scope='engineered artificial mechanism stress cases; manufactured probabilities; not independent rail accuracy or trained-model evidence')
    write(out/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=Path(os.path.abspath(str(a.out)));out.relative_to(ROOT/'runs');run(out)
