"""Scalar cross-checks of E3 aggregates and oracle sampling contracts."""
import argparse,csv,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import evaluate_v2_screening as evaluation
import audit_fusion_training as old


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main(folder,out):
    assert not out.exists();out.mkdir(parents=True);timer=time.perf_counter();summary=old.read(folder/'summary.json');assert summary['status']=='PASS_E3_TRAINING_DIAGNOSTIC'
    by_source=defaultdict(lambda:[0]*8);by_method=defaultdict(lambda:[0]*8)
    for r in rows(folder/'per_frame_counts.csv'):
        if int(r['frame'])<2:continue
        c=[int(r[k]) for k in evaluation.FIELDS];assert sum(c[:4])==65536
        for j,v in enumerate(c):by_source[r['source_id'],r['method']][j]+=v;by_method[r['method']][j]+=v
    for r in rows(folder/'per_source_metrics.csv'):
        c=by_source[r['source_id'],r['method']];assert c==[int(r[k]) for k in evaluation.FIELDS]
        tp,fp,fn=c[:3];assert abs(tp/(tp+fp+fn)-float(r['IoU']))<1e-12
    for r in rows(folder/'method_summary.csv'):
        c=by_method[r['method']];assert c==[int(r[k]) for k in evaluation.FIELDS]
        tp,fp,fn=c[:3];assert abs(tp/(tp+fp+fn)-float(r['IoU']))<1e-12
        macro=math.fsum(c[0]/(c[0]+c[1]+c[2]) for (source,m),c in by_source.items() if m==r['method'])/3
        assert abs(macro-float(r['source_macro_IoU']))<1e-12
    opp=rows(folder/'historical_opportunity.csv');assert len(opp)==96
    for field in ['current_FN','oracle_opportunity','v2_opportunity_recovered','below_update_recovery_bound_opportunity']:
        assert sum(int(r[field]) for r in opp)==summary[field]
    assert all(0<=int(r['v2_opportunity_recovered'])<=int(r['oracle_opportunity'])<=int(r['current_FN']) for r in opp)
    # Analytic sampling fixtures, independent expected arrays and translation warp.
    a=np.arange(65536,dtype=float).reshape(256,256)/65535;mask=a>=.5;I=np.eye(3);T=I.copy();T[1,2]=8
    h,t,v,_,_=evaluation.oracle([a,a],[mask,mask],[I,T],0,1);reference,valid=old.warp(a,[(0,-8)])
    assert np.array_equal(v,valid) and np.max(abs(h[v]-reference[v]))<1e-14
    R=np.array([[0,-1,257],[1,0,0],[0,0,1]],float);h,t,v,_,_=evaluation.oracle([a,a],[mask,mask],[I,R],0,1)
    assert v.all() and np.max(abs(h-np.rot90(a)))<1e-14 and np.array_equal(t,np.rot90(mask))
    outside=I.copy();outside[0,2]=300;_,_,v,_,_=evaluation.oracle([a,a,a],[mask]*3,[I,outside,I],0,2);assert not v.any()
    candidate=rows(folder/'candidate_geometry_coverage.csv');risk=rows(folder/'risk_summary.csv');opportunity_risk=next(r for r in risk if r['scope']=='opportunity')
    total_risk=float(opportunity_risk['risk_mean']);risk_fractions={key:float(opportunity_risk[key+'_mean'])/total_risk for key in ['variance','conflict','unknown']}
    result=dict(status='PASS_E3_SCALAR_AND_ORACLE_OPERATOR_CHECKS',source_rows=21,count_rows=756,oracle_operator_contracts=['translation_sign_matches_warp','quarter_turn_and_nearest_mask','intermediate_out_of_view_rejection'],
        opportunity_fraction=summary['oracle_opportunity']/summary['current_FN'],opportunity_recovered_fraction=summary['v2_opportunity_recovered']/summary['oracle_opportunity'],
        opportunity_below_bound_fraction=summary['below_update_recovery_bound_opportunity']/summary['oracle_opportunity'],
        candidate_within_1px_fraction=sum(int(r['candidate_within_1px']) for r in candidate)/sum(int(r['oracle_visible_pixels']) for r in candidate),
        candidate_within_2px_fraction=sum(int(r['candidate_within_2px']) for r in candidate)/sum(int(r['oracle_visible_pixels']) for r in candidate),
        opportunity_candidate_within_1px_fraction=sum(int(r['opportunity_candidate_within_1px']) for r in candidate)/sum(int(r['opportunity_pixels']) for r in candidate),
        opportunity_risk_component_fractions=risk_fractions,wall_seconds=time.perf_counter()-timer,
        scope='descriptive training-source diagnostic; component sizes do not establish causal benefit of removing a term')
    old.write(out/'audit.json',result);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.evaluation.resolve(),a.out.resolve())
