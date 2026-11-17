"""Independently audit numeric outputs with NumPy; never train or tune a model."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def metric(c):
    tp,fp,fn,tn=np.moveaxis(np.asarray(c,dtype=float),-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.stack([tp/(tp+fp+fn),2*tp/(2*tp+fp+fn),tp/(tp+fp),tp/(tp+fn)],axis=-1)
def close(a,b):assert np.allclose(a,b,rtol=0,atol=5e-12,equal_nan=True)
def main(run):
    out=ROOT/'runs'/run;s=read(out/'summary.json');assert s['status']=='PASS' and s['completed_checkpoints']==12
    thresholds=np.arange(1,20,dtype=float)/20;pidx=9;rows=[];case_pixels=0;case_comparisons=0
    models={'small_unet':'full_stage2_v2_01','mobilenet_deeplab':'mature_stage3_v1_01'}
    expected={(m,seed,kind) for m in models for seed in [20261114,20261115,20261116] for kind in ['best_validation','last']}
    assert {(r['model'],r['seed'],r['checkpoint_kind']) for r in s['checkpoints']}==expected
    for r in s['checkpoints']:
        assert float(r['seed']).is_integer();r['seed']=int(r['seed'])
        d=out/r['name'];layout=read(d/'binary_layout.json')
        C=np.fromfile(d/'counts.u32',dtype='<u4').reshape((400,19,4),order='F').astype(np.int64)
        G=np.fromfile(d/'group_counts.u32',dtype='<u4').reshape((15,19,4),order='F').astype(np.int64)
        per=list(csv.DictReader((d/'per_image_primary.csv').open(encoding='utf-8-sig',newline='')))
        curve=list(csv.DictReader((d/'threshold_curve.csv').open(encoding='utf-8-sig',newline='')))
        assert len(per)==400 and len(curve)==19 and len({x['id'] for x in per})==400
        assert np.all(C>=0) and np.all(np.diff(C[:,:,:2],axis=1)<=0)
        valid=np.array([int(x['valid_pixels']) for x in per]);target=np.array([int(x['target_pixels']) for x in per])
        assert np.all(C.sum(axis=2)==valid[:,None])
        assert np.all(C[:,:,0]+C[:,:,2]==target[:,None])
        primary=np.array([[int(x[k]) for k in ['TP','FP','FN','TN']] for x in per])
        assert np.array_equal(C[:,pidx,:],primary)
        groups=list(dict.fromkeys(x['group'] for x in per));assert len(groups)==15
        for i,g in enumerate(groups):assert np.array_equal(G[i],C[[x['group']==g for x in per]].sum(axis=0))
        sums=C.sum(axis=0);M=metric(sums)
        for j,record in enumerate(curve):
            close(float(record['threshold']),thresholds[j]);assert [int(record[k]) for k in ['TP','FP','FN','TN']]==sums[j].tolist()
            close([float(record[k]) for k in ['IoU','Dice','precision','recall']],M[j])
            close(float(record['macro_group_IoU']),np.nanmean(metric(G)[:,j,0]))
        history=read(ROOT/'runs'/models[r['model']]/'seeds'/f"seed_{r['seed']}"/'validation_evaluations.json')
        old=next(x for x in history if x['global_update']==r['update'])
        oldC=np.array([[x[k] for k in ['TP','FP','FN','TN']] for x in old['per_image']])
        assert [x['id'] for x in old['per_image']]==[x['id'] for x in per]
        assert np.array_equal(primary,oldC)
        close(M[pidx],[old[k] for k in ['micro_foreground_IoU','micro_foreground_Dice','micro_precision','micro_recall']])
        assert sha(ROOT/r['checkpoint_relative'])==r['checkpoint_sha256']
        assert r['checkpoint_bytes_and_network_state_unchanged']
        candidates=np.flatnonzero(np.abs(M[:,0]-np.max(M[:,0]))<=1e-12)
        best=min(candidates,key=lambda i:(abs(thresholds[i]-.5),-thresholds[i]))
        close(r['diagnostic_selected_threshold'],thresholds[best]);close(r['diagnostic_selected_metrics'],M[best]);close(r['primary_counts'],sums[pidx])
        shape=tuple(layout['case_shape']);indices=np.array(layout['case_indices_one_based'],ndmin=1,dtype=int)-1
        P=np.fromfile(d/'case_P.f32',dtype='<f4').reshape(shape,order='F').astype(np.float64)
        T=np.fromfile(d/'case_T.u8',dtype='u1').reshape(shape,order='F').astype(bool)
        V=np.fromfile(d/'case_V.u8',dtype='u1').reshape(shape,order='F').astype(bool)
        for k,index in enumerate(indices):
            for j,t in enumerate(thresholds):
                pred=P[:,:,k]>=t;gt=T[:,:,k];v=V[:,:,k]
                actual=[np.count_nonzero(v&gt&pred),np.count_nonzero(v&~gt&pred),np.count_nonzero(v&gt&~pred),np.count_nonzero(v&~gt&~pred)]
                assert np.array_equal(actual,C[index,j,:]);case_comparisons+=1
            case_pixels+=np.count_nonzero(V[:,:,k])
        assert sum(layout['foreground_histogram'])==target.sum()
        assert sum(layout['background_histogram'])==valid.sum()-target.sum()
        row={'model':r['model'],'seed':r['seed'],'checkpoint':r['checkpoint_kind'],'epoch':r['epoch'],'update':r['update'],'threshold_diagnostic':thresholds[best]}
        for i,k in enumerate(['IoU','Dice','precision','recall']):row['primary_'+k]=M[pidx,i];row['tuned_'+k]=M[best,i]
        row.update(FP_primary=int(sums[pidx,1]),FN_primary=int(sums[pidx,2]),FP_tuned=int(sums[best,1]),FN_tuned=int(sums[best,2]))
        rows.append(row)
    access=read(out/'input_access.json');assert access['training_pairs_opened']==access['heldout_pairs_opened']==0
    assert len(access['validation'])==400
    for x in access['validation']:
        assert x['assigned_split']=='validation'
        for field,key in [('image','image_file_sha256'),('mask','mask_file_sha256')]:assert sha(ROOT/'..'/'..'/'work'/'data_audit'/x[field])==x[key]
    for key in ['optimizer_updates','training_pairs_opened','heldout_pairs_opened']:assert s[key]==0
    with (out/'checkpoint_overview.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats=[]
    for model in models:
        for kind in ['best_validation','last']:
            rr=[r for r in rows if r['model']==model and r['checkpoint']==kind];assert len(rr)==3
            for phase in ['primary','tuned']:
                z={'model':model,'checkpoint':kind,'phase':phase,'seeds':3}
                for k in ['IoU','Dice','precision','recall']:
                    values=[r[phase+'_'+k] for r in rr];z[k+'_mean']=statistics.mean(values);z[k+'_sample_SD']=statistics.stdev(values)
                stats.append(z)
    with (out/'three_seed_statistics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    report={'status':'PASS','checkpoints':12,'thresholds':19,'historical_primary_images_checked':4800,'case_threshold_comparisons':case_comparisons,'case_valid_pixels_checked':case_pixels,'raw_validation_file_hashes_rechecked':800,'test_files_opened':0,'training_files_opened':0,'optimizer_updates':0,'checks':['exact historical per-image counts','19-point aggregate metrics','proxy group aggregation','pixel conservation and monotonicity','predeclared tie rule','independent NumPy saved-case probabilities','foreground/background histogram totals','checkpoint byte SHA','raw validation byte SHA'],'limits':['Only selected saved probability cases independently recomputed, not all 400x12 probability tensors','Threshold-selected performance is validation tuning, not unbiased or final-test evidence','No motion compensation, fusion, millimetre positioning or real-time result']}
    report['case_valid_pixels_checked']=int(report['case_valid_pixels_checked'])
    (out/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'audit':report,'statistics':stats,'thresholds':[{k:r[k] for k in ['model','seed','checkpoint','threshold_diagnostic']} for r in rows]},ensure_ascii=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');main(p.parse_args().run)
