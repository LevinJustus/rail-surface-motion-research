"""Independent E2 reconstruction for artificial examples and four RGB frames.

No actual rail dataset, model, validation or held-out data is accessed.
"""
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import loadmat
import audit_fusion_training as old


def candidates(C,delta,cfg):
    yy,xx=np.mgrid[-20:21,-20:21];ranking=sorted(zip(-C.ravel(),yy.ravel(),xx.ravel()));mode=np.array([ranking[0][2],ranking[0][1]])
    v=np.exp((C-C.max())/cfg['visual_temperature']);v/=v.sum();w=v.copy()
    if delta is not None:
        center=np.array([0.,-delta]);r=1/(1+np.sum((mode-center)**2)/8)
        e=-.5*((xx-center[0])**2/4+(yy-center[1])**2/4);g=np.exp(e-e.max());g/=g.sum();w=v*(r*g+(1-r)/C.size);w/=w.sum()
    ranked=sorted(zip(-w.ravel(),yy.ravel(),xx.ravel()));chosen=[];mass=0.
    for neg,dy,dx in ranked[:25]:
        if neg==0:break
        chosen.append((float(dx),float(dy),float(-neg)));mass-=neg
        if mass>=.95:break
    return w,chosen


def ncc(previous,current):
    a,b=old.gray(previous),old.gray(current);patch=a[64:192,64:192];t=patch-patch.mean();n=patch.size
    num=signal.correlate(b,t,mode='valid',method='fft')
    integral=np.pad(b,((1,0),(1,0)),mode='constant').cumsum(0).cumsum(1)
    square=np.pad(b*b,((1,0),(1,0)),mode='constant').cumsum(0).cumsum(1)
    def box(x):return x[128:,128:]-x[:-128,128:]-x[128:,:-128]+x[:-128,:-128]
    sums=box(integral);var=np.maximum(box(square)-sums*sums/n,0);den=np.sqrt(var*np.sum(t*t))
    C=np.divide(num,den,out=np.zeros_like(num),where=den>0)
    return C[44:85,44:85]


def compose(edges):
    paths=[([],1.)]
    for e in edges:
        expanded=[(shifts+[(dx,dy)],weight*w) for shifts,weight in paths for dx,dy,w in e]
        expanded=sorted(enumerate(expanded),key=lambda x:(-x[1][1],sum(s[1] for s in x[1][0]),sum(s[0] for s in x[1][0]),x[0]))
        paths=[];mass=0.
        for _,value in expanded[:25]:
            paths.append(value);mass+=value[1]
            if mass>=.95:break
    return paths


def fuse(P,edges,cfg):
    p=P[-1].astype(float);delta=np.zeros_like(p);n=len(P);risk_maps=[]
    for j in range(n-1):
        q=np.zeros_like(p);first=np.zeros_like(p);known=np.zeros_like(p)
        for shifts,w in compose(edges[j:]):
            hp,v=old.warp(P[j],shifts);wv=w*v;q+=wv;first+=wv*hp;known+=wv*(hp-p)**2
        q=np.clip(q,0,1);h=p.copy();covered=q>0;h[covered]=first[covered]/q[covered]
        risk=known+(1-q)*np.maximum(p*p,(1-p)**2);g=q*np.exp(-risk/(2*cfg['risk_sigma']**2))
        delta+=(1/n)*g*(h-p);risk_maps.append((risk,g))
    value=p+delta;strong=p>=cfg['strong_current_foreground'];value[strong]=np.maximum(value[strong],cfg['protected_decision_threshold'])
    return np.clip(value,0,1).astype(np.float32),risk_maps


def main(root,run,out):
    assert not out.exists();out.mkdir(parents=True);timer=time.perf_counter();folder=root/'runs'/run;cfg=old.read(folder/'frozen_design.json')
    report=old.read(folder/'contracts.json');assert report['status']=='PASS_V2_MINIMAL_IMPLEMENTATION_CONTRACTS'
    assert old.sha(folder/'frozen_design.json')==report['config_sha256']
    for row in report['source_files']:assert old.sha(folder/'source_snapshot'/Path(row['path']).name)==row['sha256']
    fixture=loadmat(str(folder/'synthetic_stream.mat'),squeeze_me=True,struct_as_record=False)
    images=list(fixture['streamImages']);P=list(fixture['streamRaw']);actual=list(fixture['streamOutput']);stored=list(fixture['streamEdges']);edges=[];max_w=0.;max_p=0.
    for k in range(4):
        if k:
            delta=4 if k==3 else None;C=ncc(images[k-1],images[k]);w,chosen=candidates(C,delta,cfg)
            max_w=max(max_w,float(np.max(np.abs(w-stored[k-1].full_weights))));assert max_w<2e-10
            assert np.max(np.abs(np.asarray(chosen)-np.atleast_2d(stored[k-1].candidates)))<2e-10
            edges.append(chosen)
        start=max(0,k-2);prediction,_=fuse(P[start:k+1],edges[start:k],cfg);err=float(np.max(np.abs(prediction.astype(float)-actual[k].astype(float))));max_p=max(max_p,err);assert err<=2e-6
    data=loadmat(str(folder/'controlled_examples.mat'),squeeze_me=True,struct_as_record=False)['examples'];normal=np.full((256,256),.05,np.float32);target=np.zeros((256,256),bool);target[79:179,127:129]=True
    one=[(0.,0.,1.)];fixtures=[]
    h=normal.copy();h[target]=np.float32(.8);p=normal.copy();p[target]=np.float32(.45);fixtures.append(('recover_near_threshold',[h,h,p],[one,one]))
    h=normal.copy();h[79:179,121:123]=np.float32(.9);p=normal.copy();p[target]=np.float32(.55);fixtures.append(('ambiguous_wrong_single_motion',[h,p],[[(0.,0.,.5),(6.,0.,.5)]]))
    h=normal.copy();h[target]=np.float32(.8);p=h.copy();p[79:179,99:101]=np.float32(.95);fixtures.append(('confident_current_false_positive_failure',[h,h,p],[one,one]))
    counts=[]
    for name,P,edge in fixtures:
        item=getattr(data,name);pred,risk=fuse(P,edge,cfg);err=float(np.max(np.abs(pred.astype(float)-item.v2.astype(float))));max_p=max(max_p,err);assert err<=2e-6
        assert np.array_equal(item.target.astype(bool),target)
        for m in ['single','single_motion_fixed','v2']:
            a=getattr(item,m);b=a.astype(float)>=.5;tp=int(np.count_nonzero(b&target));fp=int(np.count_nonzero(b&~target));fn=int(np.count_nonzero(~b&target))
            counts.append(dict(example=name,method=m,TP=tp,FP=fp,FN=fn,IoU=tp/(tp+fp+fn) if tp+fp+fn else 1.))
    existing=list(csv.DictReader((folder/'controlled_examples.csv').open(encoding='utf-8-sig',newline='')))
    for a,b in zip(counts,existing):
        assert a['example']==b['example'] and a['method']==b['method']
        for k in ['TP','FP','FN']:assert a[k]==int(b[k])
        assert abs(a['IoU']-float(b['IoU']))<1e-12
    result=dict(status='PASS_V2_INDEPENDENT_ARTIFICIAL_AUDIT',RGB_frames_reconstructed=4,artificial_examples=3,count_rows=9,
        maximum_motion_weight_error=max_w,maximum_probability_error=max_p,three_frame_update_bound=2/3*.25/math.sqrt(math.e),
        real_dataset_payloads_opened=0,model_forward_calls=0,optimizer_updates=0,wall_seconds=time.perf_counter()-timer,
        scope='independent artificial/mechanism checks only; no rail accuracy, novelty or real-time validation',numpy_version=np.__version__)
    old.write(out/'audit.json',result);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.project_root.resolve(),a.run_id,a.out.resolve())
