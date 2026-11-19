"""Causal, one-pass historical-evidence-guided motion prototype.

Inputs are RGB, RAW segmentation probabilities and optional observed displacement.
No trained weights, labels, geometry files or repository-specific helpers are used.
The scores below are heuristic scores, not calibrated posterior probabilities.
"""
import numpy as np
from scipy import ndimage, signal

DEFAULTS = dict(max_shift=20, maximum_hypotheses=5, peak_separation=4,
                minimum_ncc=.2, visual_temperature=.02, feedback_weight=.1,
                history_threshold=.5, minimum_roi_pixels=16, roi_dilation=2,
                ring_dilation=8, risk_sigma=.25, strong_current=.7, decision=.5)
VARIANTS = ('B_base_conservative', 'C_feedback_conservative',
            'E_base_strong', 'D_feedback_strong', 'F_uniform_strong')
Y, X = np.mgrid[0:256, 0:256]


def gray(rgb):
    a=np.asarray(rgb)
    if a.dtype!=np.uint8 or a.shape!=(256,256,3):raise ValueError('RGB must be uint8 256x256x3')
    return np.dot(a.astype(float),np.array([.2989,.5870,.1140]))/255.


def warp(a, shifts):
    """Forward [dx,dy] edges; every intermediate view must remain valid."""
    qx=X.astype(float).copy();qy=Y.astype(float).copy();valid=np.ones((256,256),bool)
    for dx,dy in reversed(shifts):
        qx-=dx;qy-=dy;valid&=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255)
    result=ndimage.map_coordinates(np.asarray(a,float),[qy,qx],order=1,mode='constant',cval=0.,prefilter=False)
    result[~valid]=0.
    return result,valid


def ncc_grid(a,b,limit):
    patch=a[64:192,64:192];t=patch-patch.mean();num=signal.correlate(b,t,mode='valid',method='fft')
    def box(v):
        c=np.pad(v,((1,0),(1,0)),mode='constant').cumsum(0).cumsum(1)
        return c[128:,128:]-c[:-128,128:]-c[128:,:-128]+c[:-128,:-128]
    sums=box(b);var=np.maximum(box(b*b)-sums*sums/16384.,0);den=np.sqrt(var*np.sum(t*t))
    C=np.divide(num,den,out=np.zeros_like(num),where=den>1e-15)
    return np.clip(C[64-limit:65+limit,64-limit:65+limit],-1,1)


def motion_hypotheses(a,b,observed_delta,cfg):
    if np.std(a[64:192,64:192],ddof=1)<1e-6:return []
    limit=cfg['max_shift'];C=ncc_grid(a,b,limit)
    iy,ix=np.unravel_index(np.argmax(C),C.shape)
    if not np.isfinite(C).all() or C[iy,ix]<cfg['minimum_ncc'] or ix in (0,2*limit) or iy in (0,2*limit):return []
    score=C.copy();yy,xx=np.mgrid[-limit:limit+1,-limit:limit+1]
    if observed_delta is not None and np.isfinite(observed_delta) and abs(observed_delta)<=limit:
        mode=np.array([ix-limit,iy-limit]);center=np.array([0.,-observed_delta]);r=1/(1+np.sum((mode-center)**2)/8)
        logprior=-.5*(xx*xx+(yy+observed_delta)**2)/4
        prior=np.exp(logprior-logprior.max());prior/=prior.sum();mixture=r*prior+(1-r)/C.size
        score+=cfg['visual_temperature']*np.log(np.maximum(mixture/mixture.max(),1e-300))
    peaks=(score>=ndimage.maximum_filter(score,size=3,mode='constant',cval=-np.inf))
    peaks[[0,-1],:]=False;peaks[:,[0,-1]]=False
    candidates=sorted([(float(-score[y,x]),int(y),int(x)) for y,x in zip(*np.nonzero(peaks))])
    result=[]
    for neg,y,x in candidates:
        if any(max(abs(x-h['integer_xy'][0]-limit),abs(y-h['integer_xy'][1]-limit))<cfg['peak_separation'] for h in result):continue
        if C[y,x]<cfg['minimum_ncc']:continue
        def offset(left,middle,right):
            den=left-2*middle+right
            return float(np.clip(.5*(left-right)/den,-.5,.5)) if den<-1e-12 else 0.
        xy=[x-limit+offset(C[y,x-1],C[y,x],C[y,x+1]),y-limit+offset(C[y-1,x],C[y,x],C[y+1,x])]
        result.append(dict(xy=xy,integer_xy=[x-limit,y-limit],base_score=-neg,ncc=float(C[y,x])))
        if len(result)==cfg['maximum_hypotheses']:break
    return result


def historical_roi(older,previous,previous_base_edge,cfg):
    if older is None or previous_base_edge is None:return None
    aligned,valid=warp(older,[previous_base_edge]);persistent=valid&(np.minimum(aligned,previous)>=cfg['history_threshold'])
    if np.count_nonzero(persistent)<cfg['minimum_roi_pixels']:return None
    core=ndimage.binary_dilation(persistent,iterations=cfg['roi_dilation']);ring=ndimage.binary_dilation(core,iterations=cfg['ring_dilation'])&~core
    weights=core.astype(float)/np.count_nonzero(core)
    if ring.any():weights=.5*weights+.5*ring/np.count_nonzero(ring)
    return weights


def image_support(a,b,shift,weights):
    # Sample current image at previous coordinates + forward displacement.
    aligned,valid=warp(b,[(-shift[0],-shift[1])]);w=weights*valid
    if w.sum()<.5:return None
    w=w/w.sum();am=float((w*a).sum());bm=float((w*aligned).sum());u=a-am;v=aligned-bm
    den=np.sqrt((w*u*u).sum()*(w*v*v).sum())
    if den<1e-12:return None
    intensity=float(np.clip((w*u*v).sum()/den,-1,1))
    ax=ndimage.sobel(a,axis=1)/8.;ay=ndimage.sobel(a,axis=0)/8.
    bx=ndimage.sobel(b,axis=1)/8.;by=ndimage.sobel(b,axis=0)/8.
    bx,_=warp(bx,[(-shift[0],-shift[1])]);by,_=warp(by,[(-shift[0],-shift[1])])
    denominator=np.sqrt((w*(ax*ax+ay*ay)).sum()*(w*(bx*bx+by*by)).sum())
    edge=float(np.clip((w*(ax*bx+ay*by)).sum()/denominator,-1,1)) if denominator>1e-12 else intensity
    return .5*(intensity+edge)


def select_motion(a,b,hypotheses,weights,cfg):
    if not hypotheses:return None,[]
    scores=[]
    for h in hypotheses:
        local=image_support(a,b,h['xy'],weights) if weights is not None else None
        score=h['base_score']+(cfg['feedback_weight']*local if local is not None else 0.)
        scores.append(dict(xy=h['xy'],base_score=h['base_score'],local_score=local,selection_score=float(score)))
    best=max(range(len(scores)),key=lambda i:(scores[i]['selection_score'],-i))
    return list(scores[best]['xy']),scores


def fuse(raw,edges,conservative,cfg):
    p=np.asarray(raw[-1],float);value=p.copy();n=len(raw)
    for j in range(n-1):
        if any(e is None for e in edges[j:]):continue
        h,valid=warp(raw[j],edges[j:]);gate=valid.astype(float)
        if conservative:gate*=np.exp(-(h-p)**2/(2*cfg['risk_sigma']**2))
        value+=gate*(h-p)/n
    assert np.isfinite(value).all() and value.min()>=-1e-12 and value.max()<=1+1e-12
    strong=p>=cfg['strong_current'];value[strong]=np.maximum(value[strong],cfg['decision'])
    return np.clip(value,0,1).astype(np.float32)


class MotionFeedback:
    """New instance resets a stream. No predicted output is reused as evidence."""
    def __init__(self,config=None):
        self.cfg=dict(DEFAULTS)
        if config:self.cfg.update(config)
        self.images=[];self.raw=[];self.edges=dict(base=[],feedback=[],uniform=[]);self.time=None;self.last_odometry=None

    def step(self,rgb,raw_probability,timestamp_s,odometry_px=None):
        if not np.isfinite(timestamp_s) or (self.time is not None and timestamp_s<=self.time):raise ValueError('timestamps must increase')
        p=np.asarray(raw_probability,dtype=np.float32)
        if p.shape!=(256,256) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('probability must be finite 256x256 in [0,1]')
        current=gray(rgb);diagnostic=dict(feedback_available=False,roi_pixels=0,hypotheses=[],base=None,feedback=None,uniform=None,feedback_scores=[],uniform_scores=[])
        if self.images:
            previous=self.images[-1];delta=None
            if self.last_odometry is not None and odometry_px is not None:delta=odometry_px-self.last_odometry
            hypotheses=motion_hypotheses(previous,current,delta,self.cfg);base=hypotheses[0]['xy'] if hypotheses else None
            roi=historical_roi(self.raw[-2],self.raw[-1],self.edges['base'][-1],self.cfg) if len(self.raw)>1 else None
            feedback,scores=select_motion(previous,current,hypotheses,roi,self.cfg)
            uniform,uscores=select_motion(previous,current,hypotheses,np.ones((256,256))/65536 if roi is not None else None,self.cfg)
            for key,edge in [('base',base),('feedback',feedback),('uniform',uniform)]:self.edges[key].append(edge)
            diagnostic.update(feedback_available=roi is not None,roi_pixels=int(np.count_nonzero(roi)) if roi is not None else 0,hypotheses=hypotheses,base=base,feedback=feedback,uniform=uniform,feedback_scores=scores,uniform_scores=uscores)
        self.images.append(current.copy());self.raw.append(p.copy());self.time=float(timestamp_s);self.last_odometry=odometry_px
        if len(self.raw)>3:
            self.raw=self.raw[-3:];self.images=self.images[-3:]
            for key in self.edges:self.edges[key]=self.edges[key][-2:]
        result={}
        for name,key,conservative in [('B_base_conservative','base',True),('C_feedback_conservative','feedback',True),('E_base_strong','base',False),('D_feedback_strong','feedback',False),('F_uniform_strong','uniform',False)]:
            result[name]=fuse(self.raw,self.edges[key],conservative,self.cfg)
        return result,diagnostic
