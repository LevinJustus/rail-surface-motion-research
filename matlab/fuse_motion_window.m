function [out,diagnostic]=fuse_motion_window(images,probabilities,edges,cfg,constant)
% Pure bounded-window arithmetic; no paths, labels, source/profile or truth.
n=numel(images);assert(n>=1 && n<=cfg.window_frames && numel(probabilities)==n && numel(edges)==n-1);
assert(isscalar(constant)&&isfinite(constant)&&constant>=0&&constant<=1);
methods={'S','U','V','P','GV','GVP','GC','R'};p=double(probabilities{n});
for k=1:n
    assert(isa(probabilities{k},'single') && isequal(size(probabilities{k}),[256 256]));
    assert(all(isfinite(probabilities{k}),'all')&&all(probabilities{k}>=0 & probabilities{k}<=1,'all'));
end
out=struct;for m=methods,out.(m{1})=p;end
diagnostic=struct([]);K=n-1;if K==0,for m=methods,out.(m{1})=single(p);end;return;end
gray=cellfun(@motion_gray,images,'UniformOutput',false);w=1/(K+1);
for j=1:n-1
    out.U=out.U+w*(double(probabilities{j})-p);
    path=edges(j:end);shifts=zeros(numel(path),2);priorShifts=shifts;available=true;priorAvailable=true;Q=1;z=1;observedCount=0;
    for e=1:numel(path)
        edge=path{e};available=available&&edge.visual.available;priorAvailable=priorAvailable&&edge.prior.available;
        shifts(e,:)=[edge.visual.dx edge.visual.dy];priorShifts(e,:)=[edge.prior.dx edge.prior.dy];
        Q=Q*edge.quality;z=z*edge.observed_factor;observedCount=observedCount+edge.observed_available;
    end
    valid=false(256);pv=valid;g=zeros(256);h=zeros(256);dv=[NaN NaN];dp=dv;
    gd=struct('photo_mean',NaN,'confidence_mean',NaN);
    if available
        [h,valid,dv]=warp_motion_history(probabilities{j},shifts);
        [hg,vg]=warp_motion_history(gray{j},shifts);assert(isequal(valid,vg));
        [g,gd]=motion_history_gate(p,h,gray{n},hg,valid,Q,cfg.visual_gate);
        difference=h-p;difference(~valid)=0;
        out.V=out.V+w*difference;
        out.GV=out.GV+w*g.*difference;
        out.GVP=out.GVP+w*z*g.*difference;
        out.GC=out.GC+w*constant*g.*difference;
    end
    if priorAvailable
        [hp,pv,dp]=warp_motion_history(probabilities{j},priorShifts);
        d=hp-p;d(~pv)=0;out.P=out.P+w*d;
    end
    rr=struct('history_index',j,'age',n-j,'visual_available',available,'prior_available',priorAvailable,...
        'dx',dv(1),'dy',dv(2),'prior_dx',dp(1),'prior_dy',dp(2),...
        'valid_pixels',nnz(valid),'prior_valid_pixels',nnz(pv),'quality',Q,'observed_factor',z,...
        'observed_edges',observedCount,'gate_mean_all',mean(g,'all'),...
        'gate_mean_valid',mean(g(valid),'omitnan'),'photo_mean_valid',gd.photo_mean,'confidence_mean_valid',gd.confidence_mean);
    if isempty(diagnostic),diagnostic=rr;else,diagnostic(end+1)=rr;end %#ok<AGROW>
end
% Rebuild the bounded recursive mechanism from the oldest raw observation.
acc=double(probabilities{1});
for k=2:n
    current=double(probabilities{k});edge=edges{k-1};
    if edge.visual.available
        shift=[edge.visual.dx edge.visual.dy];[h,v]=warp_motion_history(acc,shift);
        [hg,~]=warp_motion_history(gray{k-1},shift);
        g=motion_history_gate(current,h,gray{k},hg,v,edge.quality,cfg.visual_gate);
        d=h-current;d(~v)=0;acc=current+cfg.bounded_recursive.historical_fraction_each_step*g.*d;
    else,acc=current;end
end
out.R=acc;
for m=methods
    a=out.(m{1});assert(all(isfinite(a),'all')&&all(a>=-cfg.range_tolerance & a<=1+cfg.range_tolerance,'all'));
    out.(m{1})=single(min(1,max(0,a)));
end
end
