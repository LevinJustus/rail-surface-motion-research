function [output,diagnostic]=v2_fuse_window(probabilities,edges,cfg)
% Probability-level compensation core; no images/labels/networks are loaded.
n=numel(probabilities);assert(n>=1&&n<=cfg.window_frames&&numel(edges)==n-1);p=double(probabilities{end});
assert(isequal(size(p),[256 256])&&all(isfinite(p)&p>=0&p<=1,'all'));delta=zeros(size(p));histories=cell(n-1,1);
for j=1:n-1
    paths=v2_compose_motion(edges(j:end),cfg);e=v2_history_evidence(probabilities{j},p,paths,cfg);
    delta=delta+(1/n)*e.gate.*(e.mean_history-p);histories{j}=e;
end
unprotected=p+delta;protected=p>=cfg.strong_current_foreground;value=unprotected;
value(protected)=max(value(protected),cfg.protected_decision_threshold);value=min(1,max(0,value));output=single(value);
bound=((n-1)/n)*cfg.risk_sigma/sqrt(exp(1));assert(max(abs(delta),[],'all')<=bound+1e-12);
diagnostic=struct('histories',{histories},'unprotected_probability',unprotected,'delta',delta,...
    'protected_pixels',nnz(value~=unprotected),'analytic_update_bound',bound,'arrived_window_frames',n);
end
