function [output,diagnostic]=v2_no_conflict_fuse_window(probabilities,edges,cfg)
% E4 single-factor ablation. Original v2 and its bound remain unchanged.
n=numel(probabilities);assert(n>=1&&n<=cfg.window_frames&&numel(edges)==n-1);
p=double(probabilities{end});assert(isequal(size(p),[256 256])&&all(isfinite(p)&p>=0&p<=1,'all'));
delta=zeros(size(p));histories=cell(n-1,1);lower=p;upper=p;gateSum=zeros(size(p));
for j=1:n-1
    paths=v2_compose_motion(edges(j:end),cfg);e=v2_history_evidence(probabilities{j},p,paths,cfg);
    % Only q*(mean_history-current)^2 is removed; unknown mass stays intact.
    e.risk=e.known_mass.*e.conditional_variance+e.unknown_risk;
    e.gate=e.known_mass.*exp(-e.risk/(2*cfg.risk_sigma^2));
    assert(all(isfinite(e.risk)&e.risk>=0&e.risk<=1+1e-12,'all')&&all(e.gate>=0&e.gate<=1+1e-12,'all'));
    delta=delta+(1/n)*e.gate.*(e.mean_history-p);gateSum=gateSum+e.gate/n;
    lower=min(lower,e.mean_history);upper=max(upper,e.mean_history);histories{j}=e;
end
unprotected=p+delta;
assert(all(gateSum<=(n-1)/n+1e-12,'all')&&all(unprotected>=lower-1e-12&unprotected<=upper+1e-12,'all'));
% The old sigma/sqrt(e) bound does not apply after this formula change.
assert(max(abs(delta),[],'all')<=(n-1)/n+1e-12);
protected=p>=cfg.strong_current_foreground;value=unprotected;
value(protected)=max(value(protected),cfg.protected_decision_threshold);value=min(1,max(0,value));output=single(value);
diagnostic=struct('histories',{histories},'unprotected_probability',unprotected,'delta',delta,...
    'protected_pixels',nnz(value~=unprotected),'convex_update_bound',(n-1)/n,'arrived_window_frames',n);
end
