function evidence=v2_history_evidence(history,current,paths,cfg)
% Risk bound is on probability disagreement, not on true segmentation error.
p=double(current);assert(isequal(size(p),[256 256])&&isequal(size(history),size(p)));
assert(all(isfinite(p)&p>=0&p<=1,'all')&&all(isfinite(history)&history>=0&history<=1,'all'));
q=zeros(size(p));first=q;second=q;known=q;
for k=1:numel(paths.weights)
    [h,valid]=warp_motion_history(history,paths.shifts{k});weight=paths.weights(k)*double(valid);
    q=q+weight;first=first+weight.*h;second=second+weight.*h.^2;known=known+weight.*(h-p).^2;
end
assert(all(q>=-1e-12&q<=1+1e-12,'all'));q=min(1,max(0,q));covered=q>0;meanHistory=p;
meanHistory(covered)=first(covered)./q(covered);variance=zeros(size(p));
variance(covered)=max(0,second(covered)./q(covered)-meanHistory(covered).^2);
unknown=(1-q).*max(p.^2,(1-p).^2);risk=known+unknown;gate=q.*exp(-risk/(2*cfg.risk_sigma^2));
assert(all(isfinite(risk)&risk>=0&risk<=1+1e-12,'all')&&all(gate>=0&gate<=1+1e-12,'all'));
evidence=struct('mean_history',meanHistory,'known_mass',q,'risk',risk,'gate',gate,...
    'conditional_variance',variance,'known_disagreement',known,'unknown_risk',unknown);
end
