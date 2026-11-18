function context=ablation_history_context(history,historyGray,current,currentGray,path,cfg,alignment)
shifts=zeros(numel(path),2);available=true;Q=1;Qnc=1;Qnm=1;z=1;
for i=1:numel(path)
    e=path{i};estimate=e.(alignment);available=available&&estimate.available;
    shifts(i,:)=[estimate.dx estimate.dy];q=ablation_quality_factors(e,cfg.visual_gate);
    Q=Q*q.full;Qnc=Qnc*q.without_cycle;Qnm=Qnm*q.without_margin;z=z*e.observed_factor;
end
context=struct('available',available,'valid',false(256),'history',zeros(256),'photo',zeros(256),...
    'confidence',zeros(256),'Q',Q,'Qnc',Qnc,'Qnm',Qnm,'z',z);
if ~available,return;end
[h,valid]=warp_motion_history(history,shifts);[hg,vg]=warp_motion_history(historyGray,shifts);assert(isequal(valid,vg));
kernel=ones(cfg.visual_gate.photo_window_hw(:)');den=conv2(double(valid),kernel,'same');
residual=abs(currentGray-hg);residual(~valid)=0;
residual=conv2(residual,kernel,'same')./max(den,1);
photo=exp(-residual.^2/(2*cfg.visual_gate.photo_sigma^2));photo(den==0)=0;
eps=cfg.visual_gate.confidence_epsilon;
conf=min(1,max(0,2*(eps+motion_confidence(h))./(2*eps+motion_confidence(h)+motion_confidence(current))));
context.valid=valid;context.history=h;context.photo=photo;context.confidence=conf;
end
