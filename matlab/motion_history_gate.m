function [g,diagnostic]=motion_history_gate(current,history,currentGray,historyGray,valid,quality,cfg)
assert(isscalar(quality) && isfinite(quality) && quality>=0 && quality<=1);
kernel=ones(cfg.photo_window_hw(:)');
den=conv2(double(valid),kernel,'same');
res=abs(double(currentGray)-double(historyGray));res(~valid)=0;
r=conv2(res,kernel,'same')./max(den,1);
photo=exp(-r.^2/(2*cfg.photo_sigma^2));photo(den==0)=0;
eps=cfg.confidence_epsilon;
confidence=min(1,max(0,2*(eps+motion_confidence(history))./(2*eps+motion_confidence(history)+motion_confidence(current))));
g=quality*photo.*confidence;g(~valid)=0;
diagnostic=struct('photo_mean',mean(photo(valid),'omitnan'),'confidence_mean',mean(confidence(valid),'omitnan'));
end
