function [state,output]=ablation_recurrent_step(previous,current,previousGray,currentGray,edge,cfg)
current=double(current);state=current;
if ~isempty(previous)&&edge.visual.available
    shift=[edge.visual.dx edge.visual.dy];[h,v]=warp_motion_history(previous,shift);[hg,~]=warp_motion_history(previousGray,shift);
    g=motion_history_gate(current,h,currentGray,hg,v,edge.quality,cfg.visual_gate);
    difference=h-current;difference(~v)=0;state=current+.5*g.*difference;
end
assert(all(isfinite(state),'all')&&all(state>=-1e-6 & state<=1+1e-6,'all'));
state=min(1,max(0,state));output=single(state);
end
