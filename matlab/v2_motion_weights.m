function edge=v2_motion_weights(previous,current,observedDelta,cfg)
% Same grayscale/template and forward displacement sign as frozen v1.
a=motion_gray(previous);b=motion_gray(current);patch=a(65:192,65:192);s=std(patch(:));
lo=cfg.motion_grid_xy(1);hi=cfg.motion_grid_xy(2);
if s<cfg.minimum_template_std,C=zeros(hi-lo+1);else,full=normxcorr2(patch,b);C=full(192+(lo:hi),192+(lo:hi));end
edge=v2_select_motion(C,s,observedDelta,cfg);
end
