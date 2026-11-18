function edge=register_fusion_pair(previous,current,delta,cfg)
% Same forward selection as frozen 13B; two shared maps for the arrived pair.
assert(isscalar(delta) && (isfinite(delta)||isnan(delta)));
a=motion_gray(previous);b=motion_gray(current);r=cfg.registration;
[visual,prior,forward,m1]=direction(a,b,delta,r,cfg.visual_gate);
[back,~,reverse,m2]=direction(b,a,NaN,r,cfg.visual_gate);
quality=0;cycle=NaN;
if visual.available && back.available
    cycle=hypot(visual.dx+back.dx,visual.dy+back.dy);
    if ~visual.boundary_peak && ~back.boundary_peak && isfinite(forward.second_score) && isfinite(reverse.second_score)
        score=min(clip((visual.score-.2)/cfg.visual_gate.score_scale),clip((back.score-.2)/cfg.visual_gate.score_scale));
        margin=min(clip((visual.score-forward.second_score)/cfg.visual_gate.peak_margin_scale),clip((back.score-reverse.second_score)/cfg.visual_gate.peak_margin_scale));
        quality=score*margin*exp(-cycle^2/(2*cfg.visual_gate.cycle_sigma_px^2));
    end
end
has=isfinite(delta) && abs(delta)<=r.max_shift_px;z=1;
if has && visual.available,z=exp(-(visual.dy+delta)^2/(2*cfg.observed_gate.sigma_px^2));end
edge=struct('visual',visual,'prior',prior,'reverse',back,'forward_diagnostic',forward,...
    'reverse_diagnostic',reverse,'observed_delta',delta,'observed_available',has,...
    'cycle_error_px',cycle,'quality',quality,'observed_factor',z,'ncc_maps',m1+m2);
end
function [visual,prior,diagnostic,maps]=direction(a,b,delta,r,g)
visual=blank();prior=blank();maps=0;
patch=a(65:192,65:192);diagnostic=struct('template_std',std(patch(:)),'second_score',NaN,'integer_dx',NaN,'integer_dy',NaN);
if diagnostic.template_std<r.minimum_template_std
    visual.fallback='textureless_template';prior=visual;return;
end
C=normxcorr2(patch,b);maps=1;
visual=peak(C,-r.max_shift_px:r.max_shift_px,-r.max_shift_px:r.max_shift_px,r.minimum_ncc);
diagnostic.integer_dx=visual.integer_dx;diagnostic.integer_dy=visual.integer_dy;
xs=-r.max_shift_px:r.max_shift_px;ys=xs;sub=C(192+ys,192+xs);
[xx,yy]=meshgrid(xs,ys);outside=max(abs(xx-visual.integer_dx),abs(yy-visual.integer_dy))>g.peak_exclusion_chebyshev_radius_px;
if any(outside,'all') && all(isfinite(sub),'all'),diagnostic.second_score=max(sub(outside));end
has=isfinite(delta) && abs(delta)<=r.max_shift_px;
if has
    low=max(-r.max_shift_px,ceil(-delta-r.prior_radius_px));high=min(r.max_shift_px,floor(-delta+r.prior_radius_px));
    prior=peak(C,-r.prior_radius_px:r.prior_radius_px,low:high,r.minimum_ncc);prior.used_prior=true;
    if ~prior.available,prior=visual;prior.fallback='low_local_ncc_to_visual';prior.used_prior=false;end
else
    prior=visual;prior.fallback='missing_prior_to_visual';prior.used_prior=false;
end
end
function e=blank()
e=struct('available',false,'dx',NaN,'dy',NaN,'score',NaN,'used_prior',false,...
    'fallback','none','boundary_peak',false,'candidate_count',0,'integer_dx',NaN,'integer_dy',NaN);
end
function e=peak(C,xs,ys,minimum)
e=blank();if isempty(xs)||isempty(ys),return;end
sub=C(192+ys,192+xs);[score,k]=max(sub(:));[row,col]=ind2sub(size(sub),k);
e.score=score;e.candidate_count=numel(sub);e.boundary_peak=row==1||col==1||row==numel(ys)||col==numel(xs);
e.integer_dx=xs(col);e.integer_dy=ys(row);
if ~isfinite(score)||score<minimum,e.fallback='low_ncc';return;end
ox=0;oy=0;
if col>1&&col<numel(xs),ox=quadratic(sub(row,col-1),score,sub(row,col+1));end
if row>1&&row<numel(ys),oy=quadratic(sub(row-1,col),score,sub(row+1,col));end
e.available=true;e.dx=xs(col)+ox;e.dy=ys(row)+oy;
end
function d=quadratic(a,b,c)
den=a-2*b+c;d=0;if den < -1e-12,d=max(-.5,min(.5,.5*(a-c)/den));end
end
function y=clip(x),y=min(1,max(0,x));end
