function edge=ablation_pressure_edge(base,C,delta,cfg)
% Re-select only prior NCC from the same 41x41 forward map; global visual fixed.
edge=base;r=cfg.registration;has=isfinite(delta)&&abs(delta)<=r.max_shift_px;
edge.observed_delta=delta;edge.observed_available=has;edge.observed_factor=1;edge.ncc_maps=0;
if has && base.visual.available
    edge.observed_factor=exp(-(base.visual.dy+delta)^2/(2*cfg.observed_gate.sigma_px^2));
end
if isempty(C),return;end
if has
    ys=max(-20,ceil(-delta-r.prior_radius_px)):min(20,floor(-delta+r.prior_radius_px));
    prior=select(C,-r.prior_radius_px:r.prior_radius_px,ys,r.minimum_ncc);prior.used_prior=true;
    if ~prior.available,prior=base.visual;prior.fallback='low_local_ncc_to_visual';prior.used_prior=false;end
else
    prior=base.visual;prior.fallback='missing_prior_to_visual';prior.used_prior=false;
end
edge.prior=prior;
end
function e=select(C,xs,ys,minimum)
e=struct('available',false,'dx',NaN,'dy',NaN,'score',NaN,'used_prior',false,...
    'fallback','none','boundary_peak',false,'candidate_count',0,'integer_dx',NaN,'integer_dy',NaN);
if isempty(xs)||isempty(ys),return;end
sub=C(21+ys,21+xs);[score,index]=max(sub(:));[row,col]=ind2sub(size(sub),index);
e.score=score;e.candidate_count=numel(sub);e.integer_dx=xs(col);e.integer_dy=ys(row);
e.boundary_peak=row==1||col==1||row==numel(ys)||col==numel(xs);
if ~isfinite(score)||score<minimum,e.fallback='low_ncc';return;end
ox=0;oy=0;
if col>1&&col<numel(xs),ox=quad(sub(row,col-1),score,sub(row,col+1));end
if row>1&&row<numel(ys),oy=quad(sub(row-1,col),score,sub(row+1,col));end
e.available=true;e.dx=xs(col)+ox;e.dy=ys(row)+oy;
end
function d=quad(a,b,c)
den=a-2*b+c;d=0;if den < -1e-12,d=max(-.5,min(.5,.5*(a-c)/den));end
end
