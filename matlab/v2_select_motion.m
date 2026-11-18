function edge=v2_select_motion(C,templateStd,observedDelta,cfg)
% Integer-grid candidate weights. C is the fixed NCC score grid, not labels.
lo=cfg.motion_grid_xy(1);hi=cfg.motion_grid_xy(2);axis=lo:hi;[dx,dy]=meshgrid(axis,axis);
assert(isequal(size(C),size(dx)));edge=struct('available',false,'reason','','candidates',zeros(0,3),...
    'retained_mass',0,'visual_mode_xy',[NaN NaN],'observed_reliability',0,'full_weights',zeros(size(C)),'visual_weights',zeros(size(C)));
if ~isfinite(templateStd)||templateStd<cfg.minimum_template_std,edge.reason='textureless';return;end
if any(~isfinite(C),'all'),edge.reason='nonfinite_scores';return;end
rank=sortrows([-C(:),dy(:),dx(:)],[1 2 3]);mode=[rank(1,3) rank(1,2)];maximum=-rank(1,1);edge.visual_mode_xy=mode;
if maximum<cfg.minimum_NCC,edge.reason='low_NCC';return;end
if cfg.reject_visual_boundary_peak&&any(mode==lo|mode==hi),edge.reason='boundary_peak';return;end
visual=exp((double(C)-maximum)/cfg.visual_temperature);visual=visual/sum(visual,'all');weights=visual;ro=0;
if isscalar(observedDelta)&&isfinite(observedDelta)&&abs(observedDelta)<=hi
    center=[0 -observedDelta];ro=1/(1+sum((mode-center).^2)/cfg.odometry_conflict_distance_scale_px2);
    variance=cfg.odometry_covariance_diagonal_px2;
    exponent=-.5*((dx-center(1)).^2/variance(1)+(dy-center(2)).^2/variance(2));
    prior=exp(exponent-max(exponent,[],'all'));prior=prior/sum(prior,'all');
    mixture=ro*prior+(1-ro)/numel(C);weights=visual.*mixture;weights=weights/sum(weights,'all');
end
rank=sortrows([-weights(:),dy(:),dx(:)],[1 2 3]);mass=-rank(:,1);count=find(cumsum(mass)>=cfg.candidate_mass_target,1);
if isempty(count),count=numel(mass);end;count=min(count,cfg.maximum_candidates_per_history);
selected=rank(1:count,:);selected=selected(selected(:,1)<0,:);
edge.available=~isempty(selected);edge.reason='accepted';edge.candidates=[selected(:,3) selected(:,2) -selected(:,1)];
edge.retained_mass=sum(edge.candidates(:,3));edge.observed_reliability=ro;edge.full_weights=weights;edge.visual_weights=visual;
assert(edge.retained_mass<=1+1e-12&&edge.retained_mass>0&&all(edge.candidates(:,3)>0));
end
