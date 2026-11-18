function q=ablation_quality_factors(edge,cfg)
q=struct('full',0,'without_cycle',0,'without_margin',0);
a=edge.visual;b=edge.reverse;
if ~a.available||~b.available||a.boundary_peak||b.boundary_peak,return;end
sf=edge.forward_diagnostic.second_score;sb=edge.reverse_diagnostic.second_score;
if ~isfinite(sf)||~isfinite(sb),return;end
clip=@(x)min(1,max(0,x));
score=min(clip((a.score-.2)/cfg.score_scale),clip((b.score-.2)/cfg.score_scale));
margin=min(clip((a.score-sf)/cfg.peak_margin_scale),clip((b.score-sb)/cfg.peak_margin_scale));
cycle=exp(-hypot(a.dx+b.dx,a.dy+b.dy)^2/(2*cfg.cycle_sigma_px^2));
q.full=score*margin*cycle;q.without_cycle=score*margin;q.without_margin=score*cycle;
assert(abs(q.full-edge.quality)<1e-12);
end
