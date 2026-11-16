function better = pilot_is_better(candidate,best)
%PILOT_IS_BETTER Frozen validation-only diagnostic ordering, including update0.
assert(isfinite(candidate.total_loss),'Nonfinite validation loss is an error');
score=candidate.micro_foreground_IoU;
assert(isnan(score)||(isfinite(score)&&score>=0&&score<=1),'Invalid validation IoU');
if isnan(score),better=false;return;end
if isempty(best),better=true;return;end
assert(isfinite(best.micro_foreground_IoU)&&isfinite(best.total_loss));
better=score>best.micro_foreground_IoU || ...
    (score==best.micro_foreground_IoU && candidate.total_loss<best.total_loss) || ...
    (score==best.micro_foreground_IoU && candidate.total_loss==best.total_loss && ...
    candidate.global_update<best.global_update);
end
