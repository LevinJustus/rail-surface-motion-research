function [est,diag]=register_motion_pair(previous,current,deltaOdom,cfg)
% RGB + optional observed scalar odometry only; no labels/geometry/source ID.
assert(isa(previous,'uint8') && isa(current,'uint8') && isequal(size(previous),[256 256 3]) && isequal(size(current),[256 256 3]));
assert(isscalar(deltaOdom) && (isfinite(deltaOdom)||isnan(deltaOdom)));
templateStart=65;templateSize=128;basePeak=192;
g=@(a).2989*double(a(:,:,1))/255+.5870*double(a(:,:,2))/255+.1140*double(a(:,:,3))/255;
a=g(previous);b=g(current);patch=a(templateStart:templateStart+templateSize-1,templateStart:templateStart+templateSize-1);
methods={'odometry','visual_ncc','prior_ncc'};
blank=struct('method','','available',false,'dx',NaN,'dy',NaN,'score',NaN,'used_prior',false,'fallback','none','boundary_peak',false,'candidate_count',0,'selection_seconds',0);
est=repmat(blank,3,1);for j=1:3,est(j).method=methods{j};end
has=isfinite(deltaOdom) && abs(deltaOdom)<=cfg.max_shift_px;
if has,est(1).available=true;est(1).dx=0;est(1).dy=-deltaOdom;est(1).used_prior=true;else,est(1).fallback='odometry_unavailable_or_out_of_range';end
timer=tic;
if std(patch(:))<cfg.minimum_template_std
    diag=struct('shared_ncc_seconds',toc(timer),'template_std',std(patch(:)),'ncc_computed',false);
    est(2).fallback='textureless_template';est(3).fallback='textureless_template';return;
end
C=normxcorr2(patch,b);nccSeconds=toc(timer);
timer=tic;visual=selectPeak(C,basePeak,[-cfg.max_shift_px cfg.max_shift_px],[-cfg.max_shift_px cfg.max_shift_px],cfg.minimum_ncc);
visual.selection_seconds=toc(timer);visual.method=methods{2};est(2)=visual;
timer=tic;
if has
    lowY=max(-cfg.max_shift_px,ceil(-deltaOdom-cfg.prior_radius_px));highY=min(cfg.max_shift_px,floor(-deltaOdom+cfg.prior_radius_px));
    prior=selectPeak(C,basePeak,[-cfg.prior_radius_px cfg.prior_radius_px],[lowY highY],cfg.minimum_ncc);prior.used_prior=true;
    if ~prior.available
        prior=visual;prior.fallback='low_local_ncc_to_visual';prior.used_prior=false;
    end
else
    prior=visual;prior.fallback='missing_prior_to_visual';prior.used_prior=false;
end
prior.selection_seconds=toc(timer);prior.method=methods{3};est(3)=prior;
diag=struct('shared_ncc_seconds',nccSeconds,'template_std',std(patch(:)),'ncc_computed',true);
end
function e=selectPeak(C,base,xRange,yRange,minScore)
e=struct('method','','available',false,'dx',NaN,'dy',NaN,'score',NaN,'used_prior',false,'fallback','none','boundary_peak',false,'candidate_count',0,'selection_seconds',0);
xs=xRange(1):xRange(2);ys=yRange(1):yRange(2);if isempty(xs)||isempty(ys),return;end
sub=C(base+ys,base+xs);[score,k]=max(sub(:));[row,col]=ind2sub(size(sub),k);
e.score=score;e.candidate_count=numel(sub);e.boundary_peak=(row==1||col==1||row==numel(ys)||col==numel(xs));
if ~isfinite(score)||score<minScore,e.fallback='low_ncc';return;end
dx=xs(col);dy=ys(row);ox=0;oy=0;
if col>1&&col<numel(xs),ox=quadratic(sub(row,col-1),score,sub(row,col+1));end
if row>1&&row<numel(ys),oy=quadratic(sub(row-1,col),score,sub(row+1,col));end
e.available=true;e.dx=dx+ox;e.dy=dy+oy;
end
function d=quadratic(a,b,c)
den=a-2*b+c;d=0;if den < -1e-12,d=max(-.5,min(.5,.5*(a-c)/den));end
end
