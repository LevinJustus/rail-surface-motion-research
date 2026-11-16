function report = check_masked_segmentation_loss(outDir)
%CHECK_MASKED_SEGMENTATION_LOSS Synthetic value, masking and gradient checks.
% No real image, model training, optimizer update, or held-out data is used.
% Returns a report and writes loss_check.json (including failure diagnostics).
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<1,outDir=fullfile(root,'runs','masked_loss_check');end
if ~exist(outDir,'dir'),mkdir(outDir);end
cfg=jsondecode(fileread(fullfile(root,'configs','training_stage1_v1.json')));
lossCfg=cfg.loss;
report=struct('status','RUNNING','synthetic_inputs_only',true, ...
    'real_images_opened',0,'optimizer_updates',0,'matlab_version',version, ...
    'logits_layout','H-by-W-by-2-by-B; SSCB; pre-softmax; background then defect', ...
    'target_valid_layout','H-by-W-by-1-by-B; binary numeric or dlarray', ...
    'diagnostics','bce and dice are unweighted per-image means');
try
    % One positive pixel with p(defect)=3/4 has an independent closed-form
    % value and derivative. This also catches a swapped class or second sigmoid.
    z=reshape([0 log(3)],[1 1 2 1]);
    target=ones(1,1,1,1);valid=ones(1,1,1,1);
    [L,B,D,P,G]=dlfeval(@loss_gradient,dlarray(z,'SSCB'),target,valid,lossCfg);
    epsD=lossCfg.dice_smoothing;prob=0.75;
    expectB=-log(prob);expectD=1-(2*prob+epsD)/(prob+1+epsD);
    expectL=lossCfg.bce_weight_positive_target*expectB+ ...
        lossCfg.dice_weight_positive_target*expectD;
    gDef=lossCfg.bce_weight_positive_target*(prob-1) - ...
        lossCfg.dice_weight_positive_target*(2+epsD)/(prob+1+epsD)^2*prob*(1-prob);
    expectG=reshape([-gDef gDef],[1 1 2 1]);
    valError=max(abs([scalar(L)-expectL scalar(B)-expectB scalar(D)-expectD scalar(P)-prob]));
    gradError=max(abs(raw(G)-expectG),[],'all');
    assert(valError<1e-10 && gradError<1e-10,'rail:losscheck:Analytic', ...
        'Analytic positive-pixel value or gradient check failed.');
    report.analytic=struct('max_value_abs_error',valError, ...
        'max_gradient_abs_error',gradError,'tolerance',1e-10);

    % Two images have different valid areas and one has an empty target.
    % The answer is the mean of image losses, not a pooled spatial loss.
    z=zeros(2,2,2,2);
    target=zeros(2,2,1,2);target(:,:,1,1)=[1 0;0 1];
    valid=zeros(2,2,1,2);valid(:,:,1,1)=[1 1;1 0];valid(:,:,1,2)=[1 0;1 0];
    [L,B,D,P]=masked_segmentation_loss(dlarray(z,'SSCB'),target,valid,lossCfg);
    d1=1-(1+epsD)/(2.5+epsD);d2=1-epsD/(1+epsD);
    expectL=(lossCfg.bce_weight_positive_target*log(2)+ ...
        lossCfg.dice_weight_positive_target*d1+log(2))/2;
    err=max(abs([scalar(L)-expectL scalar(B)-log(2) scalar(D)-(d1+d2)/2]));
    assert(err<1e-10 && max(abs(raw(P)-0.5),[],'all')<1e-12, ...
        'rail:losscheck:ImageMean','Per-image averaging or component diagnostics are incorrect.');
    assert(strcmp(dims(P),'SSCB') && isequal(size(P,[1 2 3 4]),[2 2 1 2]), ...
        'rail:losscheck:ProbabilityShape','Defect probability layout changed.');
    report.per_image_mixed_batch=struct('max_abs_error',err,'tolerance',1e-10);

    % Central finite differences on every logit check an independently
    % perturbed, nonuniform batch with both foreground and empty target cases.
    z=reshape(sin(1:24)*0.8+cos((1:24)*0.7)*0.3,[2 3 2 2]);
    target=zeros(2,3,1,2);target(:,:,1,1)=[1 0 1;0 1 0];
    valid=ones(2,3,1,2);valid(2,3,1,1)=0;valid(1,2,1,2)=0;
    [L,~,~,P,G]=dlfeval(@loss_gradient,dlarray(z,'SSCB'),target,valid,lossCfg);
    autoGrad=raw(G);finiteGrad=zeros(size(z));h=1e-5;
    for k=1:numel(z)
        plus=z;minus=z;plus(k)=plus(k)+h;minus(k)=minus(k)-h;
        lp=masked_segmentation_loss(dlarray(plus,'SSCB'),target,valid,lossCfg);
        lm=masked_segmentation_loss(dlarray(minus,'SSCB'),target,valid,lossCfg);
        finiteGrad(k)=(scalar(lp)-scalar(lm))/(2*h);
    end
    absError=max(abs(autoGrad-finiteGrad),[],'all');
    relError=norm(autoGrad(:)-finiteGrad(:))/max(norm(finiteGrad(:)),1e-12);
    assert(all(isfinite(autoGrad),'all') && norm(autoGrad(:))>0 && ...
        absError<1e-6 && relError<1e-5,'rail:losscheck:FiniteDifference', ...
        'Backward derivative disagrees with central finite differences.');
    report.finite_difference=struct('step',h,'elements_checked',numel(z), ...
        'max_abs_error',absError,'relative_l2_error',relError, ...
        'max_abs_tolerance',1e-6,'relative_l2_tolerance',1e-5);

    % Change both probability and truth only in padding. Valid-pixel results
    % must be unchanged, and every padding logit derivative must be zero.
    changed=z;targetChanged=target;
    for batch=1:2
        invalid=~logical(valid(:,:,1,batch));
        for channel=1:2
            plane=changed(:,:,channel,batch);plane(invalid)=25*(2*channel-3);
            changed(:,:,channel,batch)=plane;
        end
        plane=targetChanged(:,:,1,batch);plane(invalid)=1;targetChanged(:,:,1,batch)=plane;
    end
    [newL,~,~,newP,newG]=dlfeval(@loss_gradient,dlarray(changed,'SSCB'), ...
        targetChanged,valid,lossCfg);
    padLogits=repmat(~logical(valid),[1 1 2 1]);
    padProb=~logical(valid);newGrad=raw(newG);
    paddingGrad=max(abs([autoGrad(padLogits);newGrad(padLogits)]));
    validGradChange=max(abs(autoGrad(~padLogits)-newGrad(~padLogits)));
    padProbabilityChange=max(abs(raw(newP).*double(padProb)-raw(P).*double(padProb)),[],'all');
    lossChange=abs(scalar(newL)-scalar(L));
    assert(padProbabilityChange>0.1 && lossChange<1e-12 && ...
        paddingGrad<1e-12 && validGradChange<1e-12, ...
        'rail:losscheck:Padding','Padding changed loss or valid gradient, or received nonzero gradient.');
    report.padding=struct('probability_change_on_padding',padProbabilityChange, ...
        'loss_abs_change',lossChange,'max_padding_gradient',paddingGrad, ...
        'max_valid_gradient_change',validGradChange,'tolerance',1e-12);

    % Labels/masks passed as SSCB dlarrays must agree with numeric constants.
    [dlL,dlB,dlD,dlP]=masked_segmentation_loss(dlarray(z,'SSCB'), ...
        dlarray(single(target),'SSCB'),dlarray(single(valid),'SSCB'),lossCfg);
    [numL,numB,numD,numP]=masked_segmentation_loss(dlarray(z,'SSCB'),target,valid,lossCfg);
    dlError=max(abs([scalar(dlL)-scalar(numL) scalar(dlB)-scalar(numB) scalar(dlD)-scalar(numD)]));
    dlError=max(dlError,max(abs(raw(dlP)-raw(numP)),[],'all'));
    assert(dlError<1e-12,'rail:losscheck:TargetContainer','Numeric and dlarray targets disagree.');
    report.target_container=struct('max_abs_error',dlError,'tolerance',1e-12);

    % Empty foreground uses the complete BCE, not half BCE or Dice weighting.
    [eL,eB,~,~,eG]=dlfeval(@loss_gradient,dlarray(zeros(2,2,2,1),'SSCB'), ...
        zeros(2,2,1,1),ones(2,2,1,1),lossCfg);
    emptyExpectedG=zeros(2,2,2,1);emptyExpectedG(:,:,1,1)=-1/8;emptyExpectedG(:,:,2,1)=1/8;
    emptyErr=max(abs([scalar(eL)-log(2) scalar(eB)-log(2)]));
    emptyGradErr=max(abs(raw(eG)-emptyExpectedG),[],'all');
    assert(emptyErr<1e-10 && emptyGradErr<1e-10,'rail:losscheck:EmptyTarget', ...
        'Empty-target loss is not full-weight BCE.');
    report.empty_target=struct('max_value_abs_error',emptyErr, ...
        'max_gradient_abs_error',emptyGradErr,'tolerance',1e-10);

    % Production single precision: large but finite logits must not overflow
    % softmax/log or sever the gradient path. This is not a bound for realmax.
    extreme=zeros(2,2,2,2,'single');
    extreme(:,:,1,1)=[1000 -1000;1000 -1000];
    extreme(:,:,2,1)=-extreme(:,:,1,1);
    extreme(:,:,:,2)=-extreme(:,:,:,1);
    exTarget=zeros(2,2,1,2,'single');exTarget(:,:,1,1)=[1 0;0 1];
    [exL,exB,exD,exP,exG]=dlfeval(@loss_gradient,dlarray(extreme,'SSCB'), ...
        exTarget,ones(2,2,1,2,'single'),lossCfg);
    vals=[scalar(exL) scalar(exB) scalar(exD)];exGrad=raw(exG);exProb=raw(exP);
    assert(all(isfinite(vals)) && all(isfinite(exGrad),'all') && ...
        all(isfinite(exProb),'all') && all(exProb>=0 & exProb<=1,'all') && ...
        norm(double(exGrad(:)))>0,'rail:losscheck:ExtremeLogits', ...
        'Extreme finite single logits produced nonfinite values or unusable gradients.');
    report.extreme_single=struct('max_abs_logit',1000,'loss',vals(1), ...
        'bce',vals(2),'dice',vals(3),'gradient_l2',norm(double(exGrad(:))));

    % Contract failures are explicit and must not silently drop samples.
    dz=dlarray(zeros(2,2,2,2,'single'),'SSCB');
    goodT=zeros(2,2,1,2,'single');goodV=ones(2,2,1,2,'single');
    badV=goodV;badV(:,:,:,2)=0;
    expect_error(@()masked_segmentation_loss(dz,goodT,badV,lossCfg),'rail:loss:EmptyValidRegion');
    badT=goodT;badT(1)=0.25;
    expect_error(@()masked_segmentation_loss(dz,badT,goodV,lossCfg),'rail:loss:TargetValues');
    badT(1)=NaN;
    expect_error(@()masked_segmentation_loss(dz,badT,goodV,lossCfg),'rail:loss:TargetValues');
    badV=goodV;badV(1)=0.5;
    expect_error(@()masked_segmentation_loss(dz,goodT,badV,lossCfg),'rail:loss:ValidMaskValues');
    expect_error(@()masked_segmentation_loss(dz,zeros(2,2,2,2,'single'),goodV,lossCfg), ...
        'rail:loss:TargetShape');
    report.rejected_invalid_inputs={'empty valid region in one image', ...
        'fractional target','NaN target','fractional valid mask','two-channel target'};
    report.status='PASS';
catch err
    report.status='FAIL';report.error_identifier=err.identifier;report.error_message=err.message;
    write_report(outDir,report);
    rethrow(err);
end
write_report(outDir,report);
fprintf('MASKED_LOSS_CHECK_OK analytic finiteDifference padding emptyTarget extremeSingle targetContract\n');
end

function [loss,bce,dice,p,g]=loss_gradient(z,target,valid,cfg)
[loss,bce,dice,p]=masked_segmentation_loss(z,target,valid,cfg);
g=dlgradient(loss,z);
end

function x=raw(x)
if isa(x,'dlarray'),x=extractdata(x);end
if isa(x,'gpuArray'),x=gather(x);end
end

function n=scalar(x)
n=double(raw(x));
assert(isscalar(n),'rail:losscheck:Scalar','Expected a scalar loss or component.');
end

function expect_error(callback,identifier)
try
    callback();
catch err
    assert(strcmp(err.identifier,identifier),'rail:losscheck:WrongError', ...
        'Expected %s but received %s.',identifier,err.identifier);
    return
end
error('rail:losscheck:MissingError','Expected rejection %s did not occur.',identifier);
end

function write_report(outDir,report)
fid=fopen(fullfile(outDir,'loss_check.json'),'w','n','UTF-8');
assert(fid>=0,'rail:losscheck:ReportFile','Cannot open loss-check report.');
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
end
