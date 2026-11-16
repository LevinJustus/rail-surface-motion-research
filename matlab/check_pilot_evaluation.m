function report = check_pilot_evaluation(outDir)
%CHECK_PILOT_EVALUATION Independent constant-network aggregation fixtures.
% Five images, unequal valid areas, groups crossing batch boundaries and
% one empty target test actual aggregation, not a second implementation.
if nargin<1,outDir=fullfile(fileparts(fileparts(mfilename('fullpath'))),'runs','pilot_evaluation_check');end
if ~exist(outDir,'dir'),mkdir(outDir);end
root=fileparts(fileparts(mfilename('fullpath')));
cfg=jsondecode(fileread(fullfile(root,'configs','training_stage1_v1.json')));
report.status='RUNNING';report.synthetic_images_only=true;
try
    layers=[imageInputLayer([2 2 3],Normalization='none',Name='input')
        convolution2dLayer(1,2,Weights=zeros(1,1,3,2,'single'),Bias=zeros(1,1,2,'single'),Name='logits')
        softmaxLayer(Name='probabilities')];
    net=dlnetwork(layers);
    X=zeros(2,2,3,5,'single');T=zeros(2,2,1,5,'single');V=T;
    valid=[4 4 2 3 1];fg=[1 2 1 0 1];
    for k=1:5
        v=zeros(2,2,'single');t=v;v(1:valid(k))=1;t(1:fg(k))=1;
        V(:,:,1,k)=v;T(:,:,1,k)=t;
    end
    ids=["a1";"b1";"a2";"c1";"b2"];groups=["A";"B";"A";"C";"B"];
    one=evaluate_pilot(net,X,T,V,ids,groups,cfg.loss,1);
    two=evaluate_pilot(net,X,T,V,ids,groups,cfg.loss,2);
    five=evaluate_pilot(net,X,T,V,ids,groups,cfg.loss,5);
    expected=[5/14,10/19,5/14,1,0.45,(1/3+3/5)/3];
    fields={'micro_foreground_IoU','micro_foreground_Dice','micro_precision', ...
        'micro_recall','macro_image_foreground_IoU','macro_group_foreground_IoU'};
    err=0;
    for k=1:numel(fields)
        values=[one.(fields{k}),two.(fields{k}),five.(fields{k})];
        err=max(err,max(abs(values-expected(k))));
    end
    epsD=cfg.loss.dice_smoothing;
    expectedPerImage=zeros(1,5);
    for k=1:5
        if fg(k)==0,expectedPerImage(k)=log(2);
        else,expectedPerImage(k)=0.5*log(2)+0.5*(1-(fg(k)+epsD)/(0.5*valid(k)+fg(k)+epsD));end
    end
    lossError=max(abs([one.total_loss,two.total_loss,five.total_loss]-mean(expectedPerImage)));
    assert(err<1e-10 && lossError<1e-6,'Batched metric/loss aggregation mismatch');
    assert(two.TP==5 && two.FP==9 && two.FN==0 && two.TN==0 && two.valid_pixels==14);
    report.aggregation=struct('batch_sizes',[1 2 5],'sample_count',5, ...
        'metric_max_abs_error',err,'loss_max_abs_error',lossError, ...
        'expected_micro_IoU',5/14,'expected_macro_group_IoU',(1/3+3/5)/3);
    layers(2).Bias=reshape(single([10 -10]),1,1,2);
    backgroundNet=dlnetwork(layers);
    bg=evaluate_pilot(backgroundNet,X,T,V,ids,groups,cfg.loss,2);
    assert(bg.all_background_image_count==5 && bg.micro_foreground_IoU==0 && ...
        isnan(bg.micro_precision) && bg.micro_recall==0);
    empty=evaluate_pilot(backgroundNet,X,zeros(size(T),'single'),V,ids,groups,cfg.loss,2);
    assert(isnan(empty.micro_foreground_IoU) && isnan(empty.micro_foreground_Dice));
    reference=struct('global_update',0,'total_loss',0.7,'micro_foreground_IoU',0.2);
    assert(pilot_is_better(reference,[]));
    candidate=reference;candidate.global_update=64;
    assert(~pilot_is_better(candidate,reference));
    candidate.total_loss=0.6;assert(pilot_is_better(candidate,reference));
    candidate.micro_foreground_IoU=0.1;assert(~pilot_is_better(candidate,reference));
    candidate.micro_foreground_IoU=NaN;assert(~pilot_is_better(candidate,[]));
    candidate=reference;candidate.micro_foreground_IoU=0;assert(pilot_is_better(candidate,[]));
    report.nan_and_best_rules='PASS; empty IoU excluded, zero IoU eligible, update0 included, validation IoU/loss/earlier ordering';
    report.status='PASS';
catch err
    report.status='FAIL';report.error_identifier=err.identifier;report.error_message=err.message;
    writeReport(outDir,report);rethrow(err);
end
writeReport(outDir,report);
fprintf('PILOT_EVALUATION_CHECK_OK batching crossBatchGroups remainder loss NaN bestIncludingZero\n');
end

function writeReport(outDir,report)
fid=fopen(fullfile(outDir,'pilot_evaluation_check.json'),'w','n','UTF-8');assert(fid>=0);
c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
end
