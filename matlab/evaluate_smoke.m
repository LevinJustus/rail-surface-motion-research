function report = evaluate_smoke(net,X,T,V,ids,groupIds,lossCfg)
%EVALUATE_SMOKE Fixed training-subset diagnostics; never a test-set score.
% X is H-by-W-by-3-by-B; T/V are binary H-by-W-by-1-by-B arrays.
% H-by-W-by-B target arrays are also accepted and explicitly canonicalized.
% Channel 1 is background, channel 2 defect. Prediction threshold is >=0.5.
% Group IoU first pools TP/FP/FN within a proxy group, then averages groups.
% Undefined ratios remain NaN (MATLAB jsonencode normally writes null).

assert(isnumeric(X) && isreal(X) && ndims(X)<=4 && size(X,3)==3, ...
    'evaluate_smoke:InputShape','Expected real H-by-W-by-3-by-B input');
assert(all(isfinite(X(:))) && all(X(:)>=0 & X(:)<=1), ...
    'evaluate_smoke:InputRange','Inputs must be finite and normalized once to [0,1]');
[h,w,~,n]=size(X);
assert(n>=1,'evaluate_smoke:EmptyBatch','At least one image is required');
ids=string(ids(:)); groupIds=string(groupIds(:));
assert(numel(ids)==n && numel(groupIds)==n && ...
    all(strlength(ids)>0) && all(strlength(groupIds)>0), ...
    'evaluate_smoke:IDs','One nonempty sample/group ID is required per image');
assert(numel(unique(ids))==n,'evaluate_smoke:DuplicateIDs', ...
    'Fixed evaluation IDs must be unique');
T=canonical_mask(T,h,w,n,'target'); V=canonical_mask(V,h,w,n,'valid');
validCounts=reshape(sum(V,[1 2]),n,1);
assert(all(validCounts>0),'evaluate_smoke:EmptyValidRegion', ...
    'An image has no valid pixels; samples must not be silently dropped');

% Locate the final softmax and its immediate input; do not hard-code a
% proprietary network's internal layer name or apply another sigmoid.
[logitsName,softmaxName]=output_layers(net);
dlX=dlarray(single(X),'SSCB');
[logits,probability]=predict(net,dlX,Outputs=[logitsName softmaxName]);
P=host_numeric(probability);
assert(isequal(size(P,[1 2 3 4]),[h w 2 n]), ...
    'evaluate_smoke:ProbabilityShape','Expected H-by-W-by-2-by-B probabilities');
assert(all(isfinite(P(:))),'evaluate_smoke:NonfiniteProbability', ...
    'Nonfinite probability is a numerical error, not an undefined metric');
sumError=max(abs(sum(P,3)-1),[],'all');
assert(min(P,[],'all')>=-1e-6 && max(P,[],'all')<=1+1e-6 && sumError<=1e-4, ...
    'evaluate_smoke:InvalidProbability','Probabilities must lie in [0,1] and sum to one');

[totalLoss,bce,dice,lossProbability]=masked_segmentation_loss( ...
    logits,dlarray(T,'SSCB'),dlarray(V,'SSCB'),lossCfg);
lossValues=[scalar_value(totalLoss) scalar_value(bce) scalar_value(dice)];
assert(all(isfinite(lossValues)),'evaluate_smoke:NonfiniteLoss', ...
    'All diagnostic losses must be finite');
defectProbability=reshape(P(:,:,2,:),h,w,1,n);
fromLoss=host_numeric(lossProbability);
% The helper may return defect probabilities or the complete two-channel map.
if isequal(size(fromLoss,[1 2 3 4]),[h w 2 n])
    fromLoss=reshape(fromLoss(:,:,2,:),h,w,1,n);
end
assert(isequal(size(fromLoss,[1 2 3 4]),[h w 1 n]), ...
    'evaluate_smoke:LossProbabilityShape','Loss helper returned an unexpected probability shape');
probabilityDifference=max(abs(double(fromLoss(:))-double(defectProbability(:))));
assert(isfinite(probabilityDifference) && probabilityDifference<=1e-5, ...
    'evaluate_smoke:ProbabilityMismatch','Loss and inference probabilities disagree');

template=image_record(); perImage=repmat(template,n,1);
for k=1:n
    truth=T(:,:,1,k)>0; valid=V(:,:,1,k)>0;
    foreground=defectProbability(:,:,1,k)>=0.5;
    counts=[nnz(valid & truth & foreground),nnz(valid & ~truth & foreground), ...
        nnz(valid & truth & ~foreground),nnz(valid & ~truth & ~foreground)];
    perImage(k)=populate_record(template,counts);
    perImage(k).id=char(ids(k)); perImage(k).group_id=char(groupIds(k));
    perImage(k).valid_pixels=sum(counts);
    perImage(k).target_foreground_pixels=counts(1)+counts(3);
    perImage(k).predicted_foreground_pixels=counts(1)+counts(2);
    perImage(k).target_foreground_fraction=ratio(counts(1)+counts(3),sum(counts));
    perImage(k).predicted_foreground_fraction=ratio(counts(1)+counts(2),sum(counts));
    perImage(k).all_background_prediction=(counts(1)+counts(2)==0);
    pixelProb=defectProbability(:,:,1,k);
    perImage(k).mean_defect_probability=mean(double(pixelProb(valid)));
end

groups=unique(groupIds,'stable'); perGroup=repmat(template,numel(groups),1);
for k=1:numel(groups)
    selected=groupIds==groups(k);
    counts=[sum([perImage(selected).TP]),sum([perImage(selected).FP]), ...
        sum([perImage(selected).FN]),sum([perImage(selected).TN])];
    perGroup(k)=populate_record(template,counts);
    perGroup(k).id=char(groups(k)); perGroup(k).group_id=char(groups(k));
    perGroup(k).image_count=nnz(selected);
    perGroup(k).valid_pixels=sum(counts);
    perGroup(k).target_foreground_pixels=counts(1)+counts(3);
    perGroup(k).predicted_foreground_pixels=counts(1)+counts(2);
    perGroup(k).target_foreground_fraction=ratio(counts(1)+counts(3),sum(counts));
    perGroup(k).predicted_foreground_fraction=ratio(counts(1)+counts(2),sum(counts));
    perGroup(k).all_background_prediction=(counts(1)+counts(2)==0);
    means=[perImage(selected).mean_defect_probability];
    weights=[perImage(selected).valid_pixels];
    perGroup(k).mean_defect_probability=sum(means.*weights)/sum(weights);
end

tp=sum([perImage.TP]); fp=sum([perImage.FP]);
fn=sum([perImage.FN]); tn=sum([perImage.TN]); nv=tp+fp+fn+tn;
report.schema_version=1;
report.scope='fixed smoke training images only; not validation/test or generalization';
report.sample_count=n; report.proxy_group_count=numel(groups);
report.output_channel_order={'background','defect'};
report.foreground_threshold=0.5; report.threshold_comparison='>=';
report.total_loss=lossValues(1); report.BCE_component=lossValues(2);
report.Dice_component=lossValues(3);
report.TP=tp; report.FP=fp; report.FN=fn; report.TN=tn;
report.valid_pixels=nv;
report.micro_foreground_IoU=ratio(tp,tp+fp+fn);
report.micro_foreground_Dice=ratio(2*tp,2*tp+fp+fn);
report.micro_precision=ratio(tp,tp+fp);
report.micro_recall=ratio(tp,tp+fn);
report.macro_image_foreground_IoU=defined_mean([perImage.foreground_IoU]);
report.macro_group_foreground_IoU=defined_mean([perGroup.foreground_IoU]);
report.macro_group_definition='pool confusion counts within each proxy group, then equal mean over defined group IoUs';
report.predicted_foreground_fraction=ratio(tp+fp,nv);
report.target_foreground_fraction=ratio(tp+fn,nv);
report.all_background_prediction=(tp+fp==0);
report.all_background_image_count=nnz([perImage.all_background_prediction]);
report.zero_target_image_count=nnz([perImage.target_foreground_pixels]==0);
report.undefined_image_metrics=undefined_counts(perImage);
report.undefined_group_metrics=undefined_counts(perGroup);
report.micro_metric_denominators=struct('IoU',tp+fp+fn, ...
    'Dice',2*tp+fp+fn,'precision',tp+fp,'recall',tp+fn);
report.probability_checks=struct('all_finite',true,'minimum',min(P,[],'all'), ...
    'maximum',max(P,[],'all'),'maximum_channel_sum_error',sumError, ...
    'loss_inference_maximum_difference',probabilityDifference, ...
    'logits_layer',char(logitsName),'softmax_layer',char(softmaxName));
report.per_image=perImage; report.per_group=perGroup;
end

function M=canonical_mask(M,h,w,n,description)
assert((isnumeric(M)||islogical(M)) && isreal(M) && ndims(M)<=4, ...
    'evaluate_smoke:MaskType','Invalid %s array type',description);
if isequal(size(M,[1 2 3 4]),[h w 1 n])
    M=reshape(single(M),h,w,1,n);
elseif n>1 && isequal(size(M,[1 2 3 4]),[h w n 1])
    M=reshape(single(M),h,w,1,n);
else
    error('evaluate_smoke:MaskShape','Unexpected %s dimensions',description);
end
assert(all(isfinite(M(:))) && all(M(:)==0 | M(:)==1), ...
    'evaluate_smoke:MaskValues','%s must contain finite binary 0/1 values',description);
end

function [logitsName,softmaxName]=output_layers(net)
assert(isscalar(net.OutputNames),'evaluate_smoke:NetworkOutputs','Expected one final softmax output');
softmaxName=string(net.OutputNames{1});
names=string({net.Layers.Name});
idx=find(names==softmaxName);
assert(isscalar(idx) && isa(net.Layers(idx),'nnet.cnn.layer.SoftmaxLayer'), ...
    'evaluate_smoke:FinalSoftmax','The network must end in a two-class softmax layer');
destinations=string(net.Connections.Destination);
rows=(destinations==softmaxName)|startsWith(destinations,softmaxName+"/");
assert(nnz(rows)==1,'evaluate_smoke:LogitsConnection','Expected one input connection to final softmax');
sources=string(net.Connections.Source);
logitsName=sources(rows);
end

function value=host_numeric(value)
if isa(value,'dlarray'),value=extractdata(value);end
if isa(value,'gpuArray'),value=gather(value);end
end

function value=scalar_value(value)
value=host_numeric(value);
assert(isscalar(value),'evaluate_smoke:LossScalar','Expected scalar loss components');
value=double(value);
end

function value=ratio(numerator,denominator)
if denominator==0,value=NaN;else,value=double(numerator)/double(denominator);end
end

function value=defined_mean(values)
values=values(~isnan(values));
if isempty(values),value=NaN;else,value=mean(values);end
end

function record=image_record()
record=struct('id','','group_id','','image_count',1,'valid_pixels',0, ...
    'target_foreground_pixels',0,'predicted_foreground_pixels',0, ...
    'TP',0,'FP',0,'FN',0,'TN',0,'foreground_IoU',NaN,'foreground_Dice',NaN, ...
    'precision',NaN,'recall',NaN,'target_foreground_fraction',NaN, ...
    'predicted_foreground_fraction',NaN,'all_background_prediction',false, ...
    'mean_defect_probability',NaN);
end

function record=populate_record(record,counts)
record.TP=counts(1);record.FP=counts(2);record.FN=counts(3);record.TN=counts(4);
record.foreground_IoU=ratio(counts(1),counts(1)+counts(2)+counts(3));
record.foreground_Dice=ratio(2*counts(1),2*counts(1)+counts(2)+counts(3));
record.precision=ratio(counts(1),counts(1)+counts(2));
record.recall=ratio(counts(1),counts(1)+counts(3));
end

function output=undefined_counts(records)
output=struct('foreground_IoU',nnz(isnan([records.foreground_IoU])), ...
    'foreground_Dice',nnz(isnan([records.foreground_Dice])), ...
    'precision',nnz(isnan([records.precision])),'recall',nnz(isnan([records.recall])));
end
