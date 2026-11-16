function report = evaluate_pilot(net,X,T,V,ids,groupIds,lossCfg,batchSize)
%EVALUATE_PILOT Batched fixed-validation diagnostics, not final test evidence.
% X: H-by-W-by-3-by-N. T/V: binary H-by-W-by-1-by-N; H-by-W-by-N
% is accepted and canonicalized. Intended frozen validation N is 64.
% Loss components are weighted by image count, NOT by valid-pixel count.
% Micro and group metrics are recomputed from all per-image confusion counts;
% neither batch IoUs nor batch group IoUs are averaged to form these metrics.
if nargin<8 || isempty(batchSize),batchSize=2;end
assert(isnumeric(batchSize) && isscalar(batchSize) && isfinite(batchSize) && ...
    batchSize>=1 && batchSize==fix(batchSize), ...
    'evaluate_pilot:BatchSize','batchSize must be a positive integer');
assert(isnumeric(X) && isreal(X) && ndims(X)<=4 && size(X,3)==3, ...
    'evaluate_pilot:InputShape','Expected real H-by-W-by-3-by-N input');
[h,w,~,n]=size(X);
assert(n>=1,'evaluate_pilot:EmptySubset','The validation subset must not be empty');
ids=string(ids(:));groupIds=string(groupIds(:));
assert(numel(ids)==n && numel(groupIds)==n && ...
    all(strlength(ids)>0) && all(strlength(groupIds)>0), ...
    'evaluate_pilot:IDs','One nonempty sample/group ID is required per image');
assert(numel(unique(ids))==n,'evaluate_pilot:DuplicateIDs', ...
    'Fixed validation IDs must be unique');
T=canonical_mask(T,h,w,n,'target');V=canonical_mask(V,h,w,n,'valid');

started=tic;
batchSize=double(batchSize);batchCount=ceil(n/batchSize);
lossSum=zeros(1,3);perImage=struct([]);batchRecords=struct([]);
probabilityMinimum=Inf;probabilityMaximum=-Inf;sumError=0;lossProbabilityError=0;
for first=1:batchSize:n
    last=min(first+batchSize-1,n);selected=first:last;nb=numel(selected);
    batch=evaluate_smoke(net,X(:,:,:,selected),T(:,:,:,selected),V(:,:,:,selected), ...
        ids(selected),groupIds(selected),lossCfg);
    assert(batch.sample_count==nb && numel(batch.per_image)==nb, ...
        'evaluate_pilot:BatchCount','Batch diagnostics omitted or duplicated samples');
    assert(isequal(string({batch.per_image.id})',ids(selected)), ...
        'evaluate_pilot:BatchOrder','Batch evaluation changed the fixed sample order');
    lossSum=lossSum+nb*[batch.total_loss batch.BCE_component batch.Dice_component];
    if isempty(perImage),perImage=repmat(batch.per_image(1),n,1);end
    perImage(selected)=batch.per_image(:);
    checks=batch.probability_checks;
    assert(checks.all_finite && all(isfinite([batch.total_loss ...
        batch.BCE_component batch.Dice_component checks.minimum checks.maximum ...
        checks.maximum_channel_sum_error checks.loss_inference_maximum_difference])), ...
        'evaluate_pilot:NonfiniteBatch','Nonfinite validation output is a numerical error');
    probabilityMinimum=min(probabilityMinimum,double(checks.minimum));
    probabilityMaximum=max(probabilityMaximum,double(checks.maximum));
    sumError=max(sumError,double(checks.maximum_channel_sum_error));
    lossProbabilityError=max(lossProbabilityError,double(checks.loss_inference_maximum_difference));
    record=struct('first_index',first,'last_index',last,'sample_count',nb, ...
        'total_loss',batch.total_loss,'BCE_component',batch.BCE_component, ...
        'Dice_component',batch.Dice_component, ...
        'maximum_channel_sum_error',double(checks.maximum_channel_sum_error));
    batchIndex=(first-1)/batchSize+1;
    if isempty(batchRecords),batchRecords=repmat(record,batchCount,1);end
    batchRecords(batchIndex)=record;
end
assert(all(isfinite(lossSum)),'evaluate_pilot:NonfiniteLoss','Aggregated validation loss is nonfinite');

% A proxy group can straddle many inference batches. Pool its counts here.
groups=unique(groupIds,'stable');
perGroup=repmat(perImage(1),numel(groups),1);
for k=1:numel(groups)
    selected=groupIds==groups(k);
    records=perImage(selected);
    tp=sum([records.TP]);fp=sum([records.FP]);fn=sum([records.FN]);tn=sum([records.TN]);
    valid=tp+fp+fn+tn;
    perGroup(k).id=char(groups(k));perGroup(k).group_id=char(groups(k));
    perGroup(k).image_count=nnz(selected);perGroup(k).valid_pixels=valid;
    perGroup(k).TP=tp;perGroup(k).FP=fp;perGroup(k).FN=fn;perGroup(k).TN=tn;
    perGroup(k).target_foreground_pixels=tp+fn;
    perGroup(k).predicted_foreground_pixels=tp+fp;
    perGroup(k).foreground_IoU=ratio(tp,tp+fp+fn);
    perGroup(k).foreground_Dice=ratio(2*tp,2*tp+fp+fn);
    perGroup(k).precision=ratio(tp,tp+fp);perGroup(k).recall=ratio(tp,tp+fn);
    perGroup(k).target_foreground_fraction=ratio(tp+fn,valid);
    perGroup(k).predicted_foreground_fraction=ratio(tp+fp,valid);
    perGroup(k).all_background_prediction=(tp+fp==0);
    means=[records.mean_defect_probability];weights=[records.valid_pixels];
    perGroup(k).mean_defect_probability=sum(means.*weights)/sum(weights);
end
tp=sum([perImage.TP]);fp=sum([perImage.FP]);
fn=sum([perImage.FN]);tn=sum([perImage.TN]);valid=tp+fp+fn+tn;
report.schema_version=1;
report.scope='pilot validation: fixed subset, descriptive only; not heldout test or generalization evidence';
report.sample_count=n;report.proxy_group_count=numel(groups);
report.inference_batch_size=batchSize;report.evaluation_batch_count=batchCount;
report.output_channel_order={'background','defect'};
report.foreground_threshold=0.5;report.threshold_comparison='>=';
report.total_loss=lossSum(1)/n;report.BCE_component=lossSum(2)/n;
report.Dice_component=lossSum(3)/n;
report.loss_aggregation='image-count-weighted batch means, equivalent to equal per-image mean';
report.TP=tp;report.FP=fp;report.FN=fn;report.TN=tn;report.valid_pixels=valid;
report.micro_foreground_IoU=ratio(tp,tp+fp+fn);
report.micro_foreground_Dice=ratio(2*tp,2*tp+fp+fn);
report.micro_precision=ratio(tp,tp+fp);report.micro_recall=ratio(tp,tp+fn);
report.macro_image_foreground_IoU=defined_mean([perImage.foreground_IoU]);
report.macro_group_foreground_IoU=defined_mean([perGroup.foreground_IoU]);
report.macro_group_definition='pool confusion counts across all batches within a proxy group, then equal mean over defined group IoUs';
report.predicted_foreground_fraction=ratio(tp+fp,valid);
report.target_foreground_fraction=ratio(tp+fn,valid);
report.all_background_prediction=(tp+fp==0);
report.all_background_image_count=nnz([perImage.all_background_prediction]);
report.zero_target_image_count=nnz([perImage.target_foreground_pixels]==0);
report.undefined_image_metrics=undefined_counts(perImage);
report.undefined_group_metrics=undefined_counts(perGroup);
report.micro_metric_denominators=struct('IoU',tp+fp+fn,'Dice',2*tp+fp+fn, ...
    'precision',tp+fp,'recall',tp+fn);
report.probability_checks=struct('all_finite',true,'minimum',probabilityMinimum, ...
    'maximum',probabilityMaximum,'maximum_channel_sum_error',sumError, ...
    'loss_inference_maximum_difference',lossProbabilityError, ...
    'logits_layer',checks.logits_layer,'softmax_layer',checks.softmax_layer);
report.per_image=perImage;report.per_group=perGroup;
report.batch_diagnostics=batchRecords;
report.evaluation_time_s=toc(started);
report.timing_scope='batched inference, loss and metric aggregation; not end-to-end detection FPS';
end

function M=canonical_mask(M,h,w,n,description)
assert((isnumeric(M)||islogical(M)) && isreal(M) && ndims(M)<=4, ...
    'evaluate_pilot:MaskType','Invalid %s array type',description);
if isequal(size(M,[1 2 3 4]),[h w 1 n]) || ...
        (n>1 && isequal(size(M,[1 2 3 4]),[h w n 1]))
    M=reshape(single(M),h,w,1,n);
else
    error('evaluate_pilot:MaskShape','Unexpected %s dimensions',description);
end
assert(all(isfinite(M(:))) && all(M(:)==0|M(:)==1), ...
    'evaluate_pilot:MaskValues','%s must contain finite binary 0/1 values',description);
end

function value=ratio(numerator,denominator)
if denominator==0,value=NaN;else,value=double(numerator)/double(denominator);end
end

function value=defined_mean(values)
assert(~any(isinf(values)),'evaluate_pilot:InfiniteMetric','Infinite metric is invalid');
values=values(~isnan(values));
if isempty(values),value=NaN;else,value=mean(values);end
end

function output=undefined_counts(records)
output=struct('foreground_IoU',nnz(isnan([records.foreground_IoU])), ...
    'foreground_Dice',nnz(isnan([records.foreground_Dice])), ...
    'precision',nnz(isnan([records.precision])),'recall',nnz(isnan([records.recall])));
end
