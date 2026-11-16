function [trainData,valData,context] = load_pilot_inputs(dataRoot)
%LOAD_PILOT_INPUTS Load only the frozen 128 training and 64 validation pairs.
% Original directory prefixes do not determine the revised assigned split.
% Keep fixed sample order, valid-pixel masks, and vanished foreground cases.
root=fileparts(fileparts(mfilename('fullpath')));
e=jsondecode(fileread(fullfile(root,'docs','evidence','stage1_plan_check.json')));
cfg=jsondecode(fileread(fullfile(root,'configs','training_stage1_v1.json')));
files={cfg.model_config,cfg.source_split,cfg.subsets,cfg.batch_schedule, ...
    'configs/training_stage1_v1.json'};
keys={'model_config_sha256','source_split_sha256','subset_sha256', ...
    'schedule_sha256','training_config_sha256'};
hashes=struct;
for k=1:numel(files)
    actual=file_sha256(fullfile(root,files{k}));
    assert(strcmp(actual,e.(keys{k})),'Frozen file hash mismatch: %s',files{k});
    hashes.(keys{k})=actual;
end
assert(strcmp(cfg.model_config_sha256,hashes.model_config_sha256));
assert(strcmp(cfg.source_split_sha256,hashes.source_split_sha256));
assert(strcmp(cfg.subsets_sha256,hashes.subset_sha256));
assert(strcmp(cfg.batch_schedule_sha256,hashes.schedule_sha256));
subsets=jsondecode(fileread(fullfile(root,cfg.subsets)));
parent=jsondecode(fileread(fullfile(root,cfg.source_split)));
schedule=jsondecode(fileread(fullfile(root,cfg.batch_schedule)));
assert(strcmp(subsets.source_split_sha256,hashes.source_split_sha256));
assert(strcmp(schedule.subset_sha256,hashes.subset_sha256));
profile=cfg.profiles.pilot;
assert(profile.unique_train_images==128 && profile.unique_validation_images==64);
assert(profile.batch_size==2 && profile.passes==3 && profile.target_optimizer_updates==192);
trainRecords=subsets.(profile.training_subset);
valRecords=subsets.(profile.validation_subset);
parentIds=string({parent.records.id})';
assert(numel(unique(parentIds))==numel(parentIds),'Parent IDs must be unique');
validateRecords(trainRecords,'training',128,45,parent.records,parentIds);
validateRecords(valRecords,'validation',64,15,parent.records,parentIds);
trainIds=string({trainRecords.id})';valIds=string({valRecords.id})';
trainGroups=string({trainRecords.group_id})';valGroups=string({valRecords.group_id})';
assert(isempty(intersect(trainIds,valIds)),'Training/validation ID overlap');
assert(isempty(intersect(trainGroups,valGroups)),'Training/validation group overlap');
assert(isempty(intersect(string({trainRecords.image}),string({valRecords.image}))), ...
    'Training/validation image-path overlap');
assert(isempty(intersect(string({trainRecords.mask}),string({valRecords.mask}))), ...
    'Training/validation mask-path overlap');
assert(numel(schedule.pilot)==192 && ~schedule.replacement && schedule.batch_size==2);
assert(isequal(double([schedule.pilot.update]),1:192),'Pilot updates must be consecutive');
assert(isequal(double([schedule.pilot.epoch]),ceil((1:192)/64)), ...
    'Pilot epoch boundaries must be fixed at 64 updates');
for epoch=1:profile.passes
    rows=schedule.pilot([schedule.pilot.epoch]==epoch);
    epochIds=strings(0,1);
    for j=1:numel(rows)
        batchIds=string(rows(j).sample_ids(:));
        assert(numel(batchIds)==2 && numel(unique(batchIds))==2);
        assert(all(ismember(batchIds,trainIds)),'Batch includes a non-training ID');
        epochIds=[epochIds;batchIds]; %#ok<AGROW>
    end
    assert(isequal(sort(epochIds),sort(trainIds)), ...
        'Each pilot epoch must visit every training sample exactly once');
end
validationOrder=string(schedule.validation_order(:));
assert(isequal(validationOrder,valIds),'Validation order differs from the frozen subset');
modelCfg=jsondecode(fileread(fullfile(root,cfg.model_config)));
hw=double(modelCfg.input_size_hwc(1:2)');
assert(isequal(hw,[256 256]) && modelCfg.input_size_hwc(3)==3);
% All metadata checks above finish before the first raw image or mask read.
[trainData,trainAccess]=loadPartition(dataRoot,trainRecords,hw,cfg.jpeg_label_threshold);
[valData,valAccess]=loadPartition(dataRoot,valRecords,hw,cfg.jpeg_label_threshold);
context.config=cfg;context.model_config=modelCfg;context.hashes=hashes;
context.schedule=schedule.pilot;context.validation_order=validationOrder;
context.train_access=trainAccess;context.validation_access=valAccess;
context.unique_training_images_opened=numel(trainRecords);
context.validation_images_opened=numel(valRecords);
context.heldout_test_images_opened=0;
context.access_rule='assigned_split in frozen protocol; original directory prefixes preserved';
fprintf('PILOT_INPUTS_OK train=%d train_groups=%d validation=%d validation_groups=%d heldout=0\n', ...
    numel(trainRecords),numel(unique(trainGroups)),numel(valRecords),numel(unique(valGroups)));
end

function validateRecords(records,assignedSplit,n,groupCount,parentRecords,parentIds)
assert(numel(records)==n && numel(unique(string({records.id})))==n);
assert(numel(unique(string({records.group_id})))==groupCount);
assert(numel(unique(string({records.image})))==n && numel(unique(string({records.mask})))==n);
assert(all(strcmp({records.assigned_split},assignedSplit)),'Revised split mismatch');
fields={'group_id','image','mask','assigned_split'};
for k=1:n
    j=find(parentIds==string(records(k).id));assert(isscalar(j),'Sample absent or ambiguous in parent split');
    for f=1:numel(fields)
        assert(strcmp(records(k).(fields{f}),parentRecords(j).(fields{f})), ...
            'Sample differs from parent split');
    end
    validateRelativePath(records(k).image);validateRelativePath(records(k).mask);
end
end

function validateRelativePath(path)
% Frozen records use forward-slash paths below the caller's local data root.
assert(ischar(path) && ~isempty(path));
assert(~contains(path,'..') && ~contains(path,':') && ~contains(path,'\') && path(1)~='/', ...
    'Sample path must remain relative to the data root');
end

function [data,access]=loadPartition(dataRoot,records,hw,threshold)
n=numel(records);
data.X=zeros([hw 3 n],'single');data.T=zeros([hw 1 n],'single');data.V=data.T;
data.records=records;data.ids=string({records.id})';data.groupIds=string({records.group_id})';
access=repmat(struct,1,n);
for k=1:n
    r=records(k);ip=fullfile(dataRoot,r.image);mp=fullfile(dataRoot,r.mask);
    [I,M,W,meta]=read_segmentation_sample(ip,mp,hw,threshold);
    assert(isa(I,'single') && isequal(size(I),[hw 3]) && all(isfinite(I),'all'));
    assert(all(I>=0 & I<=1,'all') && islogical(M) && islogical(W));
    assert(isequal(size(M),hw) && isequal(size(W),hw) && nnz(W)>0 && ~any(M(~W)));
    assert(nnz(W)==prod(meta.resized_hw) && nnz(M)==meta.foreground_pixels_after);
    data.X(:,:,:,k)=I;data.T(:,:,1,k)=single(M);data.V(:,:,1,k)=single(W);
    access(k).id=r.id;access(k).group_id=r.group_id;access(k).assigned_split=r.assigned_split;
    access(k).image=r.image;access(k).mask=r.mask;
    access(k).image_file_sha256=file_sha256(ip);access(k).mask_file_sha256=file_sha256(mp);
    access(k).valid_pixels=nnz(W);access(k).preprocessing=meta;
    % Record this diagnostic without dropping, replacing, or resampling an ID.
    access(k).foreground_disappeared=meta.foreground_pixels_before>0 && nnz(M)==0;
end
end
