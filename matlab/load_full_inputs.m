function [trainData,valData,context] = load_full_inputs(dataRoot,configPath)
%LOAD_FULL_INPUTS Preload the frozen 1396 training and 400 validation pairs.
% Split membership comes from assigned_split, never original path prefixes.
% Records are sorted by ID; this function never reads heldout image/mask data.
root=fileparts(fileparts(mfilename('fullpath')));
assert(nargin>=2 && ~isempty(configPath),'An explicit stage2 configuration is required');
configPath=resolveProjectFile(root,configPath);
cfg=jsondecode(fileread(configPath));
required={'model_config','model_config_sha256','source_split','source_split_sha256'};
assert(all(isfield(cfg,required)),'Stage2 configuration lacks required input references');
modelPath=resolveProjectFile(root,cfg.model_config);
splitPath=resolveProjectFile(root,cfg.source_split);
hashes.model_config_sha256=file_sha256(modelPath);
hashes.source_split_sha256=file_sha256(splitPath);
hashes.training_config_sha256=file_sha256(configPath);
assert(strcmp(hashes.model_config_sha256,cfg.model_config_sha256),'Model configuration hash mismatch');
assert(strcmp(hashes.source_split_sha256,cfg.source_split_sha256),'Parent split hash mismatch');
frozenParentSha='c65f8c439ac3f245541a6678f3fe937bd3dcf7136584f088fd90eae1609270a6';
assert(strcmp(hashes.source_split_sha256,frozenParentSha), ...
    'Full inputs require the frozen conservative_group_v1 parent split');
modelCfg=jsondecode(fileread(modelPath));
assert(isequal(double(modelCfg.input_size_hwc(:)'),[256 256 3]), ...
    'Full inputs require 256-by-256 RGB input');
assert(modelCfg.num_classes==2 && ...
    isequal(string(modelCfg.classes_in_channel_order(:)),["background";"defect"]));
threshold=128;
if isfield(cfg,'jpeg_label_threshold')
    assert(isequal(cfg.jpeg_label_threshold,128),'Frozen JPEG label threshold must remain 128');
end
if isfield(modelCfg,'nrsd_jpeg_mask_threshold')
    assert(isequal(modelCfg.nrsd_jpeg_mask_threshold,128));
end
parent=jsondecode(fileread(splitPath));
parentIds=string({parent.records.id})';
assert(numel(unique(parentIds))==numel(parentIds),'Parent IDs must be unique');
trainRecords=sortedPartition(parent.records,'training');
valRecords=sortedPartition(parent.records,'validation');
validateRecords(trainRecords,'training',1396,45);
validateRecords(valRecords,'validation',400,15);
assert(isempty(intersect(string({trainRecords.id}),string({valRecords.id}))), ...
    'Training/validation ID overlap');
assert(isempty(intersect(string({trainRecords.group_id}),string({valRecords.group_id}))), ...
    'Training/validation proxy-group overlap');
assert(isempty(intersect(string({trainRecords.image}),string({valRecords.image}))), ...
    'Training/validation image-path overlap');
assert(isempty(intersect(string({trainRecords.mask}),string({valRecords.mask}))), ...
    'Training/validation mask-path overlap');
% Configuration/parent hash and membership checks precede all raw-pair reads.
hw=[256 256];
[trainData,trainAccess]=loadPartition(dataRoot,trainRecords,hw,threshold);
[valData,valAccess]=loadPartition(dataRoot,valRecords,hw,threshold);
context.config=cfg;context.model_config=modelCfg;context.hashes=hashes;
context.train_access=trainAccess;context.validation_access=valAccess;
context.unique_training_images_opened=numel(trainRecords);
context.validation_images_opened=numel(valRecords);
context.heldout_test_images_opened=0;
context.sample_order='Ascending MATLAB string sort of frozen sample IDs';
context.access_rule='assigned_split in frozen conservative_group_v1; original path prefixes preserved';
fprintf('FULL_INPUTS_OK train=%d train_groups=%d validation=%d validation_groups=%d heldout=0\n', ...
    numel(trainRecords),numel(unique(trainData.groupIds)), ...
    numel(valRecords),numel(unique(valData.groupIds)));
end

function path=resolveProjectFile(root,path)
assert((ischar(path) && isrow(path)) || (isstring(path) && isscalar(path)), ...
    'Configuration file references must be scalar paths');
path=char(path);assert(~isempty(path),'Empty configuration path');
if isempty(regexp(path,'^([A-Za-z]:[\\/]|[\\/])','once'))
    path=fullfile(root,path);
end
assert(isfile(path),'Required configuration or metadata file is absent: %s',path);
end

function records=sortedPartition(parentRecords,assignedSplit)
records=parentRecords(strcmp({parentRecords.assigned_split},assignedSplit));
records=records(:);
[~,order]=sort(string({records.id})');records=records(order);
end

function validateRecords(records,assignedSplit,n,groupCount)
assert(numel(records)==n && numel(unique(string({records.id})))==n, ...
    'Unexpected full-partition image count or duplicate ID');
assert(numel(unique(string({records.group_id})))==groupCount, ...
    'Unexpected full-partition proxy-group count');
assert(numel(unique(string({records.image})))==n && numel(unique(string({records.mask})))==n, ...
    'Repeated raw path within a partition');
assert(all(strcmp({records.assigned_split},assignedSplit)),'Revised split mismatch');
ids=string({records.id})';assert(isequal(ids,sort(ids)),'IDs must remain sorted');
for k=1:n
    validateRelativePath(records(k).image);validateRelativePath(records(k).mask);
end
end

function validateRelativePath(path)
assert(ischar(path) && ~isempty(path));
assert(~contains(path,'..') && ~contains(path,':') && ~contains(path,'\') && path(1)~='/', ...
    'Raw sample path must remain relative to the local data root');
end

function [data,access]=loadPartition(dataRoot,records,hw,threshold)
n=numel(records);
data.X=zeros([hw 3 n],'single');data.T=zeros([hw 1 n],'single');data.V=data.T;
data.records=records(:);data.ids=string({records.id})';data.groupIds=string({records.group_id})';
access=repmat(struct,1,n);
for k=1:n
    r=records(k);ip=fullfile(dataRoot,r.image);mp=fullfile(dataRoot,r.mask);
    [I,M,W,meta]=read_segmentation_sample(ip,mp,hw,threshold);
    assert(isa(I,'single') && isequal(size(I),[hw 3]) && all(isfinite(I),'all'));
    assert(all(I>=0 & I<=1,'all') && islogical(M) && islogical(W));
    assert(isequal(size(M),hw) && isequal(size(W),hw) && nnz(W)>0 && ~any(M(~W)));
    assert(meta.scale>0 && meta.scale<=1 && all(meta.resized_hw<=meta.source_hw), ...
        'Preprocessing must not enlarge raw images');
    assert(all(meta.resized_hw<=hw) && all(meta.offset_rc>=0));
    assert(nnz(W)==prod(meta.resized_hw) && nnz(M)==meta.foreground_pixels_after);
    assert(meta.threshold==128,'Unexpected label threshold');
    data.X(:,:,:,k)=I;data.T(:,:,1,k)=single(M);data.V(:,:,1,k)=single(W);
    access(k).id=r.id;access(k).group_id=r.group_id;access(k).assigned_split=r.assigned_split;
    access(k).image=r.image;access(k).mask=r.mask;
    access(k).image_file_sha256=file_sha256(ip);access(k).mask_file_sha256=file_sha256(mp);
    access(k).valid_pixels=nnz(W);access(k).preprocessing=meta;
    % Keep every frozen ID even when a small foreground vanishes on resize.
    access(k).foreground_disappeared=meta.foreground_pixels_before>0 && nnz(M)==0;
    if mod(k,128)==0 || k==n
        fprintf('FULL_LOADING partition=%s pairs=%d/%d\n',r.assigned_split,k,n);
    end
end
end
