function [X,T,V,records,context] = load_smoke_inputs(dataRoot)
%LOAD_SMOKE_INPUTS Open only the eight frozen assigned-training samples.
root=fileparts(fileparts(mfilename('fullpath')));
e=jsondecode(fileread(fullfile(root,'docs','evidence','stage1_plan_check.json')));
cfgPath=fullfile(root,'configs','training_stage1_v1.json');
cfg=jsondecode(fileread(cfgPath));
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
subsets=jsondecode(fileread(fullfile(root,cfg.subsets)));
parent=jsondecode(fileread(fullfile(root,cfg.source_split)));
schedule=jsondecode(fileread(fullfile(root,cfg.batch_schedule)));
records=subsets.smoke_train;
assert(numel(records)==8 && numel(unique(string({records.id})))==8);
assert(numel(unique(string({records.group_id})))==8);
assert(all(strcmp({records.assigned_split},'training')));
parentIds=string({parent.records.id});
fields={'group_id','image','mask','assigned_split'};
for k=1:numel(records)
    j=find(parentIds==string(records(k).id));assert(isscalar(j));
    for f=1:numel(fields)
        assert(strcmp(records(k).(fields{f}),parent.records(j).(fields{f})), ...
            'Sample differs from parent split');
    end
end
assert(numel(schedule.smoke)==cfg.profiles.smoke.target_optimizer_updates);
for epoch=1:cfg.profiles.smoke.passes
    rows=schedule.smoke([schedule.smoke.epoch]==epoch);
    epochIds=strings(0,1);
    for j=1:numel(rows)
        assert(numel(rows(j).sample_ids)==cfg.profiles.smoke.batch_size);
        epochIds=[epochIds;string(rows(j).sample_ids(:))]; %#ok<AGROW>
    end
    assert(isequal(sort(epochIds),sort(string({records.id})')));
end
modelCfg=jsondecode(fileread(fullfile(root,cfg.model_config)));
hw=double(modelCfg.input_size_hwc(1:2)');n=numel(records);
X=zeros([hw 3 n],'single');T=zeros([hw 1 n],'single');V=T;
access=repmat(struct,1,n);
for k=1:n
    r=records(k);
    assert(~contains(r.image,'..') && ~contains(r.mask,'..'));
    ip=fullfile(dataRoot,r.image);mp=fullfile(dataRoot,r.mask);
    [I,M,W,meta]=read_segmentation_sample(ip,mp,hw,cfg.jpeg_label_threshold);
    assert(isa(I,'single') && all(isfinite(I),'all') && all(I>=0 & I<=1,'all'));
    assert(nnz(W)>0 && ~any(M(~W)) && isequal(size(M),hw));
    X(:,:,:,k)=I;T(:,:,1,k)=single(M);V(:,:,1,k)=single(W);
    access(k).id=r.id;access(k).group_id=r.group_id;
    access(k).assigned_split=r.assigned_split;
    access(k).image=r.image;access(k).mask=r.mask;
    access(k).image_file_sha256=file_sha256(ip);
    access(k).mask_file_sha256=file_sha256(mp);
    access(k).valid_pixels=nnz(W);
    access(k).preprocessing=meta;
    access(k).foreground_disappeared=meta.foreground_pixels_before>0 && nnz(M)==0;
end
context.config=cfg;context.model_config=modelCfg;context.hashes=hashes;
context.schedule=schedule.smoke;context.access=access;
context.unique_training_images_opened=n;
context.validation_images_opened=0;context.heldout_test_images_opened=0;
context.access_rule='assigned_split in frozen protocol; original directory prefixes preserved';
fprintf('SMOKE_INPUTS_OK images=%d groups=%d valid_pixels=',n,n);
fprintf('%d ',[access.valid_pixels]);fprintf('\n');
end
