function report=check_mature_baseline_contract(dataRoot,runId)
%CHECK_MATURE_BASELINE_CONTRACT Construction, BN, gradients, tail and compute.
% Only the existing fixed8 training pairs are read. Networks are disposable;
% five optimizer updates here never initialize a formal comparison run.
root=fileparts(fileparts(mfilename('fullpath')));
assert(~isempty(regexp(runId,'^[A-Za-z0-9_-]+$','once')));
out=fullfile(root,'runs',runId);assert(~isfolder(out),'Preflight already exists');mkdir(out);
diary(fullfile(out,'run.log'));cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
modelPath=fullfile(root,'configs','mobilenet_deeplab_scratch_v1.json');
report.status='RUNNING';report.optimizer_updates=0;report.raw_training_images_opened=0;
report.validation_images_opened=0;report.heldout_test_images_opened=0;clock=tic;
try
    report.matlab_version=version;report.execution_environment='CPU';report.precision='single';
    report.max_computational_threads=maxNumCompThreads;[report.gpu_count,report.gpu_query_available]=publicOptionalGpuCount();
    report.parent_model_sha256=file_sha256(modelPath);
    mkdir(fullfile(out,'source_snapshot'));files=dir(fullfile(root,'matlab','*.m'));
    for k=1:numel(files)
        target=fullfile(out,'source_snapshot',files(k).name);
        copyfile(fullfile(files(k).folder,files(k).name),target);
        report.source_files(k).name=files(k).name;report.source_files(k).sha256=file_sha256(target);
    end
    seeds=[20261114 20261115 20261116 20261114];fingerprints=cell(1,4);initialStates=cell(1,4);
    for k=1:4
        [net,modelCfg,~,architecture]=build_seeded_mature_baseline(seeds(k),modelPath, ...
            fullfile(out,sprintf('seed_fixture_%d',k)));
        fingerprints{k}=network_parameter_sha256(net);initialStates{k}=stateSummary(net.State);
        assert(initialStates{k}.all_finite&&initialStates{k}.entry_count>0);
        assert(initialStates{k}.means_all_zero&&initialStates{k}.variances_all_one, ...
            'Fresh random model must start with zero BN means and unit variances');
    end
    assert(strcmp(fingerprints{1},fingerprints{4}));
    assert(numel(unique(string(fingerprints(1:3))))==3);
    assert(strcmp(report.parent_model_sha256,file_sha256(modelPath)));
    report.initialization.seeds=seeds;report.initialization.fingerprints=fingerprints;
    report.initialization.bn_initial_states=initialStates;
    report.initialization.same_seed_identical=true;report.initialization.distinct_seeds_different=true;
    report.architecture=architecture;
    [X,T,V,records,context]=load_smoke_inputs(dataRoot);
    report.raw_training_images_opened=8;report.training_ids=cellstr(string({records.id}));
    report.input_access=context;
    cfg=context.config;cfg.optimizer.learning_rate=0.0003;
    softmaxName=string(net.OutputNames{1});rows=string(net.Connections.Destination)==softmaxName;
    assert(isscalar(net.OutputNames)&&nnz(rows)==1);logitsName=string(net.Connections.Source(rows));
    beforeState=net.State;savedRng=rng;
    [Z,P,H,L]=predict(net,dlarray(X(:,:,:,1:2),'SSCB'), ...
        Outputs=[logitsName softmaxName string(modelCfg.high_level_endpoint) string(modelCfg.low_level_endpoint)]);
    assert(isequal(size(Z,[1 2 3 4]),[256 256 2 2]));
    assert(isequal(size(P,[1 2 3 4]),[256 256 2 2]));
    assert(isequal(size(H,[1 2 3 4]),[16 16 320 2]));
    assert(isequal(size(L,[1 2 3 4]),[64 64 144 2]));
    [loss,~,~,fromLoss]=masked_segmentation_loss(Z,T(:,:,:,1:2),V(:,:,:,1:2),cfg.loss);
    p=extractdata(P);q=extractdata(fromLoss);
    assert(all(isfinite(p),'all')&&all(p>=0 & p<=1,'all'));
    report.probability_checks.maximum_channel_sum_error=double(max(abs(sum(p,3)-1),[],'all'));
    report.probability_checks.loss_probability_difference=double(max(abs(q-p(:,:,2,:)),[],'all'));
    assert(report.probability_checks.maximum_channel_sum_error<=1e-4);
    assert(report.probability_checks.loss_probability_difference<=1e-5&&isfinite(extractdata(loss)));
    singleP=predict(net,dlarray(X(:,:,:,1),'SSCB'));
    report.probability_checks.batch1_vs2_max_difference=double(max(abs(extractdata(singleP)-p(:,:,:,1)),[],'all'));
    assert(report.probability_checks.batch1_vs2_max_difference<=1e-5);
    assert(tableError(net.State,beforeState)==0&&isequal(savedRng,rng));
    report.predict_preserves_state_and_rng=true;
    retained=string(architecture.retained_backbone_layer_names);
    backboneMask=ismember(string(net.Learnables.Layer),retained);
    assert(any(backboneMask)&&any(~backboneMask));
    avg=[];avgSq=[];times=zeros(1,4);updates=struct([]);
    for iteration=1:5
        n=8;if iteration==5,n=4;end
        old=net;timer=tic;
        [net,avg,avgSq,h]=segmentation_train_step(net,avg,avgSq,iteration, ...
            X(:,:,:,1:n),T(:,:,:,1:n),V(:,:,:,1:n),cfg,logitsName);
        h.elapsed_time_s=toc(timer);h.batch_size=n;h.update=iteration;
        h.bn_state_max_absolute_change=tableError(old.State,net.State);
        h.backbone_parameter_change_L2=parameterChange(old,net,backboneMask);
        h.head_parameter_change_L2=parameterChange(old,net,~backboneMask);
        ss=stateSummary(net.State);assert(ss.all_finite&&h.bn_state_max_absolute_change>0);
        assert(h.backbone_parameter_change_L2>0&&h.head_parameter_change_L2>0);
        h.bn_state_all_finite=true;
        if isempty(updates),updates=h;else,updates(end+1)=h;end %#ok<AGROW>
        report.optimizer_updates=iteration;
        if iteration<=4,times(iteration)=h.elapsed_time_s;end
        fprintf('MATURE_PREFLIGHT_UPDATE iteration=%d batch=%d loss=%.6f time=%.3fs BN=%.6g backbone=%.6g head=%.6g\n', ...
            iteration,n,h.total_loss,h.elapsed_time_s,h.bn_state_max_absolute_change, ...
            h.backbone_parameter_change_L2,h.head_parameter_change_L2);
    end
    report.updates=updates;report.bn_after_preflight=stateSummary(net.State);
    report.timing.batch_size=8;report.timing.warmup_updates=1;report.timing.measured_updates=3;
    report.timing.warmup_time_s=times(1);report.timing.measured_times_s=times(2:4);
    report.timing.median_update_time_s=median(times(2:4));
    report.timing.training_images_per_second=8/report.timing.median_update_time_s;
    report.timing.maximum_pure_updates_s_per_seed=3500*report.timing.median_update_time_s;
    report.timing.maximum_pure_updates_s_three_seeds=10500*report.timing.median_update_time_s;
    report.timing.note='Disposable training-update timing, not inference FPS; full loading/evaluation/checkpoints/plots excluded';
    savedRng=rng;state=net.State;pBefore=predict(net,dlarray(X(:,:,:,1),'SSCB'));
    assert(tableError(net.State,state)==0&&isequal(savedRng,rng));
    checkpoint.net=net;checkpoint.averageGrad=avg;checkpoint.averageSqGrad=avgSq;
    checkpoint.rng_state=savedRng;checkpoint.completed_updates=5;checkpoint.next_update=6;
    checkpoint.model_config=modelCfg;checkpoint.model_config_sha256=report.parent_model_sha256;
    path=fullfile(out,'preflight_checkpoint.mat');save(path,'checkpoint','-v7.3');
    restored=load(path,'checkpoint');restored=restored.checkpoint;
    assert(strcmp(network_parameter_sha256(net),network_parameter_sha256(restored.net)));
    assert(tableError(net.State,restored.net.State)==0&&tableError(avg,restored.averageGrad)==0);
    assert(tableError(avgSq,restored.averageSqGrad)==0&&isequal(savedRng,restored.rng_state));
    pAfter=predict(restored.net,dlarray(X(:,:,:,1),'SSCB'));
    report.reload.prediction_max_absolute_error=double(max(abs(extractdata(pBefore-pAfter)),[],'all'));
    assert(report.reload.prediction_max_absolute_error<=1e-6);report.reload.status='PASS';
    report.status='PASS';report.formal_training_started=false;report.formal_initialization_must_be_fresh=true;
    report.scope='Engineering contract and compute-only preflight on fixed8 training pairs; no model selection or validation score';
    report.total_wall_time_s=toc(clock);writeJson(fullfile(out,'summary.json'),report);
    fprintf('MATURE_PREFLIGHT_PASS updates=5 training_images=8 validation=0 test=0 median_batch8_update=%.3fs\n', ...
        report.timing.median_update_time_s);
catch err
    report.status='FAILED';report.message=err.message;report.identifier=err.identifier;
    report.stack=err.stack;report.total_wall_time_s=toc(clock);
    writeJson(fullfile(out,'failure.json'),report);rethrow(err);
end
end

function s=stateSummary(state)
s.entry_count=height(state);s.element_count=0;s.all_finite=true;
s.means_all_zero=true;s.variances_all_one=true;
for k=1:height(state)
    v=extractdata(state.Value{k});s.element_count=s.element_count+numel(v);
    s.all_finite=s.all_finite&&all(isfinite(v),'all');
    if strcmp(state.Parameter(k),'TrainedMean'),s.means_all_zero=s.means_all_zero&&all(v==0,'all');end
    if strcmp(state.Parameter(k),'TrainedVariance'),s.variances_all_one=s.variances_all_one&&all(v==1,'all');end
end
end
function value=parameterChange(before,after,mask)
sumSquared=0;
for k=find(mask(:))'
    a=double(extractdata(before.Learnables.Value{k}));b=double(extractdata(after.Learnables.Value{k}));
    sumSquared=sumSquared+sum((a-b).^2,'all');
end
value=sqrt(sumSquared);
end
function value=tableError(a,b)
assert(istable(a)&&istable(b)&&isequal(a(:,1:2),b(:,1:2)));value=0;
for k=1:height(a)
    av=extractdata(a.Value{k});bv=extractdata(b.Value{k});assert(isequal(size(av),size(bv)));
    if ~isempty(av),value=max(value,double(max(abs(av-bv),[],'all')));end
end
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end

function [count,available]=publicOptionalGpuCount()
% Public-export adapter: optional environment logging; CPU math is unchanged.
count=0;available=false;
try
    count=gpuDeviceCount;
    available=true;
catch
end
end
