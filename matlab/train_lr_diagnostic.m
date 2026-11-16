function report = train_lr_diagnostic(dataRoot,configPath,runId)
%TRAIN_LR_DIAGNOSTIC Step10 controlled LR comparison on the frozen pilot.
% Validation is descriptive model selection; never heldout-test evaluation.
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<3,runId=['pilot_' char(datetime('now',Format='yyyyMMdd_HHmmss'))];end
assert(~isempty(regexp(runId,'^[A-Za-z0-9_-]+$','once')),'Unsafe run ID');
out=fullfile(root,'runs',runId);assert(~exist(out,'dir'),'Run directory exists; do not overwrite');
mkdir(out);mkdir(fullfile(out,'checkpoints'));mkdir(fullfile(out,'source_snapshot'));
diary(fullfile(out,'run.log'));cleanDiary=onCleanup(@()diary('off')); %#ok<NASGU>
clock=tic;completed=0;lastCheckpoint='';
try
    evaluationChecks=check_pilot_evaluation(out);
    loadClock=tic;[trainData,valData,context]=load_pilot_inputs(dataRoot);
    dataTime=toc(loadClock);
    parentConfig=context.config;parentHashes=context.hashes;
    if ~isfile(configPath),configPath=fullfile(root,configPath);end
    cfg=jsondecode(fileread(configPath));
    assert(strcmp(cfg.mode,'lr_diagnostic'));
    token=split(strtrim(fileread([configPath(1:end-5) '.sha256'])));
    assert(strcmp(file_sha256(configPath),char(token(1))),'Diagnostic config hash mismatch');
    for field={'model_config_sha256','source_split_sha256','subsets_sha256','batch_schedule_sha256'}
        assert(strcmp(cfg.(field{1}),parentConfig.(field{1})));
    end
    assert(isequal(cfg.loss,parentConfig.loss)&&isequal(cfg.profiles,parentConfig.profiles));
    assert(isequal(rmfield(cfg.optimizer,'learning_rate'),rmfield(parentConfig.optimizer,'learning_rate')));
    assert(ismember(cfg.optimizer.learning_rate,[0.0003 0.0001]));
    context.parent_training_config=parentConfig;context.parent_hashes=parentHashes;
    context.config=cfg;context.hashes.training_config_sha256=file_sha256(configPath);
    p=cfg.profiles.pilot;
    assert(p.target_optimizer_updates==192 && p.batch_size==2);
    environment.matlab_version=version;environment.computer=computer;
    environment.cpu_identifier=getenv('PROCESSOR_IDENTIFIER');
    environment.execution_environment='CPU';environment.precision='single';
    environment.max_computational_threads=maxNumCompThreads;
    environment.gpu_count=gpuDeviceCount;environment.products=ver;
    try,[~,mem]=memory;environment.physical_memory_bytes=mem.PhysicalMemory.Total;catch,end
    assert(~contains(root,'"'));
    [status,revision]=system(sprintf('git -C "%s" rev-parse HEAD',root));assert(status==0);
    [status,dirty]=system(sprintf('git -C "%s" status --porcelain',root));assert(status==0);
    source.git_commit=strtrim(revision);source.git_dirty=~isempty(strtrim(dirty));
    mfiles=dir(fullfile(root,'matlab','*.m'));
    for k=1:numel(mfiles)
        original=fullfile(mfiles(k).folder,mfiles(k).name);
        copied=fullfile(out,'source_snapshot',mfiles(k).name);copyfile(original,copied);
        source.files(k).name=mfiles(k).name;source.files(k).sha256=file_sha256(copied);
        assert(strcmp(source.files(k).sha256,file_sha256(original)),'Source changed while snapshotting');
    end
    writeJson(fullfile(out,'input_access.json'),context);
    writeJson(fullfile(out,'environment.json'),environment);
    writeJson(fullfile(out,'source_version.json'),source);
    % Constructor resets the frozen initialization seed. No smoke MAT is read.
    [net,~]=build_baseline();avg=[];avgSq=[];
    names=string({net.Layers.Name});softmaxName=string(net.OutputNames{1});
    assert(isa(net.Layers(names==softmaxName),'nnet.cnn.layer.SoftmaxLayer'));
    rows=string(net.Connections.Destination)==softmaxName;
    assert(nnz(rows)==1);logitsName=string(net.Connections.Source(rows));
    fixedTrain=string(cfg.visualization.fixed_train_ids(:));
    fixedVal=string(cfg.visualization.fixed_validation_ids(:));
    [ok,train8]=ismember(fixedTrain,trainData.ids);assert(all(ok)&&numel(train8)==8);
    [ok,val8]=ismember(fixedVal,valData.ids);assert(all(ok)&&numel(val8)==8);
    assert(isequal(fixedVal,valData.ids(1:8)));
    history=struct([]);valEvals=struct([]);trainEvals=struct([]);reloads=struct([]);
    best=[];bestHistory=struct([]);checkpointSaveTime=0;
    [ve,te]=evaluatePoint(0);valEvals=ve;trainEvals=te;
    writeJson(fullfile(out,'validation_evaluations.json'),valEvals);
    writeJson(fullfile(out,'training8_evaluations.json'),trainEvals);
    [point,lastCheckpoint]=savePoint(0,0);
    reloads=verifyReload(point,lastCheckpoint,trainData.X(:,:,:,1),cfg);
    chooseBest(ve,lastCheckpoint);
    rng(point.rng_state);
    writeJson(fullfile(out,'status.json'),struct('status','RUNNING','completed_updates',0));
    fprintf('LR_DIAGNOSTIC_START fresh_initialization=1 train=128 validation=64 updates=192 batch=2\n');
    for iteration=1:p.target_optimizer_updates
        batch=context.schedule(iteration);
        assert(batch.update==iteration && iteration==completed+1);
        [ok,indices]=ismember(string(batch.sample_ids),trainData.ids);assert(all(ok));
        stepClock=tic;
        [newNet,newAvg,newAvgSq,stats]=segmentation_train_step(net,avg,avgSq,iteration, ...
            trainData.X(:,:,:,indices),trainData.T(:,:,:,indices),trainData.V(:,:,:,indices),cfg,logitsName);
        stats.update_time_s=toc(stepClock);
        net=newNet;avg=newAvg;avgSq=newAvgSq;completed=iteration;
        stats.global_update=iteration;stats.epoch=batch.epoch;
        stats.sample_ids=cellstr(string(batch.sample_ids));
        stats.valid_pixel_count=squeeze(sum(trainData.V(:,:,:,indices),[1 2]));
        stats.elapsed_time_s=toc(clock);
        if isempty(history),history=stats;else,history(end+1)=stats;end %#ok<AGROW>
        writeJson(fullfile(out,'history.json'),history);
        if mod(iteration,8)==0 || iteration==1
            fprintf('PILOT_UPDATE %03d/192 loss=%.7f grad=%.6g clip=%.6g delta=%.6g time=%.3fs\n', ...
                iteration,stats.total_loss,stats.unclipped_global_gradient_norm, ...
                stats.clip_factor,stats.parameter_delta_L2,stats.update_time_s);
        end
        validationNow=ismember(iteration,p.validation_updates);
        if validationNow
            [ve,te]=evaluatePoint(iteration);
            valEvals(end+1)=ve;trainEvals(end+1)=te; %#ok<AGROW>
            writeJson(fullfile(out,'validation_evaluations.json'),valEvals);
            writeJson(fullfile(out,'training8_evaluations.json'),trainEvals);
        end
        if mod(iteration,p.checkpoint_every_updates)==0 || iteration==p.target_optimizer_updates
            [point,lastCheckpoint]=savePoint(iteration,batch.epoch);
            reloads(end+1)=verifyReload(point,lastCheckpoint,trainData.X(:,:,:,1),cfg); %#ok<AGROW>
            writeJson(fullfile(out,'reload_checks.json'),reloads);
            if validationNow,chooseBest(ve,lastCheckpoint);end
            rng(point.rng_state);
        end
        writeJson(fullfile(out,'status.json'),struct('status','RUNNING','completed_updates',completed));
    end
    assert(completed==192 && numel(history)==192);
    assert(isequal([valEvals.global_update],[0 64 128 192]));
    assert(isequal([reloads.global_update],[0 32 64 96 128 160 192]));
    writetable(struct2table(rmfield(history,{'sample_ids','valid_pixel_count'})), ...
        fullfile(out,'training_history.csv'));
    % Keep only scalar numeric/logical fields in the convenient plotting CSV.
    fieldNames=fieldnames(valEvals);numericFields={};
    for k=1:numel(fieldNames)
        value=valEvals(1).(fieldNames{k});
        if (isnumeric(value)||islogical(value)) && isscalar(value)
            numericFields{end+1}=fieldNames{k}; %#ok<AGROW>
        end
    end
    writetable(struct2table(rmfield(valEvals,setdiff(fieldNames,numericFields))), ...
        fullfile(out,'validation_history.csv'));
    if isempty(best)
        bestSummary.status='best_unavailable';bestSummary.reason='all predefined candidate IoUs undefined';
    else
        bestSummary=best;bestSummary.status='available';
        stored=load(fullfile(out,'checkpoints','best_validation.mat'),'checkpoint');
        assert(stored.checkpoint.global_update==best.global_update);
        original=load(fullfile(out,'checkpoints',sprintf('update_%03d.mat',best.global_update)),'checkpoint');
        bestSummary.reload_check=verifyReload(original.checkpoint, ...
            fullfile(out,'checkpoints','best_validation.mat'),trainData.X(:,:,:,1),cfg);
    end
    plotClock=tic;
    artifacts=plot_pilot_run(out,history,valEvals,trainEvals,net, ...
        trainData.X(:,:,:,train8),trainData.T(:,:,:,train8),trainData.V(:,:,:,train8),trainData.ids(train8), ...
        valData.X(:,:,:,val8),valData.T(:,:,:,val8),valData.V(:,:,:,val8),valData.ids(val8));
    plotTime=toc(plotClock);
    report.status='PASS';report.scope='Step10 controlled learning-rate diagnostic; fixed pilot, no heldout test';
    report.learning_rate=cfg.optimizer.learning_rate;
    report.parent_training_config_sha256=parentHashes.training_config_sha256;
    report.run_id=runId;report.fresh_initialization=true;report.main_optimizer_updates=completed;
    report.unique_training_images=numel(trainData.ids);report.training_proxy_groups=numel(unique(trainData.groupIds));
    report.unique_validation_images=numel(valData.ids);report.validation_proxy_groups=numel(unique(valData.groupIds));
    report.heldout_test_images_opened=0;report.primary_model='checkpoints/update_192.mat';
    report.initial_validation=valEvals(1);report.final_validation=valEvals(end);
    report.initial_training8=trainEvals(1);report.final_training8=trainEvals(end);
    report.best_validation=bestSummary;report.all_validation_candidates=bestHistory;
    report.all_gradients_and_parameter_updates_finite_nonzero=true;
    report.reload_checks=reloads;report.evaluation_contract_checks=evaluationChecks;
    report.source=source;report.hashes=context.hashes;
    report.data_preparation_time_s=dataTime;report.main_update_time_s=sum([history.update_time_s]);
    report.validation_time_s=sum([valEvals.evaluation_time_s]);
    report.training8_diagnostic_time_s=sum([trainEvals.evaluation_time_s]);
    report.checkpoint_save_time_s=checkpointSaveTime;report.plot_time_s=plotTime;
    report.total_wall_time_s=toc(clock);
    report.artifacts=structfun(@(x)strrep(x,[out filesep],''),artifacts,UniformOutput=false);
    report.limitations='Pilot validation only; not final heldout performance, significance, motion-fusion benefit or inference FPS.';
    writeJson(fullfile(out,'summary.json'),report);
    writeJson(fullfile(out,'status.json'),struct('status','PASS','completed_updates',completed));
    fprintf('LR_DIAGNOSTIC_COMPLETE updates=192 validation_IoU=%.7f validation_Dice=%.7f all_background=%d/64 elapsed=%.3fs\n', ...
        valEvals(end).micro_foreground_IoU,valEvals(end).micro_foreground_Dice, ...
        valEvals(end).all_background_image_count,report.total_wall_time_s);
catch err
    failure.status='FAILED';failure.completed_main_updates=completed;
    failure.last_consistent_checkpoint=lastCheckpoint;
    failure.identifier=err.identifier;failure.message=err.message;failure.stack=err.stack;
    failure.elapsed_time_s=toc(clock);
    if completed>0 && exist('net','var') && exist('history','var')
        try
            [~,safePath]=savePoint(completed,context.schedule(completed).epoch,'interrupted');
            failure.last_consistent_checkpoint=safePath;
        catch saveErr
            failure.checkpoint_save_error=saveErr.message;
        end
    end
    writeJson(fullfile(out,'failure.json'),failure);
    writeJson(fullfile(out,'status.json'),struct('status','FAILED', ...
        'completed_updates',completed,'last_consistent_checkpoint',failure.last_consistent_checkpoint));
    fprintf('%s\n',getReport(err,'extended','hyperlinks','off'));rethrow(err);
end

    function [ve,te]=evaluatePoint(update)
        trainingRng=rng;stateBefore=net.State;
        valClock=tic;
        ve=evaluate_pilot(net,valData.X,valData.T,valData.V, ...
            valData.ids,valData.groupIds,cfg.loss,2);
        ve.evaluation_time_s=toc(valClock);ve.global_update=update;
        trainClock=tic;
        te=evaluate_smoke(net,trainData.X(:,:,:,train8),trainData.T(:,:,:,train8), ...
            trainData.V(:,:,:,train8),trainData.ids(train8),trainData.groupIds(train8),cfg.loss);
        te.scope='Fixed eight training images within pilot; not all128 training metrics';
        te.evaluation_time_s=toc(trainClock);te.global_update=update;
        assert(tableError(net.State,stateBefore)==0,'Validation altered network State');
        rng(trainingRng);
        fprintf('PILOT_VALIDATION update=%d loss=%.7f IoU=%.7f Dice=%.7f precision=%.7f recall=%.7f all_background=%d/64 time=%.3fs\n', ...
            update,ve.total_loss,ve.micro_foreground_IoU,ve.micro_foreground_Dice, ...
            ve.micro_precision,ve.micro_recall,ve.all_background_image_count,ve.evaluation_time_s);
    end

    function [point,path]=savePoint(update,epoch,suffix)
        if nargin<3,suffix='';end
        saveClock=tic;
        point.net=net;point.averageGrad=avg;point.averageSqGrad=avgSq;
        point.global_update=update;point.epoch=epoch;point.next_schedule_position=update+1;
        point.rng_state=rng;point.model_config=context.model_config;point.training_config=cfg;
        point.hashes=context.hashes;point.source=source;point.environment=environment;
        point.class_channel_order=cfg.output_channel_order;point.elapsed_time_s=toc(clock);
        point.history=history;point.validation_evaluations=valEvals;
        point.training8_diagnostics=trainEvals;point.profile='pilot';point.schedule=context.schedule;
        point.validation_order=context.validation_order;
        if isempty(suffix),name=sprintf('update_%03d.mat',update);
        else,name=sprintf('%s_update_%03d.mat',suffix,update);end
        path=fullfile(out,'checkpoints',name);checkpoint=point;save(path,'checkpoint','-v7.3');
        checkpointSaveTime=checkpointSaveTime+toc(saveClock);
    end

    function chooseBest(ve,pointPath)
        candidate.global_update=ve.global_update;candidate.total_loss=ve.total_loss;
        candidate.micro_foreground_IoU=ve.micro_foreground_IoU;
        candidate.micro_foreground_Dice=ve.micro_foreground_Dice;
        candidate.is_untrained_reference=ve.global_update==0;
        candidate.eligible=~isnan(ve.micro_foreground_IoU);
        candidate.selected_now=pilot_is_better(candidate,best);
        if candidate.selected_now
            best=candidate;
            copyfile(pointPath,fullfile(out,'checkpoints','best_validation.mat'),'f');
        end
        if isempty(bestHistory),bestHistory=candidate;else,bestHistory(end+1)=candidate;end %#ok<AGROW>
        selection.candidates=bestHistory;selection.best=best;
        selection.best_unavailable=isempty(best);
        selection.rule='validation micro IoU descending, valid loss ascending, update ascending; includes0';
        writeJson(fullfile(out,'best_selection.json'),selection);
    end
end

function check=verifyReload(point,path,X,cfg)
s=load(path,'checkpoint');q=s.checkpoint;
assert(strcmp(q.profile,'pilot') && isequal(q.hashes,point.hashes));
assert(q.global_update==point.global_update && q.next_schedule_position==point.global_update+1);
assert(isequal(q.rng_state,point.rng_state));
assert(tableError(point.net.Learnables,q.net.Learnables)==0 && tableError(point.net.State,q.net.State)==0);
assert(tableError(point.averageGrad,q.averageGrad)==0 && tableError(point.averageSqGrad,q.averageSqGrad)==0);
a=predict(point.net,dlarray(X,'SSCB'));b=predict(q.net,dlarray(X,'SSCB'));
error=double(max(abs(extractdata(a-b)),[],'all'));
assert(isfinite(error) && error<=cfg.checkpoints.prediction_reload_max_abs_error);
check.global_update=point.global_update;check.next_schedule_position=q.next_schedule_position;
check.prediction_max_abs_error=error;check.parameters_state_moments_exact=true;check.status='PASS';
fprintf('PILOT_RELOAD_OK update=%d next=%d prediction_error=%.9g\n',q.global_update,q.next_schedule_position,error);
end

function error=tableError(a,b)
if isempty(a)&&isempty(b),error=0;return;end
assert(istable(a)&&istable(b)&&isequal(a(:,1:2),b(:,1:2))&&height(a)==height(b));
error=0;
for k=1:height(a)
    av=extractdata(a.Value{k});bv=extractdata(b.Value{k});assert(isequal(size(av),size(bv)));
    if ~isempty(av),error=max(error,double(max(abs(av-bv),[],'all')));end
end
end

function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
