function report = train_smoke(dataRoot,runId)
%TRAIN_SMOKE Execute Step08 only: 16 main updates on eight frozen images.
% Includes one isolated replay of batch9 to verify checkpoint8 continuation.
% No pilot-validation images and no heldout-test images are opened.
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<2,runId=['smoke_' char(datetime('now',Format='yyyyMMdd_HHmmss'))];end
assert(~isempty(regexp(runId,'^[A-Za-z0-9_-]+$','once')),'Unsafe run ID');
out=fullfile(root,'runs',runId);assert(~exist(out,'dir'),'Run directory exists; do not overwrite');
mkdir(out);mkdir(fullfile(out,'checkpoints'));mkdir(fullfile(out,'source_snapshot'));
diary(fullfile(out,'run.log'));diaryCleanup=onCleanup(@()diary('off')); %#ok<NASGU>
runClock=tic;completed=0;lastCheckpoint='';
try
    [X,T,V,records,context]=load_smoke_inputs(dataRoot);
    cfg=context.config;profile=cfg.profiles.smoke;
    ids=string({records.id})';groups=string({records.group_id})';
    environment.matlab_version=version;environment.computer=computer;
    environment.cpu_identifier=getenv('PROCESSOR_IDENTIFIER');
    environment.execution_environment='CPU';environment.precision='single';
    environment.max_computational_threads=maxNumCompThreads;
    environment.gpu_count=gpuDeviceCount;
    environment.products=ver;
    try,[~,mem]=memory;environment.physical_memory_bytes=mem.PhysicalMemory.Total;catch,end
    assert(~contains(root,'"'));
    [s,revision]=system(sprintf('git -C "%s" rev-parse HEAD',root));assert(s==0);
    [s,dirty]=system(sprintf('git -C "%s" status --porcelain',root));assert(s==0);
    source.git_commit=strtrim(revision);source.git_dirty=~isempty(strtrim(dirty));
    mfiles=dir(fullfile(root,'matlab','*.m'));
    for k=1:numel(mfiles)
        original=fullfile(mfiles(k).folder,mfiles(k).name);
        copyfile(original,fullfile(out,'source_snapshot',mfiles(k).name));
        source.files(k).name=mfiles(k).name;source.files(k).sha256=file_sha256(original);
    end
    writeJson(fullfile(out,'input_access.json'),context);
    writeJson(fullfile(out,'environment.json'),environment);
    writeJson(fullfile(out,'source_version.json'),source);
    lossChecks=check_masked_segmentation_loss(out);
    [net,~]=build_baseline();
    logitsName=findLogits(net);
    history=struct([]);evals=struct([]);reloads=struct([]);
    averageGrad=[];averageSqGrad=[];
    evalClock=tic;
    entry=evaluate_smoke(net,X,T,V,ids,groups,cfg.loss);
    entry.global_update=0;entry.evaluation_time_s=toc(evalClock);evals=entry;
    writeJson(fullfile(out,'evaluations.json'),evals);
    [checkpoint,lastCheckpoint]=savePoint(0,0);
    reloads=verifyReload(checkpoint,lastCheckpoint,X(:,:,:,1),cfg);
    writeJson(fullfile(out,'reload_checks.json'),reloads);
    rng(checkpoint.rng_state);
    writeJson(fullfile(out,'status.json'),struct('status','RUNNING','completed_updates',0));
    resumeCheck=struct('status','NOT_RUN');
    fprintf('SMOKE_START main_update_budget=16 diagnostic_replay_budget=1\n');
    for iteration=1:profile.target_optimizer_updates
        batch=context.schedule(iteration);
        assert(batch.update==iteration && completed+1==iteration);
        [found,indices]=ismember(string(batch.sample_ids),ids);assert(all(found));
        Xb=X(:,:,:,indices);Tb=T(:,:,:,indices);Vb=V(:,:,:,indices);
        updateClock=tic;
        [newNet,newAvg,newAvgSq,stats]=updateOnce(net,averageGrad,averageSqGrad, ...
            iteration,Xb,Tb,Vb,cfg,logitsName);
        updateTime=toc(updateClock);
        if iteration==9
            % The primary trajectory is untouched by this isolated replay.
            mainRng=rng;
            restored=load(fullfile(out,'checkpoints','update_008.mat'),'checkpoint');
            restored=restored.checkpoint;
            assert(restored.global_update==8 && restored.next_schedule_position==9);
            rng(restored.rng_state);replayClock=tic;
            [probeNet,probeAvg,probeAvgSq,probeStats]=updateOnce(restored.net, ...
                restored.averageGrad,restored.averageSqGrad,9,Xb,Tb,Vb,cfg,logitsName);
            resumeCheck.replay_time_s=toc(replayClock);rng(mainRng);
            resumeCheck.parameter_max_abs_error=tableError(newNet.Learnables,probeNet.Learnables);
            resumeCheck.state_max_abs_error=tableError(newNet.State,probeNet.State);
            resumeCheck.first_moment_max_abs_error=tableError(newAvg,probeAvg);
            resumeCheck.second_moment_max_abs_error=tableError(newAvgSq,probeAvgSq);
            resumeCheck.loss_abs_error=abs(stats.total_loss-probeStats.total_loss);
            pa=predict(newNet,dlarray(Xb,'SSCB'));pb=predict(probeNet,dlarray(Xb,'SSCB'));
            resumeCheck.probability_max_abs_error=double(max(abs(extractdata(pa-pb)),[],'all'));
            errors=[resumeCheck.parameter_max_abs_error,resumeCheck.state_max_abs_error, ...
                resumeCheck.first_moment_max_abs_error,resumeCheck.second_moment_max_abs_error, ...
                resumeCheck.loss_abs_error,resumeCheck.probability_max_abs_error];
            assert(all(isfinite(errors)) && all(errors<=1e-6),'Checkpoint replay mismatch');
            resumeCheck.status='PASS';resumeCheck.checkpoint_update=8;
            resumeCheck.replayed_schedule_position=9;
            resumeCheck.main_optimizer_updates_added=0;
            resumeCheck.diagnostic_optimizer_updates=1;
            writeJson(fullfile(out,'resume_replay_check.json'),resumeCheck);
            fprintf('RESUME_REPLAY_OK checkpoint=8 batch=9 max_error=%.9g\n',max(errors));
        end
        net=newNet;averageGrad=newAvg;averageSqGrad=newAvgSq;completed=iteration;
        stats.global_update=iteration;stats.epoch=batch.epoch;
        stats.sample_ids=cellstr(string(batch.sample_ids));
        stats.valid_pixel_count=squeeze(sum(Vb,[1 2]));
        stats.update_time_s=updateTime;stats.elapsed_time_s=toc(runClock);
        if isempty(history),history=stats;else,history(end+1)=stats;end %#ok<AGROW>
        writeJson(fullfile(out,'history.json'),history);
        writetable(struct2table(rmfield(history,{'sample_ids','valid_pixel_count'})), ...
            fullfile(out,'training_history.csv'));
        fprintf('SMOKE_UPDATE %02d/16 loss=%.7f BCE=%.7f DiceLoss=%.7f grad=%.6g clip=%.6g delta=%.6g time=%.3fs\n', ...
            iteration,stats.total_loss,stats.BCE_component,stats.Dice_component, ...
            stats.unclipped_global_gradient_norm,stats.clip_factor,stats.parameter_delta_L2,updateTime);
        if ismember(iteration,profile.evaluation_updates)
            evalClock=tic;entry=evaluate_smoke(net,X,T,V,ids,groups,cfg.loss);
            entry.global_update=iteration;entry.evaluation_time_s=toc(evalClock);
            evals(end+1)=entry;writeJson(fullfile(out,'evaluations.json'),evals); %#ok<AGROW>
            fprintf('SMOKE_TRAIN_DIAGNOSTIC update=%d loss=%.7f IoU=%.7f Dice=%.7f\n', ...
                iteration,entry.total_loss,entry.micro_foreground_IoU,entry.micro_foreground_Dice);
        end
        if ismember(iteration,profile.checkpoint_updates)
            [checkpoint,lastCheckpoint]=savePoint(iteration,batch.epoch);
            reloads(end+1)=verifyReload(checkpoint,lastCheckpoint,X(:,:,:,1),cfg); %#ok<AGROW>
            writeJson(fullfile(out,'reload_checks.json'),reloads);
            rng(checkpoint.rng_state);
        end
        writeJson(fullfile(out,'status.json'),struct('status','RUNNING','completed_updates',completed));
    end
    assert(completed==16 && numel(history)==16 && strcmp(resumeCheck.status,'PASS'));
    assert(all([history.unclipped_global_gradient_norm]>0) && all([history.parameter_delta_L2]>0));
    plot_smoke_run(out,history,evals,net,X,T,V,ids);
    report.status='PASS';report.scope='Step08 engineering smoke test; training images only';
    report.main_optimizer_updates=completed;report.diagnostic_replay_updates=1;
    report.unique_training_images=8;report.training_proxy_groups=8;
    report.validation_images_opened=0;report.heldout_test_images_opened=0;
    report.initial_training_diagnostics=evals(1);report.final_training_diagnostics=evals(end);
    report.all_main_gradients_finite_nonzero=true;report.all_main_parameter_updates_finite_nonzero=true;
    report.loss_contract_checks=lossChecks;report.reload_checks=reloads;
    report.resume_replay=resumeCheck;report.source=source;report.hashes=context.hashes;
    report.main_update_time_s=sum([history.update_time_s]);
    report.total_wall_time_s=toc(runClock);report.run_id=runId;
    report.limitations='No validation/test/generalization/fusion/realtime claim. CPU training timings only.';
    writeJson(fullfile(out,'summary.json'),report);
    writeJson(fullfile(out,'status.json'),struct('status','PASS','completed_updates',completed));
    fprintf('SMOKE_COMPLETE updates=16 training_images=8 validation_images=0 heldout_images=0 elapsed=%.3fs\n',report.total_wall_time_s);
catch err
    failure.status='FAILED';failure.completed_main_updates=completed;
    failure.last_consistent_checkpoint=lastCheckpoint;
    failure.identifier=err.identifier;failure.message=err.message;
    failure.stack=err.stack;failure.elapsed_time_s=toc(runClock);
    writeJson(fullfile(out,'failure.json'),failure);
    writeJson(fullfile(out,'status.json'),struct('status','FAILED', ...
        'completed_updates',completed,'last_consistent_checkpoint',lastCheckpoint));
    fprintf('%s\n',getReport(err,'extended','hyperlinks','off'));rethrow(err);
end

    function [point,path] = savePoint(update,epoch)
        point.net=net;point.averageGrad=averageGrad;point.averageSqGrad=averageSqGrad;
        point.global_update=update;point.epoch=epoch;point.next_schedule_position=update+1;
        point.rng_state=rng;point.model_config=context.model_config;
        point.training_config=cfg;point.hashes=context.hashes;
        point.source=source;point.environment=environment;
        point.class_channel_order=cfg.output_channel_order;
        point.elapsed_time_s=toc(runClock);point.history=history;point.evaluations=evals;
        point.profile='smoke';point.schedule=context.schedule;
        path=fullfile(out,'checkpoints',sprintf('update_%03d.mat',update));
        checkpoint=point;save(path,'checkpoint','-v7.3');
    end
end

function name=findLogits(net)
layers=net.Layers;
idx=find(arrayfun(@(l)isa(l,'nnet.cnn.layer.SoftmaxLayer'),layers));
assert(isscalar(idx),'Expected one softmax layer');
c=net.Connections;source=c.Source(strcmp(c.Destination,layers(idx).Name));
assert(numel(source)==1);name=string(source(1));
end

function [newNet,newAvg,newAvgSq,stats]=updateOnce(net,avg,avgSq,iteration,X,T,V,cfg,logitsName)
[loss,gradients,state,bce,dice]=dlfeval(@lossAndGradient,net, ...
    dlarray(X,'SSCB'),T,V,cfg.loss,logitsName);
stats.total_loss=double(extractdata(loss));
stats.BCE_component=double(extractdata(bce));stats.Dice_component=double(extractdata(dice));
assert(all(isfinite([stats.total_loss stats.BCE_component stats.Dice_component])),'Nonfinite loss');
sq=0;nonzero=0;
for k=1:height(gradients)
    g=extractdata(gradients.Value{k});assert(all(isfinite(g),'all'),'Nonfinite gradient');
    sq=sq+sum(double(g).^2,'all');nonzero=nonzero+nnz(g);
end
norm=sqrt(sq);assert(isfinite(norm) && norm>0 && nonzero>0,'No effective gradient');
factor=min(1,cfg.optimizer.max_gradient_norm/norm);
for k=1:height(gradients),gradients.Value{k}=gradients.Value{k}*single(factor);end
[newNet,newAvg,newAvgSq]=adamupdate(net,gradients,avg,avgSq,iteration, ...
    cfg.optimizer.learning_rate,cfg.optimizer.beta1,cfg.optimizer.beta2,cfg.optimizer.epsilon);
newNet.State=state;deltaSq=0;
for k=1:height(net.Learnables)
    old=extractdata(net.Learnables.Value{k});new=extractdata(newNet.Learnables.Value{k});
    assert(all(isfinite(new),'all'),'Nonfinite new parameter');
    deltaSq=deltaSq+sum((double(new)-double(old)).^2,'all');
end
assert(deltaSq>0,'Parameters did not change');
stats.unclipped_global_gradient_norm=norm;stats.clip_factor=factor;
stats.gradient_nonzero_elements=nonzero;stats.parameter_delta_L2=sqrt(deltaSq);
end

function [loss,gradients,state,bce,dice]=lossAndGradient(net,X,T,V,cfg,logitsName)
[Z,state]=forward(net,X,Outputs=logitsName);
[loss,bce,dice]=masked_segmentation_loss(Z,T,V,cfg);
gradients=dlgradient(loss,net.Learnables);
end

function check=verifyReload(point,path,X,cfg)
s=load(path,'checkpoint');loaded=s.checkpoint;
assert(loaded.global_update==point.global_update && loaded.next_schedule_position==point.global_update+1);
assert(strcmp(loaded.profile,'smoke') && isequal(loaded.hashes,point.hashes));
assert(isequal(loaded.rng_state,point.rng_state));
assert(tableError(point.net.Learnables,loaded.net.Learnables)==0);
assert(tableError(point.net.State,loaded.net.State)==0);
assert(tableError(point.averageGrad,loaded.averageGrad)==0);
assert(tableError(point.averageSqGrad,loaded.averageSqGrad)==0);
before=predict(point.net,dlarray(X,'SSCB'));after=predict(loaded.net,dlarray(X,'SSCB'));
error=double(max(abs(extractdata(before-after)),[],'all'));
assert(isfinite(error) && error<=cfg.checkpoints.prediction_reload_max_abs_error);
check.global_update=point.global_update;check.next_schedule_position=loaded.next_schedule_position;
check.prediction_max_abs_error=error;check.parameters_state_moments_exact=true;check.status='PASS';
fprintf('CHECKPOINT_RELOAD_OK update=%d next=%d prediction_error=%.9g\n',point.global_update,loaded.next_schedule_position,error);
end

function error=tableError(a,b)
if isempty(a)&&isempty(b),error=0;return;end
assert(istable(a)&&istable(b) && isequal(a(:,1:2),b(:,1:2)) && height(a)==height(b));
error=0;
for k=1:height(a)
    av=extractdata(a.Value{k});bv=extractdata(b.Value{k});
    assert(isequal(size(av),size(bv)));
    if ~isempty(av),error=max(error,double(max(abs(av-bv),[],'all')));end
end
end

function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
