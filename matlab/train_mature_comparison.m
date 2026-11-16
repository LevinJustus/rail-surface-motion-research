function aggregate=train_mature_comparison(dataRoot,configPath,runId)
%TRAIN_MATURE_COMPARISON Project mature-structure comparison, common recipe.
% Full1396/400, three fresh seeds; not new algorithm, optimal recipe, vendor
% reproduction, heldout-test evaluation, motion fusion or realtime evidence.
% Historical U-Net runner, loader, loss, update and plotting remain unchanged.
root=fileparts(fileparts(mfilename('fullpath')));
if ~isfile(configPath),configPath=fullfile(root,configPath);end
assert(~isempty(regexp(runId,'^[A-Za-z0-9_-]+$','once')));
out=fullfile(root,'runs',runId);assert(~exist(out,'dir'),'Group run already exists');
mkdir(out);mkdir(fullfile(out,'source_snapshot'));mkdir(fullfile(out,'seeds'));
diary(fullfile(out,'run.log'));cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
timer=tic;cpuTimer=cputime;finished=cell(0,1);
try
    writeJson(fullfile(out,'status.json'),struct('status','RUNNING','stage','loading_full_data','completed_seeds',0));
    cfg=jsondecode(fileread(configPath));
    token=split(strtrim(fileread([configPath(1:end-5) '.sha256'])));
    assert(strcmp(file_sha256(configPath),char(token(1))));
    assert(strcmp(cfg.mode,'mature_comparison'));
    loadClock=tic;loadCpu=cputime;[trainData,valData,context]=load_full_inputs(dataRoot,configPath);
    dataTime=toc(loadClock);dataCpu=cputime-loadCpu;
    assert(strcmp(file_sha256(fullfile(root,cfg.batch_schedule)),cfg.batch_schedule_sha256));
    schedule=jsondecode(fileread(fullfile(root,cfg.batch_schedule)));
    assert(isequal(string(schedule.validation_order(:)),valData.ids));
    batches=schedule.full;steps=ceil(numel(trainData.ids)/cfg.batch_size);
    assert(numel(batches)==cfg.max_epochs*steps);
    for epoch=1:cfg.max_epochs
        rows=batches([batches.epoch]==epoch);visited=strings(0,1);
        assert(numel(rows)==steps);
        for j=1:numel(rows)
            assert(rows(j).update==(epoch-1)*steps+j);
            ids=string(rows(j).sample_ids(:));assert(numel(ids)>=1&&numel(ids)<=cfg.batch_size);
            assert(numel(unique(ids))==numel(ids));visited=[visited;ids]; %#ok<AGROW>
        end
        assert(isequal(sort(visited),sort(trainData.ids)));
    end
    environment.matlab_version=version;environment.computer=computer;
    environment.cpu_identifier=getenv('PROCESSOR_IDENTIFIER');
    environment.execution_environment='CPU';environment.precision='single';
    environment.max_computational_threads=maxNumCompThreads;[environment.gpu_count,environment.gpu_query_available]=publicOptionalGpuCount();
    environment.products=ver;
    [status,revision]=system(sprintf('git -C "%s" rev-parse HEAD',root));assert(status==0);
    [status,dirty]=system(sprintf('git -C "%s" status --porcelain',root));assert(status==0);
    assert(isempty(strtrim(dirty)),'Commit source and config before mature comparison');
    source.git_commit=strtrim(revision);source.git_dirty=false;
    mfiles=dir(fullfile(root,'matlab','*.m'));
    for k=1:numel(mfiles)
        original=fullfile(mfiles(k).folder,mfiles(k).name);copied=fullfile(out,'source_snapshot',mfiles(k).name);
        copyfile(original,copied);source.files(k).name=mfiles(k).name;
        source.files(k).sha256=file_sha256(copied);
        assert(strcmp(source.files(k).sha256,file_sha256(original)));
    end
    writeJson(fullfile(out,'input_access.json'),context);
    writeJson(fullfile(out,'environment.json'),environment);writeJson(fullfile(out,'source_version.json'),source);
    seeds=double(cfg.initialization_seeds(:)');assert(numel(seeds)==3&&numel(unique(seeds))==3);
    for seed=seeds
        seedDir=fullfile(out,'seeds',sprintf('seed_%d',seed));mkdir(seedDir);
        result=trainSeed(trainData,valData,context,cfg,batches,seedDir,seed,environment,source,root);
        finished{end+1,1}=result; %#ok<AGROW>
        writeJson(fullfile(out,'completed_seed_summaries.json'),finished);
        writeJson(fullfile(out,'status.json'),struct('status','RUNNING','completed_seeds',numel(finished),'total_seeds',3));
    end
    fingerprints=cellfun(@(r)r.initial_parameter_sha256,finished,UniformOutput=false);
    assert(numel(unique(string(fingerprints)))==3,'Initialization seeds produced identical weights');
    aggregate.status='PASS';
    aggregate.scope='Project implementation of a mature structure under a common training recipe; validation only';
    aggregate.initialization_seeds=seeds;aggregate.reference_seed=seeds(1);
    aggregate.reference_rule='First predefined seed, not highest validation score';
    aggregate.seed_summaries=finished;aggregate.data_preparation_time_s=dataTime;
    aggregate.data_preparation_cpu_time_s=dataCpu;
    aggregate.total_wall_time_s=toc(timer);aggregate.total_cpu_time_s=cputime-cpuTimer;
    aggregate.cpu_timing_scope='MATLAB process cputime differences, distinct from elapsed wall time; neither is inference FPS';
    aggregate.source=source;aggregate.hashes=context.hashes;
    assert(numel(unique(cellfun(@(r)r.parameter_count,finished)))==1);
    assert(numel(unique(cellfun(@(r)r.bn_count,finished)))==1);
    aggregate.parameter_count=finished{1}.parameter_count;aggregate.bn_count=finished{1}.bn_count;
    aggregate.training_images=1396;aggregate.validation_images=400;aggregate.heldout_test_images_opened=0;
    names={'micro_foreground_IoU','micro_foreground_Dice','micro_precision','micro_recall'};
    for k=1:numel(names)
        key=names{k};last=cellfun(@(r)r.final_validation.(key),finished);
        selected=cellfun(@(r)r.best_validation.(key),finished);
        aggregate.last_statistics.(key)=stats(last);
        aggregate.selected_statistics.(key)=stats(selected);
    end
    aggregate.limitations='Common-recipe project implementation of a mature structure; not a new algorithm, per-model optimum, official bitwise reproduction, heldout test, significance, real-video, hardware or realtime claim.';
    writeJson(fullfile(out,'summary.json'),aggregate);
    writeJson(fullfile(out,'status.json'),struct('status','PASS','completed_seeds',3));
    fprintf('MATURE_COMPARISON_COMPLETE seeds=3 selected_IoU_mean=%.6f selected_IoU_sd=%.6f wall=%.1fs cpu=%.1fs\n', ...
        aggregate.selected_statistics.micro_foreground_IoU.mean, ...
        aggregate.selected_statistics.micro_foreground_IoU.sample_std,aggregate.total_wall_time_s,aggregate.total_cpu_time_s);
catch err
    failure.status='FAILED';failure.completed_seeds=numel(finished);failure.message=err.message;
    failure.identifier=err.identifier;failure.stack=err.stack;failure.elapsed_time_s=toc(timer);
    failure.cpu_time_s=cputime-cpuTimer;
    writeJson(fullfile(out,'failure.json'),failure);writeJson(fullfile(out,'status.json'),failure);
    rethrow(err);
end
end

function report=trainSeed(td,vd,context,cfg,batches,out,seed,environment,source,root)
timer=tic;cpuTimer=cputime;completed=0;lastPath='';history=struct([]);ve=struct([]);te=struct([]);
reloads=struct([]);best=[];bestHistory=struct([]);epochHistory=struct([]);
mkdir(fullfile(out,'checkpoints'));
fid=fopen(fullfile(out,'history.jsonl'),'w','n','UTF-8');assert(fid>=0);
closeHistory=onCleanup(@()fclose(fid)); %#ok<NASGU>
try
    constructClock=tic;constructCpu=cputime;
    [net,modelCfg,modelPath,architecture]=build_seeded_mature_baseline(seed,fullfile(root,cfg.model_config),out);
    constructionWall=toc(constructClock);constructionCpu=cputime-constructCpu;
    assert(architecture.bn_count==65,'Unexpected approved mature-model BN layer count');
    expectedBN=65;initialState=stateSummary(net.State,expectedBN);
    assert(initialState.means_all_zero&&initialState.variances_all_one, ...
        'Fresh mature comparison requires zero BN means and unit variances');
    architecturePath=fullfile(out,'architecture.json');writeJson(architecturePath,architecture);
    construction.wall_time_s=constructionWall;construction.cpu_time_s=constructionCpu;
    construction.parameter_count=architecture.parameter_count;construction.bn_count=architecture.bn_count;
    construction.initial_state_summary=initialState;
    writeJson(fullfile(out,'construction.json'),construction);
    writeJson(fullfile(out,'status.json'),struct('status','RUNNING','seed',seed, ...
        'stage','construction_complete','completed_updates',0,'wall_time_s',toc(timer),'cpu_time_s',cputime-cpuTimer));
    fprintf('MATURE_CONSTRUCTED seed=%d parameters=%d BN=%d wall=%.3fs cpu=%.3fs\n', ...
        seed,architecture.parameter_count,architecture.bn_count,constructionWall,constructionCpu);
    initialHash=network_parameter_sha256(net);avg=[];avgSq=[];
    hashes=context.hashes;hashes.seed_model_config_sha256=file_sha256(modelPath);
    hashes.batch_schedule_sha256=cfg.batch_schedule_sha256;
    hashes.architecture_sha256=file_sha256(architecturePath);
    [ok,t8]=ismember(string(cfg.fixed_train_ids(:)),td.ids);assert(all(ok)&&numel(t8)==8);
    [ok,v8]=ismember(string(cfg.fixed_validation_ids(:)),vd.ids);assert(all(ok)&&numel(v8)==8);
    softmax=string(net.OutputNames{1});outputRow=string(net.Connections.Destination)==softmax;
    assert(nnz(outputRow)==1);logits=string(net.Connections.Source(outputRow));
    assert(numel(net.InputNames)==1&&numel(net.OutputNames)==1);
    assert(strcmp(softmax,architecture.softmax_layer)&&strcmp(logits,architecture.logits_layer));
    wait=0;meaningfulBest=-Inf;stopped=false;stopReason='running';
    [v,t]=evaluateAt(0,0);ve=v;te=t;
    writeJson(fullfile(out,'history.json'),history);
    writeJson(fullfile(out,'validation_evaluations.json'),ve);
    writeJson(fullfile(out,'training8_evaluations.json'),te);
    if isfinite(v.micro_foreground_IoU),meaningfulBest=v.micro_foreground_IoU;end
    selected=selectBest(v);
    [point,lastPath]=savePoint(0,0,'');reloads=reloadCheck(point,lastPath,td.X(:,:,:,1),cfg);
    writeJson(fullfile(out,'reload_checks.json'),reloads);
    if selected,copyfile(lastPath,fullfile(out,'checkpoints','best_validation.mat'),'f');end
    rng(point.rng_state);steps=ceil(numel(td.ids)/cfg.batch_size);
    for epoch=1:cfg.max_epochs
        lr=cfg.optimizer.learning_rate*cfg.learning_rate_factors(epoch);
        stepCfg=cfg;stepCfg.optimizer.learning_rate=lr;
        first=completed+1;
        for pos=(epoch-1)*steps+(1:steps)
            batch=batches(pos);assert(batch.update==completed+1&&batch.epoch==epoch);
            [ok,idx]=ismember(string(batch.sample_ids),td.ids);assert(all(ok));
            stepClock=tic;stepCpu=cputime;beforeUpdateRng=rng;
            try
                [newNet,newAvg,newAvgSq,h]=segmentation_train_step(net,avg,avgSq,batch.update, ...
                    td.X(:,:,:,idx),td.T(:,:,:,idx),td.V(:,:,:,idx),stepCfg,logits);
                % Validate the candidate before committing network, moments,
                % update number or history. A failure keeps the old net and
                % restores the pre-update dropout RNG in the same catch.
                candidateState=stateSummary(newNet.State,expectedBN);
                h.bn_state_max_absolute_change=tableError(net.State,newNet.State);
                h.bn_state_all_finite=candidateState.all_finite;
                h.bn_mean_state_count=candidateState.mean_count;
                h.bn_variance_state_count=candidateState.variance_count;
            catch stepError
                rng(beforeUpdateRng);rethrow(stepError);
            end
            h.update_time_s=toc(stepClock);h.update_cpu_time_s=cputime-stepCpu;
            net=newNet;avg=newAvg;avgSq=newAvgSq;completed=batch.update;
            h.global_update=completed;h.epoch=epoch;h.learning_rate=lr;h.sample_count=numel(idx);
            h.sample_ids=cellstr(string(batch.sample_ids));h.valid_pixel_count=squeeze(sum(td.V(:,:,:,idx),[1 2]));
            h.elapsed_time_s=toc(timer);h.elapsed_cpu_time_s=cputime-cpuTimer;
            if isempty(history),history=h;else,history(end+1)=h;end %#ok<AGROW>
            fprintf(fid,'%s\n',jsonencode(h));
            if mod(completed,16)==0 || completed==1
                fprintf('MATURE_UPDATE seed=%d epoch=%d/%d update=%d/%d loss=%.6f lr=%.3g wall=%.3fs cpu=%.3fs BN=%.6g\n', ...
                    seed,epoch,cfg.max_epochs,completed,numel(batches),h.total_loss,lr, ...
                    h.update_time_s,h.update_cpu_time_s,h.bn_state_max_absolute_change);
                writeJson(fullfile(out,'status.json'),struct('status','RUNNING','seed',seed, ...
                    'stage','training_updates','epoch',epoch,'completed_updates',completed, ...
                    'elapsed_time_s',toc(timer),'cpu_time_s',cputime-cpuTimer));
            end
        end
        [v,t]=evaluateAt(completed,epoch);ve(end+1)=v;te(end+1)=t; %#ok<AGROW>
        if isfinite(v.micro_foreground_IoU)&&v.micro_foreground_IoU>meaningfulBest+cfg.early_stopping.min_delta
            meaningfulBest=v.micro_foreground_IoU;wait=0;
        else,wait=wait+1;end
        row.epoch=epoch;row.global_update=completed;row.learning_rate=lr;
        row.validation_IoU=v.micro_foreground_IoU;row.validation_loss=v.total_loss;
        row.no_meaningful_improvement_epochs=wait;row.meaningful_best_IoU=meaningfulBest;
        row.elapsed_time_s=toc(timer);row.cpu_time_s=cputime-cpuTimer;
        part=history(first:completed);
        row.training_loss_sample_weighted=sum([part.total_loss].*[part.sample_count])/sum([part.sample_count]);
        if isempty(epochHistory),epochHistory=row;else,epochHistory(end+1)=row;end %#ok<AGROW>
        selected=selectBest(v);
        stopped=epoch>=cfg.early_stopping.min_epochs && wait>=cfg.early_stopping.patience && epoch<cfg.max_epochs;
        if stopped,stopReason='predeclared_validation_patience';
        elseif epoch==cfg.max_epochs,stopReason='max_epochs_reached';end
        writeJson(fullfile(out,'history.json'),history);writeJson(fullfile(out,'validation_evaluations.json'),ve);
        writeJson(fullfile(out,'training8_evaluations.json'),te);writeJson(fullfile(out,'epoch_history.json'),epochHistory);
        writetable(struct2table(rmfield(history,{'sample_ids','valid_pixel_count'})),fullfile(out,'training_history.csv'));
        writetable(struct2table(epochHistory),fullfile(out,'epoch_history.csv'));
        [point,lastPath]=savePoint(completed,epoch,'');
        reloads(end+1)=reloadCheck(point,lastPath,td.X(:,:,:,1),cfg); %#ok<AGROW>
        if selected,copyfile(lastPath,fullfile(out,'checkpoints','best_validation.mat'),'f');end
        writeJson(fullfile(out,'reload_checks.json'),reloads);rng(point.rng_state);
        if stopped,break;end
    end
    assert(completed==epoch*steps&&numel(history)==completed);
    copyfile(lastPath,fullfile(out,'checkpoints','last.mat'),'f');
    if ~isempty(best)
        bestState=load(fullfile(out,'checkpoints','best_validation.mat'),'checkpoint');
        assert(bestState.checkpoint.global_update==best.global_update);
    else
        best=struct('global_update',NaN,'epoch',NaN,'total_loss',NaN, ...
            'micro_foreground_IoU',NaN,'micro_foreground_Dice',NaN, ...
            'micro_precision',NaN,'micro_recall',NaN,'status','best_unavailable');
    end
    plot_full_run(out,history,ve,te,net,td.X(:,:,:,t8),td.T(:,:,:,t8),td.V(:,:,:,t8),td.ids(t8), ...
        vd.X(:,:,:,v8),vd.T(:,:,:,v8),vd.V(:,:,:,v8),vd.ids(v8));
    report.status='PASS';report.initialization_seed=seed;report.initial_parameter_sha256=initialHash;
    report.scope='Project mature-structure implementation under the common recipe, validation only';
    report.parameter_count=architecture.parameter_count;report.bn_count=architecture.bn_count;
    report.initial_state_summary=initialState;report.final_state_summary=stateSummary(net.State,expectedBN);
    report.architecture_file='architecture.json';report.architecture_sha256=hashes.architecture_sha256;
    report.construction_wall_time_s=constructionWall;report.construction_cpu_time_s=constructionCpu;
    report.main_optimizer_updates=completed;report.completed_epochs=epoch;report.max_epochs=cfg.max_epochs;
    report.stopped_early=stopped;report.stop_reason=stopReason;report.best_validation=best;
    report.initial_validation=ve(1);report.final_validation=ve(end);
    report.initial_training8=te(1);report.final_training8=te(end);
    report.reload_checks=reloads;report.source=source;report.hashes=hashes;
    report.primary_last_checkpoint='checkpoints/last.mat';
    if isfield(best,'status')&&strcmp(best.status,'best_unavailable'),report.selected_checkpoint='';
    else,report.selected_checkpoint='checkpoints/best_validation.mat';end
    report.update_time_s=sum([history.update_time_s]);report.validation_time_s=sum([ve.evaluation_time_s]);
    report.update_cpu_time_s=sum([history.update_cpu_time_s]);
    report.validation_cpu_time_s=sum([ve.evaluation_cpu_time_s]);
    report.training8_diagnostic_time_s=sum([te.evaluation_time_s]);
    report.training8_diagnostic_cpu_time_s=sum([te.evaluation_cpu_time_s]);
    report.total_wall_time_s=toc(timer);report.total_cpu_time_s=cputime-cpuTimer;report.heldout_test_images_opened=0;
    report.limitations='Common recipe, not architecture-specific optimal tuning, official/author bitwise reproduction, new algorithm or final-test evidence';
    writeJson(fullfile(out,'summary.json'),report);writeJson(fullfile(out,'status.json'),struct('status','PASS','completed_updates',completed));
    fprintf('MATURE_SEED_COMPLETE seed=%d epochs=%d best_epoch=%d best_IoU=%.6f last_IoU=%.6f wall=%.1fs cpu=%.1fs\n', ...
        seed,epoch,best.epoch,best.micro_foreground_IoU,ve(end).micro_foreground_IoU,report.total_wall_time_s,report.total_cpu_time_s);
catch err
    failure.status='FAILED';failure.seed=seed;failure.completed_updates=completed;
    failure.message=err.message;failure.identifier=err.identifier;failure.stack=err.stack;failure.last_checkpoint=lastPath;
    failure.elapsed_time_s=toc(timer);failure.cpu_time_s=cputime-cpuTimer;
    if completed>0
        try,[~,failure.last_checkpoint]=savePoint(completed,ceil(completed/steps),'interrupted');catch,end
    end
    writeJson(fullfile(out,'failure.json'),failure);writeJson(fullfile(out,'status.json'),failure);rethrow(err);
end

    function [v,t]=evaluateAt(update,ep)
        savedRng=rng;restoreRng=onCleanup(@()rng(savedRng)); %#ok<NASGU>
        savedState=net.State;stateSummary(savedState,expectedBN);
        writeJson(fullfile(out,'status.json'),struct('status','RUNNING','seed',seed, ...
            'stage','complete_validation','epoch',ep,'completed_updates',update, ...
            'elapsed_time_s',toc(timer),'cpu_time_s',cputime-cpuTimer));
        fprintf('MATURE_VALIDATION_START seed=%d epoch=%d update=%d validation_images=400\n',seed,ep,update);
        valCpu=cputime;
        v=evaluate_pilot(net,vd.X,vd.T,vd.V,vd.ids,vd.groupIds,cfg.loss,2);
        v.evaluation_cpu_time_s=cputime-valCpu;
        v.global_update=update;v.epoch=ep;v.scope='Full fixed400 validation, not heldout test';
        ticTrain=tic;trainCpu=cputime;
        t=evaluate_smoke(net,td.X(:,:,:,t8),td.T(:,:,:,t8),td.V(:,:,:,t8),td.ids(t8),td.groupIds(t8),cfg.loss);
        t.evaluation_time_s=toc(ticTrain);t.evaluation_cpu_time_s=cputime-trainCpu;
        t.global_update=update;t.epoch=ep;t.scope='Fixed8 training diagnostic, not full1396 metric';
        stateSummary(net.State,expectedBN);assert(tableError(net.State,savedState)==0);rng(savedRng);
        fprintf('MATURE_VALIDATION seed=%d epoch=%d update=%d loss=%.6f IoU=%.6f Dice=%.6f P=%.6f R=%.6f background=%d/400\n', ...
            seed,ep,update,v.total_loss,v.micro_foreground_IoU,v.micro_foreground_Dice,v.micro_precision,v.micro_recall,v.all_background_image_count);
    end
    function [point,path]=savePoint(update,ep,suffix)
        point.net=net;point.averageGrad=avg;point.averageSqGrad=avgSq;point.global_update=update;
        point.epoch=ep;point.next_schedule_position=update+1;point.rng_state=rng;
        point.model_config=modelCfg;point.training_config=cfg;point.hashes=hashes;point.source=source;
        point.environment=environment;point.initialization_seed=seed;point.class_channel_order=cfg.output_channel_order;
        point.architecture=architecture;point.bn_count=expectedBN;
        point.initial_state_summary=initialState;point.current_state_summary=stateSummary(net.State,expectedBN);
        point.construction_wall_time_s=constructionWall;point.construction_cpu_time_s=constructionCpu;
        % Full per-image history lives once in JSON/JSONL, not duplicated in
        % every MAT checkpoint (large HDF5 object overhead on CPU Windows).
        point.history_log='history.jsonl';point.validation_log='validation_evaluations.json';
        point.training8_log='training8_evaluations.json';
        if isempty(history),point.last_update_diagnostic=struct([]);
        else,point.last_update_diagnostic=history(end);end
        point.validation_summaries=rmfield(ve,{'per_image','per_group','batch_diagnostics'});
        point.training8_summaries=rmfield(te,{'per_image','per_group'});
        point.early_stopping_wait=wait;point.meaningful_best_IoU=meaningfulBest;
        point.epoch_history=epochHistory;point.best_validation=best;point.best_candidates=bestHistory;
        point.stopped_early=stopped;point.stop_reason=stopReason;
        point.schedule_file=cfg.batch_schedule;point.elapsed_time_s=toc(timer);
        point.cpu_time_s=cputime-cpuTimer;point.profile='mature_comparison';
        if isempty(suffix),name=sprintf('epoch_%03d.mat',ep);else,name=sprintf('%s_update_%06d.mat',suffix,update);end
        path=fullfile(out,'checkpoints',name);checkpoint=point;save(path,'checkpoint','-v7.3');
    end
    function selected=selectBest(v)
        c.global_update=v.global_update;c.epoch=v.epoch;c.total_loss=v.total_loss;
        for field={'micro_foreground_IoU','micro_foreground_Dice','micro_precision','micro_recall'}
            c.(field{1})=v.(field{1});
        end
        c.is_untrained_reference=v.global_update==0;c.selected_now=pilot_is_better(c,best);
        selected=c.selected_now;if selected,best=c;end
        if isempty(bestHistory),bestHistory=c;else,bestHistory(end+1)=c;end %#ok<AGROW>
        writeJson(fullfile(out,'best_selection.json'),struct('candidates',bestHistory,'best',best));
    end
end

function c=reloadCheck(p,path,X,cfg)
s=load(path,'checkpoint');q=s.checkpoint;
assert(strcmp(q.profile,'mature_comparison')&&isequal(p.hashes,q.hashes)&&isequal(p.rng_state,q.rng_state));
assert(q.global_update==p.global_update&&q.next_schedule_position==p.global_update+1);
assert(isequaln(p.model_config,q.model_config)&&isequaln(p.training_config,q.training_config));
assert(isequaln(p.architecture,q.architecture)&&p.bn_count==q.bn_count);
assert(isequaln(p.initial_state_summary,q.initial_state_summary)&& ...
    isequaln(p.current_state_summary,q.current_state_summary));
assert(isequaln(p.early_stopping_wait,q.early_stopping_wait)&& ...
    isequaln(p.meaningful_best_IoU,q.meaningful_best_IoU)&& ...
    isequaln(p.best_validation,q.best_validation)&&isequaln(p.best_candidates,q.best_candidates));
assert(p.initialization_seed==q.initialization_seed&&isequaln(p.epoch_history,q.epoch_history));
assert(tableError(p.net.Learnables,q.net.Learnables)==0&&tableError(p.net.State,q.net.State)==0);
assert(tableError(p.averageGrad,q.averageGrad)==0&&tableError(p.averageSqGrad,q.averageSqGrad)==0);
stateSummary(q.net.State,p.bn_count);
savedRng=rng;restoreRng=onCleanup(@()rng(savedRng)); %#ok<NASGU>
pState=p.net.State;qState=q.net.State;
a=predict(p.net,dlarray(X,'SSCB'));b=predict(q.net,dlarray(X,'SSCB'));
assert(tableError(p.net.State,pState)==0&&tableError(q.net.State,qState)==0&&isequal(savedRng,rng));
e=double(max(abs(extractdata(a-b)),[],'all'));assert(isfinite(e)&&e<=cfg.prediction_reload_max_abs_error);
c.global_update=p.global_update;c.epoch=p.epoch;c.next_schedule_position=q.next_schedule_position;
c.prediction_max_abs_error=e;c.parameters_state_moments_exact=true;c.status='PASS';
c.bn_state_all_finite=true;c.bn_mean_state_count=p.bn_count;c.bn_variance_state_count=p.bn_count;
c.predict_preserved_state_and_rng=true;
end
function s=stateSummary(state,expectedBN)
assert(istable(state)&&all(ismember({'Layer','Parameter','Value'},state.Properties.VariableNames)));
parameterNames=string(state.Parameter);
s.entry_count=height(state);s.mean_count=nnz(parameterNames=="TrainedMean");
s.variance_count=nnz(parameterNames=="TrainedVariance");
assert(s.entry_count==2*expectedBN&&s.mean_count==expectedBN&&s.variance_count==expectedBN, ...
    'rail:mature:BNStateCount','Expected %d BN mean and %d variance state entries',expectedBN,expectedBN);
s.element_count=0;s.all_finite=true;s.means_all_zero=true;s.variances_all_one=true;
s.mean_max_absolute=0;s.variance_min=Inf;s.variance_max=-Inf;
for k=1:height(state)
    value=extractdata(state.Value{k});
    assert(isnumeric(value)&&isreal(value)&&~isempty(value)&&all(isfinite(value),'all'), ...
        'rail:mature:NonfiniteBNState','BN state must be nonempty, real and finite');
    s.element_count=s.element_count+numel(value);
    if parameterNames(k)=="TrainedMean"
        s.means_all_zero=s.means_all_zero&&all(value==0,'all');
        s.mean_max_absolute=max(s.mean_max_absolute,double(max(abs(value),[],'all')));
    else
        s.variances_all_one=s.variances_all_one&&all(value==1,'all');
        s.variance_min=min(s.variance_min,double(min(value,[],'all')));
        s.variance_max=max(s.variance_max,double(max(value,[],'all')));
    end
end
end
function e=tableError(a,b)
if isempty(a)&&isempty(b),e=0;return;end
assert(istable(a)&&istable(b)&&isequal(a(:,1:2),b(:,1:2)));e=0;
for k=1:height(a)
    av=extractdata(a.Value{k});bv=extractdata(b.Value{k});assert(isequal(size(av),size(bv)));
    assert(all(isfinite(av),'all')&&all(isfinite(bv),'all'),'Nonfinite values in exact state/parameter comparison');
    if ~isempty(av),e=max(e,max(abs(double(av)-double(bv)),[],'all'));end
end
end
function s=stats(v)
s.values=v;s.defined_count=nnz(~isnan(v));s.mean=mean(v,'omitnan');s.sample_std=std(v,0,'omitnan');
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
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
