function report = benchmark_training_batches(dataRoot,outDir)
%BENCHMARK_TRAINING_BATCHES Compute-only batch selection, not detection FPS.
% Frozen eight training images, repeated for timing only; no validation use.
if ~exist(outDir,'dir'),mkdir(outDir);end
[X,T,V,~,context]=load_smoke_inputs(dataRoot);
cfg=context.config;cfg.optimizer.learning_rate=0.0003;
report.status='RUNNING';report.scope='Training-update throughput only; not inference or inspection FPS';
report.matlab_version=version;report.max_computational_threads=maxNumCompThreads;
report.unique_training_images=8;report.validation_images_opened=0;report.heldout_test_images_opened=0;
report.warmup_updates_per_batch=1;report.measured_updates_per_batch=3;
report.optimizer_updates_for_timing_only=12;
rows=struct([]);
for batchSize=[2 4 8]
    [net,~]=build_baseline();avg=[];avgSq=[];
    softmaxName=string(net.OutputNames{1});
    row=string(net.Connections.Destination)==softmaxName;
    assert(nnz(row)==1);logitsName=string(net.Connections.Source(row));
    selected=1:batchSize;times=zeros(1,4);
    for iteration=1:4
        timer=tic;
        [net,avg,avgSq,~]=segmentation_train_step(net,avg,avgSq,iteration, ...
            X(:,:,:,selected),T(:,:,:,selected),V(:,:,:,selected),cfg,logitsName);
        times(iteration)=toc(timer);
    end
    candidate.batch_size=batchSize;candidate.status='PASS';
    candidate.warmup_time_s=times(1);candidate.measured_update_times_s=times(2:4);
    candidate.median_update_time_s=median(times(2:4));
    candidate.median_images_per_s=batchSize/candidate.median_update_time_s;
    if isempty(rows),rows=candidate;else,rows(end+1)=candidate;end %#ok<AGROW>
    fprintf('TRAIN_BATCH_BENCHMARK batch=%d median_update_s=%.6f images_per_s=%.4f\n', ...
        batchSize,candidate.median_update_time_s,candidate.median_images_per_s);
end
report.candidates=rows;report.status='PASS';
[~,idx]=max([rows.median_images_per_s]);report.preferred_batch_size=rows(idx).batch_size;
report.selection_rule='Highest measured median training images/s among successful2/4/8; tie smaller batch';
report.note='Independent disposable networks; these12 timing updates are not included in any formal training trajectory.';
fid=fopen(fullfile(outDir,'batch_benchmark.json'),'w','n','UTF-8');assert(fid>=0);
c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
end
