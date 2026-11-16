function report=verify_mature_preflight_checkpoint(runId)
%VERIFY_MATURE_PREFLIGHT_CHECKPOINT Postflight metadata/BN checks, zero updates.
% No raw images are read. This is save/reload verification, not replay training.
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'runs',runId);
path=fullfile(out,'checkpoint_postcheck.json');assert(~isfile(path));
summary=jsondecode(fileread(fullfile(out,'summary.json')));assert(strcmp(summary.status,'PASS'));
checkpointPath=fullfile(out,'preflight_checkpoint.mat');loaded=load(checkpointPath,'checkpoint');c=loaded.checkpoint;
assert(c.completed_updates==5&&c.next_update==6);
parent=fullfile(root,'configs','mobilenet_deeplab_scratch_v1.json');
assert(strcmp(c.model_config_sha256,file_sha256(parent)));
assert(isequaln(c.model_config,jsondecode(fileread(parent))));
savedRng=rng;cleanup=onCleanup(@()rng(savedRng)); %#ok<NASGU>
wall=tic;cpu=cputime;
[fresh,~,arch]=build_mobilenet_deeplab(fullfile(out,'seed_fixture_4','model_config_seed.json'));
report.fresh_construction_wall_s=toc(wall);report.fresh_construction_cpu_s=cputime-cpu;
assert(strcmp(network_parameter_sha256(fresh),summary.initialization.fingerprints{4}));
parameters=string(fresh.State.Parameter);
report.bn_count=arch.bn_count;report.state_entry_count=height(fresh.State);
report.mean_entries=nnz(parameters=="TrainedMean");report.variance_entries=nnz(parameters=="TrainedVariance");
assert(report.mean_entries==arch.bn_count&&report.variance_entries==arch.bn_count);
assert(height(fresh.State)==2*arch.bn_count);
for k=1:height(fresh.State)
    value=extractdata(fresh.State.Value{k});assert(all(isfinite(value),'all'));
    if parameters(k)=="TrainedMean",assert(all(value==0,'all'));
    elseif parameters(k)=="TrainedVariance",assert(all(value==1,'all'));end
end
assert(isequal(string(c.net.State.Parameter),parameters));
rng(c.rng_state);state=c.net.State;before=rng;
X=dlarray(zeros(256,256,3,1,'single'),'SSCB');predict(c.net,X);
assert(isequaln(c.net.State,state)&&isequaln(rng,before));
report.status='PASS';report.raw_images_opened=0;report.optimizer_updates=0;
report.checkpoint_sha256=file_sha256(checkpointPath);report.model_config_sha256=file_sha256(parent);
report.update_counter=5;report.next_update=6;report.postreload_predict_preserves_rng_and_state=true;
report.initial_state_labels_and_values_verified=true;report.initial_weights_match_original_preflight=true;
report.scope='Additional postflight metadata and BN-state validation; not a resumed optimizer-step experiment';
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);closeFile=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));disp(jsonencode(report,PrettyPrint=true));
end
