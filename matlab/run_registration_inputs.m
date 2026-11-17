function report=run_registration_inputs(inputRoot,configPath,mode,runId)
% This entry reads algorithm_inputs only. Evaluation is a separate process.
root=fileparts(fileparts(mfilename('fullpath')));cfg=jsondecode(fileread(configPath));
assert(cfg.template_size==128 && cfg.template_start_one_based==65 && cfg.max_shift_px==20 && cfg.prior_radius_px==4);
[~,leaf]=fileparts(inputRoot);assert(strcmp(leaf,'algorithm_inputs'));
assert(any(strcmp(mode,{'contract','validation'})) && ~isempty(regexp(runId,'^[A-Za-z0-9_]+$','once')));
out=fullfile(root,'runs',runId);assert(~isfolder(out));mkdir(out);timer=tic;
try
    writeJson(fullfile(out,'status.json'),struct('status','RUNNING'));
    copyfile(configPath,fullfile(out,'frozen_registration_config.json'));
    for f={'run_registration_inputs.m','register_motion_pair.m','file_sha256.m'},copyfile(fullfile(root,'matlab',f{1}),fullfile(out,f{1}));end
    rng(47);texture=uint8(255*rand(320,320,3));previous=texture(33:288,33:288,:);
    [r,~]=register_motion_pair(previous,texture(41:296,30:285,:),8,cfg);
    assert(r(2).available && abs(r(2).dx-3)<.05 && abs(r(2).dy+8)<.05 && abs(r(3).dx-3)<.05 && abs(r(3).dy+8)<.05);
    [r,~]=register_motion_pair(previous,texture(25:280,33:288,:),-8,cfg);assert(abs(r(2).dy-8)<.05);
    [r,~]=register_motion_pair(previous,previous,0,cfg);assert(abs(r(2).dx)<.05 && abs(r(2).dy)<.05);
    [r,~]=register_motion_pair(previous,texture(41:296,33:288,:),NaN,cfg);assert(~r(1).available && isequaln([r(2).dx r(2).dy],[r(3).dx r(3).dy]));
    [r,~]=register_motion_pair(zeros(256,256,3,'uint8'),zeros(256,256,3,'uint8'),0,cfg);assert(r(1).available && ~r(2).available && ~r(3).available);
    writeJson(fullfile(out,'analytic_contracts.json'),struct('status','PASS','checks',{{'x/y order and forward shift sign','reverse direction','zero motion','missing prior visual fallback','textureless invalid without identity fabrication'}}));
    ids=string(cfg.([mode '_sequence_ids']));ids=ids(:);allRows=struct([]);access=struct([]);rowCount=0;accessCount=0;
    for i=1:numel(ids)
        assert(~isempty(regexp(ids(i),'^seq[0-9]{4}$','once')));seq=char(ids(i));folder=fullfile(inputRoot,seq);motionPath=fullfile(folder,'motion_input.json');
        input=jsondecode(fileread(motionPath));assert(isequal(sort(string(fieldnames(input))),sort(["schema_version";"frames"])) && numel(input.frames)==9);
        mh=file_sha256(motionPath);images=cell(9,1);
        for k=1:9
            f=input.frames(k);assert(isequal(sort(string(fieldnames(f))),sort(["image";"timestamp_s";"simulated_odometry_px";"odometry_available"])));
            assert(strcmp(f.image,sprintf('frame_%03d.png',k)) && isscalar(f.timestamp_s) && isfinite(f.timestamp_s));
            if k>1,assert(f.timestamp_s>input.frames(k-1).timestamp_s);end
            path=fullfile(folder,f.image);before=file_sha256(path);images{k}=imread(path);assert(strcmp(before,file_sha256(path)));
            ar=struct('path',[seq '/' f.image],'sha256',before);accessCount=accessCount+1;if isempty(access),access=ar;else,access(accessCount)=ar;end %#ok<AGROW>
        end
        ar=struct('path',[seq '/motion_input.json'],'sha256',mh);accessCount=accessCount+1;access(accessCount)=ar;
        for k=2:9
            f=input.frames(k);p=input.frames(k-1);delta=NaN;
            if f.odometry_available && p.odometry_available
                assert(isscalar(f.simulated_odometry_px)&&isscalar(p.simulated_odometry_px));delta=f.simulated_odometry_px-p.simulated_odometry_px;
            end
            [estimated,diag]=register_motion_pair(images{k-1},images{k},delta,cfg);
            for j=1:3
                e=estimated(j);rr=struct('sequence_id',seq,'previous_frame',k-1,'current_frame',k,'method',e.method,'available',e.available,'dx',e.dx,'dy',e.dy,'observed_delta_odometry',delta,'score',e.score,'used_prior',e.used_prior,'fallback',e.fallback,'boundary_peak',e.boundary_peak,'candidate_count',e.candidate_count,'shared_ncc_seconds',diag.shared_ncc_seconds,'selection_seconds',e.selection_seconds);
                rowCount=rowCount+1;if isempty(allRows),allRows=rr;else,allRows(rowCount)=rr;end %#ok<AGROW>
            end
        end
        assert(strcmp(mh,file_sha256(motionPath)));
        if mod(i,12)==0,fprintf('REGISTRATION_INPUTS %s sequences=%d/%d\n',mode,i,numel(ids));end
    end
    writetable(struct2table(allRows),fullfile(out,'predictions.csv'));
    writeJson(fullfile(out,'input_access.json'),access);
    report=struct('status','PASS','mode',mode,'sequences',numel(ids),'frame_pairs',numel(ids)*8,'method_rows',rowCount,'input_files_opened',accessCount,'geometry_or_masks_opened',0,'optimizer_updates',0,'wall_seconds',toc(timer),'predictions_sha256',file_sha256(fullfile(out,'predictions.csv')),'config_sha256',file_sha256(configPath),'timing_scope','NCC map shared across methods; selection costs logged separately, not standalone end-to-end latency');
    writeJson(fullfile(out,'summary.json'),report);writeJson(fullfile(out,'status.json'),struct('status','PASS'));
catch err
    writeJson(fullfile(out,'status.json'),struct('status','FAILED'));
    writeJson(fullfile(out,'failure.json'),struct('status','FAILED','message',err.message,'stack',err.stack));rethrow(err);
end
end
function writeJson(path,v)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(v,PrettyPrint=true));
end
