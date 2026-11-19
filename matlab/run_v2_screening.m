function report=run_v2_screening(runId)
% E3 input-only predictor. No mask or accurate geometry file is opened.
root=fileparts(fileparts(mfilename('fullpath')));planPath=fullfile(root,'configs','v2_screening_v1.json');plan=jsondecode(fileread(planPath));
assert(plan.seed==20261114&&numel(plan.selection)==12&&~isempty(regexp(runId,'^[A-Za-z0-9_]+$','once')));
out=fullfile(root,'runs',runId);assert(~isfolder(out));mkdir(out);mkdir(fullfile(out,'predictions'));mkdir(fullfile(out,'registration'));mkdir(fullfile(out,'source_snapshot'));timer=tic;
diary(fullfile(out,'run.log'));cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
try
    assert(strcmp(file_sha256(fullfile(root,plan.design_config)),plan.design_sha256));cfg=jsondecode(fileread(fullfile(root,plan.design_config)));
    assert(strcmp(file_sha256(fullfile(root,plan.protocol_config)),plan.protocol_sha256));protocol=jsondecode(fileread(fullfile(root,plan.protocol_config)));
    base=fullfile(root,'runs',plan.base_run);assert(strcmp(file_sha256(fullfile(base,'prediction_manifest.json')),plan.base_manifest_sha256));reference=jsondecode(fileread(fullfile(base,'prediction_manifest.json')));
    assert(strcmp(file_sha256(fullfile(root,plan.checkpoint.checkpoint_relative)),plan.checkpoint.checkpoint_sha256));
    expected=jsondecode(fileread(fullfile(root,protocol.data.expected_input_manifest)));expected=containers.Map({expected.files.path},{expected.files.sha256});
    sourceNames={'v2_select_motion.m','v2_motion_weights.m','v2_compose_motion.m','v2_history_evidence.m','v2_fuse_window.m','v2_compensation_step.m','warp_motion_history.m','motion_gray.m','run_v2_screening.m'};
    sources=struct([]);for j=1:numel(sourceNames),p=fullfile(root,'matlab',sourceNames{j});copyfile(p,fullfile(out,'source_snapshot',sourceNames{j}));sources(j).path=['matlab/' sourceNames{j}];sources(j).sha256=file_sha256(p);end
    methods=cellstr(string(plan.methods));rows=struct([]);access=struct([]);ai=0;ri=0;
    for si=1:numel(plan.selection)
        seq=plan.selection(si).sequence_id;assert(strcmp(plan.selection(si).assigned_split,'training'));folder=fullfile(root,protocol.data.input_root_relative,seq);
        motionPath=fullfile(folder,'motion_input.json');assert(strcmp(file_sha256(motionPath),expected(['algorithm_inputs/' seq '/motion_input.json'])));motion=jsondecode(fileread(motionPath));
        state=[];visualState=[];P=cell(9,1);edgeMain=cell(8,1);edgeVisual=cell(8,1);mkdir(fullfile(out,'predictions',seq));
        for k=1:9
            imagePath=fullfile(folder,motion.frames(k).image);ih=file_sha256(imagePath);assert(strcmp(ih,expected(['algorithm_inputs/' seq '/' motion.frames(k).image])));
            r=reference.raw(strcmp(string({reference.raw.sequence_id}),seq)&[reference.raw.frame]==k);assert(isscalar(r)&&strcmp(r.image_sha256,ih));P{k}=readP(fullfile(base,r.path),r.sha256);
            rgb=imread(imagePath);f=motion.frames(k);odom=NaN;if f.odometry_available&&~isempty(f.simulated_odometry_px),odom=f.simulated_odometry_px;end
            obs=struct('timestamp_s',f.timestamp_s,'odometry_px',odom,'odometry_available',f.odometry_available);
            [main,state]=v2_compensation_step(rgb,P{k},obs,state,cfg);obs.odometry_available=false;[visual,visualState]=v2_compensation_step(rgb,P{k},obs,visualState,cfg);
            if k>1,edgeMain{k-1}=state.edges{end};edgeVisual{k-1}=visualState.edges{end};end
            current=double(P{k});integer=current;start=max(1,k-2);n=k-start+1;
            for h=start:k-1
                path=edgeMain(h:k-1);if ~all(cellfun(@(e)e.available,path)),continue;end
                shifts=zeros(numel(path),2);for j=1:numel(path),shifts(j,:)=path{j}.candidates(1,1:2);end
                [warped,valid]=warp_motion_history(P{h},shifts);d=warped-current;d(~valid)=0;integer=integer+(1/n)*d;
            end
            values=struct('MAP_integer',single(integer),'DECAY05_integer',single(current+plan.simple_decay*(integer-current)),'V2',main,'V2_visual',visual);
            for m=methods'
                method=m{1};
                if strcmp(method,'S'),relative=['runs/' plan.base_run '/' r.path];hash=r.sha256;
                elseif any(strcmp(method,{'V_subpixel','GV_subpixel'}))
                    old='V';if strcmp(method,'GV_subpixel'),old='GV';end
                    entry=reference.fused(strcmp(string({reference.fused.sequence_id}),seq)&[reference.fused.frame]==k&strcmp(string({reference.fused.method}),old));assert(isscalar(entry));
                    relative=['runs/' plan.base_run '/' entry.path];hash=entry.sha256;assert(strcmp(file_sha256(fullfile(root,relative)),hash));
                else
                    relative=sprintf('runs/%s/predictions/%s/frame_%03d_%s.f32',runId,seq,k,method);p=fullfile(root,relative);fid=fopen(p,'wb','ieee-le');assert(fid>=0);assert(fwrite(fid,values.(method),'single')==65536);fclose(fid);hash=file_sha256(p);
                end
                ri=ri+1;rows(ri).sequence_id=seq;rows(ri).frame=k;rows(ri).method=method;rows(ri).path=relative;rows(ri).sha256=hash; %#ok<AGROW>
            end
            ai=ai+1;access(ai).path=['algorithm_inputs/' seq '/' motion.frames(k).image];access(ai).sha256=ih; %#ok<AGROW>
        end
        save(fullfile(out,'registration',[seq '.mat']),'edgeMain','edgeVisual','-v7');writeJson(fullfile(out,'registration',[seq '.json']),struct('sequence_id',seq,'main',{edgeMain},'visual',{edgeVisual}));
        fprintf('E3_PREDICTION %d/12 %s\n',si,seq);
    end
    for f=sources,assert(strcmp(file_sha256(fullfile(root,f.path)),f.sha256));end
    writeJson(fullfile(out,'predictions.json'),rows);writeJson(fullfile(out,'input_access.json'),access);copyfile(planPath,fullfile(out,'frozen_plan.json'));
    writeJson(fullfile(out,'source_version.json'),sources);
    report=struct('status','PASS_E3_PREDICTION_PENDING_EVALUATION','sequences',12,'frames',108,'methods',7,'method_frames',ri,'seed',plan.seed,...
        'predictions_sha256',file_sha256(fullfile(out,'predictions.json')),'plan_sha256',file_sha256(planPath),'network_forward_calls',0,'optimizer_updates',0,...
        'mask_and_geometry_payloads_opened',0,'validation_and_heldout_payloads_opened',0,'matlab_version',version,'threads',maxNumCompThreads,'wall_seconds',toc(timer));
    assert(ri==756);writeJson(fullfile(out,'summary.json'),report);
catch err
    writeJson(fullfile(out,'failure.json'),struct('message',err.message,'stack',err.stack));rethrow(err);
end
end
function p=readP(path,hash)
assert(strcmp(file_sha256(path),hash));f=fopen(path,'rb','ieee-le');assert(f>=0);p=fread(f,[256 256],'*single');fclose(f);assert(isequal(size(p),[256 256]));
end
function writeJson(path,value)
f=fopen(path,'w','n','UTF-8');assert(f>=0);c=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(value,PrettyPrint=true));
end
