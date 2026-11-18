function report=run_fusion_frozen_inference(mode,seed,runId,calibrationFreezePath)
% C4: frozen algorithms, training calibration or validation inference only.
root=fileparts(fileparts(mfilename('fullpath')));
assert(~isempty(regexp(runId,'^[A-Za-z0-9_]+$','once')));
assert(any(strcmp(mode,{'training','validation'})) && any(seed==[20261114 20261115 20261116]));
out=fullfile(root,'runs',runId);assert(~isfolder(out),'No overwrite');mkdir(out);
diary(fullfile(out,'run.log'));cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
timer=tic;forwardCalls=0;mapCount=0;
try
    writeJson(fullfile(out,'status.json'),struct('status','RUNNING','phase','preflight'));
    configPath=fullfile(root,'configs','fusion_protocol_v1.json');
    assert(strcmp(file_sha256(configPath),'20fd3a8839131d294d64053d2b9cde0b4adf6ef837482f2d4604cb22d37cb6c1'));
    protocol=jsondecode(fileread(configPath));cfg=protocol.inference;
    assert(strcmp(file_sha256(fullfile(root,'matlab','register_motion_pair.m')),'5e3df5492cc478d087a4699ed11bc3454680f5e4e4ed2cae5a6a1471eb6b289c'));
    assert(strcmp(file_sha256(fullfile(root,'configs','registration_translation_v1.json')),cfg.registration.config_sha256));
    core={'motion_gray.m','register_fusion_pair.m','warp_motion_history.m','motion_confidence.m','motion_history_gate.m','fuse_motion_window.m','motion_fusion_step.m'};
    for f=core
        assert(strcmp(file_sha256(fullfile(root,'matlab',f{1})),file_sha256(fullfile(root,'runs','step13c3_implementation_20261118_01','source_snapshot','matlab',f{1}))),'C3 algorithm changed');
    end
    if strcmp(mode,'training')
        assert(any(seed==[20261115 20261116]));ids=string(protocol.data.contract_and_calibration_sequence_ids(:));sourceCount=3;
    else
        ids=string(protocol.data.validation_sequence_ids(:));sourceCount=15;
        freeze=read_fusion_calibration(calibrationFreezePath);
        assert(strcmp(freeze.status,'FROZEN_THREE_SEED_TRAINING_CALIBRATION')&&numel(freeze.choices)==24);
        assert(isequal(sort(unique([freeze.choices.seed])),[20261114 20261115 20261116]));
        assert(strcmp(freeze.protocol_sha256,file_sha256(configPath)));
        copyfile(calibrationFreezePath,fullfile(out,'frozen_calibration.json'));
    end
    nSeq=numel(ids);assert(nSeq==12*sourceCount);
    constantPath=fullfile(root,'runs','fusion_training_v1_02','constant_control.json');
    constantRecord=jsondecode(fileread(constantPath));constant=constantRecord.value;
    assert(abs(constant-0.9597680584784836)<1e-15 && constantRecord.eligible_paths==516);
    copyfile(constantPath,fullfile(out,'constant_control.json'));
    copyfile(configPath,fullfile(out,'frozen_protocol.json'));
    mkdir(fullfile(out,'source_snapshot'));sources=struct([]);
    names={'motion_gray.m','register_fusion_pair.m','warp_motion_history.m','motion_confidence.m','motion_history_gate.m',...
        'fuse_motion_window.m','motion_fusion_step.m','run_fusion_frozen_inference.m','check_motion_fusion_contracts.m',...
        'fusion_binary_counts.m','register_motion_pair.m','network_parameter_sha256.m','file_sha256.m','read_fusion_calibration.m'};
    for i=1:numel(names)
        path=fullfile(root,'matlab',names{i});copyfile(path,fullfile(out,'source_snapshot',names{i}));
        sources(i).path=['matlab/' names{i}];sources(i).sha256=file_sha256(path); %#ok<AGROW>
    end
    [s,commit]=system(sprintf('git -C "%s" rev-parse HEAD',root));assert(s==0);
    writeJson(fullfile(out,'source_version.json'),struct('git_commit',strtrim(commit),'dirty_tree_expected',true,'source_files',sources));
    writeJson(fullfile(out,'environment.json'),struct('matlab_version',version,'computer',computer,...
        'threads',maxNumCompThreads,'execution_environment','CPU','products',ver,'precision','single net, double fusion, single output'));
    selected=jsondecode(fileread(fullfile(root,protocol.data.selection_file)));
    expected=jsondecode(fileread(fullfile(root,protocol.data.expected_input_manifest)));expected=expected.files;
    for id=ids'
        sel=selected(strcmp(string({selected.sequence_id}),id));assert(isscalar(sel)&&strcmp(sel.assigned_split,mode));
    end
    inputRoot=fullfile(root,protocol.data.input_root_relative);preflight=struct([]);count=0;
    for id=ids'
        seq=char(id);
        for name=[compose('frame_%03d.png',1:9) "motion_input.json"]
            relative=['algorithm_inputs/' seq '/' char(name)];entry=expected(strcmp(string({expected.path}),relative));assert(isscalar(entry));
            path=fullfile(inputRoot,seq,char(name));assert(strcmp(file_sha256(path),entry.sha256),'Training payload SHA mismatch');
            count=count+1;rr=struct('path',relative,'sha256',entry.sha256);
            if isempty(preflight),preflight=rr;else,preflight(count)=rr;end %#ok<AGROW>
        end
    end
    writeJson(fullfile(out,'input_preflight.json'),struct('files',preflight,'file_count',count,...
        'assigned_split',mode,'heldout_payloads_opened',0));
    cp=protocol.segmentation.checkpoints([protocol.segmentation.checkpoints.seed]==seed);assert(isscalar(cp));
    checkpointPath=fullfile(root,cp.checkpoint_relative);assert(strcmp(file_sha256(checkpointPath),cp.checkpoint_sha256));
    loaded=load(checkpointPath,'checkpoint');q=loaded.checkpoint;net=q.net;clear loaded;
    assert(q.initialization_seed==seed&&isequal(string(q.class_channel_order(:)),["background";"defect"]));
    beforeParameter=network_parameter_sha256(net);beforeState=net.State;
    assert(strcmp(beforeParameter,cp.network_parameter_sha256));
    mkdir(fullfile(out,'raw'));mkdir(fullfile(out,'registration'));rawEntries=struct([]);ri=0;
    metadata=struct([]);edgesBySequence=cell(nSeq,1);
    for i=1:nSeq
        seq=char(ids(i));folder=fullfile(inputRoot,seq);motionPath=fullfile(folder,'motion_input.json');
        motion=jsondecode(fileread(motionPath));assert(numel(motion.frames)==9);
        mkdir(fullfile(out,'raw',seq));previous=[];edges=cell(8,1);
        for k=1:9
            f=motion.frames(k);assert(strcmp(f.image,sprintf('frame_%03d.png',k)));
            path=fullfile(folder,f.image);originalHash=file_sha256(path);image=imread(path);
            assert(isa(image,'uint8')&&isequal(size(image),[256 256 3]));
            prob=extractdata(predict(net,dlarray(single(image)/255,'SSCB')));forwardCalls=forwardCalls+1;
            assert(isequal(size(prob),[256 256 2])&&all(isfinite(prob),'all')&&max(abs(sum(prob,3)-1),[],'all')<=1e-4);
            P=single(prob(:,:,2));relative=sprintf('raw/%s/frame_%03d.f32',seq,k);writeBinary(fullfile(out,relative),P);
            ri=ri+1;rr=struct('sequence_id',seq,'frame',k,'path',relative,'sha256',file_sha256(fullfile(out,relative)),...
                'image_sha256',originalHash);
            if isempty(rawEntries),rawEntries=rr;else,rawEntries(ri)=rr;end %#ok<AGROW>
            if k>1
                pf=motion.frames(k-1);assert(f.timestamp_s>pf.timestamp_s);delta=NaN;
                if f.odometry_available&&pf.odometry_available,delta=f.simulated_odometry_px-pf.simulated_odometry_px;end
                edge=register_fusion_pair(previous,image,delta,cfg);edges{k-1}=edge;mapCount=mapCount+edge.ncc_maps;
                % C3 established core equivalence; C4 independently audits frozen predictions later.
            end
            assert(strcmp(originalHash,file_sha256(path)));previous=image;
        end
        edgesBySequence{i}=edges;
        save(fullfile(out,'registration',[seq '.mat']),'edges','-v7');
        writeJson(fullfile(out,'registration',[seq '.json']),struct('sequence_id',seq,'edges',{edges}));
        metadata(i).sequence_id=seq;metadata(i).motion=motion; %#ok<AGROW>
        if mod(i,12)==0,fprintf('FUSION_CACHE %s seed=%d %d/%d forward_calls=%d\n',mode,seed,i,nSeq,forwardCalls);end
        writeJson(fullfile(out,'status.json'),struct('status','RUNNING','phase','cache_and_registration','sequences',i,'forward_calls',forwardCalls));
    end
    assert(forwardCalls==nSeq*9);
    assert(strcmp(beforeParameter,network_parameter_sha256(net))&&isequaln(beforeState,net.State));
    assert(strcmp(file_sha256(checkpointPath),cp.checkpoint_sha256));
    clear net q beforeState;
    writeJson(fullfile(out,'raw_manifest.json'),rawEntries);
    fusedEntries=struct([]);fi=0;allPaths=struct([]);pi=0;
    mkdir(fullfile(out,'fused'));methods={'S','U','V','P','GV','GVP','GC','R'};
    for i=1:nSeq
        seq=char(ids(i));mkdir(fullfile(out,'fused',seq));state=[];motion=metadata(i).motion;edges=edgesBySequence{i};
        for k=1:9
            raw=rawEntries(strcmp(string({rawEntries.sequence_id}),seq)&[rawEntries.frame]==k);assert(isscalar(raw));
            imagePath=fullfile(inputRoot,seq,motion.frames(k).image);assert(strcmp(file_sha256(imagePath),raw.image_sha256));
            assert(strcmp(file_sha256(fullfile(out,raw.path)),raw.sha256));
            fid=fopen(fullfile(out,raw.path),'rb','ieee-le');P=fread(fid,[256 256],'*single');fclose(fid);
            f=motion.frames(k);odom=NaN;if f.odometry_available,odom=f.simulated_odometry_px;end
            frame=struct('image',imread(imagePath),'timestamp_s',f.timestamp_s,...
                'simulated_odometry_px',odom,'odometry_available',logical(f.odometry_available));
            cached=[];if k>1,cached=edges{k-1};end
            [state,result,diagnostic]=motion_fusion_step(state,frame,P,cfg,constant,cached);
            for m=methods
                relative=sprintf('fused/%s/%s_%03d.f32',seq,m{1},k);writeBinary(fullfile(out,relative),result.(m{1}));
                fi=fi+1;rr=struct('sequence_id',seq,'frame',k,'method',m{1},'path',relative,'sha256',file_sha256(fullfile(out,relative)));
                if isempty(fusedEntries),fusedEntries=rr;else,fusedEntries(fi)=rr;end %#ok<AGROW>
            end
            for p=diagnostic.paths
                pi=pi+1;rr=p;rr.sequence_id=seq;rr.current_frame=k;rr.history_frame=k-p.age;
                if isempty(allPaths),allPaths=rr;else,allPaths(pi)=rr;end %#ok<AGROW>
            end
        end
        if mod(i,12)==0,fprintf('FUSION_STREAM %s seed=%d %d/%d outputs=%d\n',mode,seed,i,nSeq,fi);end
    end
    assert(fi==nSeq*9*8&&pi==nSeq*15);
    for r=rawEntries,assert(strcmp(file_sha256(fullfile(out,r.path)),r.sha256),'Raw cache changed');end
    for r=preflight,assert(strcmp(file_sha256(fullfile(root,'runs','step12_full_v2_01',r.path)),r.sha256),'Input changed');end
    for r=sources,assert(strcmp(file_sha256(fullfile(root,r.path)),r.sha256),'Source changed during run');end
    writetable(struct2table(allPaths),fullfile(out,'path_diagnostics.csv'));
    manifest=struct('seed',seed,'mode',mode,'raw',rawEntries,'fused',fusedEntries,...
        'protocol_sha256',file_sha256(configPath),'constant_sha256',file_sha256(fullfile(out,'constant_control.json')),...
        'layout',struct('type','float32','endian','little','shape',[256 256],'order','column_major'));
    writeJson(fullfile(out,'prediction_manifest.json'),manifest);
    report=struct('status','PASS_FROZEN_INFERENCE_PENDING_EVALUATION','mode',mode,'seed',seed,...
        'sequences',nSeq,'sources',sourceCount,'unique_image_payloads',nSeq*9,...
        'rgb_imread_calls',nSeq*18,'input_files_sha_verified',count,'raw_outputs',forwardCalls,'fused_outputs',fi,...
        'model_forward_calls',forwardCalls,'optimizer_updates',0,'network_parameters_and_state_unchanged',true,...
        'checkpoint_bytes_unchanged',true,'algorithm_ncc_maps',mapCount,'constant',constant,...
        'diagnostic_paths',pi,'heldout_payloads_opened',0,'geometry_or_masks_opened',0,...
        'wall_seconds',toc(timer),'prediction_manifest_sha256',file_sha256(fullfile(out,'prediction_manifest.json')),...
        'scope','Frozen causal inference; shared cached pipeline wall time is not end-to-end detector FPS');
    writeJson(fullfile(out,'summary.json'),report);writeJson(fullfile(out,'status.json'),struct('status',report.status));
catch err
    writeJson(fullfile(out,'status.json'),struct('status','FAILED','model_forward_calls',forwardCalls));
    writeJson(fullfile(out,'failure.json'),struct('message',err.message,'identifier',err.identifier,'stack',err.stack));rethrow(err);
end
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
function writeBinary(path,value)
fid=fopen(path,'wb','ieee-le');assert(fid>=0);c=onCleanup(@()fclose(fid));assert(fwrite(fid,value,'single')==65536);
end
