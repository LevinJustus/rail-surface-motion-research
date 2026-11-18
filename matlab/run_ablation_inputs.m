function report=run_ablation_inputs(mode,seed,runId,registrationRunId,trainingRunId)
% C5 predictor: cached raw probabilities and arrived image prefixes only, no GT.
root=fileparts(fileparts(mfilename('fullpath')));assert(any(strcmp(mode,{'training','validation'})));
if nargin<5,trainingRunId='fusion_c5_training_v1_01';end
assert(~isempty(regexp(runId,'^[A-Za-z0-9_]+$','once')));
if strcmp(mode,'training'),assert(seed==20261114);baseId='fusion_training_v1_02';
elseif seed==20261114,baseId='fusion_validation_seed20261114_v1_02';
else,assert(any(seed==[20261115 20261116]));baseId=sprintf('fusion_validation_seed%d_v1_01',seed);end
out=fullfile(root,'runs',runId);assert(~isfolder(out),'No overwrite');mkdir(out);
diary(fullfile(out,'run.log'));cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
timer=tic;newMaps=0;logicalOutputs=0;newPayloads=0;newBytes=0;aliases=0;
mf=-1;bf=-1;
try
    configPath=fullfile(root,'configs','fusion_ablation_v1.json');configHash=file_sha256(configPath);
    assert(strcmp(configHash,'82e4badd60122876c9aca97d05b36cfe443a38caa55fd95f1ccb4a47b8bb3994'));
    plan=jsondecode(fileread(configPath));protocol=jsondecode(fileread(fullfile(root,plan.scientific_plan_source)));cfg=protocol.inference;
    assert(strcmp(file_sha256(fullfile(root,plan.scientific_plan_source)),plan.scientific_plan_sha256));
    assert(strcmp(file_sha256(fullfile(root,plan.calibration)),plan.calibration_sha256));
    constant=.9597680584784836;
    if strcmp(mode,'training'),ids=string(protocol.data.contract_and_calibration_sequence_ids(:));expectedSources=3;
    else
        trainingContract=jsondecode(fileread(fullfile(root,'runs',trainingRunId,'contracts.json')));
        assert(strcmp(trainingContract.status,'PASS_C5_TRAINING_CONTRACTS'));
        ids=string(protocol.data.validation_sequence_ids(:));expectedSources=15;
    end
    selected=jsondecode(fileread(fullfile(root,protocol.data.selection_file)));
    for id=ids',s=selected(strcmp(string({selected.sequence_id}),id));assert(isscalar(s)&&strcmp(s.assigned_split,mode));end
    baseFolder=fullfile(root,'runs',baseId);baseSummary=jsondecode(fileread(fullfile(baseFolder,'summary.json')));
    baseManifestPath=fullfile(baseFolder,'prediction_manifest.json');assert(strcmp(file_sha256(baseManifestPath),baseSummary.prediction_manifest_sha256));
    reference=jsondecode(fileread(baseManifestPath));assert(reference.seed==seed&&numel(reference.raw)==numel(ids)*9);
    expected=jsondecode(fileread(fullfile(root,protocol.data.expected_input_manifest)));expected=containers.Map({expected.files.path},{expected.files.sha256});
    inputRoot=fullfile(root,protocol.data.input_root_relative);
    mkdir(fullfile(out,'blobs'));mkdir(fullfile(out,'registration'));mkdir(fullfile(out,'source_snapshot'));
    names={'run_ablation_inputs.m','ablation_frame.m','ablation_observation_variants.m','ablation_pressure_edge.m',...
        'ablation_quality_factors.m','ablation_history_context.m','ablation_recurrent_step.m',...
        'fuse_motion_window.m','motion_gray.m','motion_history_gate.m','motion_confidence.m','warp_motion_history.m','file_sha256.m'};
    sources=struct([]);
    for j=1:numel(names)
        path=fullfile(root,'matlab',names{j});copyfile(path,fullfile(out,'source_snapshot',names{j}));
        sources(j).path=['matlab/' names{j}];sources(j).sha256=file_sha256(path); %#ok<AGROW>
    end
    for name={'fuse_motion_window.m','motion_gray.m','motion_history_gate.m','motion_confidence.m','warp_motion_history.m'}
        assert(strcmp(file_sha256(fullfile(root,'matlab',name{1})),file_sha256(fullfile(root,'runs','step13c3_implementation_20261118_01','source_snapshot','matlab',name{1}))));
    end
    copyfile(configPath,fullfile(out,'frozen_ablation_config.json'));
    writeJson(fullfile(out,'source_version.json'),struct('source_files',sources,'base_prediction_run',baseId,'base_manifest_sha256',file_sha256(baseManifestPath)));
    writeJson(fullfile(out,'environment.json'),struct('matlab_version',version,'threads',maxNumCompThreads,'computer',computer,'network_forward_calls',0));
    manifestFile=fullfile(out,'predictions.jsonl');blobFile=fullfile(out,'payloads.jsonl');
    mf=fopen(manifestFile,'w','n','UTF-8');bf=fopen(blobFile,'w','n','UTF-8');assert(mf>=0&&bf>=0);
    addresses=containers.Map('KeyType','char','ValueType','char');registrationIndex=struct([]);inputAccess=struct([]);ai=0;
    cacheIndex=[];
    if ~isempty(registrationRunId)
        assert(~isempty(regexp(registrationRunId,'^[A-Za-z0-9_]+$','once')));
        cacheSummary=jsondecode(fileread(fullfile(root,'runs',registrationRunId,'summary.json')));
        assert(strcmp(cacheSummary.config_sha256,configHash)&&strcmp(cacheSummary.mode,mode));
        cacheIndex=jsondecode(fileread(fullfile(root,'runs',registrationRunId,'registration_index.json')));
    end
    for si=1:numel(ids)
        seq=char(ids(si));folder=fullfile(inputRoot,seq);motionPath=fullfile(folder,'motion_input.json');
        assert(strcmp(file_sha256(motionPath),expected(['algorithm_inputs/' seq '/motion_input.json'])));
        motion=jsondecode(fileread(motionPath));assert(numel(motion.frames)==9);images=cell(9,1);P=cell(9,1);base=cell(9,1);basePaths=cell(9,1);baseHashes=cell(9,1);
        for k=1:9
            f=motion.frames(k);path=fullfile(folder,f.image);ih=file_sha256(path);
            assert(strcmp(ih,expected(['algorithm_inputs/' seq '/' f.image])));images{k}=imread(path);
            assert(strcmp(ih,file_sha256(path)));
            ai=ai+1;inputAccess(ai).path=[seq '/' f.image];inputAccess(ai).sha256=ih; %#ok<AGROW>
            r=reference.raw((si-1)*9+k);assert(strcmp(r.sequence_id,seq)&&r.frame==k&&strcmp(r.image_sha256,ih));
            rawPath=['runs/' baseId '/' r.path];P{k}=readProbability(rawPath,r.sha256);
            addresses(r.sha256)=rawPath;base{k}=struct;paths=struct;hashes=struct;
            for mi=1:8
                m=char(protocol.inference.methods(mi).id);entry=reference.fused(((si-1)*9+k-1)*8+mi);
                assert(strcmp(entry.method,m)&&strcmp(entry.sequence_id,seq)&&entry.frame==k);
                if isKey(addresses,entry.sha256)
                    target=addresses(entry.sha256);value=readProbability(target,entry.sha256);
                else
                    target=['runs/' baseId '/' entry.path];value=readProbability(target,entry.sha256);addresses(entry.sha256)=target;
                end
                base{k}.(m)=value;paths.(m)=target;hashes.(m)=entry.sha256;
            end
            basePaths{k}=paths;baseHashes{k}=hashes;
        end
        ai=ai+1;inputAccess(ai).path=[seq '/motion_input.json'];inputAccess(ai).sha256=file_sha256(motionPath); %#ok<AGROW>
        if isempty(registrationRunId)
            loaded=load(fullfile(baseFolder,'registration',[seq '.mat']),'edges');edges=loaded.edges;
            assert(isequaln(jsondecode(jsonencode(struct('sequence_id',seq,'edges',{edges}))),jsondecode(fileread(fullfile(baseFolder,'registration',[seq '.json'])))));
            variants=ablation_observation_variants(motion.frames);stressEdges=cell(7,1);for j=1:7,stressEdges{j}=cell(8,1);end
            maps=cell(8,1);
            for k=2:9
                a=motion_gray(images{k-1});b=motion_gray(images{k});patch=a(65:192,65:192);C=[];
                if std(patch(:))>=cfg.registration.minimum_template_std
                    full=normxcorr2(patch,b);C=full(192+(-20:20),192+(-20:20));newMaps=newMaps+1;
                end
                maps{k-1}=C;original=motion.frames;delta=NaN;
                if original(k-1).odometry_available&&original(k).odometry_available,delta=original(k).simulated_odometry_px-original(k-1).simulated_odometry_px;end
                check=ablation_pressure_edge(edges{k-1},C,delta,cfg);
                assert(isequaln([check.prior.dx check.prior.dy check.prior.score],[edges{k-1}.prior.dx edges{k-1}.prior.dy edges{k-1}.prior.score]));
                for j=1:7
                    f=variants{j};delta=NaN;
                    if f(k-1).odometry_available&&f(k).odometry_available,delta=f(k).simulated_odometry_px-f(k-1).simulated_odometry_px;end
                    stressEdges{j}{k-1}=ablation_pressure_edge(edges{k-1},C,delta,cfg);
                end
            end
            cache=struct('sequence_id',seq,'base_edges',{edges},'stress_edges',{stressEdges},'candidate_maps',{maps},'motion_sha256',file_sha256(motionPath));
            save(fullfile(out,'registration',[seq '.mat']),'cache','-v7');
            writeJson(fullfile(out,'registration',[seq '.json']),rmfield(cache,'candidate_maps'));
            regRoot=['runs/' runId '/registration/'];
        else
            ix=cacheIndex(strcmp(string({cacheIndex.sequence_id}),seq));assert(isscalar(ix));
            regRoot=['runs/' registrationRunId '/registration/'];
            assert(strcmp(file_sha256(fullfile(root,[regRoot seq '.mat'])),ix.mat_sha256)&&strcmp(file_sha256(fullfile(root,[regRoot seq '.json'])),ix.json_sha256));
            loaded=load(fullfile(root,[regRoot seq '.mat']),'cache');cache=loaded.cache;
            assert(strcmp(cache.motion_sha256,file_sha256(motionPath)));edges=cache.base_edges;stressEdges=cache.stress_edges;
        end
        rr=struct('sequence_id',seq,'mat_path',[regRoot seq '.mat'],'mat_sha256',file_sha256(fullfile(root,[regRoot seq '.mat'])),...
            'json_path',[regRoot seq '.json'],'json_sha256',file_sha256(fullfile(root,[regRoot seq '.json'])));
        if isempty(registrationIndex),registrationIndex=rr;else,registrationIndex(si)=rr;end %#ok<AGROW>
        state=[];
        for k=1:9
            prefix=cellfun(@(e)e(1:k-1),stressEdges,'UniformOutput',false);
            [results,state]=ablation_frame(images(1:k),P(1:k),edges(1:k-1),prefix,base{k},cfg,constant,state);
            badSingle=P{k};if k==3,badSingle=single(1-double(badSingle));end
            [badSPath,badSHash]=store(badSingle);cleanUnboundedPath='';cleanUnboundedHash='';
            for j=1:79
                value=results(j);assert(strcmp(value.scenario,plan.arms(j).scenario)&&strcmp(value.method,plan.arms(j).method));
                [probabilityPath,probabilityHash]=store(value.probability);
                parent=value.parent_method;cleanPath=basePaths{k}.(parent);cleanHash=baseHashes{k}.(parent);
                singlePath=basePaths{k}.S;singleHash=baseHashes{k}.S;
                if strcmp(value.scenario,'unbounded'),cleanUnboundedPath=probabilityPath;cleanUnboundedHash=probabilityHash;end
                if strcmp(value.scenario,'pollution')
                    singlePath=badSPath;singleHash=badSHash;
                    if strcmp(value.method,'R_unbounded'),cleanPath=cleanUnboundedPath;cleanHash=cleanUnboundedHash;end
                end
                row=struct('seed',seed,'sequence_id',seq,'frame',k,'scenario',value.scenario,'method',value.method,'parent_method',parent,...
                    'probability_path',probabilityPath,'probability_sha256',probabilityHash,...
                    'single_reference_path',singlePath,'single_reference_sha256',singleHash,...
                    'clean_reference_path',cleanPath,'clean_reference_sha256',cleanHash);
                fprintf(mf,'%s\n',jsonencode(row));logicalOutputs=logicalOutputs+1;
            end
        end
        if mod(si,12)==0
            fprintf('C5_PREDICTION %s seed=%d %d/%d logical=%d new_blobs=%d\n',mode,seed,si,numel(ids),logicalOutputs,newPayloads);
            writeJson(fullfile(out,'status.json'),struct('status','RUNNING','sequences',si,'logical_outputs',logicalOutputs,'new_payload_bytes',newBytes));
        end
    end
    fclose(mf);mf=-1;fclose(bf);bf=-1;
    assert(logicalOutputs==numel(ids)*9*79);
    for f=sources,assert(strcmp(file_sha256(fullfile(root,f.path)),f.sha256));end
    writeJson(fullfile(out,'registration_index.json'),registrationIndex);writeJson(fullfile(out,'input_access.json'),inputAccess);
    report=struct('status','PASS_C5_PREDICTION_PENDING_AUDIT','mode',mode,'seed',seed,'sequences',numel(ids),'source_count',expectedSources,...
        'logical_outputs',logicalOutputs,'new_payloads',newPayloads,'new_payload_bytes',newBytes,'aliases',aliases,...
        'new_forward_ncc_maps',newMaps,'network_forward_calls',0,'optimizer_updates',0,'mask_or_geometry_payloads_opened',0,'heldout_payloads_opened',0,...
        'base_prediction_run',baseId,'base_manifest_sha256',file_sha256(baseManifestPath),'config_sha256',configHash,...
        'predictions_sha256',file_sha256(manifestFile),'payloads_index_sha256',file_sha256(blobFile),...
        'registration_index_sha256',file_sha256(fullfile(out,'registration_index.json')),'wall_seconds',toc(timer),...
        'scope','Cached ablation predictor; aliases and shared computation preclude a detector FPS claim');
    writeJson(fullfile(out,'summary.json'),report);writeJson(fullfile(out,'status.json'),struct('status',report.status));
catch err
    if mf>=0,fclose(mf);end;if bf>=0,fclose(bf);end
    writeJson(fullfile(out,'status.json'),struct('status','FAILED','logical_outputs',logicalOutputs));
    writeJson(fullfile(out,'failure.json'),struct('message',err.message,'identifier',err.identifier,'stack',err.stack));rethrow(err);
end
    function probability=readProbability(relative,expectedHash)
        path=fullfile(root,relative);assert(strcmp(file_sha256(path),expectedHash),'Referenced probability SHA mismatch');
        fid=fopen(path,'rb','ieee-le');assert(fid>=0);c=onCleanup(@()fclose(fid));probability=fread(fid,[256 256],'*single');
        assert(isequal(size(probability),[256 256])&&all(isfinite(probability),'all')&&all(probability>=0&probability<=1,'all'));
    end
    function [relative,hash]=store(probability)
        probability=single(probability);md=java.security.MessageDigest.getInstance('SHA-256');md.update(typecast(probability(:),'int8'));
        hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
        if isKey(addresses,hash),relative=addresses(hash);aliases=aliases+1;return;end
        relative=['runs/' runId '/blobs/' hash '.f32'];path=fullfile(root,relative);assert(~isfile(path));
        fid=fopen(path,'wb','ieee-le');assert(fid>=0);assert(fwrite(fid,probability,'single')==65536);fclose(fid);
        assert(strcmp(file_sha256(path),hash));addresses(hash)=relative;newPayloads=newPayloads+1;newBytes=newBytes+262144;
        fprintf(bf,'%s\n',jsonencode(struct('path',relative,'sha256',hash,'bytes',262144)));
    end
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
