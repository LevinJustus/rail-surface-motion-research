function report=evaluate_fusion_frozen_counts(predictionRunId,evaluationRunId,calibrationFreezePath)
% Separate C4 evaluator. Opens only the declared split after prediction freeze.
root=fileparts(fileparts(mfilename('fullpath')));run=fullfile(root,'runs',predictionRunId);
assert(~isempty(regexp(evaluationRunId,'^[A-Za-z0-9_]+$','once')));
out=fullfile(root,'runs',evaluationRunId);assert(~isfolder(out));mkdir(out);timer=tic;
try
    config=fullfile(root,'configs','fusion_protocol_v1.json');
    assert(strcmp(file_sha256(config),'20fd3a8839131d294d64053d2b9cde0b4adf6ef837482f2d4604cb22d37cb6c1'));
    protocol=jsondecode(fileread(config));summary=jsondecode(fileread(fullfile(run,'summary.json')));
    assert(strcmp(summary.status,'PASS_FROZEN_INFERENCE_PENDING_EVALUATION'));
    mode=summary.mode;seed=summary.seed;
    if strcmp(mode,'training'),ids=string(protocol.data.contract_and_calibration_sequence_ids(:));variants=1;freeze=[];
    else
        assert(strcmp(mode,'validation'));ids=string(protocol.data.validation_sequence_ids(:));variants=2;
        freeze=read_fusion_calibration(calibrationFreezePath);assert(strcmp(freeze.status,'FROZEN_THREE_SEED_TRAINING_CALIBRATION'));
        assert(strcmp(file_sha256(calibrationFreezePath),file_sha256(fullfile(run,'frozen_calibration.json'))));
        copyfile(calibrationFreezePath,fullfile(out,'frozen_calibration.json'));
    end
    selection=jsondecode(fileread(fullfile(root,protocol.data.selection_file)));
    manifestPath=fullfile(run,'prediction_manifest.json');predictionHash=file_sha256(manifestPath);
    assert(strcmp(predictionHash,summary.prediction_manifest_sha256));manifest=jsondecode(fileread(manifestPath));
    nSeq=numel(ids);assert(numel(manifest.raw)==nSeq*9 && numel(manifest.fused)==nSeq*9*8);
    for r=manifest.raw(:)',assert(ismember(string(r.sequence_id),ids)&&strcmp(file_sha256(fullfile(run,r.path)),r.sha256));end
    for r=manifest.fused(:)',assert(ismember(string(r.sequence_id),ids)&&strcmp(file_sha256(fullfile(run,r.path)),r.sha256));end
    writeJson(fullfile(out,'prediction_freeze_before_masks.json'),struct('prediction_manifest_sha256',predictionHash,...
        'probability_files_verified',nSeq*81,'declared_split',mode,'seed',seed));
    payload=jsondecode(fileread(fullfile(root,'runs','step12_full_v2_01','evaluation_only','payload_manifest.json')));
    expected=containers.Map({payload.path},{payload.sha256});
    template=struct('sequence_id','','frame',0,'method','','threshold_kind','','threshold',0,...
        'TP',0,'FP',0,'FN',0,'TN',0,'FN_to_TP',0,'FP_to_TN',0,'TP_to_FN',0,'TN_to_FP',0);
    rows=repmat(template,nSeq*9*8*variants,1);access=repmat(struct('path','','sha256',''),nSeq*9,1);
    ri=0;ai=0;methods={'S','U','V','P','GV','GVP','GC','R'};
    for i=1:nSeq
        seq=char(ids(i));sel=selection(strcmp(string({selection.sequence_id}),seq));assert(isscalar(sel)&&strcmp(sel.assigned_split,mode));
        for k=1:9
            relative=sprintf('evaluation_only/sequences/%s/mask_%03d.png',seq,k);
            path=fullfile(root,'runs','step12_full_v2_01',relative);assert(strcmp(file_sha256(path),expected(relative)));
            t=imread(path);if islogical(t),target=t;else,assert(all(ismember(unique(t),[0 255])));target=t>=128;end
            assert(isequal(size(target),[256 256]));ai=ai+1;access(ai)=struct('path',relative,'sha256',expected(relative));
            raw=manifest.raw((i-1)*9+k);assert(strcmp(raw.sequence_id,seq)&&raw.frame==k);
            singleProbability=readProbability(fullfile(run,raw.path));
            for mi=1:8
                m=methods{mi};f=manifest.fused(((i-1)*9+k-1)*8+mi);
                assert(strcmp(f.sequence_id,seq)&&f.frame==k&&strcmp(f.method,m));probability=readProbability(fullfile(run,f.path));
                for v=1:variants
                    threshold=.5;kind='primary';
                    if v==2
                        choice=freeze.choices([freeze.choices.seed]==seed & strcmp(string({freeze.choices.method}),m));assert(isscalar(choice));
                        threshold=choice.threshold;kind='training_calibrated';
                    end
                    [c,tr]=fusion_binary_counts(probability,target,singleProbability,threshold);
                    [base,~]=fusion_binary_counts(singleProbability,target,singleProbability,threshold);
                    assert(c(3)==base(3)-tr(1)+tr(3)&&c(2)==base(2)-tr(2)+tr(4));
                    ri=ri+1;rows(ri)=struct('sequence_id',seq,'frame',k,'method',m,'threshold_kind',kind,'threshold',threshold,...
                        'TP',c(1),'FP',c(2),'FN',c(3),'TN',c(4),'FN_to_TP',tr(1),'FP_to_TN',tr(2),'TP_to_FN',tr(3),'TN_to_FP',tr(4));
                end
            end
        end
        if mod(i,12)==0,fprintf('FUSION_COUNTS %s seed=%d %d/%d rows=%d\n',mode,seed,i,nSeq,ri);end
    end
    assert(ri==numel(rows)&&ai==nSeq*9&&strcmp(file_sha256(manifestPath),predictionHash));
    writetable(struct2table(rows),fullfile(out,'per_frame_counts.csv'));writeJson(fullfile(out,'mask_access.json'),access);
    copyfile(fullfile(root,'matlab','evaluate_fusion_frozen_counts.m'),fullfile(out,'evaluate_fusion_frozen_counts.m'));
    copyfile(fullfile(root,'matlab','fusion_binary_counts.m'),fullfile(out,'fusion_binary_counts.m'));
    report=struct('status','PASS_MATLAB_FROZEN_COUNTS','mode',mode,'seed',seed,'mask_files',ai,'count_rows',ri,...
        'prediction_manifest_sha256',predictionHash,'counts_sha256',file_sha256(fullfile(out,'per_frame_counts.csv')),...
        'heldout_payloads_opened',0,'accurate_geometry_payloads_opened',0,'model_forward_calls',0,'optimizer_updates',0,...
        'wall_seconds',toc(timer),'scope','Evaluation at frozen thresholds; calibrated transitions are not pure-fusion attribution');
    writeJson(fullfile(out,'summary.json'),report);
catch err
    writeJson(fullfile(out,'failure.json'),struct('message',err.message,'identifier',err.identifier,'stack',err.stack));rethrow(err);
end
end
function p=readProbability(path)
fid=fopen(path,'rb','ieee-le');assert(fid>=0);c=onCleanup(@()fclose(fid));p=fread(fid,[256 256],'*single');
assert(isequal(size(p),[256 256])&&all(isfinite(p),'all')&&all(p>=0&p<=1,'all'));
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
