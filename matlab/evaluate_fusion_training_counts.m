function report=evaluate_fusion_training_counts(predictionRunId,evaluationRunId)
% Separate evaluator, training masks only, after immutable prediction manifest.
root=fileparts(fileparts(mfilename('fullpath')));run=fullfile(root,'runs',predictionRunId);
assert(~isempty(regexp(evaluationRunId,'^[A-Za-z0-9_]+$','once')));
out=fullfile(root,'runs',evaluationRunId);assert(~isfolder(out));mkdir(out);timer=tic;
try
    config=fullfile(root,'configs','fusion_protocol_v1.json');
    assert(strcmp(file_sha256(config),'20fd3a8839131d294d64053d2b9cde0b4adf6ef837482f2d4604cb22d37cb6c1'));
    protocol=jsondecode(fileread(config));ids=string(protocol.data.contract_and_calibration_sequence_ids(:));
    selected=jsondecode(fileread(fullfile(root,protocol.data.selection_file)));
    summary=jsondecode(fileread(fullfile(run,'summary.json')));
    assert(strcmp(summary.status,'PASS_TRAINING_INPUT_PIPELINE_PENDING_INDEPENDENT_AUDIT')&&summary.model_forward_calls==324);
    manifestPath=fullfile(run,'prediction_manifest.json');predictionHash=file_sha256(manifestPath);
    assert(strcmp(predictionHash,summary.prediction_manifest_sha256));manifest=jsondecode(fileread(manifestPath));
    assert(numel(manifest.raw)==324&&numel(manifest.fused)==2592);
    % Check every saved probability before opening any label.
    for r=manifest.raw(:)',assert(ismember(string(r.sequence_id),ids)&&strcmp(file_sha256(fullfile(run,r.path)),r.sha256));end
    for r=manifest.fused(:)',assert(ismember(string(r.sequence_id),ids)&&strcmp(file_sha256(fullfile(run,r.path)),r.sha256));end
    writeJson(fullfile(out,'prediction_freeze_before_masks.json'),struct('prediction_manifest_sha256',predictionHash,'probability_files_verified',2916));
    payload=jsondecode(fileread(fullfile(root,'runs','step12_full_v2_01','evaluation_only','payload_manifest.json')));
    rows=struct([]);access=struct([]);ri=0;ai=0;methods={'S','U','V','P','GV','GVP','GC','R'};
    for id=ids'
        seq=char(id);sel=selected(strcmp(string({selected.sequence_id}),seq));assert(isscalar(sel)&&strcmp(sel.assigned_split,'training'));
        for k=1:9
            relative=sprintf('evaluation_only/sequences/%s/mask_%03d.png',seq,k);
            p=fullfile(root,'runs','step12_full_v2_01',relative);expected=payload(strcmp(string({payload.path}),relative));assert(isscalar(expected));
            assert(strcmp(file_sha256(p),expected.sha256));t=imread(p);
            if islogical(t),target=t;else,assert(all(ismember(unique(t),[0 255])));target=t>=128;end
            assert(isequal(size(target),[256 256]));ai=ai+1;rr=struct('path',relative,'sha256',expected.sha256);
            if isempty(access),access=rr;else,access(ai)=rr;end %#ok<AGROW>
            raw=manifest.raw(strcmp(string({manifest.raw.sequence_id}),seq)&[manifest.raw.frame]==k);assert(isscalar(raw));
            singleProbability=readProbability(fullfile(run,raw.path));
            for m=methods
                f=manifest.fused(strcmp(string({manifest.fused.sequence_id}),seq)&[manifest.fused.frame]==k&strcmp(string({manifest.fused.method}),m{1}));assert(isscalar(f));
                probability=readProbability(fullfile(run,f.path));[c,tr]=fusion_binary_counts(probability,target,singleProbability,.5);
                [b,~]=fusion_binary_counts(singleProbability,target,singleProbability,.5);
                assert(c(3)==b(3)-tr(1)+tr(3)&&c(2)==b(2)-tr(2)+tr(4));
                rr=struct('sequence_id',seq,'frame',k,'method',m{1},'TP',c(1),'FP',c(2),'FN',c(3),'TN',c(4),...
                    'FN_to_TP',tr(1),'FP_to_TN',tr(2),'TP_to_FN',tr(3),'TN_to_FP',tr(4));
                ri=ri+1;if isempty(rows),rows=rr;else,rows(ri)=rr;end %#ok<AGROW>
            end
        end
    end
    assert(ri==2592&&ai==324&&strcmp(file_sha256(manifestPath),predictionHash));
    writetable(struct2table(rows),fullfile(out,'per_frame_counts.csv'));
    writeJson(fullfile(out,'mask_access.json'),access);
    copyfile(fullfile(root,'matlab','evaluate_fusion_training_counts.m'),fullfile(out,'evaluate_fusion_training_counts.m'));
    copyfile(fullfile(root,'matlab','fusion_binary_counts.m'),fullfile(out,'fusion_binary_counts.m'));
    report=struct('status','PASS_MATLAB_TRAINING_COUNTS','training_mask_files',324,'count_rows',2592,...
        'prediction_manifest_sha256',predictionHash,'counts_sha256',file_sha256(fullfile(out,'per_frame_counts.csv')),...
        'validation_or_heldout_payloads_opened',0,'accurate_geometry_payloads_opened',0,'model_forward_calls',0,'optimizer_updates',0,...
        'wall_seconds',toc(timer),'scope','Training evaluator only; not validation improvement');
    writeJson(fullfile(out,'summary.json'),report);fprintf('MATLAB_TRAINING_COUNTS PASS %d rows\n',ri);
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
