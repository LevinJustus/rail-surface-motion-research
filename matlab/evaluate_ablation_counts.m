function report=evaluate_ablation_counts(predictionRunId,evaluationRunId)
% Separate C5 evaluator; hashes every referenced probability before mask access.
root=fileparts(fileparts(mfilename('fullpath')));run=fullfile(root,'runs',predictionRunId);out=fullfile(root,'runs',evaluationRunId);
assert(~isempty(regexp(evaluationRunId,'^[A-Za-z0-9_]+$','once'))&&~isfolder(out));mkdir(out);timer=tic;fid=-1;csv=-1;
try
    summary=jsondecode(fileread(fullfile(run,'summary.json')));assert(strcmp(summary.mode,'validation')&&strcmp(summary.status,'PASS_C5_PREDICTION_PENDING_AUDIT'));
    writeJson(fullfile(out,'environment.json'),struct('matlab_version',version,'threads',maxNumCompThreads,'computer',computer));
    predictionHash=file_sha256(fullfile(run,'predictions.jsonl'));assert(strcmp(predictionHash,summary.predictions_sha256));
    freeze=read_fusion_calibration(fullfile(root,'configs','fusion_calibration_v1.json'));
    assert(strcmp(file_sha256(fullfile(root,'configs','fusion_calibration_v1.json')),'66a21cb635398bf953a1b7e1b2256a7cf20a1f05ca08330ed4d087b47c5aa9a6'));
    protocol=jsondecode(fileread(fullfile(root,'configs','fusion_protocol_v1.json')));selection=jsondecode(fileread(fullfile(root,protocol.data.selection_file)));
    verified=containers.Map('KeyType','char','ValueType','char');fid=fopen(fullfile(run,'predictions.jsonl'),'r','n','UTF-8');nr=0;
    while true
        line=fgetl(fid);if ~ischar(line),break;end;r=jsondecode(line);nr=nr+1;
        for prefix={'probability','single_reference','clean_reference'}
            path=r.([prefix{1} '_path']);hash=r.([prefix{1} '_sha256']);assert(~contains(path,'..')&&~startsWith(path,'/')&&~contains(path,':'));
            if isKey(verified,path),assert(strcmp(verified(path),hash));else,assert(strcmp(file_sha256(fullfile(root,path)),hash));verified(path)=hash;end
        end
    end
    fclose(fid);fid=-1;assert(nr==180*9*79);
    writeJson(fullfile(out,'prediction_freeze_before_masks.json'),struct('predictions_sha256',predictionHash,'unique_paths_verified',verified.Count,'mask_files_opened',0));
    expected=jsondecode(fileread(fullfile(root,'runs','step12_full_v2_01','evaluation_only','payload_manifest.json')));expected=containers.Map({expected.path},{expected.sha256});
    fid=fopen(fullfile(run,'predictions.jsonl'),'r','n','UTF-8');csv=fopen(fullfile(out,'per_frame_counts.csv'),'w','n','UTF-8');assert(fid>=0&&csv>=0);
    fprintf(csv,'seed,sequence_id,source_id,group_id,profile,frame,scenario,method,parent_method,threshold_kind,threshold,TP,FP,FN,TN,FN_to_TP,FP_to_TN,TP_to_FN,TN_to_FP,probability_changed_pixels,max_probability_delta,mean_probability_delta,prediction_flip_pixels,clean_FP,clean_FN,delta_FP,delta_FN\n');
    masks=0;rows=0;seqFrames=0;lastKey='';access=struct([]);
    while true
        line=fgetl(fid);if ~ischar(line),break;end;r=jsondecode(line);key=sprintf('%s_%d',r.sequence_id,r.frame);
        if ~strcmp(key,lastKey)
            lastKey=key;seqFrames=seqFrames+1;pcache=containers.Map('KeyType','char','ValueType','any');ccache=containers.Map('KeyType','char','ValueType','any');classcache=containers.Map('KeyType','char','ValueType','any');
            sel=selection(strcmp(string({selection.sequence_id}),r.sequence_id));assert(isscalar(sel)&&strcmp(sel.assigned_split,'validation'));
            relative=sprintf('evaluation_only/sequences/%s/mask_%03d.png',r.sequence_id,r.frame);path=fullfile(root,'runs','step12_full_v2_01',relative);assert(strcmp(file_sha256(path),expected(relative)));
            t=imread(path);if islogical(t),target=t;else,assert(all(ismember(unique(t),[0 255])));target=t>=128;end
            masks=masks+1;access(masks).path=relative;access(masks).sha256=expected(relative); %#ok<AGROW>
        end
        p=getProbability(r,'probability');s=getProbability(r,'single_reference');clean=getProbability(r,'clean_reference');
        d=abs(p-clean);changed=nnz(d);maxDelta=max(d,[],'all');meanDelta=mean(d,'all');
        choice=freeze.choices([freeze.choices.seed]==r.seed & strcmp(string({freeze.choices.method}),r.parent_method));assert(isscalar(choice));
        for v=1:2
            threshold=.5;kind='primary';if v==2,threshold=choice.threshold;kind='training_calibrated';end
            ck=sprintf('%s_%s_%.17g',r.probability_sha256,r.single_reference_sha256,threshold);
            if isKey(ccache,ck),counts=ccache(ck);c=counts(1:4);tr=counts(5:8);
            else,[c,tr]=fusion_binary_counts(p,target,s,threshold);ccache(ck)=[c tr];end
            bk=sprintf('%s_%.17g',r.single_reference_sha256,threshold);lk=sprintf('%s_%.17g',r.clean_reference_sha256,threshold);
            if isKey(classcache,bk),b=classcache(bk);else,[b,~]=fusion_binary_counts(s,target,s,threshold);classcache(bk)=b;end
            if isKey(classcache,lk),cl=classcache(lk);else,[cl,~]=fusion_binary_counts(clean,target,clean,threshold);classcache(lk)=cl;end
            assert(c(3)==b(3)-tr(1)+tr(3)&&c(2)==b(2)-tr(2)+tr(4));flips=nnz((p>=threshold)~=(clean>=threshold));
            fprintf(csv,'%d,%s,%s,%s,%s,%d,%s,%s,%s,%s,%.17g,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g,%d,%d,%d,%d,%d\n',...
                r.seed,r.sequence_id,sel.source_id,sel.group_id,sel.profile,r.frame,r.scenario,r.method,r.parent_method,kind,threshold,...
                c(1),c(2),c(3),c(4),tr(1),tr(2),tr(3),tr(4),changed,maxDelta,meanDelta,flips,cl(2),cl(3),c(2)-cl(2),c(3)-cl(3));rows=rows+1;
        end
        if mod(rows,17064)==0,fprintf('C5_COUNTS seed=%d masks=%d rows=%d\n',summary.seed,masks,rows);end
    end
    fclose(fid);fid=-1;fclose(csv);csv=-1;assert(rows==255960&&masks==1620&&seqFrames==1620);
    assert(strcmp(file_sha256(fullfile(run,'predictions.jsonl')),predictionHash));writeJson(fullfile(out,'mask_access.json'),access);
    copyfile(fullfile(root,'matlab','evaluate_ablation_counts.m'),fullfile(out,'evaluate_ablation_counts.m'));copyfile(fullfile(root,'matlab','fusion_binary_counts.m'),fullfile(out,'fusion_binary_counts.m'));
    report=struct('status','PASS_MATLAB_C5_COUNTS','seed',summary.seed,'count_rows',rows,'mask_files',masks,'predictions_sha256',predictionHash,...
        'counts_sha256',file_sha256(fullfile(out,'per_frame_counts.csv')),'wall_seconds',toc(timer),'heldout_payloads_opened',0,'accurate_geometry_payloads_opened',0,'model_forward_calls',0,'optimizer_updates',0);
    writeJson(fullfile(out,'summary.json'),report);
catch err
    if fid>=0,fclose(fid);end;if csv>=0,fclose(csv);end
    writeJson(fullfile(out,'failure.json'),struct('message',err.message,'stack',err.stack));rethrow(err);
end
    function value=getProbability(r,prefix)
        hash=r.([prefix '_sha256']);
        if isKey(pcache,hash),value=pcache(hash);return;end
        f=fopen(fullfile(root,r.([prefix '_path'])),'rb','ieee-le');assert(f>=0);value=fread(f,[256 256],'*single');fclose(f);
        assert(isequal(size(value),[256 256])&&all(isfinite(value),'all')&&all(value>=0&value<=1,'all'));value=double(value);pcache(hash)=value;
    end
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
