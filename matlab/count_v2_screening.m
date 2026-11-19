function report=count_v2_screening(predictionRunId,evaluationRunId)
% Independent evaluation entry; prediction index freezes before mask access.
root=fileparts(fileparts(mfilename('fullpath')));run=fullfile(root,'runs',predictionRunId);out=fullfile(root,'runs',evaluationRunId);assert(~isfolder(out));mkdir(out);timer=tic;
summary=jsondecode(fileread(fullfile(run,'summary.json')));assert(strcmp(summary.status,'PASS_E3_PREDICTION_PENDING_EVALUATION'));
plan=jsondecode(fileread(fullfile(run,'frozen_plan.json')));manifest=fullfile(run,'predictions.json');hash=file_sha256(manifest);assert(strcmp(hash,summary.predictions_sha256));rows=jsondecode(fileread(manifest));
for r=rows(:)',assert(strcmp(file_sha256(fullfile(root,r.path)),r.sha256));end
payload=jsondecode(fileread(fullfile(root,'runs','step12_full_v2_01','evaluation_only','payload_manifest.json')));expected=containers.Map({payload.path},{payload.sha256});
fid=fopen(fullfile(out,'per_frame_counts.csv'),'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'sequence_id,frame,method,TP,FP,FN,TN,FN_to_TP,FP_to_TN,TP_to_FN,TN_to_FP\n');n=0;
for s=plan.selection(:)'
    assert(strcmp(s.assigned_split,'training'));seq=s.sequence_id;
    for k=1:9
        relative=sprintf('evaluation_only/sequences/%s/mask_%03d.png',seq,k);path=fullfile(root,'runs','step12_full_v2_01',relative);assert(strcmp(file_sha256(path),expected(relative)));
        t=imread(path)>0;base=rows(strcmp(string({rows.sequence_id}),seq)&[rows.frame]==k&strcmp(string({rows.method}),'S'));singleP=readP(fullfile(root,base.path));
        selected=rows(strcmp(string({rows.sequence_id}),seq)&[rows.frame]==k);
        for r=selected(:)'
            [c,tr]=fusion_binary_counts(readP(fullfile(root,r.path)),t,singleP,plan.threshold);fprintf(fid,'%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d\n',seq,k,r.method,c(1),c(2),c(3),c(4),tr(1),tr(2),tr(3),tr(4));n=n+1;
        end
    end
end
clear cleanup;assert(n==756&&strcmp(file_sha256(manifest),hash));
report=struct('status','PASS_E3_MATLAB_COUNTS','count_rows',n,'mask_files',108,'predictions_sha256',hash,'counts_sha256',file_sha256(fullfile(out,'per_frame_counts.csv')),'heldout_and_validation_payloads_opened',0,'wall_seconds',toc(timer));
f=fopen(fullfile(out,'summary.json'),'w','n','UTF-8');fprintf(f,'%s\n',jsonencode(report,PrettyPrint=true));fclose(f);
copyfile(fullfile(root,'matlab','count_v2_screening.m'),fullfile(out,'count_v2_screening.m'));
end
function p=readP(path)
f=fopen(path,'rb','ieee-le');assert(f>=0);p=double(fread(f,[256 256],'*single'));fclose(f);assert(isequal(size(p),[256 256]));
end
