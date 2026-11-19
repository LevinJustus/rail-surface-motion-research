function report=run_v2_conflict_ablation(runId)
% Only frozen probabilities/edges are read. No RGB, masks, or geometry.
root=fileparts(fileparts(mfilename('fullpath')));planPath=fullfile(root,'configs','v2_conflict_ablation_v1.json');plan=jsondecode(fileread(planPath));
out=fullfile(root,'runs',runId);assert(~isfolder(out));mkdir(out);mkdir(fullfile(out,'predictions'));timer=tic;
diary(fullfile(out,'run.log'));cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
try
    for item=plan.frozen_files(:)',assert(strcmp(file_sha256(fullfile(root,item.path)),item.sha256));end
    cfg=jsondecode(fileread(fullfile(root,plan.design_config)));base=fullfile(root,'runs',plan.base_prediction_run);
    rows=jsondecode(fileread(fullfile(base,'predictions.json')));assert(numel(rows)==756);
    for item=rows(:)',assert(strcmp(file_sha256(fullfile(root,item.path)),item.sha256));end
    diagnostic=struct([]);di=0;
    for si=1:numel(plan.selection)
        seq=plan.selection(si).sequence_id;assert(strcmp(plan.selection(si).assigned_split,'training'));mkdir(fullfile(out,'predictions',seq));
        reg=load(fullfile(base,'registration',[seq '.mat']),'edgeMain');P=cell(9,1);
        for k=1:9
            r=rows(strcmp(string({rows.sequence_id}),seq)&[rows.frame]==k&strcmp(string({rows.method}),'S'));assert(isscalar(r));P{k}=readP(fullfile(root,r.path));
            first=max(1,k-2);[value,d]=v2_no_conflict_fuse_window(P(first:k),reg.edgeMain(first:k-1),cfg);
            relative=sprintf('runs/%s/predictions/%s/frame_%03d_V2_no_conflict.f32',runId,seq,k);f=fopen(fullfile(root,relative),'wb','ieee-le');assert(f>=0);assert(fwrite(f,value,'single')==65536);fclose(f);
            rows(end+1)=struct('sequence_id',seq,'frame',k,'method','V2_no_conflict','path',relative,'sha256',file_sha256(fullfile(root,relative))); %#ok<AGROW>
            di=di+1;diagnostic(di).sequence_id=seq;diagnostic(di).frame=k;diagnostic(di).protected_pixels=d.protected_pixels;diagnostic(di).maximum_absolute_update=max(abs(d.delta),[],'all'); %#ok<AGROW>
        end
        fprintf('E4_PREDICTION %d/12 %s\n',si,seq);
    end
    for item=plan.frozen_files(:)',assert(strcmp(file_sha256(fullfile(root,item.path)),item.sha256));end
    writeJson(fullfile(out,'predictions.json'),rows);writeJson(fullfile(out,'diagnostics.json'),diagnostic);copyfile(planPath,fullfile(out,'frozen_plan.json'));
    report=struct('status','PASS_E4_PREDICTION_PENDING_EVALUATION','method_frames',numel(rows),'new_probability_outputs',di,'predictions_sha256',file_sha256(fullfile(out,'predictions.json')),...
        'plan_sha256',file_sha256(planPath),'network_forward_calls',0,'optimizer_updates',0,'mask_and_geometry_payloads_opened',0,'validation_and_heldout_payloads_opened',0,'matlab_version',version,'wall_seconds',toc(timer));
    assert(di==108&&numel(rows)==864);writeJson(fullfile(out,'summary.json'),report);
catch err
    writeJson(fullfile(out,'failure.json'),struct('message',err.message,'stack',err.stack));rethrow(err);
end
end
function p=readP(path)
f=fopen(path,'rb','ieee-le');assert(f>=0);p=fread(f,[256 256],'*single');fclose(f);assert(isequal(size(p),[256 256]));
end
function writeJson(path,value)
f=fopen(path,'w','n','UTF-8');assert(f>=0);c=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(value,PrettyPrint=true));
end
