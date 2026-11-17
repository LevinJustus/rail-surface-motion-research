function summary=run_validation_diagnostics(dataRoot,runId)
% Frozen validation only. Reuse existing checkpoints; zero optimizer updates.
root=fileparts(fileparts(mfilename('fullpath')));
assert(~isempty(regexp(runId,'^[A-Za-z0-9_]+$','once')));
out=fullfile(root,'runs',runId);assert(~isfolder(out),'Refuse to overwrite a run');mkdir(out);
started=tic;writeJson(fullfile(out,'status.json'),struct('status','RUNNING','completed_checkpoints',0));
try
    splitPath=fullfile(root,'data','splits','conservative_group_v1.json');
    assert(strcmp(file_sha256(splitPath),'c65f8c439ac3f245541a6678f3fe937bd3dcf7136584f088fd90eae1609270a6'));
    frozen=jsondecode(fileread(splitPath));records=frozen.records(strcmp({frozen.records.assigned_split},'validation'));
    [~,order]=sort(string({records.id}));records=records(order);records=records(:);
    assert(numel(records)==400 && numel(unique(string({records.group_id})))==15);
    reference=jsondecode(fileread(fullfile(root,'runs','full_stage2_v2_01','input_access.json')));
    access=reference.validation_access;assert(isequal(string({access.id}),string({records.id})));
    thresholds=(1:19)/20;primary=find(thresholds==0.5);assert(isscalar(primary));
    % Explicit hand-counted test includes an invalid pixel and equality at .5.
    testP=reshape(single([.25 .5 .75 .875]),[2 2]);testT=logical(reshape([0 1 1 0],[2 2]));testV=logical(reshape([1 1 1 0],[2 2]));
    actual=threshold_confusion_counts(testP,testT,testV,[.25 .5 .75]);
    assert(isequal(reshape(actual,3,4),[2 1 0 0;2 0 0 1;1 0 1 1]));
    X=zeros(256,256,3,400,'single');T=false(256,256,400);V=T;
    for i=1:400
        r=records(i);assert(strcmp(r.assigned_split,'validation'));
        assert(~contains(r.image,'..') && ~contains(r.mask,'..') && startsWith(r.image,'mcnet/'));
        ip=fullfile(dataRoot,r.image);mp=fullfile(dataRoot,r.mask);
        assert(strcmp(file_sha256(ip),access(i).image_file_sha256) && strcmp(file_sha256(mp),access(i).mask_file_sha256));
        [X(:,:,:,i),T(:,:,i),V(:,:,i),meta]=read_segmentation_sample(ip,mp,[256 256],128);
        assert(nnz(V(:,:,i))==access(i).valid_pixels && nnz(T(:,:,i))==meta.foreground_pixels_after);
    end
    ids=string({records.id})';groups=string({records.group_id})';
    copyfile(splitPath,fullfile(out,'frozen_split.json'));
    copyfile(fullfile(root,'matlab','run_validation_diagnostics.m'),fullfile(out,'run_validation_diagnostics.m'));
    copyfile(fullfile(root,'matlab','threshold_confusion_counts.m'),fullfile(out,'threshold_confusion_counts.m'));
    copyfile(fullfile(root,'matlab','read_segmentation_sample.m'),fullfile(out,'read_segmentation_sample.m'));
    writeJson(fullfile(out,'input_access.json'),struct('validation',access,'validation_pairs',400,'proxy_groups',15,'training_pairs_opened',0,'heldout_pairs_opened',0,'raw_hashes_verified',true));
    modelNames={'small_unet','mobilenet_deeplab'};runNames={'full_stage2_v2_01','mature_stage3_v1_01'};
    kinds={'best_validation','last'};results=struct([]);index=0;
    for model=1:2
        for seed=[20261114 20261115 20261116]
            seedDir=fullfile(root,'runs',runNames{model},'seeds',sprintf('seed_%d',seed));
            historical=jsondecode(fileread(fullfile(seedDir,'validation_evaluations.json')));
            for kind=1:2
                index=index+1;name=sprintf('%s_%d_%s',modelNames{model},seed,kinds{kind});dest=fullfile(out,name);mkdir(dest);
                cp=fullfile(seedDir,'checkpoints',[kinds{kind} '.mat']);cpHash=file_sha256(cp);
                loaded=load(cp,'checkpoint');q=loaded.checkpoint;net=q.net;clear loaded;
                assert(q.initialization_seed==seed && isequal(string(q.class_channel_order(:)),["background";"defect"]));
                preHash=network_parameter_sha256(net);preState=net.State;
                old=historical([historical.global_update]==q.global_update);assert(isscalar(old));
                assert(isequal(string({old.per_image.id})',ids));
                P=zeros(256,256,400,'single');inferenceTimer=tic;
                for first=1:2:400
                    sel=first:min(first+1,400);
                    prob=extractdata(predict(net,dlarray(X(:,:,:,sel),'SSCB')));
                    assert(all(isfinite(prob),'all') && max(abs(sum(prob,3)-1),[],'all')<=1e-4);
                    P(:,:,sel)=reshape(prob(:,:,2,:),256,256,numel(sel));
                    if mod(first-1,100)==0,fprintf('DIAGNOSTIC %d/12 %s images=%d/400\n',index,name,first-1);end
                end
                inferenceSeconds=toc(inferenceTimer);
                assert(strcmp(preHash,network_parameter_sha256(net)) && isequaln(preState,net.State));
                assert(strcmp(cpHash,file_sha256(cp)),'Checkpoint modified');
                C=threshold_confusion_counts(P,T,V,thresholds);
                oldC=[[old.per_image.TP]' [old.per_image.FP]' [old.per_image.FN]' [old.per_image.TN]'];
                assert(isequal(reshape(C(:,primary,:),400,4),oldC),'Historical 0.5 counts failed to reproduce');
                sums=reshape(sum(C,1),19,4);micro=metrics(sums);
                target=sum(oldC(:,[1 3]),2);valid=sum(oldC,2);
                gnames=unique(groups,'stable');groupCounts=zeros(15,19,4);
                for g=1:15,groupCounts(g,:,:)=sum(C(groups==gnames(g),:,:),1);end
                groupIou=groupCounts(:,:,1)./(groupCounts(:,:,1)+groupCounts(:,:,2)+groupCounts(:,:,3));
                candidates=find(abs(micro(:,1)-max(micro(:,1)))<=1e-12);
                [~,tie]=sortrows([abs(thresholds(candidates)'-0.5) -thresholds(candidates)'],[1 2]);best=candidates(tie(1));
                curve=table(thresholds',sums(:,1),sums(:,2),sums(:,3),sums(:,4),micro(:,1),micro(:,2),micro(:,3),micro(:,4),mean(groupIou,1,'omitnan')', ...
                    'VariableNames',{'threshold','TP','FP','FN','TN','IoU','Dice','precision','recall','macro_group_IoU'});
                writetable(curve,fullfile(dest,'threshold_curve.csv'));
                base=metrics(oldC);perImage=table(ids,groups,target,valid,oldC(:,1),oldC(:,2),oldC(:,3),oldC(:,4),base(:,1),base(:,2),base(:,3),base(:,4),oldC(:,2)./(oldC(:,2)+oldC(:,4)),oldC(:,3)./target, ...
                    'VariableNames',{'id','group','target_pixels','valid_pixels','TP','FP','FN','TN','IoU','Dice','precision','recall','FP_over_background','FN_over_target'});
                writetable(perImage,fullfile(dest,'per_image_primary.csv'));
                % Predeclared top3 each by lowest IoU, highest FP/bg and FN/GT.
                [~,a]=sort(base(:,1),'ascend');[~,b]=sort(perImage.FP_over_background,'descend');[~,c]=sort(perImage.FN_over_target,'descend');
                cases=unique([a(1:3);b(1:3);c(1:3)],'stable');
                case_X=X(:,:,:,cases);case_T=T(:,:,cases);case_V=V(:,:,cases);case_P=P(:,:,cases);case_counts=C(cases,:,:);case_ids=ids(cases); %#ok<NASGU>
                save(fullfile(dest,'qualitative_cases.mat'),'case_X','case_T','case_V','case_P','case_counts','case_ids','thresholds','-v7');
                edges=0:.01:1;fgHist=histcounts(P(T&V),edges);bgHist=histcounts(P(~T&V),edges);
                save(fullfile(dest,'counts.mat'),'C','groupCounts','ids','groups','thresholds','fgHist','bgHist','edges','-v7');
                writeBinary(fullfile(dest,'counts.u32'),C,'uint32');
                writeBinary(fullfile(dest,'group_counts.u32'),groupCounts,'uint32');
                writeBinary(fullfile(dest,'case_P.f32'),case_P,'single');
                writeBinary(fullfile(dest,'case_T.u8'),case_T,'uint8');
                writeBinary(fullfile(dest,'case_V.u8'),case_V,'uint8');
                writeJson(fullfile(dest,'binary_layout.json'),struct('endian','little','array_order','MATLAB column major','counts_shape',[400 19 4],'group_counts_shape',[15 19 4],'case_shape',[256 256 numel(cases)],'channels',{{'TP','FP','FN','TN'}},'case_indices_one_based',cases','foreground_histogram',fgHist,'background_histogram',bgHist));
                r=struct('name',name,'model',modelNames{model},'seed',seed,'checkpoint_kind',kinds{kind},'checkpoint_relative',strrep(erase(cp,[root filesep]),filesep,'/'),'checkpoint_sha256',cpHash,'network_parameter_sha256',preHash,'epoch',q.epoch,'update',q.global_update,'validation_pairs',400,'proxy_groups',15,'primary_threshold',.5,'primary_counts',sums(primary,:),'primary_metrics',micro(primary,:),'diagnostic_selected_threshold',thresholds(best),'diagnostic_selected_metrics',micro(best,:),'diagnostic_selected_counts',sums(best,:),'historical_per_image_count_match',true,'checkpoint_bytes_and_network_state_unchanged',true,'inference_seconds',inferenceSeconds,'qualitative_case_indices',cases','scope','Validation descriptive/tuned diagnostics, not heldout performance or a new algorithm');
                writeJson(fullfile(dest,'summary.json'),r);
                if isempty(results),results=r;else,results(index)=r;end %#ok<AGROW>
                writeJson(fullfile(out,'completed_checkpoints.json'),results);
                writeJson(fullfile(out,'status.json'),struct('status','RUNNING','completed_checkpoints',index));
                fprintf('DIAGNOSTIC_DONE %s primary_IoU=%.6f selected_t=%.2f tuned_IoU=%.6f\n',name,micro(primary,1),thresholds(best),micro(best,1));
                clear P net q preState case_X case_T case_V case_P;
            end
        end
    end
    summary=struct('status','PASS','checkpoints',results,'completed_checkpoints',12,'validation_pairs',400,'proxy_groups',15,'thresholds',thresholds,'training_pairs_opened',0,'heldout_pairs_opened',0,'optimizer_updates',0,'total_wall_seconds',toc(started),'scope','Validation diagnostics only; threshold tuning is optimistic and no primary table is replaced.');
    writeJson(fullfile(out,'summary.json'),summary);writeJson(fullfile(out,'status.json'),struct('status','PASS','completed_checkpoints',12));
catch err
    writeJson(fullfile(out,'failure.json'),struct('status','FAILED','message',err.message,'identifier',err.identifier,'stack',err.stack));
    writeJson(fullfile(out,'status.json'),struct('status','FAILED'));rethrow(err);
end
end
function M=metrics(C)
tp=C(:,1);fp=C(:,2);fn=C(:,3);
M=[tp./(tp+fp+fn) 2*tp./(2*tp+fp+fn) tp./(tp+fp) tp./(tp+fn)];
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
function writeBinary(path,value,precision)
fid=fopen(path,'wb','ieee-le');assert(fid>=0);c=onCleanup(@()fclose(fid));assert(fwrite(fid,value,precision)==numel(value));
end
