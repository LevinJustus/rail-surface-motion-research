function report=check_v2_conflict_ablation(runId)
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'runs',runId);assert(~isfolder(out));mkdir(out);
cfg=jsondecode(fileread(fullfile(root,'configs','algorithm_design_v2.json')));names={};
try
    edge=struct('available',true,'candidates',[0 0 1]);two={edge,edge};p=single(.1*ones(256));h=single(.8*ones(256));
    [a,d]=v2_no_conflict_fuse_window({p},{},cfg);assert(isequal(a,p)&&d.convex_update_bound==0);names{end+1}='first_frame_identity';
    [a,d]=v2_no_conflict_fuse_window({h,h,p},two,cfg);expected=single((double(p)+2*double(h))/3);
    assert(isequal(a,expected)&&all(a>=.5,'all')&&max(d.delta,[],'all')>.1011);names{end+1}='severe_miss_can_cross_threshold';
    [old,oldD]=v2_fuse_window({h,h,p},two,cfg);assert(all(old<.5,'all')&&max(abs(oldD.delta),[],'all')<=oldD.analytic_update_bound+1e-12);names{end+1}='old_v2_bound_preserved';
    % The same evidence at a normal pixel creates an FP: no semantic guarantee.
    normal=false(256);[c,~]=fusion_binary_counts(double(a),normal,double(p),.5);assert(c(2)==65536);names{end+1}='stable_wrong_history_can_create_false_positives';
    absent=struct('available',false,'candidates',zeros(0,3));[a,~]=v2_no_conflict_fuse_window({h,p},{absent},cfg);assert(isequal(a,p));names{end+1}='missing_motion_identity';
    outside=edge;outside.candidates=[300 0 1];[a,~]=v2_no_conflict_fuse_window({h,p},{outside},cfg);assert(isequal(a,p));names{end+1}='out_of_view_identity';
    [a,d]=v2_no_conflict_fuse_window({zeros(256,'single'),zeros(256,'single'),single(.75*ones(256))},two,cfg);
    assert(d.protected_pixels==65536&&all(a==.5,'all')&&all(abs(d.unprotected_probability-.25)<1e-12,'all'));names{end+1}='unchanged_strong_current_protection';
    partial=edge;partial.candidates=[0 0 .4];[a,d]=v2_no_conflict_fuse_window({h,p},{partial},cfg);
    risk=.6*max(double(p).^2,(1-double(p)).^2);g=.4*exp(-risk/(2*cfg.risk_sigma^2));
    assert(max(abs(double(a)-double(single(double(p)+.5*g.*(double(h)-double(p))))),[],'all')==0&&all(d.histories{1}.unknown_risk>0,'all'));names{end+1}='discarded_mass_penalty_not_removed';
    split=edge;split.candidates=[0 0 .5;20 0 .5];[xx,~]=meshgrid(1:256);texture=single(mod(xx,40)<20);
    [~,d]=v2_no_conflict_fuse_window({texture,p},{split},cfg);assert(any(d.histories{1}.conditional_variance>.2,'all'));names{end+1}='candidate_variance_not_removed';
    report=struct('status','PASS_E4_ANALYTIC_CONTRACTS','contracts',{names},'contract_count',numel(names),'old_core_unchanged',true);
    writeJson(fullfile(out,'contracts.json'),report);copyfile([mfilename('fullpath') '.m'],fullfile(out,'check_v2_conflict_ablation.m'));
catch err
    writeJson(fullfile(out,'failure.json'),struct('message',err.message,'stack',err.stack));rethrow(err);
end
end
function writeJson(path,value)
f=fopen(path,'w','n','UTF-8');assert(f>=0);c=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(value,PrettyPrint=true));
end
