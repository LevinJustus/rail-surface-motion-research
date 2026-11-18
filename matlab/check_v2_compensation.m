function report=check_v2_compensation(outFolder)
% E2 mechanism contracts and labelled artificial examples, not dataset accuracy.
root=fileparts(fileparts(mfilename('fullpath')));cfg=jsondecode(fileread(fullfile(root,'configs','algorithm_design_v2.json')));
assert(~isfolder(outFolder));mkdir(outFolder);diary(fullfile(outFolder,'run.log'));cleanup=onCleanup(@()diary('off'));timer=tic;checks={};
try
    rng(6182,'twister');C=zeros(41);C(17,21)=1;e=v2_select_motion(C,.2,4,cfg);
    assert(e.available&&isequal(e.visual_mode_xy,[0 -4])&&abs(sum(e.full_weights,'all')-1)<1e-12);pass('motion_sign_and_normalization');
    missing=v2_select_motion(C,.2,NaN,cfg);assert(isequal(missing.full_weights,missing.visual_weights)&&missing.observed_reliability==0);pass('missing_observation_visual_fallback');
    wrong=v2_select_motion(C,.2,-16,cfg);assert(wrong.observed_reliability<.02&&wrong.available);pass('conflicting_observation_attenuation');
    low=v2_select_motion(zeros(41),.2,0,cfg);flat=v2_select_motion(C,0,0,cfg);B=zeros(41);B(1,21)=1;boundary=v2_select_motion(B,.2,0,cfg);
    assert(~low.available&&~flat.available&&~boundary.available);pass('weak_texture_and_boundary_rejection');
    A=.2*ones(41);A(21,21)=.21;diffuse=v2_select_motion(A,.2,NaN,cfg);
    assert(size(diffuse.candidates,1)==25&&diffuse.retained_mass<.02&&abs(sum(diffuse.candidates(:,3))-diffuse.retained_mass)<1e-14);pass('diffuse_mass_not_renormalized');
    first=manual([3 0 .4;0 0 .3]);second=manual([-3 0 .6]);path=v2_compose_motion({first,second},cfg);
    assert(abs(path.retained_mass-.42)<1e-12&&abs(sum(path.weights)-.42)<1e-12);pass('composed_path_mass_product');
    p=single(.45*ones(256));known=manual([0 0 1]);unknown=manual([0 0 .25]);
    a=v2_history_evidence(p,p,v2_compose_motion({known},cfg),cfg);b=v2_history_evidence(p,p,v2_compose_motion({unknown},cfg),cfg);
    assert(max(a.risk,[],'all')<1e-20&&all(b.risk>0,'all')&&all(b.gate<a.gate,'all'));pass('omitted_mass_increases_risk');
    history=single(rand(256));current=single(rand(256));path=v2_compose_motion({manual([0 0 .3;1 0 .25;-1 0 .1])},cfg);
    ev=v2_history_evidence(history,current,path,cfg);q=ev.known_mass;h=ev.mean_history;
    decomposition=q.*(ev.conditional_variance+(h-double(current)).^2);
    decompositionError=max(abs(decomposition-ev.known_disagreement),[],'all');assert(decompositionError<1e-12);
    actual=ev.known_disagreement+(1-q).*(.37-double(current)).^2;assert(all(actual<=ev.risk+1e-12,'all'));pass('risk_decomposition_and_unknown_value_bound');
    backtrack=v2_compose_motion({manual([300 0 1]),manual([-300 0 1])},cfg);
    ev=v2_history_evidence(history,current,backtrack,cfg);assert(~any(ev.known_mass,'all')&&~any(ev.gate,'all'));pass('intermediate_view_coverage');
    ev=v2_history_evidence(history,current,v2_compose_motion({flat},cfg),cfg);assert(~any(ev.gate,'all'));pass('unavailable_history_fallback');
    [out,d]=v2_fuse_window({current},{},cfg);assert(isequal(out,current)&&d.analytic_update_bound==0);pass('first_frame_identity');
    [out,d]=v2_fuse_window({history,current},{known},cfg);assert(all(out>=0&out<=1,'all')&&max(abs(d.delta),[],'all')<=d.analytic_update_bound+1e-12);pass('convex_range_and_update_bound');
    two={known,known};[out,d]=v2_fuse_window({zeros(256,'single'),zeros(256,'single'),single(.75*ones(256))},two,cfg);
    assert(d.protected_pixels==0&&d.analytic_update_bound<.2&&all(out>=.5,'all'));pass('default_strong_protection_is_inactive');
    alternative=cfg;alternative.risk_sigma=1;[out,d]=v2_fuse_window({zeros(256,'single'),zeros(256,'single'),single(.75*ones(256))},two,alternative);
    assert(d.protected_pixels==65536&&all(out==.5,'all'));pass('explicit_protection_guard_with_contract_only_parameter_copy');
    [out,~]=v2_fuse_window({ones(256,'single'),ones(256,'single'),single(.1*ones(256))},two,cfg);assert(all(out<.5,'all'));pass('severe_current_miss_not_recovered');
    % Actual RGB-to-NCC-to-stream path, with known synthetic texture translation.
    texture=uint8(randi([0 255],400,400,3));state=[];streamImages=cell(4,1);streamRaw=cell(4,1);streamOutput=cell(4,1);streamEdges=cell(3,1);
    for k=1:4
        rgb=texture(73+4*(k-1):328+4*(k-1),73:328,:);raw=single(.2+.1*k)*ones(256,'single');
        observation=struct('timestamp_s',.05*(k-1),'odometry_px',4*(k-1),'odometry_available',k~=2);
        [out,state,~]=v2_compensation_step(rgb,raw,observation,state,cfg);
        streamImages{k}=rgb;streamRaw{k}=raw;streamOutput{k}=out;if k>1,streamEdges{k-1}=state.edges{end};end
        assert(numel(state.images)==min(k,3)&&numel(state.edges)==min(k-1,2)&&isequal(state.raw_probabilities{end},raw));
        if k>1,assert(isequal(state.edges{end}.visual_mode_xy,[0 -4]));end
    end
    pass('RGB_NCC_stream_sign_missing_and_bounded_raw_state');
    save(fullfile(outFolder,'synthetic_stream.mat'),'streamImages','streamRaw','streamOutput','streamEdges','-v7');
    [reset,resetState,~]=v2_compensation_step(rgb,raw,observation,[],cfg);assert(isequal(reset,raw)&&resetState.frames_seen==1);pass('explicit_sequence_reset');
    rejected=false;try,v2_compensation_step(rgb,raw,observation,state,cfg);catch,rejected=true;end;assert(rejected);pass('nonincreasing_timestamp_rejection');
    % Controlled probability fixtures: two useful cases and an explicit failure.
    target=false(256);target(80:179,128:129)=true;normal=single(.05*ones(256));rows=struct([]);examples=struct;
    h=normal;h(target)=.8;p=normal;p(target)=.45;compareExample('recover_near_threshold',target,{h,h,p},two);
    h=normal;h(80:179,122:123)=.9;p=normal;p(target)=.55;ambiguous=manual([0 0 .5;6 0 .5]);compareExample('ambiguous_wrong_single_motion',target,{h,p},{ambiguous});
    h=normal;h(target)=.8;p=h;falseRegion=false(256);falseRegion(80:179,100:101)=true;p(falseRegion)=.95;
    compareExample('confident_current_false_positive_failure',target,{h,h,p},two);
    writetable(struct2table(rows),fullfile(outFolder,'controlled_examples.csv'));save(fullfile(outFolder,'controlled_examples.mat'),'examples','-v7');
    writeJson(fullfile(outFolder,'controlled_examples_summary.json'),struct('scope','artificial probability/known-motion examples only, not rail dataset or trained accuracy','rows',rows));
    files={'v2_select_motion.m','v2_motion_weights.m','v2_compose_motion.m','v2_history_evidence.m','v2_fuse_window.m','v2_compensation_step.m','check_v2_compensation.m'};
    mkdir(fullfile(outFolder,'source_snapshot'));sources=struct([]);
    for j=1:numel(files),copyfile(fullfile(root,'matlab',files{j}),fullfile(outFolder,'source_snapshot',files{j}));sources(j).path=['matlab/' files{j}];sources(j).sha256=file_sha256(fullfile(root,'matlab',files{j}));end
    copyfile(fullfile(root,'configs','algorithm_design_v2.json'),fullfile(outFolder,'frozen_design.json'));
    report=struct('status','PASS_V2_MINIMAL_IMPLEMENTATION_CONTRACTS','contract_count',numel(checks),'contracts',{checks},...
        'risk_decomposition_max_error',decompositionError,'three_frame_update_bound',2/3*cfg.risk_sigma/sqrt(exp(1)),...
        'config_sha256',file_sha256(fullfile(root,'configs','algorithm_design_v2.json')),'source_files',sources,...
        'matlab_version',version,'threads',maxNumCompThreads,'wall_seconds',toc(timer),'real_dataset_payloads_opened',0,'model_forward_calls',0,'optimizer_updates',0,...
        'scope','minimum core/analytic contracts and three artificial examples; novelty, rail accuracy and full real-time efficiency not validated');
    writeJson(fullfile(outFolder,'contracts.json'),report);fprintf('V2_CONTRACTS PASS count=%d bound=%.12g\n',numel(checks),report.three_frame_update_bound);
catch err
    writeJson(fullfile(outFolder,'failure.json'),struct('message',err.message,'identifier',err.identifier,'stack',err.stack,'passed_contracts',{checks}));rethrow(err);
end
    function pass(name),checks{end+1}=name;fprintf('V2_CONTRACT %s PASS\n',name);end
    function compareExample(name,target,probabilities,edges)
        [v2,diag]=v2_fuse_window(probabilities,edges,cfg);singleP=probabilities{end};n=numel(probabilities);fixed=double(singleP);
        for t=1:n-1
            shift=zeros(n-t,2);
            for j=t:n-1,shift(j-t+1,:)=edges{j}.candidates(1,1:2);end
            [warped,valid]=warp_motion_history(probabilities{t},shift);difference=warped-double(singleP);difference(~valid)=0;fixed=fixed+(1/n)*difference;
        end
        maps={singleP,single(fixed),v2};methods={'single','single_motion_fixed','v2'};
        for m=1:3
            binary=double(maps{m})>=.5;tp=nnz(binary&target);fp=nnz(binary&~target);fn=nnz(~binary&target);union=tp+fp+fn;iou=1;if union>0,iou=tp/union;end
            row=struct('example',name,'method',methods{m},'TP',tp,'FP',fp,'FN',fn,'IoU',iou,'predicted_positive_pixels',nnz(binary));
            if isempty(rows),rows=row;else,rows(end+1)=row;end
        end
        examples.(name)=struct('target',target,'single',singleP,'single_motion_fixed',single(fixed),'v2',v2,'delta',diag.delta,'protected_pixels',diag.protected_pixels);
        if ~isempty(diag.histories),examples.(name).risk=diag.histories{end}.risk;examples.(name).gate=diag.histories{end}.gate;end
    end
end
function e=manual(candidates)
e=struct('available',true,'candidates',candidates,'retained_mass',sum(candidates(:,3)));
end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
