function report=check_motion_fusion_contracts(outFolder)
% Analytic checks only, with explicitly synthetic images/probabilities/edges.
root=fileparts(fileparts(mfilename('fullpath')));protocol=jsondecode(fileread(fullfile(root,'configs','fusion_protocol_v1.json')));cfg=protocol.inference;
assert(~isfolder(outFolder),'No overwrite');mkdir(outFolder);
results=struct([]);timer=tic;rng(218,'twister');rgb=uint8(randi([0 255],256,256,3));
[x,y]=meshgrid(1:256);plane=(x+2*y)/768;edge=fakeEdge(0,0);flat=zeros(256,256,3,'uint8');
[v,m]=warp_motion_history(plane,[3 -8]);expected=((x-3)+2*(y+8))/768;
assert(max(abs(v(m)-expected(m)))<1e-12);assert(~any(m(:,1:3),'all'));
pass('integer_translation_xy_sign');
[v,m]=warp_motion_history(plane,[.25 -.5]);expected=((x-.25)+2*(y+.5))/768;
assert(max(abs(v(m)-expected(m)))<1e-12);[v0,m0]=warp_motion_history(plane,[0 0]);assert(isequal(v0,plane)&&all(m0,'all'));
pass('subpixel_bilinear_and_closed_domain_edges');
[v,m,d]=warp_motion_history(plane,[8 0;-8 0]);assert(isequal(d,[0 0]));assert(~any(m(:,249:end),'all'));assert(isequal(v(m),plane(m)));
nonlinear=double(mod(x,2));[once,m]=warp_motion_history(nonlinear,[.25 0;.25 0]);[onceRef,~]=warp_motion_history(nonlinear,[.5 0]);assert(isequal(once(m),onceRef(m)));
pass('two_hop_composition_once_only_and_intermediate_coverage');
texture=uint8(randi([0 255],320,320,3));prev=texture(33:288,33:288,:);
ez=register_fusion_pair(prev,prev,0,cfg);er=register_fusion_pair(prev,texture(25:280,33:288,:),-8,cfg);
assert(abs(ez.visual.dx)<.05&&abs(ez.visual.dy)<.05&&abs(er.visual.dy-8)<.05);
pass('zero_motion_and_reverse');
state=[];
for k=1:5
    p=single(.1*k*ones(256));frame=makeFrame(rgb,k);
    [state,o,dd]=motion_fusion_step(state,frame,p,cfg,1,edge);
    assert(dd.window_size==min(k,3));assert(isequal(o.S,p));
    expected=single(mean((max(1,k-2):k)*.1));assert(max(abs(o.U-expected),[],'all')<1e-7);
    if k==1,assert(all(structfun(@(a)isequal(a,p),o)));end
end
pass('first_frame_and_partial_full_window');
[~,part]=warp_motion_history(plane,[4 0]);[~,none]=warp_motion_history(plane,[300 0]);assert(nnz(part)==252*256&&~any(none,'all'));
eall=fakeEdge(300,0);oo=fuse_motion_window({rgb,rgb},{single(plane),single(.2*ones(256))},{eall},cfg,1);
assert(all(oo.V==single(.2),'all'));
pass('partial_and_all_out_of_view');
eu=register_fusion_pair(flat,flat,NaN,cfg);assert(~eu.visual.available);
oo=fuse_motion_window({flat,flat},{single(.9*ones(256)),single(.1*ones(256))},{eu},cfg,1);
for name={'V','P','GV','GVP','GC','R'},assert(all(oo.(name{1})==single(.1),'all'));end
pass('unavailable_visual_edge');
for delta=[NaN 25]
    ee=register_fusion_pair(prev,prev,delta,cfg);
    assert(~ee.observed_available&&ee.observed_factor==1&&isequaln([ee.prior.dx ee.prior.dy],[ee.visual.dx ee.visual.dy]));
end
pass('missing_or_out_of_range_odometry');
for delta=[0 -8 NaN]
    current=prev;if delta==-8,current=texture(25:280,33:288,:);end
    ee=register_fusion_pair(prev,current,delta,cfg);rr=register_motion_pair(prev,current,delta,cfg.registration);
    compare(ee.visual,rr(2));compare(ee.prior,rr(3));
end
pass('forward_results_match_frozen_13b_core');
assert(isequal(motion_confidence([0 .5 1]),[1 0 1]));pass('entropy_endpoints_and_half');
same=single(.37*ones(256));oo=fuse_motion_window({rgb,rgb,rgb},{same,same,same},{edge,edge},cfg,.7);
assert(all(structfun(@(a)isequal(a,same),oo)));pass('equal_probability_no_drift');
rejected=edge;rejected.quality=0;current=single(.2*ones(256));
oo=fuse_motion_window({rgb,rgb},{single(.9*ones(256)),current},{rejected},cfg,1);
for name={'GV','GVP','GC','R'},assert(isequal(oo.(name{1}),current));end
assert(any(oo.V~=current,'all'));pass('zero_gate_exact_current_fallback');
% Register from RGB in this test, rather than injecting analytic cached edges.
frames=cell(7,1);probs=cell(7,1);
for k=1:7,frames{k}=makeFrame(rgb,k);probs{k}=single((.1+.03*k)*ones(256));end
normal=stream(frames,probs,cfg);
mutated=frames;mutP=probs;
for k=5:7,mutated{k}.image=255-frames{k}.image;mutated{k}.simulated_odometry_px=1000*k;mutP{k}=1-probs{k};end
changed=stream(mutated,mutP,cfg);prefix=stream(frames(1:4),probs(1:4),cfg);
for k=1:4,assert(isequal(normal{k},changed{k})&&isequal(normal{k},prefix{k}));end
pass('future_rgb_and_motion_perturbation_prefix_invariance');
frame=makeFrame(rgb,1);frame.timestamp_s=100;
[reset,out,dd]=motion_fusion_step(state,frame,current,cfg,1);
assert(dd.reset_for_time_gap&&numel(reset.frames)==1&&all(structfun(@(a)isequal(a,current),out)));
[fresh,out]=motion_fusion_step([],makeFrame(rgb,1),current,cfg,1);
assert(numel(fresh.frames)==1&&all(structfun(@(a)isequal(a,current),out)));
pass('sequence_reset_and_timestamp_gap_reset');
didFail=false;try,motion_fusion_step(fresh,makeFrame(rgb,1),current,cfg,1);catch,didFail=true;end
assert(didFail);pass('nonincreasing_timestamp_fails');
before=fresh.probabilities;[stored,out]=motion_fusion_step(fresh,makeFrame(rgb,2),single(1-current),cfg,1,edge);
out.V(:)=0;assert(isequal(stored.probabilities{1},before{1})&&isequal(stored.probabilities{2},single(1-current)));
assert(~isfield(stored,'fused'));pass('raw_and_fused_probability_cache_separation');
oo=fuse_motion_window({rgb,rgb,rgb},{single(1-plane),single(plane),current},{edge,edge},cfg,.8);
for name=fieldnames(oo)',a=oo.(name{1});assert(all(isfinite(a),'all')&&all(a>=0 & a<=1,'all'));end
pass('inverted_history_finite_bounded');
bad=makeFrame(rgb,1);bad.profile='not_allowed';didFail=false;
try,motion_fusion_step([],bad,current,cfg,1);catch,didFail=true;end;assert(didFail);
badCfg=cfg;badCfg.mask=true;didFail=false;
try,motion_fusion_step([],makeFrame(rgb,1),current,badCfg,1);catch,didFail=true;end;assert(didFail);
pass('no_source_profile_or_gt_algorithm_fields');
t=logical([1 1 0 0 1 0 1 0]);s=single([0 1 1 0 1 0 0 1]);p=single([.5 .1 .1 .5 .9 .2 .3 .8]);
[counts,transitions]=fusion_binary_counts(p,t,s,.5);[base,~]=fusion_binary_counts(s,t,s,.5);
assert(isequal(counts,[2 2 2 2])&&isequal(transitions,[1 1 1 1]));
assert(counts(3)==base(3)-transitions(1)+transitions(3)&&counts(2)==base(2)-transitions(2)+transitions(4));
fid=fopen(fullfile(outFolder,'fixture_probability.f32'),'wb','ieee-le');fwrite(fid,p,'single');fclose(fid);
fid=fopen(fullfile(outFolder,'fixture_probability.f32'),'rb','ieee-le');roundtrip=fread(fid,8,'*single')';fclose(fid);assert(isequal(roundtrip,p));
writeJson(fullfile(outFolder,'count_fixture.json'),struct('target',t,'single_probability',s,'probability',p,'counts',counts,'transitions',transitions));
pass('binary_float32_roundtrip_and_count_identities');
frames2=frames;probs2=probs;probs2{1}=1-probs{1};frames2{1}.image=255-rgb;
changed=stream(frames2,probs2,cfg);
assert(isequal(normal{4},changed{4})&&isequal(normal{7},changed{7}));
pass('bounded_recursive_excludes_window_external_state');
assert(numel(results)==20);
report=struct('status','PASS_20_MATLAB_ANALYTIC_CONTRACTS','checks',results,'wall_seconds',toc(timer),...
    'synthetic_only',true,'real_images_opened',0,'model_forward_calls',0,...
    'remaining_python_contracts',{{'training_calibration_excludes_validation_gt','region_metrics_empty_and_split_merge_cases'}});
writeJson(fullfile(outFolder,'contracts.json'),report);
fprintf('FUSION_ANALYTIC_CONTRACTS PASS %d\n',numel(results));
    function pass(id)
        assert(any(strcmp(protocol.required_c3_contract_ids,id)));
        item=struct('id',id,'status','PASS','scope','synthetic analytic contract, not accuracy result');
        if isempty(results),results=item;else,results(end+1)=item;end
    end
end
function frame=makeFrame(rgb,k)
frame=struct('image',rgb,'timestamp_s',(k-1)*.05,'simulated_odometry_px',0,'odometry_available',true);
end
function edge=fakeEdge(dx,dy)
estimate=struct('available',true,'dx',dx,'dy',dy,'score',1,'used_prior',false,'fallback','analytic_fixture',...
    'boundary_peak',false,'candidate_count',1681,'integer_dx',round(dx),'integer_dy',round(dy));
edge=struct('visual',estimate,'prior',estimate,'reverse',estimate,'observed_delta',0,'observed_available',true,...
    'quality',1,'observed_factor',1,'ncc_maps',0);
end
function result=stream(frames,probs,cfg)
state=[];result=cell(numel(frames),1);
for k=1:numel(frames),[state,result{k}]=motion_fusion_step(state,frames{k},probs{k},cfg,1);end
end
function compare(a,b)
assert(a.available==b.available && isequaln([a.dx a.dy a.score],[b.dx b.dy b.score]) && a.boundary_peak==b.boundary_peak && strcmp(a.fallback,b.fallback));
end
function writeJson(p,v)
fid=fopen(p,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(v,PrettyPrint=true));
end
