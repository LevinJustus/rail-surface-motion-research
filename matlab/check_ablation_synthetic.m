function report=check_ablation_synthetic(outFolder)
% Analytic and frozen-reference checks; no actual dataset or model is read.
root=fileparts(fileparts(mfilename('fullpath')));protocol=jsondecode(fileread(fullfile(root,'configs','fusion_protocol_v1.json')));cfg=protocol.inference;
assert(~isfolder(outFolder));mkdir(outFolder);rng(5518,'twister');timer=tic;
texture=uint8(randi([0 255],400,400,3));images=cell(9,1);P=cell(9,1);edges=cell(8,1);maps=cell(8,1);
[x,y]=meshgrid(1:256);frames=repmat(struct('image','','timestamp_s',0,'simulated_odometry_px',0,'odometry_available',true),9,1);
for k=1:9
    images{k}=texture(73+4*(k-1):328+4*(k-1),73:328,:);P{k}=single(.5+.25*sin((x+2*y)/70+k/7));
    frames(k).timestamp_s=(k-1)*.05;frames(k).simulated_odometry_px=4*(k-1);
    if k>1
        edges{k-1}=register_fusion_pair(images{k-1},images{k},4,cfg);a=motion_gray(images{k-1});b=motion_gray(images{k});C=normxcorr2(a(65:192,65:192),b);maps{k-1}=C(192+(-20:20),192+(-20:20));
    end
end
variants=ablation_observation_variants(frames);stress=cell(7,1);
for j=1:7
    stress{j}=cell(8,1);f=variants{j};
    for k=2:9
        delta=NaN;if f(k-1).odometry_available&&f(k).odometry_available,delta=f(k).simulated_odometry_px-f(k-1).simulated_odometry_px;end
        stress{j}{k-1}=ablation_pressure_edge(edges{k-1},maps{k-1},delta,cfg);
    end
end
maxError=0;comparisons=0;state=[];
for k=1:9
    start=max(1,k-2);base=fuse_motion_window(images(start:k),P(start:k),edges(start:k-1),cfg,.9597680584784836);
    prefix=cellfun(@(e)e(1:k-1),stress,'UniformOutput',false);
    [out,state]=ablation_frame(images(1:k),P(1:k),edges(1:k-1),prefix,base,cfg,.9597680584784836,state);
    for j=1:7
        reference=fuse_motion_window(images(start:k),P(start:k),stress{j}(start:k-1),cfg,.9597680584784836);
        for m={'S','U','V','P','GV','GVP','GC','R'}
            maxError=max(maxError,compare(out((j-1)*8+find(strcmp({'S','U','V','P','GV','GVP','GC','R'},m{1}))).probability,reference.(m{1})));comparisons=comparisons+1;
        end
    end
    cfg5=cfg;cfg5.window_frames=5;start5=max(1,k-4);
    reference=fuse_motion_window(images(start5:k),P(start5:k),edges(start5:k-1),cfg5,.9597680584784836);
    for m={'S','V','GV','GVP'}
        one=findRow(out,'window_1',m{1});assert(isequal(one,base.S));
        five=findRow(out,'window_5',m{1});maxError=max(maxError,compare(five,reference.(m{1})));comparisons=comparisons+1;
    end
    assert(isequal(findRow(out,'all_missing','P'),base.V));
    assert(isequal(findRow(out,'all_missing','GVP'),base.GV));
end
assert(maxError<=2e-6);
% Missing flags survive a perturbation and alternate absolute errors yield 2A increments.
missing=frames;missing(4).odometry_available=false;missing(4).simulated_odometry_px=NaN;
vv=ablation_observation_variants(missing);for j=1:7,assert(~vv{j}(4).odometry_available);end
assert(vv{1}(2).simulated_odometry_px-vv{1}(1).simulated_odometry_px==-2);
assert(vv{6}(2).simulated_odometry_px-vv{6}(1).simulated_odometry_px==-2);
% Manual known edges isolate factors; these are not estimated from GT.
e=edges{1};e.visual.dx=0;e.visual.dy=0;e.reverse.dx=0;e.reverse.dy=0;e.prior=e.visual;
e.visual.score=1;e.reverse.score=1;e.forward_diagnostic.second_score=0;e.reverse_diagnostic.second_score=0;
e.visual.boundary_peak=false;e.reverse.boundary_peak=false;e.quality=1;e.observed_factor=1;
rgb=images{1};low=single(.1*ones(256));high=single(.9*ones(256));
fixture=struct('S',low,'U',(low+high)/2,'V',(low+high)/2,'P',(low+high)/2,'GV',(low+high)/2,'GVP',(low+high)/2,'GC',(low+high)/2,'R',(low+high)/2);
seven=repmat({{e}},7,1);
[out,~]=ablation_frame({uint8(255*ones(256,256,3)),zeros(256,256,3,'uint8')},{high,low},{e},seven,fixture,cfg,.9597680584784836,[]);
a=findRow(out,'remove_photo','GV');b=findRow(out,'remove_confidence','GV');assert(a(128,128)>b(128,128));
q=e;q.forward_diagnostic.second_score=1;q.reverse_diagnostic.second_score=1;q.quality=0;
seven=repmat({{q}},7,1);[out,~]=ablation_frame({rgb,rgb},{high,low},{q},seven,fixture,cfg,.9597680584784836,[]);
a=findRow(out,'remove_peak_margin','GV');assert(a(128,128)>.49);
q.visual.boundary_peak=true;
seven=repmat({{q}},7,1);[out,~]=ablation_frame({rgb,rgb},{high,low},{q},seven,fixture,cfg,.9597680584784836,[]);
assert(isequal(findRow(out,'remove_peak_margin','GV'),low));
% Pollution disappears after the finite window, while an unbounded state can persist.
I=repmat({rgb},9,1);zerosP=repmat({zeros(256,'single')},9,1);E=repmat({e},8,1);seven=repmat({E},7,1);state=[];recovery=[];
for k=1:9
    start=max(1,k-2);base=fuse_motion_window(I(start:k),zerosP(start:k),E(start:k-1),cfg,.9597680584784836);
    prefix=cellfun(@(a)a(1:k-1),seven,'UniformOutput',false);
    [out,state]=ablation_frame(I(1:k),zerosP(1:k),E(1:k-1),prefix,base,cfg,.9597680584784836,state);
    if k>=6
        assert(isequal(findRow(out,'pollution','R'),base.R));assert(any(findRow(out,'pollution','R_unbounded')>0,'all'));recovery(end+1)=k; %#ok<AGROW>
    end
end
report=struct('status','PASS_C5_SYNTHETIC','frozen_reference_output_comparisons',comparisons,'maximum_probability_error',maxError,...
    'checks',{{'seven observation variants vs frozen core','W1 identity','W5 vs frozen pure-window core','all-missing visual fallback',...
    'original missing flags','absolute perturbation sign','photo removal','zero margin removal','boundary rejection retained',...
    'bounded recovery and unbounded contamination after frame6'}},'wall_seconds',toc(timer),'network_forward_calls',0,'real_dataset_payloads_opened',0);
writeJson(fullfile(outFolder,'contracts.json'),report);fprintf('C5_SYNTHETIC PASS comparisons=%d error=%.9g\n',comparisons,maxError);
end
function value=findRow(rows,scenario,method)
selection=strcmp(string({rows.scenario}),scenario)&strcmp(string({rows.method}),method);assert(nnz(selection)==1);value=rows(selection).probability;
end
function error=compare(a,b),error=max(abs(double(a)-double(b)),[],'all');end
function writeJson(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
