function generate_controlled_sequences(dataRoot,configPath,runId,mode)
% Step12 only: native MCnet crops, exposure rendering, isolated evaluation truth.
% No prediction, network loading, training, heldout raw data, or hardware access.
if nargin<4,mode='contract';end
root=fileparts(fileparts(mfilename('fullpath')));
assert(any(strcmp(mode,{'contract','full'})),'Mode must be contract or full');
assert(~isempty(regexp(runId,'^step12_[A-Za-z0-9_]+$','once')),'Use a new step12_ run ID');
if ~isfile(configPath),configPath=fullfile(root,configPath);end
cfg=jsondecode(fileread(configPath));
assert(strcmp(file_sha256(configPath),'33fe958c3f68e6a415679ead76853b8991a1758a7c7b312fdbfae8f06271b172'),'Frozen configuration mismatch');
assert(strcmp(file_sha256(fullfile(root,cfg.source_plan)),cfg.source_plan_sha256));
assert(strcmp(file_sha256(fullfile(root,cfg.source_split)),cfg.source_split_sha256));
assert(strcmp(file_sha256(fullfile(root,cfg.size_audit)),cfg.size_audit_sha256));
plan=jsondecode(fileread(fullfile(root,cfg.source_plan)));
split=jsondecode(fileread(fullfile(root,cfg.source_split)));
records=plan.selected_records;
assert(numel(records)==60 && isempty(plan.missing_groups));
active=split.records(ismember({split.records.assigned_split},{'training','validation'}));
assert(numel(active)==1796);
for j=1:numel(records)
    k=find(strcmp({active.id},records(j).id));assert(isscalar(k));
    assert(strcmp(active(k).image,records(j).image)&&strcmp(active(k).mask,records(j).mask));
    assert(strcmp(active(k).group_id,records(j).group_id)&&strcmp(active(k).assigned_split,records(j).assigned_split));
end
assert(numel(unique(string({records.group_id})))==60);
allSelected=records;
if strcmp(mode,'contract'),records=records(cfg.contract_source_indices_one_based);end
dataRoot=char(java.io.File(dataRoot).getCanonicalPath());
assert(isfolder(fullfile(dataRoot,'mcnet','extracted')),'Expected local MCnet root');
out=fullfile(root,'runs',runId);assert(~isfolder(out)&&~isfile(out),'Refuse existing run');
mkdir(out);mkdir(fullfile(out,'algorithm_inputs'));mkdir(fullfile(out,'evaluation_only'));
diary(fullfile(out,'run.log'));cleanup=onCleanup(@()diary('off'));
status.state='RUNNING';status.mode=mode;status.started_utc=utcNow();
status.source_count=numel(records);status.planned_sequences=numel(records)*numel(cfg.profiles);
status.completed_sequences=0;status.completed_frames=0;status.heldout_images_opened=0;
writeJson(fullfile(out,'status.json'),status);
try
    snapshotFiles={'matlab/generate_controlled_sequences.m','matlab/file_sha256.m', ...
        'tools/prepare_step12_plan.py','tools/revise_step12_config_v2.py',cfg.source_plan,cfg.source_split,cfg.size_audit};
    sourceMeta=struct('path',{},'sha256',{});
    for k=1:numel(snapshotFiles)
        source=fullfile(root,snapshotFiles{k});dest=fullfile(out,'evaluation_only','source_snapshot',snapshotFiles{k});
        if ~isfolder(fileparts(dest)),mkdir(fileparts(dest));end
        copyfile(source,dest);assert(strcmp(file_sha256(source),file_sha256(dest)));
        sourceMeta(k)=struct('path',snapshotFiles{k},'sha256',file_sha256(source)); %#ok<AGROW>
    end
    copyfile(configPath,fullfile(out,'evaluation_only','frozen_config.json'));
    [code,commit]=system(sprintf('git -C "%s" rev-parse HEAD',root));assert(code==0);
    [~,gitstatus]=system(sprintf('git -C "%s" status --porcelain=v1',root));
    environment.matlab_version=version;environment.computer=computer;
    environment.products=ver;environment.git_commit=strtrim(commit);
    environment.git_status_at_start=gitstatus;environment.source_files=sourceMeta;
    environment.config_sha256=file_sha256(configPath);
    environment.seed=cfg.seed;environment.random_draws=false;
    writeJson(fullfile(out,'evaluation_only','environment.json'),environment);
    writeJson(fullfile(out,'evaluation_only','selected_sources.json'),records);
    writeJson(fullfile(out,'evaluation_only','all_frozen_source_candidates.json'),plan);
    [contracts,contractDetails]=geometryContracts();
    assert(contracts,'Analytic geometry contract failed');
    writeJson(fullfile(out,'evaluation_only','analytic_contracts.json'),contractDetails);
    access=struct([]);index=struct([]);checks=struct([]);seqNo=0;
    for si=1:numel(records)
        r=records(si);ip=safeRaw(dataRoot,r.image);mp=safeRaw(dataRoot,r.mask);
        imageShaBefore=file_sha256(ip);maskShaBefore=file_sha256(mp);
        rgb=imread(ip);raw=imread(mp);
        assert(isa(rgb,'uint8')&&size(rgb,3)==3);
        if ndims(raw)==3,assert(isequal(raw(:,:,1),raw(:,:,2))&&isequal(raw(:,:,1),raw(:,:,3)));raw=raw(:,:,1);end
        assert(isa(raw,'uint8')&&isequal(size(rgb,[1 2]),size(raw)));
        assert(isequal(size(raw),[r.height r.width]),'Raw dimensions differ from frozen audit');
        mask=raw>=cfg.jpeg_label_threshold;
        entry=struct('id',r.id,'group_id',r.group_id,'assigned_split',r.assigned_split, ...
            'image',r.image,'mask',r.mask,'height',r.height,'width',r.width, ...
            'image_sha256',imageShaBefore,'mask_sha256',maskShaBefore);
        if si==1,access=entry;else,access(si)=entry;end %#ok<AGROW>
        for pi=1:numel(cfg.profiles)
            seqNo=seqNo+1;seqId=sprintf('seq%04d',seqNo);profile=cfg.profiles(pi);
            [frames,masks,truth,motion,check]=renderSequence(rgb,mask,r,cfg,profile);
            inputDir=fullfile(out,'algorithm_inputs',seqId);
            truthDir=fullfile(out,'evaluation_only','sequences',seqId);mkdir(inputDir);mkdir(truthDir);
            for fi=1:cfg.frames_per_sequence
                imwrite(frames(:,:,:,fi),fullfile(inputDir,sprintf('frame_%03d.png',fi)));
                imwrite(uint8(masks(:,:,fi))*255,fullfile(truthDir,sprintf('mask_%03d.png',fi)));
            end
            writeJson(fullfile(inputDir,'motion_input.json'),motion);
            truth.source_id=r.id;truth.proxy_group=r.group_id;truth.assigned_split=r.assigned_split;
            truth.profile=profile.name;truth.native_source_hw=[r.height r.width];
            truth.note='Evaluation only. Synthetic image generation, not detection results or measured encoder data.';
            writeJson(fullfile(truthDir,'geometry_truth.json'),truth);
            check.sequence_id=seqId;check.source_id=r.id;check.profile=profile.name;
            if si==1 && any(strcmp(profile.name,{'clean','combined'}))
                [fr2,ma2,tr2,mo2]=renderSequence(rgb,mask,r,cfg,profile);
                check.exact_internal_replay=isequal(frames,fr2)&&isequal(masks,ma2)&&isequaln(truth.frames,tr2.frames)&&isequaln(motion,mo2);
                assert(check.exact_internal_replay,'Deterministic replay failed');
            else
                check.exact_internal_replay=[];
            end
            if seqNo==1,checks=check;else,checks(seqNo)=check;end %#ok<AGROW>
            entry=struct('sequence_id',seqId,'source_id',r.id,'group_id',r.group_id, ...
                'assigned_split',r.assigned_split,'profile',profile.name,'frame_count',cfg.frames_per_sequence);
            if seqNo==1,index=entry;else,index(seqNo)=entry;end %#ok<AGROW>
            status.completed_sequences=seqNo;status.completed_frames=seqNo*cfg.frames_per_sequence;
            if mod(seqNo,12)==0
                writeJson(fullfile(out,'status.json'),status);
                fprintf('SEQUENCE_PROGRESS source=%d/%d sequences=%d/%d frames=%d\n',si,numel(records),seqNo,status.planned_sequences,status.completed_frames);
            end
        end
        assert(strcmp(imageShaBefore,file_sha256(ip))&&strcmp(maskShaBefore,file_sha256(mp)),'Raw source changed during generation');
    end
    writeJson(fullfile(out,'evaluation_only','source_access.json'),access);
    writeJson(fullfile(out,'evaluation_only','sequence_index.json'),index);
    writeJson(fullfile(out,'evaluation_only','sequence_checks.json'),checks);
    inputAudit=auditInputs(out,cfg,index);
    writeJson(fullfile(out,'evaluation_only','algorithm_input_audit.json'),inputAudit);
    plotPanels(out,index,checks,cfg);
    manifest=payloadManifest(out);writeJson(fullfile(out,'evaluation_only','payload_manifest.json'),manifest);
    summary.state='PASS';summary.mode=mode;summary.sequences=seqNo;summary.frames=status.completed_frames;
    summary.source_images_opened=numel(records);summary.source_masks_opened=numel(records);
    summary.proxy_groups=numel(records);summary.physical_rail_segments_verified=0;
    summary.heldout_images_opened=0;summary.neu_images_opened=0;summary.optimizer_updates=0;
    summary.missing_groups=plan.missing_groups;summary.ineligible_candidate_count=sum(~[plan.candidate_records.eligible]);
    summary.complete_training_proxy_groups=sum(strcmp({allSelected.assigned_split},'training'));
    summary.complete_validation_proxy_groups=sum(strcmp({allSelected.assigned_split},'validation'));
    summary.minimum_source_boundary_clearance_px=min([checks.minimum_boundary_clearance_px]);
    summary.minimum_nearest_mask_pixel_count=min([checks.minimum_mask_foreground_pixels]);
    summary.input_audit=inputAudit;summary.frozen_config_sha256=environment.config_sha256;
    summary.source_plan_sha256=cfg.source_plan_sha256;
    summary.payload_count=numel(manifest);summary.payload_digest=manifestDigest(manifest);
    summary.foreground_statistics_used_for_selection=false;
    summary.source_bytes_unchanged=true;summary.completed_utc=utcNow();
    writeJson(fullfile(out,'summary.json'),summary);
    status.state='PASS';status.completed_utc=summary.completed_utc;writeJson(fullfile(out,'status.json'),status);
    fprintf('STEP12_PASS mode=%s sources=%d sequences=%d frames=%d heldout=0 digest=%s\n',mode,numel(records),seqNo,status.completed_frames,summary.payload_digest);
catch err
    status.state='FAILED';status.error_identifier=err.identifier;status.error_message=err.message;
    status.stack=err.stack;status.failed_utc=utcNow();writeJson(fullfile(out,'status.json'),status);
    fprintf(2,'%s\n',getReport(err,'extended','hyperlinks','off'));rethrow(err);
end
end

function [frames,masks,truth,motion,check]=renderSequence(rgb,mask,r,cfg,p)
h=cfg.frame_size_hw(1);w=cfg.frame_size_hw(2);n=cfg.frames_per_sequence;
[xx,yy]=meshgrid(1:w,1:h);xy=[xx(:)';yy(:)'];centre=[(w+1)/2;(h+1)/2];
span=r.longitudinal_span_px;path=cfg.trajectory_knots.(p.trajectory)(:)';
assert(numel(path)==n);offset=span*path;
origin=[(r.width-w)/2;(r.height-h-min(offset)-max(offset))/2];
t=0:n-1;
if p.jitter
    angles=cfg.rotation_amplitude_deg*sin(2*pi*t/5);
    lateral=cfg.lateral_amplitude_px*sin(2*pi*t/3);
else,angles=zeros(1,n);lateral=angles;end
exposureFrames=0;
if p.blur_px>0,exposureFrames=p.blur_px/max(abs(diff(offset)));end
if p.blur_px>0,sub=linspace(-0.5,0.5,cfg.exposure_subsamples)*exposureFrames;else,sub=0;end
frames=zeros(h,w,3,n,'uint8');masks=false(h,w,n);minClearance=inf;
frTruth=struct([]);inputFrames=struct([]);fg=zeros(1,n);blurDistance=zeros(1,n);
measured=round((p.scale*offset+p.drift_px*(t/(n-1)).^2)/p.quantization_px)*p.quantization_px;
assert(any(abs(measured-offset)>1e-12),'Simulated odometry accidentally equals exact trajectory');
for k=1:n
    acc=zeros(h,w,3);corners=0;
    for sj=1:numel(sub)
        q=t(k)+sub(sj);
        sy=interp1(t,offset,q,'linear','extrap');sx=interp1(t,lateral,q,'linear','extrap');
        angle=interp1(t,angles,q,'linear','extrap');
        T=makeTransform(origin,centre,sx,sy,angle);coords=T*[xy;ones(1,numel(xx))];
        clearance=min([coords(1,:)-1,r.width-coords(1,:),coords(2,:)-1,r.height-coords(2,:)]);
        assert(clearance>=0,'Exposure sample goes outside native image');minClearance=min(minClearance,clearance);
        sxGrid=reshape(coords(1,:),h,w);syGrid=reshape(coords(2,:),h,w);
        for c=1:3,acc(:,:,c)=acc(:,:,c)+interp2(double(rgb(:,:,c))/255,sxGrid,syGrid,'linear');end
        corners=corners+1;
    end
    image=acc/corners;
    if p.lighting
        gain=0.65+0.05*cos(2*pi*t(k)/(n-1));field=0.75+0.25*xx/w;
        glint=0.08*exp(-((xx-0.65*w)/(0.18*w)).^2);
        image=image.*(gain*field)+glint;
    end
    frames(:,:,:,k)=uint8(round(255*min(1,max(0,image))));
    T=makeTransform(origin,centre,lateral(k),offset(k),angles(k));
    coord=T*[xy;ones(1,numel(xx))];
    masks(:,:,k)=interp2(double(mask),reshape(coord(1,:),h,w),reshape(coord(2,:),h,w),'nearest')>=0.5;
    fg(k)=nnz(masks(:,:,k));
    left=interp1(t,offset,t(k)-exposureFrames/2,'linear','extrap');
    right=interp1(t,offset,t(k)+exposureFrames/2,'linear','extrap');blurDistance(k)=abs(right-left);
    frTruth(k).frame=sprintf('frame_%03d.png',k);frTruth(k).frame_to_source_3x3=T;
    frTruth(k).true_longitudinal_offset_px=offset(k);frTruth(k).true_lateral_offset_px=lateral(k);
    frTruth(k).true_rotation_deg=angles(k);frTruth(k).exposure_s=exposureFrames*cfg.sampling_interval_s;
    frTruth(k).longitudinal_exposure_distance_px=blurDistance(k);frTruth(k).mask_foreground_pixels=fg(k);
    if k>1
        Tprev=frTruth(k-1).frame_to_source_3x3;
        frTruth(k).previous_to_current_point_3x3=T\Tprev;
        frTruth(k).current_to_previous_sampling_3x3=Tprev\T;
    else
        frTruth(k).previous_to_current_point_3x3=eye(3);
        frTruth(k).current_to_previous_sampling_3x3=eye(3);
    end
    inputFrames(k).image=sprintf('frame_%03d.png',k);
    inputFrames(k).timestamp_s=t(k)*cfg.sampling_interval_s;
    inputFrames(k).simulated_odometry_px=measured(k);
    inputFrames(k).odometry_available=~ismember(k,p.drop_frames);
    if ~inputFrames(k).odometry_available,inputFrames(k).simulated_odometry_px=NaN;end
end
motion.schema_version=1;motion.frames=inputFrames;
truth.coordinate_convention='MATLAB one-based pixel centres: [source_x;source_y;1]=frame_to_source_3x3*[frame_x;frame_y;1].';
truth.temporal_mapping='Previous-to-current point = inv(T_current)*T_previous; current-to-previous sampling = inv(T_previous)*T_current.';
truth.label_definition='Clear binary source mask >=128, nearest-neighbour mapped at exposure centre, never photometrically degraded.';
truth.exposure_definition='Equal-weight integration of 9 continuous piecewise-linear trajectory samples; blur1/3 profiles have exactly 1/3 source-pixel longitudinal exposure span.';
truth.true_longitudinal_offsets_px=offset;truth.frames=frTruth;
check.minimum_boundary_clearance_px=minClearance;check.native_no_resize=true;
check.frame_count=n;check.minimum_mask_foreground_pixels=min(fg);check.maximum_mask_foreground_pixels=max(fg);
check.longitudinal_exposure_distances_px=blurDistance;check.odometry_missing_frames=p.drop_frames;
check.true_stop_intervals=find(diff(offset)==0);check.true_reverse_intervals=find(diff(offset)<0);
check.longitudinal_span_px=span;check.actual_longitudinal_range_px=max(offset)-min(offset);
check.maximum_absolute_frame_step_px=max(abs(diff(offset)));
check.approximate_minimum_longitudinal_overlap=1-max(abs(diff(offset)))/h;
check.exact_internal_replay=[];
end

function T=makeTransform(origin,centre,lateral,longitudinal,angle)
a=angle*pi/180;A=[cos(a),-sin(a);sin(a),cos(a)];
b=origin+[lateral;longitudinal]+centre-A*centre;T=[A,b;0,0,1];
end

function [pass,detail]=geometryContracts()
T0=[1 0 20;0 1 30;0 0 1];T1=[1 0 20;0 1 34;0 0 1];
q=[80;90;1];old=T0\q;cur=T1\q;
detail.positive_camera_motion_moves_source_feature_up=abs(cur(2)-old(2)+4)<1e-12;
detail.point_map_error=max(abs((T1\T0)*old-cur));
detail.backward_sampling_error=max(abs((T0\T1)*cur-old));
detail.image_interpolation_integer_identity=true;
[x,y]=meshgrid(1:10);img=reshape(1:100,10,10);
detail.image_interpolation_integer_identity=isequal(interp2(img,x,y,'linear'),img);
detail.mask_nearest_identity=isequal(interp2(double(img>50),x,y,'nearest'),double(img>50));
pass=detail.positive_camera_motion_moves_source_feature_up&&detail.point_map_error<1e-12&&detail.backward_sampling_error<1e-12&&detail.image_interpolation_integer_identity&&detail.mask_nearest_identity;
end

function audit=auditInputs(out,cfg,index)
files=dir(fullfile(out,'algorithm_inputs','**','*'));files=files(~[files.isdir]);
assert(numel(files)==numel(index)*(cfg.frames_per_sequence+1));
count=0;missing=0;
for j=1:numel(index)
    folder=fullfile(out,'algorithm_inputs',index(j).sequence_id);
    path=fullfile(folder,'motion_input.json');text=fileread(path);
    assert(~contains(text,'NaN')&&~contains(text,'Infinity'));
    item=jsondecode(text);assert(isequal(sort(fieldnames(item)),sort({'schema_version';'frames'})));
    assert(numel(item.frames)==cfg.frames_per_sequence);
    for k=1:numel(item.frames)
        r=item.frames(k);assert(isequal(sort(fieldnames(r)),sort(cfg.algorithm_frame_fields)));
        assert(strcmp(r.image,sprintf('frame_%03d.png',k)));
        assert(r.timestamp_s==(k-1)*cfg.sampling_interval_s);
        if r.odometry_available,assert(isscalar(r.simulated_odometry_px)&&isfinite(r.simulated_odometry_px));
        else,assert(isempty(r.simulated_odometry_px));missing=missing+1;end
        info=imfinfo(fullfile(folder,r.image));assert(info.Height==256&&info.Width==256);
        count=count+1;
    end
end
audit.state='PASS';audit.input_images=count;audit.json_null_odometry_count=missing;
audit.forbidden_source_path_or_mask_or_exact_geometry_fields=0;
audit.whitelist='Top-level schema_version,frames; frame image,timestamp_s,simulated_odometry_px,odometry_available only';
end

function plotPanels(out,index,checks,cfg)
% Fixed first source and predeclared profile positions; no GT/performance selection.
profiles={'clean','geometric_jitter','blur_3px','persistent_light','combined'};
f=figure('Visible','off','Color','white','Position',[50 50 1400 700]);tiledlayout(2,5,'TileSpacing','compact','Padding','compact');
for p=1:5
    q=find(strcmp({index.profile},profiles{p}),1,'first');id=index(q).sequence_id;k=5;
    nexttile(p);image(imread(fullfile(out,'algorithm_inputs',id,sprintf('frame_%03d.png',k))));axis image off;title(strrep(profiles{p},'_',' '));
    nexttile(5+p);imagesc(imread(fullfile(out,'evaluation_only','sequences',id,sprintf('mask_%03d.png',k))));axis image off;title('Evaluation mask only');
end
colormap(f,gray(256));sgtitle('Synthetic input and centre-exposure truth: NOT detection results');
exportgraphics(f,fullfile(out,'contactsheet.png'),'Resolution',120);close(f);
f=figure('Visible','off','Color','white','Position',[50 50 1200 700]);tiledlayout(2,2);
names={'variable_speed','stop_reverse','odom_dropout','combined'};
for j=1:4
    q=find(strcmp({index.profile},names{j}),1,'first');id=index(q).sequence_id;
    tr=jsondecode(fileread(fullfile(out,'evaluation_only','sequences',id,'geometry_truth.json')));
    ob=jsondecode(fileread(fullfile(out,'algorithm_inputs',id,'motion_input.json')));
    vals=nan(1,cfg.frames_per_sequence);for k=1:cfg.frames_per_sequence,if ob.frames(k).odometry_available,vals(k)=ob.frames(k).simulated_odometry_px;end,end
    nexttile;plot(0:cfg.frames_per_sequence-1,tr.true_longitudinal_offsets_px,'-o',0:cfg.frames_per_sequence-1,vals,'--s');grid on;
    xlabel('Frame index (zero based)');ylabel('Source-pixel displacement');title(strrep(names{j},'_',' '));legend('Truth (evaluation only)','Simulated odometry','Location','best');
end
sgtitle('Synthetic trajectories and missing odometry: not measured motion');
exportgraphics(f,fullfile(out,'trajectory_audit.png'),'Resolution',120);close(f);
end

function path=safeRaw(dataRoot,relative)
assert(~contains(relative,'..')&&~contains(relative,':')&&~contains(relative,'\')&&startsWith(relative,'mcnet/extracted/'));
path=char(java.io.File(fullfile(dataRoot,relative)).getCanonicalPath());
prefix=[dataRoot filesep];assert(startsWith(lower(path),lower(prefix)));assert(isfile(path));
end

function records=payloadManifest(out)
parts={'algorithm_inputs',fullfile('evaluation_only','sequences')};records=struct([]);n=0;
for p=1:numel(parts)
    files=dir(fullfile(out,parts{p},'**','*'));files=files(~[files.isdir]);
    names=string(fullfile({files.folder},{files.name}));[~,order]=sort(names);files=files(order);
    for k=1:numel(files)
        n=n+1;path=fullfile(files(k).folder,files(k).name);relative=path(numel(out)+2:end);
        entry=struct('path',strrep(relative,'\','/'),'bytes',files(k).bytes,'sha256',file_sha256(path));
        if n==1,records=entry;else,records(n)=entry;end %#ok<AGROW>
    end
end
end

function value=manifestDigest(manifest)
text='';for k=1:numel(manifest),text=[text manifest(k).path ':' manifest(k).sha256 newline];end %#ok<AGROW>
md=java.security.MessageDigest.getInstance('SHA-256');md.update(uint8(text));value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

function writeJson(path,value)
assert(~isfolder(path));fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true,'ConvertInfAndNaN',true));
end

function value=utcNow()
value=char(datetime('now','TimeZone','UTC','Format',"yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"));
end
