function infer_continuous(run, inputRoot, checkpointPath, checkpointSha)
% Frozen CPU inference from 256x256 uint8 RGB files in frame_input_manifest.csv.
% MATLAB R2024b on Windows with Deep Learning Toolbox; no JVM required.
if isfile(fullfile(run,'prediction_index.csv'))
    error('Choose an empty output directory');
end
[code,result] = system(sprintf('certutil -hashfile "%s" SHA256',checkpointPath));
assert(code==0 && contains(lower(result),lower(checkpointSha)),'Checkpoint SHA mismatch');
s = load(checkpointPath,'checkpoint');
c = s.checkpoint;
net = c.net;
assert(isa(net,'dlnetwork') && c.initialization_seed==20261114);
assert(isequal(string(c.class_channel_order(:)),["background";"defect"]));
learnables = net.Learnables;
state = net.State;
index = readtable(fullfile(inputRoot,'frame_input_manifest.csv'),'TextType','string');
mkdir(fullfile(run,'raw'));
fid = fopen(fullfile(run,'prediction_index.csv'),'w');
assert(fid>0);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'source_video,frame_index,raw_path\n');
for k = 1:height(index)
    rgb = imread(fullfile(inputRoot,index.rgb_path(k)));
    assert(isa(rgb,'uint8') && isequal(size(rgb),[256 256 3]));
    p = extractdata(predict(net,dlarray(single(rgb)/255,'SSCB')));
    assert(isequal(size(p),[256 256 2]) && all(isfinite(p),'all'));
    assert(min(p,[],'all')>=0 && max(p,[],'all')<=1 && max(abs(sum(p,3)-1),[],'all')<=1e-4);
    name = sprintf('raw/prob_%06d.f32',k);
    binary = fopen(fullfile(run,name),'w','ieee-le');
    assert(binary>0);
    count = fwrite(binary,p(:,:,2),'single');
    fclose(binary);
    assert(count==256^2);
    fprintf(fid,'%s,%d,%s\n',index.source_video(k),index.frame_index_zero_based(k),name);
    if mod(k,60)==0
        fprintf('Completed %d of %d inputs\n',k,height(index));
    end
end
clear cleanup
assert(isequaln(learnables,net.Learnables) && isequaln(state,net.State));
status = struct('status','PASS','forward_calls',height(index),'optimizer_updates',0,...
    'frozen_state_and_parameters',true,'model_sha256',lower(checkpointSha),'matlab_version',version);
fid = fopen(fullfile(run,'inference_status.json'),'w');
assert(fid>0);
fprintf(fid,'%s\n',jsonencode(status));
fclose(fid);
end
