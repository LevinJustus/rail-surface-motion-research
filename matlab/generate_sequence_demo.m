function generate_sequence_demo(dataRoot,outRoot)
% Research input demonstration only. No trained network and no detection result.
% Requires the NEU author data downloaded for this project; MATLAB base functions.
% Pixels are not millimetres. All evaluation truth is kept separate from inputs.
root = fileparts(fileparts(mfilename('fullpath')));
if nargin<2,outRoot=fullfile(root,'runs','sequence_demo');end
if ~exist(outRoot,'dir'),mkdir(outRoot);end
base=outRoot;
data = fullfile(dataRoot,'neu_rsdds','extracted');
sourceId = '10';
imagePath = fullfile(data,'2D_Image','INPUT_IMG',[sourceId '.bmp']);
maskPath = fullfile(data,'Ground Truth','Ground Truth',[sourceId '.png']);
rgb = imread(imagePath);
[mask,palette] = imread(maskPath);
if ~isempty(palette)
    % These PNGs may be indexed: 0/1 are palette indices, not image intensities.
    luminance=mean(palette,2);
    if isinteger(mask), idx=double(mask)+1; else, idx=double(mask); end
    mask=uint8(round(255*reshape(luminance(idx(:)),size(mask))));
end
if islogical(mask), mask=uint8(mask)*255; end
if ndims(mask)==3, mask=mask(:,:,1); end
assert(isequal(size(rgb,[1 2]),size(mask,[1 2])),'Image-mask mismatch');
assert(all(ismember(unique(mask),[0 255])),'Unexpected label values');
[srcH,srcW,~]=size(rgb);
frameH=256; frameW=256; nFrames=5; stepPx=32; margin=16;
assert(srcH>=frameH+(nFrames-1)*stepPx+2*margin && srcW>=frameW+2*margin);
[fgY,fgX]=find(mask>0);
originX=round(median(fgX)-(frameW-1)/2);
originY=round(median(fgY)-(frameH-1)/2-(nFrames-1)*stepPx/2);
originX=max(margin+1,min(originX,srcW-frameW-margin+1));
originY=max(margin+1,min(originY,srcH-frameH-(nFrames-1)*stepPx-margin+1));

demoRoot=fullfile(base,'motion_sequence_demo');
inputRoot=fullfile(demoRoot,'algorithm_inputs');
truthRoot=fullfile(demoRoot,'evaluation_only');
if ~exist(inputRoot,'dir'), mkdir(inputRoot); end
if ~exist(truthRoot,'dir'), mkdir(truthRoot); end

[xx,yy]=meshgrid(1:frameW,1:frameH);
xy=[xx(:)';yy(:)']; centre=[(frameW+1)/2;(frameH+1)/2];
anglesDeg=[0 .5 0 -.5 0]; lateralPx=[0 2 0 -2 0];
frames=zeros(frameH,frameW,3,nFrames,'uint8');
masks=zeros(frameH,frameW,nFrames,'uint8');
frameToSource=zeros(3,3,nFrames);
inputRecords=struct([]);
for k=1:nFrames
    a=anglesDeg(k)*pi/180;
    A=[cos(a) -sin(a);sin(a) cos(a)];
    b=[originX-1+lateralPx(k);originY-1+(k-1)*stepPx]+centre-A*centre;
    coords=A*xy+b;
    assert(all(coords(1,:)>=1 & coords(1,:)<=srcW & coords(2,:)>=1 & coords(2,:)<=srcH),'Out-of-source view');
    sx=reshape(coords(1,:),frameH,frameW); sy=reshape(coords(2,:),frameH,frameW);
    for c=1:3
        frames(:,:,c,k)=uint8(round(interp2(double(rgb(:,:,c)),sx,sy,'linear')));
    end
    masks(:,:,k)=uint8(interp2(double(mask),sx,sy,'nearest'));
    assert(all(ismember(unique(masks(:,:,k)),[0 255])));
    frameToSource(:,:,k)=[A b;0 0 1];
    filename=sprintf('frame_%02d.png',k);
    imwrite(frames(:,:,:,k),fullfile(inputRoot,filename));
    imwrite(masks(:,:,k),fullfile(truthRoot,sprintf('mask_%02d.png',k)));
    inputRecords(k).frame=filename;
    inputRecords(k).time_s=(k-1)*0.05;
    % Pixel-equivalent odometry surrogate: +2%% scale error, one-pixel quantization.
    inputRecords(k).odometry_offset_px=round((k-1)*stepPx*1.02);
end
inputs.description='Synthetic short-range views from one real image, not real video. Pixel-equivalent odometry, not measured encoder data.';
inputs.virtual_sampling_interval_s=0.05;
inputs.frame_size_wh=[frameW frameH];
inputs.frames=inputRecords;
write_json(fullfile(inputRoot,'motion_input.json'),inputs);
truth.source_id=sourceId;
truth.true_longitudinal_offset_px=(0:nFrames-1)*stepPx;
truth.lateral_offset_px=lateralPx;
truth.rotation_deg=anglesDeg;
truth.frame_to_source_3x3=frameToSource;
truth.coordinate_convention='MATLAB 1-based pixel centres. [x_source;y_source;1]=T*[x_frame;y_frame;1].';
truth.note='Evaluation only. No masks or exact transforms may be read by the tested algorithm.';
write_json(fullfile(truthRoot,'geometry_truth.json'),truth);
save(fullfile(truthRoot,'geometry_truth.mat'),'frameToSource','originX','originY','stepPx','sourceId');

f=figure('Visible','off','Color','white','Position',[100 100 1450 650], ...
    'DefaultAxesFontName','Microsoft YaHei','DefaultTextFontName','Microsoft YaHei');
tiledlayout(2,nFrames,'TileSpacing','compact','Padding','compact');
for k=1:nFrames
    nexttile(k); image(frames(:,:,:,k)); axis image off;
    title(sprintf('第%d帧 | 模拟位移 %d px',k,inputRecords(k).odometry_offset_px),'FontSize',11);
    nexttile(nFrames+k); imagesc(masks(:,:,k),[0 255]); axis image off;
    title('缺陷真值（仅供评价）','FontSize',11);
end
colormap(f,gray(256));
sgtitle('真实轨面上的合成连续帧输入示例：非检测结果','FontSize',16,'FontName','Microsoft YaHei');
exportgraphics(f,fullfile(base,'合成连续帧输入示例.png'),'Resolution',120);
close(f);
fprintf('SEQUENCE_DEMO_OK source=%s frames=%d size=%dx%d step=%dpx\n',sourceId,nFrames,frameW,frameH,stepPx);
fprintf('Foreground pixels per frame: '); fprintf('%d ',sum(masks>0,[1 2]));fprintf('\n');
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0);
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(value,PrettyPrint=true));
end
