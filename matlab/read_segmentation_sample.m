function [I,M,V,meta] = read_segmentation_sample(imagePath,maskPath,targetHW,threshold)
% Preserve raw files; output [0,1] RGB, binary defect mask, and valid-pixel mask.
% Aspect-preserving downsampling only. No invented detail through upsampling.
% Padding must be excluded from loss and metrics using V.
if nargin<3,targetHW=[256 256];end
if nargin<4,threshold=128;end
rgb=imread(imagePath);
if ndims(rgb)==2,rgb=repmat(rgb,1,1,3);end
assert(isa(rgb,'uint8') && size(rgb,3)==3,'Expected uint8 RGB or grayscale input');
[raw,palette]=imread(maskPath);
if ~isempty(palette)
    raw=im2uint8(ind2gray(raw,palette));
end
if ndims(raw)==3
    assert(isequal(raw(:,:,1),raw(:,:,2)) && isequal(raw(:,:,1),raw(:,:,3)), ...
        'Color-coded labels require an explicit class mapping');
    raw=raw(:,:,1);
end
assert(isequal(size(rgb,[1 2]),size(raw,[1 2])),'Image/mask size mismatch');
if islogical(raw) || all(ismember(unique(raw),[0 1]))
    fg=logical(raw);
else
    assert(isa(raw,'uint8'),'Expected uint8 grayscale label');
    fg=raw>=threshold;
end
sourceHW=size(rgb,[1 2]);
foregroundBefore=nnz(fg);
scale=min([1 targetHW./sourceHW]);
resizedHW=max(1,round(sourceHW*scale));
rgb=imresize(rgb,resizedHW,'bilinear');fg=imresize(fg,resizedHW,'nearest');
offset=floor((targetHW-resizedHW)/2);
rows=offset(1)+(1:resizedHW(1));cols=offset(2)+(1:resizedHW(2));
I=zeros([targetHW 3],'single');M=false(targetHW);V=false(targetHW);
I(rows,cols,:)=single(rgb)/255;
M(rows,cols)=fg;V(rows,cols)=true;
meta.source_hw=sourceHW;meta.resized_hw=resizedHW;
meta.scale=scale;meta.offset_rc=offset;
meta.foreground_pixels_before=foregroundBefore;
meta.foreground_pixels_after=nnz(M);
meta.threshold=threshold;
end
