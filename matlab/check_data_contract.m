function check_data_contract()
% Meaningful checks for label encoding and padding, independent of real test images.
root=fileparts(fileparts(mfilename('fullpath')));
folder=fullfile(root,'runs','data_contract_check');
if ~exist(folder,'dir'),mkdir(folder);end
image=uint8(zeros(2,4,3));image(:,4,:)=255;
raw=uint8(repmat([0 127 128 255],2,1));
ip=fullfile(folder,'input.png');mp=fullfile(folder,'label.png');
imwrite(image,ip);imwrite(raw,mp);
[I,M,V,meta]=read_segmentation_sample(ip,mp,[8 8],128);
assert(nnz(V)==8 && nnz(M)==4,'Grayscale label threshold or valid-pixel mapping failed');
assert(all(I>=0 & I<=1,'all') && max(I,[],'all')==1);
assert(meta.scale==1 && isequal(meta.resized_hw,[2 4]),'Small source was upscaled');
logicalMask=logical(repmat([0 0 1 1],2,1));imwrite(logicalMask,mp);
[~,L,W]=read_segmentation_sample(ip,mp,[8 8],128);
assert(isequal(M,L) && isequal(V,W),'1-bit and grayscale encodings disagree');
assert(~any(M(~V)),'Padding was incorrectly labeled foreground');
fprintf('DATA_CONTRACT_OK gray128 logicalRGB01 noUpsampling validPadding\n');
end
