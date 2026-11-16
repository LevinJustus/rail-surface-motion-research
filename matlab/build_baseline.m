function [net,cfg] = build_baseline(configPath)
%BUILD_BASELINE Small established U-Net baseline, not a proposed new method.
% MATLAB R2024b; requires Computer Vision and Deep Learning Toolboxes.
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<1, configPath=fullfile(root,'configs','baseline.json'); end
cfg=jsondecode(fileread(configPath));
rng(cfg.random_seed,'twister');
assert(cfg.num_classes==numel(cfg.classes_in_channel_order));
net=unet(double(cfg.input_size_hwc(:)'),cfg.num_classes, ...
    EncoderDepth=cfg.encoder_depth, ...
    NumFirstEncoderFilters=cfg.num_first_encoder_filters, ...
    ConvolutionPadding=cfg.convolution_padding);
% Input scaling is owned by read_segmentation_sample, never inferred twice.
inputName=net.InputNames{1};
net=replaceLayer(net,inputName,imageInputLayer(double(cfg.input_size_hwc(:)'), ...
    Normalization='none',Name=inputName));
net=initialize(net);
end
