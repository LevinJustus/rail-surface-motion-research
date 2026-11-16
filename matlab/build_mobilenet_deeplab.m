function [net,cfg,architecture] = build_mobilenet_deeplab(configPath)
%BUILD_MOBILENET_DEEPLAB Project implementation of a mature segmentation model.
% Public, untrained MobileNetV2 backbone + a DeepLabv3+ style ASPP/decoder.
% This is neither a new algorithm nor a bitwise reproduction of MathWorks'
% deeplabv3plus constructor or the authors' TensorFlow implementation.
% MATLAB R2024b; Deep Learning, Computer Vision and Image Processing Toolboxes.
% No support-package installation, pretrained weights, private API or images.
%
% Architectural sources (structure reviewed; no vendor implementation copied):
% https://arxiv.org/abs/1802.02611 (Fig.2, sections3.1/4.1, tables1/2)
% https://github.com/tensorflow/models/blob/master/research/deeplab/model.py
% https://github.com/tensorflow/models/blob/master/research/deeplab/core/utils.py
%
% The single synthetic-zero PREDICT below checks geometry only, never trains
% or estimates accuracy. It preserves RNG and BN State after initialization.

root=fileparts(fileparts(mfilename('fullpath')));
if nargin<1
    configPath=fullfile(root,'configs','mobilenet_deeplab_scratch_v1.json');
elseif ~isfile(configPath)
    configPath=fullfile(root,configPath);
end
cfg=jsondecode(fileread(configPath));
validate_config(cfg);
inputSize=double(cfg.input_size_hwc(:)');
highName='block_16_project_BN';lowName='block_3_expand_relu';
highHW=inputSize(1:2)/cfg.output_stride;lowHW=inputSize(1:2)/4;

% Always create the public NONE graph afresh. initialize() alone would not
% erase pretrained values in an already populated network.
rng(cfg.random_seed,'twister');
graph=mobilenetv2('Weights','none');
allNames=string({graph.Layers.Name});
assert(all(ismember([string(highName) string(lowName)],allNames)), ...
    'build_mobilenet_deeplab:Endpoints','Expected R2024b backbone endpoints are absent');
keep=ancestor_names(graph,highName);
removed=allNames(~ismember(allNames,keep));
if ~isempty(removed),graph=removeLayers(graph,cellstr(removed));end
assert(ismember(string(lowName),keep));
inputMask=arrayfun(@(layer)isa(layer,'nnet.cnn.layer.ImageInputLayer'),graph.Layers);
assert(nnz(inputMask)==1,'build_mobilenet_deeplab:Input','Expected one RGB image input');
inputName=graph.Layers(inputMask).Name;
graph=replaceLayer(graph,inputName,imageInputLayer(inputSize, ...
    Normalization='none',Name=inputName));

% Suppress the last stride-2 step. The removed stride affects the dilation of
% subsequent DW convolutions, not that stride-suppressed layer itself.
changeNames={'block_13_depthwise','block_14_depthwise', ...
    'block_15_depthwise','block_16_depthwise'};
changes=struct([]);
for k=1:numel(changeNames)
    old=find_layer(graph,changeNames{k});
    assert(isa(old,'nnet.cnn.layer.GroupedConvolution2DLayer'), ...
        'build_mobilenet_deeplab:Depthwise','Unexpected backbone DW layer type');
    assert(isequal(double(old.FilterSize),[3 3]) && old.NumFiltersPerGroup==1);
    if k==1
        assert(isequal(double(old.Stride),[2 2]));dilation=1;
    else
        assert(isequal(double(old.Stride),[1 1]));dilation=2;
    end
    % R2024b infers NumChannelsPerGroup; it is not a constructor argument.
    layer=groupedConvolution2dLayer(old.FilterSize,old.NumFiltersPerGroup,old.NumGroups, ...
        Stride=1, ...
        DilationFactor=dilation,Padding='same',Name=old.Name, ...
        WeightsInitializer=old.WeightsInitializer,BiasInitializer=old.BiasInitializer, ...
        WeightLearnRateFactor=old.WeightLearnRateFactor,BiasLearnRateFactor=old.BiasLearnRateFactor, ...
        WeightL2Factor=old.WeightL2Factor,BiasL2Factor=old.BiasL2Factor);
    graph=replaceLayer(graph,old.Name,layer);
    row.layer=old.Name;row.old_stride=double(old.Stride);row.new_stride=[1 1];
    row.old_dilation=double(old.DilationFactor);row.new_dilation=[dilation dilation];
    row.new_padding='same';
    if isempty(changes),changes=row;else,changes(end+1)=row;end %#ok<AGROW>
end

% Recreate every BN through its public constructor so no running statistics,
% scale or offset can survive from a previously initialized backbone.
for k=1:numel(graph.Layers)
    layer=graph.Layers(k);
    if isa(layer,'nnet.cnn.layer.BatchNormalizationLayer')
        replacement=make_bn(layer.Name,cfg);
        replacement.ScaleLearnRateFactor=layer.ScaleLearnRateFactor;
        replacement.OffsetLearnRateFactor=layer.OffsetLearnRateFactor;
        graph=replaceLayer(graph,layer.Name,replacement);
    end
end
backboneNames=string({graph.Layers.Name});

% ASPP: 1x1 branch, three atrous DW/PW branches, image-level pooling branch.
graph=addLayers(graph,pointwise_block('dlv3p_aspp_1x1',cfg.aspp_channels,cfg));
graph=connectLayers(graph,highName,'dlv3p_aspp_1x1_conv');
branchEnds={'dlv3p_aspp_1x1_relu'};
for k=1:numel(cfg.atrous_rates)
    name=sprintf('dlv3p_aspp_rate%d',cfg.atrous_rates(k));
    graph=addLayers(graph,separable_block(name,cfg.aspp_channels,cfg.atrous_rates(k),cfg));
    graph=connectLayers(graph,highName,[name '_dw']);
    branchEnds{end+1}=[name '_pw_relu']; %#ok<AGROW>
end
pool=[globalAveragePooling2dLayer(Name='dlv3p_image_pool')
    pointwise_block('dlv3p_image_pool_projection',cfg.aspp_channels,cfg)
    make_resize('dlv3p_image_pool_resize',highHW)];
graph=addLayers(graph,pool);
graph=connectLayers(graph,highName,'dlv3p_image_pool');
branchEnds{end+1}='dlv3p_image_pool_resize';
graph=addLayers(graph,depthConcatenationLayer(numel(branchEnds),Name='dlv3p_aspp_concat'));
for k=1:numel(branchEnds)
    graph=connectLayers(graph,branchEnds{k},sprintf('dlv3p_aspp_concat/in%d',k));
end
projection=[pointwise_block('dlv3p_aspp_projection',cfg.aspp_channels,cfg)
    dropoutLayer(cfg.aspp_dropout,Name='dlv3p_aspp_dropout')
    make_resize('dlv3p_aspp_to_low',lowHW)];
graph=addLayers(graph,projection);
graph=connectLayers(graph,'dlv3p_aspp_concat','dlv3p_aspp_projection_conv');

% Decoder: OS4 low-level 1x1/48 + upsampled ASPP, two DW/PW 3x3/256 blocks.
graph=addLayers(graph,pointwise_block('dlv3p_low_projection',cfg.low_level_channels,cfg));
graph=connectLayers(graph,lowName,'dlv3p_low_projection_conv');
graph=addLayers(graph,depthConcatenationLayer(2,Name='dlv3p_decoder_concat'));
graph=connectLayers(graph,'dlv3p_aspp_to_low','dlv3p_decoder_concat/in1');
graph=connectLayers(graph,'dlv3p_low_projection_relu','dlv3p_decoder_concat/in2');
decoder=[separable_block('dlv3p_decoder_1',cfg.decoder_channels,1,cfg)
    separable_block('dlv3p_decoder_2',cfg.decoder_channels,1,cfg)
    convolution2dLayer(1,cfg.num_classes,Padding='same', ...
        WeightsInitializer='glorot',BiasInitializer='zeros', ...
        WeightLearnRateFactor=1,BiasLearnRateFactor=1,Name='dlv3p_logits_os4')
    make_resize('dlv3p_logits_full',inputSize(1:2))
    softmaxLayer(Name='dlv3p_softmax')];
graph=addLayers(graph,decoder);
graph=connectLayers(graph,'dlv3p_decoder_concat','dlv3p_decoder_1_dw');
names=string({graph.Layers.Name});
headNames=names(~ismember(names,backboneNames));
assert(nnz(arrayfun(@(layer)isa(layer,'nnet.cnn.layer.SoftmaxLayer'),graph.Layers))==1);
unset=assert_unset_graph(graph);
net=dlnetwork(graph,Initialize=false);
assert(~net.Initialized,'build_mobilenet_deeplab:Initialization','Graph unexpectedly initialized');
net=initialize(net);
assert(net.Initialized && numel(net.InputNames)==1 && numel(net.OutputNames)==1);
assert(strcmp(string(net.OutputNames{1}),'dlv3p_softmax'));
assert_finite_parameters(net);

architecture.schema_version='mobilenet_deeplab_architecture_v1';
architecture.implementation='Project public-layer implementation of mature MobileNetV2/DeepLabv3+ structure, not a new algorithm or vendor/author bitwise reproduction';
architecture.matlab_version=version;
architecture.backbone_constructor='mobilenetv2(''Weights'',''none'')';
architecture.pretrained_weights_used=false;
architecture.random_seed=cfg.random_seed;
architecture.initialization='Fresh unset graph, seeded before construction, initialize exactly once';
architecture.unset_graph_check=unset;
architecture.input_layer=inputName;architecture.input_size_hwc=inputSize;
architecture.input_normalization='none';architecture.external_input_units='single image/255 once';
architecture.classes_in_channel_order=cellstr(string(cfg.classes_in_channel_order(:)));
architecture.high_level_endpoint=highName;architecture.high_level_layer=highName;
architecture.low_level_endpoint=lowName;architecture.low_level_layer=lowName;
architecture.logits_layer='dlv3p_logits_full';architecture.softmax_layer='dlv3p_softmax';
architecture.output_stride=cfg.output_stride;architecture.decoder_output_stride=4;
architecture.backbone_layer_names=cellstr(backboneNames);
architecture.retained_backbone_layer_names=cellstr(backboneNames);
architecture.head_layer_names=cellstr(headNames);
architecture.removed_classification_tail_layer_names=cellstr(removed);
architecture.backbone_dilation_changes=changes;
architecture.aspp_rates=double(cfg.atrous_rates(:)');
architecture.aspp_channels=cfg.aspp_channels;architecture.aspp_branch_count=numel(branchEnds);
architecture.image_level_pooling=true;architecture.aspp_dropout=cfg.aspp_dropout;
architecture.low_level_channels=cfg.low_level_channels;architecture.decoder_channels=cfg.decoder_channels;
architecture.separable_order='Depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> ReLU';
architecture.bn_epsilon=cfg.bn_epsilon;architecture.bn_mean_decay=cfg.bn_mean_decay;
architecture.bn_variance_decay=cfg.bn_variance_decay;
architecture.bn_count=nnz(arrayfun(@(layer)isa(layer,'nnet.cnn.layer.BatchNormalizationLayer'),net.Layers));
architecture.learn_rate_factor_policy='Public untrained backbone factors preserved, including any zero factors; new head weight/bias and BN scale/offset factors are 1. dlnetwork Adam applies factors; do not rescale gradients manually.';
architecture.resize_method='bilinear';architecture.resize_geometry='half-pixel';
architecture.parameter_count=sum(cellfun(@numel,net.Learnables.Value));
isBackbone=ismember(string(net.Learnables.Layer),backboneNames);
architecture.backbone_parameter_count=sum(cellfun(@numel,net.Learnables.Value(isBackbone)));
architecture.head_parameter_count=sum(cellfun(@numel,net.Learnables.Value(~isBackbone)));
factors=zeros(height(net.Learnables),1);counts=cellfun(@numel,net.Learnables.Value);
nonunit=struct([]);
for k=1:height(net.Learnables)
    layerName=string(net.Learnables.Layer(k));parameterName=string(net.Learnables.Parameter(k));
    factors(k)=getLearnRateFactor(net,layerName,parameterName);
    if factors(k)~=1
        factorRow.layer=char(layerName);factorRow.parameter=char(parameterName);
        factorRow.learn_rate_factor=factors(k);factorRow.parameter_count=counts(k);
        factorRow.is_backbone=isBackbone(k);
        if isempty(nonunit),nonunit=factorRow;else,nonunit(end+1)=factorRow;end %#ok<AGROW>
    end
end
architecture.nonunit_learn_rate_factors=nonunit;
architecture.nonunit_learn_rate_factor_count=nnz(factors~=1);
architecture.nonzero_factor_parameter_count=sum(counts(factors~=0));
architecture.backbone_nonzero_factor_parameter_count=sum(counts(isBackbone & factors~=0));
architecture.head_nonzero_factor_parameter_count=sum(counts(~isBackbone & factors~=0));
architecture.layer_count=numel(net.Layers);
architecture.geometry=check_geometry(net,inputSize,highName,lowName);
architecture.limitations={ ...
    'At OS16/input256 the high map is16x16: rate18 has padding-only noncentral 3x3 samples; canonical rates are fixed, not validation-tuned.', ...
    'R2024b public resize2dLayer uses half-pixel bilinear; author TensorFlow code uses align_corners=True.', ...
    'Backbone endpoints, BN decay, initialization and trainable head biases are explicit project choices, not author checkpoint reproduction; public backbone learn-rate factors are retained.', ...
    'No accuracy, convergence, realtime, K230 conversion or hardware claim follows from construction/zero-input geometry.'};
architecture.sources={ ...
    'https://arxiv.org/abs/1802.02611', ...
    'https://github.com/tensorflow/models/blob/master/research/deeplab/model.py', ...
    'https://github.com/tensorflow/models/blob/master/research/deeplab/core/utils.py', ...
    'https://www.mathworks.com/help/deeplearning/ref/mobilenetv2.html', ...
    'https://www.mathworks.com/help/images/ref/nnet.cnn.layer.resize2dlayer.html'};
end

function validate_config(cfg)
fields={'input_size_hwc','num_classes','classes_in_channel_order','random_seed', ...
    'output_stride','atrous_rates','aspp_channels','low_level_channels','decoder_channels', ...
    'aspp_dropout','bn_epsilon','bn_mean_decay','bn_variance_decay'};
assert(isstruct(cfg)&&isscalar(cfg)&&all(isfield(cfg,fields)), ...
    'build_mobilenet_deeplab:Config','Missing required model configuration fields');
assert(isequal(double(cfg.input_size_hwc(:)'),[256 256 3])&&cfg.num_classes==2);
assert(isequal(string(cfg.classes_in_channel_order(:)),["background";"defect"]));
assert(isnumeric(cfg.random_seed)&&isscalar(cfg.random_seed)&&isfinite(cfg.random_seed)&& ...
    cfg.random_seed>=0&&cfg.random_seed==fix(cfg.random_seed)&&cfg.random_seed<=2^32-1);
assert(cfg.output_stride==16&&isequal(double(cfg.atrous_rates(:)'),[6 12 18]));
assert(cfg.aspp_channels==256&&cfg.low_level_channels==48&&cfg.decoder_channels==256);
assert(cfg.aspp_dropout==0.1&&cfg.bn_epsilon==1e-5&&cfg.bn_mean_decay==0.1&&cfg.bn_variance_decay==0.1);
if isfield(cfg,'backbone_weights'),assert(strcmp(string(cfg.backbone_weights),'none'));end
if isfield(cfg,'initialization'),assert(strcmp(string(cfg.initialization),'random'));end
if isfield(cfg,'high_level_endpoint'),assert(strcmp(string(cfg.high_level_endpoint),'block_16_project_BN'));end
if isfield(cfg,'low_level_endpoint'),assert(strcmp(string(cfg.low_level_endpoint),'block_3_expand_relu'));end
end

function keep=ancestor_names(graph,endpoint)
% Follow public Connections backwards, retaining exactly the endpoint ancestry.
source=regexprep(string(graph.Connections.Source),'/.*$','');
destination=regexprep(string(graph.Connections.Destination),'/.*$','');
keep=string(endpoint);
while true
    expanded=unique([keep(:);source(ismember(destination,keep))],'stable');
    if numel(expanded)==numel(keep),break;end
    keep=expanded;
end
end

function layer=find_layer(graph,name)
mask=strcmp({graph.Layers.Name},name);assert(nnz(mask)==1);
layer=graph.Layers(mask);
end

function layer=make_bn(name,cfg)
layer=batchNormalizationLayer(Name=name,Epsilon=cfg.bn_epsilon, ...
    MeanDecay=cfg.bn_mean_decay,VarianceDecay=cfg.bn_variance_decay, ...
    ScaleInitializer='ones',OffsetInitializer='zeros', ...
    ScaleLearnRateFactor=1,OffsetLearnRateFactor=1);
end

function layers=pointwise_block(name,channels,cfg)
layers=[convolution2dLayer(1,channels,Padding='same', ...
    WeightsInitializer='he',BiasInitializer='zeros', ...
    WeightLearnRateFactor=1,BiasLearnRateFactor=1,Name=[name '_conv'])
    make_bn([name '_bn'],cfg)
    reluLayer(Name=[name '_relu'])];
end

function layers=separable_block(name,channels,rate,cfg)
layers=[groupedConvolution2dLayer(3,1,'channel-wise',Padding='same', ...
    DilationFactor=rate,Stride=1,WeightsInitializer='he',BiasInitializer='zeros', ...
    WeightLearnRateFactor=1,BiasLearnRateFactor=1,Name=[name '_dw'])
    make_bn([name '_dw_bn'],cfg)
    reluLayer(Name=[name '_dw_relu'])
    convolution2dLayer(1,channels,Padding='same',WeightsInitializer='he',BiasInitializer='zeros', ...
    WeightLearnRateFactor=1,BiasLearnRateFactor=1,Name=[name '_pw'])
    make_bn([name '_pw_bn'],cfg)
    reluLayer(Name=[name '_pw_relu'])];
end

function layer=make_resize(name,hw)
layer=resize2dLayer(OutputSize=double(hw),Method='bilinear', ...
    GeometricTransformMode='half-pixel',Name=name);
end

function report=assert_unset_graph(graph)
count=0;bnCount=0;
for k=1:numel(graph.Layers)
    layer=graph.Layers(k);
    for field={'Weights','Bias','Scale','Offset'}
        if isprop(layer,field{1})
            assert(isempty(layer.(field{1})),'build_mobilenet_deeplab:NonemptyParameters', ...
                'Untrained graph contains populated parameter %s/%s',layer.Name,field{1});
            count=count+1;
        end
    end
    if isa(layer,'nnet.cnn.layer.BatchNormalizationLayer')
        assert(isempty(layer.TrainedMean)&&isempty(layer.TrainedVariance));bnCount=bnCount+1;
    end
end
assert(count>0&&bnCount>0);
report.all_learnable_arrays_unset=true;report.checked_array_count=count;
report.all_bn_statistics_unset=true;report.checked_bn_count=bnCount;
end

function assert_finite_parameters(net)
for k=1:height(net.Learnables)
    value=extractdata(net.Learnables.Value{k});
    assert(~isempty(value)&&isa(value,'single')&&all(isfinite(value),'all'));
end
for k=1:height(net.State)
    value=extractdata(net.State.Value{k});assert(~isempty(value)&&all(isfinite(value),'all'));
end
end

function geometry=check_geometry(net,inputSize,highName,lowName)
% Predict has no dropout/BN update. Retain the post-initialization RNG even
% if an implementation error causes this diagnostic to throw.
savedRng=rng;restoreRng=onCleanup(@()rng(savedRng)); %#ok<NASGU>
savedState=net.State;
X=dlarray(zeros([inputSize 1],'single'),'SSCB');
[low,high,logits,prob]=predict(net,X,Outputs={lowName,highName,'dlv3p_logits_full','dlv3p_softmax'});
afterRng=rng;assert(isequaln(afterRng,savedRng));
assert(isequaln(net.State,savedState),'build_mobilenet_deeplab:State','Geometry predict changed BN State');
expected={[64 64 144 1],[16 16 320 1],[256 256 2 1],[256 256 2 1]};
outputs={low,high,logits,prob};names={'low_level_hwcB','high_level_hwcB','logits_hwcB','probability_hwcB'};
for k=1:numel(outputs)
    shape=[size(outputs{k},1) size(outputs{k},2) size(outputs{k},3) size(outputs{k},4)];
    assert(isequal(shape,expected{k}),'build_mobilenet_deeplab:Geometry','Unexpected shape at %s',names{k});
    value=extractdata(outputs{k});assert(all(isfinite(value),'all'));
    geometry.(names{k})=shape;
end
p=extractdata(prob);sumError=double(max(abs(sum(p,3)-1),[],'all'));
assert(all(p>=0 & p<=1,'all')&&sumError<=1e-5);
geometry.status='PASS';geometry.input='Synthetic zeros only, no source images';
geometry.probability_max_channel_sum_error=sumError;
geometry.predict_preserved_bn_state=true;geometry.predict_preserved_rng=true;
geometry.optimization_updates=0;
end
