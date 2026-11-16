function [loss,bce,dice,p] = masked_segmentation_loss(logits,T,V,lossCfg)
%MASKED_SEGMENTATION_LOSS Frozen stage1 per-image masked BCE/soft-Dice loss.
% logits: H-by-W-by-2-by-B formatted dlarray, labels SSCB, PRE-softmax.
%         Channel 1 is background; channel 2 is defect.
% T,V:    H-by-W-by-1-by-B binary numeric arrays or dlarrays. T is truth;
%         V selects real image pixels and must be nonempty in every image.
% loss:   Mean of each image's mixed loss. Nonempty target: wBCE*BCE+wD*Dice.
%         Empty target: BCE at weight 1, independently of configured weights.
% bce,dice: Unweighted per-image component means, including empty targets.
% p:      H-by-W-by-1-by-B defect probabilities, still formatted SSCB.
%
% All logits-to-loss operations remain in the differentiable dlarray graph.
% Only constant targets/masks and legality assertions inspect extracted data.
% This function performs no thresholding, resizing, or second sigmoid.
if nargin<4 || isempty(lossCfg)
    lossCfg=struct('bce_weight_positive_target',0.5, ...
        'dice_weight_positive_target',0.5,'dice_smoothing',1e-6);
end
validate_config(lossCfg);
if ~isa(logits,'dlarray') || ~strcmp(dims(logits),'SSCB')
    error('rail:loss:LogitsFormat','logits must be a formatted SSCB dlarray.');
end
shape=size(logits,[1 2 3 4]);
if shape(3)~=2 || any(shape([1 2 4])<1) || ndims(logits)>4
    error('rail:loss:LogitsShape','logits must have size H-by-W-by-2-by-B.');
end
if ~ismember(underlyingType(logits),{'single','double'})
    error('rail:loss:LogitsType','logits must have single or double underlying type.');
end
% This extracted copy is used only for the input assertion, never the loss.
logitCheck=cpu_data(logits);
if ~isreal(logitCheck) || any(~isfinite(logitCheck),'all')
    error('rail:loss:NonfiniteLogits','logits must be real and finite.');
end
expected=[shape(1:2) 1 shape(4)];
target=checked_binary(T,expected,'Target');
valid=checked_binary(V,expected,'ValidMask');
validCountData=sum(valid,[1 2]);
if any(validCountData(:)==0)
    error('rail:loss:EmptyValidRegion','Each image must contain at least one valid pixel.');
end
hasForegroundData=sum(target.*valid,[1 2])>0;
% Targets are constants. cast(...,'like',logits) matches dtype/device without
% extracting or reconstructing logits, and retains the SSCB constant labels.
target=cast(dlarray(target,'SSCB'),'like',logits);
valid=cast(dlarray(valid,'SSCB'),'like',logits);
hasForeground=cast(dlarray(single(hasForegroundData),'SSCB'),'like',logits);

% Stable two-class log-softmax; exp() receives only nonpositive arguments.
shifted=logits-max(logits,[],3);
logProb=shifted-log(sum(exp(shifted),3));
p=exp(logProb(:,:,2,:));
validCount=sum(valid,[1 2]);
bcePerImage=-sum(valid.*(target.*logProb(:,:,2,:) + ...
    (1-target).*logProb(:,:,1,:)),[1 2])./validCount;
smooth=lossCfg.dice_smoothing;
dicePerImage=1-(2*sum(valid.*target.*p,[1 2])+smooth)./ ...
    (sum(valid.*p,[1 2])+sum(valid.*target,[1 2])+smooth);
mixed=lossCfg.bce_weight_positive_target*bcePerImage + ...
    lossCfg.dice_weight_positive_target*dicePerImage;
perImage=hasForeground.*mixed+(1-hasForeground).*bcePerImage;
loss=mean(perImage,'all');
bce=mean(bcePerImage,'all');
dice=mean(dicePerImage,'all');
end

function data=checked_binary(value,expected,name)
data=cpu_data(value);
if ~isnumeric(data) && ~islogical(data)
    error(['rail:loss:' name 'Values'],'%s must be a binary numeric array or dlarray.',name);
end
if ndims(data)>4 || ~isequal(size(data,[1 2 3 4]),expected)
    error(['rail:loss:' name 'Shape'],'%s must have size H-by-W-by-1-by-B.',name);
end
if ~isreal(data) || any(~isfinite(data),'all') || any(data~=0 & data~=1,'all')
    error(['rail:loss:' name 'Values'],'%s must contain only finite binary 0/1 values.',name);
end
% Binary integers/logicals are accepted without altering their semantics.
% Matching the logits precision/device is performed by cast at the call site.
if ~isa(data,'single') && ~isa(data,'double')
    data=single(data);
end
end

function data=cpu_data(value)
if isa(value,'dlarray'), data=extractdata(value); else, data=value; end
if isa(data,'gpuArray'), data=gather(data); end
end

function validate_config(cfg)
required={'bce_weight_positive_target','dice_weight_positive_target','dice_smoothing'};
if ~isstruct(cfg) || ~isscalar(cfg) || ~all(isfield(cfg,required))
    error('rail:loss:Config','lossCfg must define BCE/Dice weights and dice_smoothing.');
end
w1=cfg.bce_weight_positive_target;w2=cfg.dice_weight_positive_target;
epsDice=cfg.dice_smoothing;
if ~valid_scalar(w1) || ~valid_scalar(w2) || w1<0 || w2<0 || ...
        abs(w1+w2-1)>1e-12 || ~valid_scalar(epsDice) || epsDice<=0
    error('rail:loss:Config','Loss weights must be nonnegative and sum to one; smoothing must be positive.');
end
end

function tf=valid_scalar(x)
tf=isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x);
end
