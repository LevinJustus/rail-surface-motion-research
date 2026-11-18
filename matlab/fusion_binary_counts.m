function [counts,transitions]=fusion_binary_counts(probability,target,singleProbability,threshold)
% Evaluation-only helper. Never called by motion_fusion_step.
assert(isequal(size(probability),size(target),size(singleProbability))&&islogical(target));
b=probability>=threshold;s=singleProbability>=threshold;t=target;
counts=[nnz(b&t) nnz(b&~t) nnz(~b&t) nnz(~b&~t)];
transitions=[nnz(~s&t&b) nnz(s&~t&~b) nnz(s&t&~b) nnz(~s&~t&b)];
end
