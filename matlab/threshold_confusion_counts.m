function C = threshold_confusion_counts(P,T,V,thresholds)
% Counts are [TP FP FN TN], per image and threshold; padding never counts.
assert(isequal(size(P),size(T),size(V)) && all(isfinite(P),'all'));
assert(all(P>=0 & P<=1,'all') && all(ismember(T(:),[0 1])) && all(ismember(V(:),[0 1])));
assert(isvector(thresholds) && all(diff(thresholds)>0) && all(thresholds>=0 & thresholds<=1));
n=size(P,3);C=zeros(n,numel(thresholds),4);
for i=1:n
    valid=logical(V(:,:,i));target=logical(T(:,:,i));p=double(P(:,:,i));
    fg=nnz(valid & target);bg=nnz(valid & ~target);
    for j=1:numel(thresholds)
        pred=p>=thresholds(j);
        tp=nnz(valid & target & pred);fp=nnz(valid & ~target & pred);
        C(i,j,:)=[tp fp fg-tp bg-fp];
    end
end
assert(all(C>=0,'all') && all(diff(C(:,:,1),1,2)<=0,'all') && all(diff(C(:,:,2),1,2)<=0,'all'));
end
