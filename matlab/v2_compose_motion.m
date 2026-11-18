function paths=v2_compose_motion(edges,cfg)
% Chronological path products; discarded probability mass is never restored.
paths=struct('available',false,'shifts',{{}},'weights',zeros(0,1),'retained_mass',0);
if isempty(edges)||any(~cellfun(@(e)e.available,edges)),return;end
shifts={zeros(0,2)};weights=1;
for k=1:numel(edges)
    candidates=edges{k}.candidates;n=numel(weights)*size(candidates,1);next=cell(n,1);w=zeros(n,1);total=zeros(n,2);index=0;
    for j=1:numel(weights)
        for c=1:size(candidates,1)
            index=index+1;next{index}=[shifts{j};candidates(c,1:2)];w(index)=weights(j)*candidates(c,3);total(index,:)=sum(next{index},1);
        end
    end
    [~,order]=sortrows([-w,total(:,2),total(:,1),(1:n)'],[1 2 3 4]);
    count=find(cumsum(w(order))>=cfg.candidate_mass_target,1);if isempty(count),count=n;end
    keep=order(1:min(count,cfg.maximum_candidates_per_history));shifts=next(keep);weights=w(keep);
end
paths.available=true;paths.shifts=shifts;paths.weights=weights;paths.retained_mass=sum(weights);
assert(paths.retained_mass<=1+1e-12&&all(weights>0));
end
