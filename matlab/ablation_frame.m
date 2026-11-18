function [outputs,state]=ablation_frame(images,P,edges,stressEdges,base,cfg,constant,state)
% Prefix-only C5 arithmetic. base contains corresponding frozen causal outputs.
k=numel(P);assert(numel(images)==k&&numel(edges)==k-1&&numel(stressEdges)==7);
if isempty(state),state=struct('clean',[],'polluted',[]);end
methods={'S','U','V','P','GV','GVP','GC','R'};
scenarios={'bias_m6','bias_m2','bias_p2','bias_p6','alternating_1','alternating_3','all_missing'};
template=struct('scenario','','method','','parent_method','','probability',single([]));outputs=repmat(template,79,1);oi=0;
gray=cellfun(@motion_gray,images,'UniformOutput',false);current=double(P{k});n3=min(3,k);start3=k-n3+1;
contexts=cell(1,max(0,k-1));
for j=max(1,k-4):k-1
    contexts{j}=ablation_history_context(P{j},gray{j},current,gray{k},edges(j:k-1),cfg,'visual');
end
for scenario=1:7
    prior=current;candidate=current;
    for j=start3:k-1
        c=contexts{j};path=stressEdges{scenario}(j:k-1);
        z=prod(cellfun(@(e)e.observed_factor,path));
        d=c.history-current;d(~c.valid)=0;
        g=c.Q*c.photo.*c.confidence;candidate=candidate+(1/n3)*z*g.*d;
        available=all(cellfun(@(e)e.prior.available,path));
        if available
            shifts=cell2mat(cellfun(@(e)[e.prior.dx e.prior.dy],path,'UniformOutput',false));
            [h,v]=warp_motion_history(P{j},shifts);d=h-current;d(~v)=0;prior=prior+(1/n3)*d;
        end
    end
    for m=methods
        value=base.(m{1});if strcmp(m{1},'P'),value=prior;elseif strcmp(m{1},'GVP'),value=candidate;end
        emit(scenarios{scenario},m{1},m{1},value);
    end
end
for window=[1 5]
    n=min(window,k);start=k-n+1;V=current;GV=current;GVP=current;
    for j=start:k-1
        c=contexts{j};d=c.history-current;d(~c.valid)=0;g=c.Q*c.photo.*c.confidence;
        V=V+(1/n)*d;GV=GV+(1/n)*g.*d;GVP=GVP+(1/n)*c.z*g.*d;
    end
    for m={'S','V','GV','GVP'}
        if strcmp(m{1},'S'),value=current;elseif strcmp(m{1},'V'),value=V;elseif strcmp(m{1},'GV'),value=GV;else,value=GVP;end
        emit(sprintf('window_%d',window),m{1},m{1},value);
    end
end
for removal={'photo','confidence','cycle','peak_margin'}
    value=current;
    for j=start3:k-1
        c=contexts{j};q=c.Q;photo=c.photo;confidence=c.confidence;
        switch removal{1}
            case 'photo',photo=ones(256);
            case 'confidence',confidence=ones(256);
            case 'cycle',q=c.Qnc;
            case 'peak_margin',q=c.Qnm;
        end
        g=q*photo.*confidence;g(~c.valid)=0;d=c.history-current;d(~c.valid)=0;value=value+(1/n3)*g.*d;
    end
    emit(['remove_' removal{1}],'GV','GV',value);
end
value=current;
for j=start3:k-1
    c=ablation_history_context(P{j},gray{j},current,gray{k},edges(j:k-1),cfg,'prior');
    d=c.history-current;d(~c.valid)=0;g=c.Q*c.photo.*c.confidence;value=value+(1/n3)*c.z*g.*d;
end
emit('prior_alignment','GVP','GVP',value);
previousGray=[];edge=[];if k>1,previousGray=gray{k-1};edge=edges{k-1};end
[state.clean,cleanR]=ablation_recurrent_step(state.clean,P{k},previousGray,gray{k},edge,cfg);
emit('unbounded','R','R',cleanR);
badCurrent=P{k};if k==3,badCurrent=single(1-double(badCurrent));end
[state.polluted,badR]=ablation_recurrent_step(state.polluted,badCurrent,previousGray,gray{k},edge,cfg);
bad=base;
if k>=3&&k<=5
    probabilities=P(start3:k);
    for j=1:numel(probabilities)
        if start3+j-1==3,probabilities{j}=single(1-double(probabilities{j}));end
    end
    bad=fuse_motion_window(images(start3:k),probabilities,edges(start3:k-1),cfg,constant);
end
for m=methods,emit('pollution',m{1},m{1},bad.(m{1}));end
emit('pollution','R_unbounded','R',badR);assert(oi==79);
    function emit(scenario,method,parent,value)
        assert(all(isfinite(value),'all')&&all(value>=-cfg.range_tolerance & value<=1+cfg.range_tolerance,'all'));
        oi=oi+1;outputs(oi)=struct('scenario',scenario,'method',method,'parent_method',parent,'probability',single(min(1,max(0,value))));
    end
end
