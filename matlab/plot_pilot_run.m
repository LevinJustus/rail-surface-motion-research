function artifacts = plot_pilot_run(runDir,history,valEvals,trainEvals,net, ...
    Xtrain8,Ttrain8,Vtrain8,trainIds8,Xval8,Tval8,Vval8,valIds8)
%PLOT_PILOT_RUN Frozen training/validation examples and descriptive curves.
% Does not load extra data, train, change thresholds, or select nice examples.
% history fields match plot_smoke_run; trainEvals/valEvals require
% global_update,total_loss,micro_foreground_IoU,micro_foreground_Dice.
% Curves represent fixed train8 / full val64 under the frozen stage1 plan.
% Panels show the supplied fixed train8 and fixed FIRST val8, never val64-wide
% cherry-picked examples. Neither figure is a heldout test result.

historyFields={'global_update','total_loss','BCE_component','Dice_component', ...
    'unclipped_global_gradient_norm','clip_factor','update_time_s'};
evalFields={'global_update','total_loss','micro_foreground_IoU','micro_foreground_Dice'};
assert_fields(history,historyFields,'history');
assert_fields(valEvals,evalFields,'valEvals');assert_fields(trainEvals,evalFields,'trainEvals');
updates=field_values(history,'global_update');
trainUpdates=check_evals(trainEvals,'trainEvals');valUpdates=check_evals(valEvals,'valEvals');
assert(all(diff(updates)>0) && all(updates>=1 & updates==fix(updates)), ...
    'plot_pilot_run:Updates','history must list increasing completed updates, excluding zero');
for name=historyFields
    assert(all(isfinite(field_values(history,name{1}))), ...
        'plot_pilot_run:NonfiniteHistory','History field %s is nonfinite',name{1});
end
assert(all(field_values(history,'clip_factor')>0 & field_values(history,'clip_factor')<=1), ...
    'plot_pilot_run:ClipFactor','clip_factor must be in (0,1]');
assert(all(field_values(history,'unclipped_global_gradient_norm')>=0) && ...
    all(field_values(history,'update_time_s')>=0), ...
    'plot_pilot_run:HistorySign','Gradient norms and elapsed times cannot be negative');
assert(all([trainUpdates valUpdates]<=max([0 updates])), ...
    'plot_pilot_run:FutureEvaluation','Evaluation cannot refer to a future optimizer update');

plotDir=fullfile(runDir,'plots');if ~exist(plotDir,'dir'),mkdir(plotDir);end
artifacts.training_panels_png=fullfile(plotDir,'pilot_training_panels.png');
artifacts.training_panels_fig=fullfile(plotDir,'pilot_training_panels.fig');
artifacts.validation_panels_png=fullfile(plotDir,'pilot_validation_panels.png');
artifacts.validation_panels_fig=fullfile(plotDir,'pilot_validation_panels.fig');
artifacts.diagnostics_png=fullfile(plotDir,'pilot_training_diagnostics.png');
artifacts.diagnostics_fig=fullfile(plotDir,'pilot_training_diagnostics.fig');
write_panels(net,Xtrain8,Ttrain8,Vtrain8,trainIds8, ...
    'Fixed eight PILOT TRAINING images: descriptive diagnostics, not validation/test', ...
    artifacts.training_panels_png,artifacts.training_panels_fig);
write_panels(net,Xval8,Tval8,Vval8,valIds8, ...
    'Fixed first eight PILOT VALIDATION images: not heldout test results', ...
    artifacts.validation_panels_png,artifacts.validation_panels_fig);

f=figure('Visible','off','Color','white','Position',[40 40 1300 900]);
closeFigure=onCleanup(@()close(f));
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
ax=nexttile;hold(ax,'on');
if ~isempty(history)
    plot(ax,updates,field_values(history,'total_loss'),'-','DisplayName','Train batch total');
    plot(ax,updates,field_values(history,'BCE_component'),'-','DisplayName','Train batch BCE component');
    plot(ax,updates,field_values(history,'Dice_component'),'-','DisplayName','Train batch Dice component');
end
if ~isempty(trainEvals)
    plot(ax,trainUpdates,field_values(trainEvals,'total_loss'),'k--o', ...
        'LineWidth',1.3,'DisplayName','Fixed train8 total');
end
if ~isempty(valEvals)
    plot(ax,valUpdates,field_values(valEvals,'total_loss'),'m-s', ...
        'LineWidth',1.3,'DisplayName','Full fixed validation total');
end
grid(ax,'on');xlabel(ax,'Completed optimizer updates');ylabel(ax,'Loss');
title(ax,'Training batches, fixed train8 and full validation');
if ~isempty(history)||~isempty(trainEvals)||~isempty(valEvals),legend(ax,'Location','best');end

ax=nexttile;
if ~isempty(history)
    yyaxis(ax,'left');plot(ax,updates,field_values(history,'unclipped_global_gradient_norm'),'-');
    ylabel(ax,'Unclipped global gradient L2 norm');
    yyaxis(ax,'right');plot(ax,updates,field_values(history,'clip_factor'),'-');
    ylabel(ax,'Applied gradient scale');ylim(ax,[0 1.05]);
else
    text(ax,.5,.5,'No completed optimizer updates','HorizontalAlignment','center');
end
grid(ax,'on');xlabel(ax,'Completed optimizer updates');title(ax,'Gradient norm and clipping');

ax=nexttile;hold(ax,'on');
if ~isempty(trainEvals)
    plot(ax,trainUpdates,field_values(trainEvals,'micro_foreground_IoU'),'--o', ...
        'DisplayName','Train8 micro defect IoU');
    plot(ax,trainUpdates,field_values(trainEvals,'micro_foreground_Dice'),'--s', ...
        'DisplayName','Train8 micro defect Dice');
end
if ~isempty(valEvals)
    plot(ax,valUpdates,field_values(valEvals,'micro_foreground_IoU'),'-o', ...
        'LineWidth',1.4,'DisplayName','Full validation micro defect IoU');
    plot(ax,valUpdates,field_values(valEvals,'micro_foreground_Dice'),'-s', ...
        'LineWidth',1.4,'DisplayName','Full validation micro defect Dice');
end
grid(ax,'on');ylim(ax,[0 1]);xlabel(ax,'Completed optimizer updates');ylabel(ax,'Descriptive subset metric');
title(ax,'Fixed train8 / full val64; NaN shown as gaps');
if ~isempty(trainEvals)||~isempty(valEvals),legend(ax,'Location','best');end
values=[field_values(trainEvals,'micro_foreground_IoU'), ...
    field_values(trainEvals,'micro_foreground_Dice'), ...
    field_values(valEvals,'micro_foreground_IoU'),field_values(valEvals,'micro_foreground_Dice')];
if ~isempty(values) && all(isnan(values))
    text(ax,.5,.5,'Metrics undefined: zero denominators','Units','normalized','HorizontalAlignment','center');
end

ax=nexttile;
if ~isempty(history),plot(ax,updates,field_values(history,'update_time_s'),'-');end
grid(ax,'on');xlabel(ax,'Completed optimizer updates');ylabel(ax,'Measured seconds per update');
title(ax,'CPU update time; NOT inference FPS');
sgtitle('Pilot training and fixed-validation trends: descriptive only, no final-test claim','FontSize',14);
savefig(f,artifacts.diagnostics_fig);
exportgraphics(f,artifacts.diagnostics_png,'Resolution',140);
clear closeFigure
end

function write_panels(net,X,T,V,ids,heading,pngPath,figPath)
assert(isnumeric(X) && isreal(X) && ndims(X)<=4 && size(X,3)==3 && size(X,4)==8, ...
    'plot_pilot_run:PanelInputs','Exactly eight RGB images are required in each fixed panel');
assert(all(isfinite(X(:))) && all(X(:)>=0 & X(:)<=1), ...
    'plot_pilot_run:InputRange','Images must be finite and normalized once to [0,1]');
[h,w,~,n]=size(X);ids=string(ids(:));
assert(numel(ids)==n && numel(unique(ids))==n && all(strlength(ids)>0), ...
    'plot_pilot_run:PanelIDs','Eight unique nonempty fixed IDs are required');
T=canonical_mask(T,h,w,n);V=canonical_mask(V,h,w,n);
assert(all(reshape(sum(V,[1 2]),[],1)>0),'plot_pilot_run:EmptyValid','An image has no valid pixels');
assert(isscalar(net.OutputNames),'plot_pilot_run:Outputs','Expected one final two-class softmax output');
P=zeros(h,w,2,n,'single');
for first=1:2:n
    selected=first:min(first+1,n);
    probabilities=predict(net,dlarray(single(X(:,:,:,selected)),'SSCB'));
    batch=host_numeric(probabilities);
    assert(isequal(size(batch,[1 2 3 4]),[h w 2 numel(selected)]) && all(isfinite(batch(:))), ...
        'plot_pilot_run:ProbabilityShape','Unexpected or nonfinite panel probabilities');
    assert(min(batch,[],'all')>=-1e-6 && max(batch,[],'all')<=1+1e-6 && ...
        max(abs(sum(batch,3)-1),[],'all')<=1e-4, ...
        'plot_pilot_run:ProbabilityRange','Panel probabilities must be normalized');
    P(:,:,:,selected)=single(batch);
end
f=figure('Visible','off','Color','white','Position',[20 20 1300 2200]);
closeFigure=onCleanup(@()close(f));
tiledlayout(n,4,'TileSpacing','compact','Padding','compact');
for k=1:n
    valid=V(:,:,1,k)>0;target=T(:,:,1,k)>0;p=double(P(:,:,2,k));
    ax=nexttile;image(ax,X(:,:,:,k));axis(ax,'image','off');
    title(ax,{sprintf('Input %d: %s',k,char(ids(k))), ...
        'Gray in label panels = excluded padding'},'Interpreter','none','FontSize',8);
    ax=nexttile;image(ax,binary_panel(target,valid));axis(ax,'image','off');
    title(ax,'Ground truth','FontSize',10);
    ax=nexttile;imagesc(ax,p,[0 1]);set(ax,'Color',[.3 .3 .3]);
    set(findobj(ax,'Type','image'),'AlphaData',double(valid));axis(ax,'image','off');
    colormap(ax,parula(256));title(ax,'Defect probability [0,1]','FontSize',10);
    ax=nexttile;image(ax,binary_panel(p>=.5,valid));axis(ax,'image','off');
    title(ax,'Prediction: p(defect) >= 0.5','FontSize',10);
end
sgtitle(heading,'FontSize',15,'FontWeight','bold');
savefig(f,figPath);exportgraphics(f,pngPath,'Resolution',130);
clear closeFigure
end

function output=check_evals(records,description)
output=field_values(records,'global_update');
assert(all(diff(output)>0) && all(output>=0 & output==fix(output)), ...
    'plot_pilot_run:EvaluationUpdates','%s updates must be increasing nonnegative integers',description);
assert(all(isfinite(field_values(records,'total_loss'))), ...
    'plot_pilot_run:EvaluationLoss','%s loss must be finite',description);
for metric={'micro_foreground_IoU','micro_foreground_Dice'}
    values=field_values(records,metric{1});
    assert(all(isnan(values)|(isfinite(values)&values>=0&values<=1)), ...
        'plot_pilot_run:MetricRange','%s %s must be [0,1] or undefined NaN',description,metric{1});
end
end

function M=canonical_mask(M,h,w,n)
assert((isnumeric(M)||islogical(M)) && isreal(M) && ndims(M)<=4, ...
    'plot_pilot_run:MaskType','Expected a real binary mask array');
assert(isequal(size(M,[1 2 3 4]),[h w 1 n]) || ...
    isequal(size(M,[1 2 3 4]),[h w n 1]), ...
    'plot_pilot_run:MaskShape','Mask dimensions do not match panel inputs');
M=reshape(single(M),h,w,1,n);
assert(all(isfinite(M(:))) && all(M(:)==0|M(:)==1), ...
    'plot_pilot_run:MaskValues','Masks must contain binary 0/1 values');
end

function panel=binary_panel(mask,valid)
gray=single(mask);gray(~valid)=.3;panel=repmat(gray,1,1,3);
end

function output=host_numeric(value)
if isa(value,'dlarray'),value=extractdata(value);end
if isa(value,'gpuArray'),value=gather(value);end
output=value;
end

function assert_fields(records,names,description)
assert(isstruct(records),'plot_pilot_run:RecordsType','%s must be a struct array',description);
if ~isempty(records)
    assert(all(isfield(records,names)),'plot_pilot_run:MissingField','Missing required %s fields',description);
end
end

function values=field_values(records,name)
if isempty(records),values=[];return;end
values=[records.(name)];
assert(numel(values)==numel(records) && isnumeric(values), ...
    'plot_pilot_run:ScalarField','Record field %s must contain numeric scalars',name);
values=reshape(double(values),1,[]);
end
