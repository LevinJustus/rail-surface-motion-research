function artifacts = plot_smoke_run(runDir,history,evals,net,X,T,V,ids)
%PLOT_SMOKE_RUN Fixed eight-image training panels and diagnostic curves.
% This function does not load additional images, train, or select examples.
% history: global_update,total_loss,BCE_component,Dice_component,
% unclipped_global_gradient_norm,clip_factor,update_time_s (completed updates).
% evals: global_update,total_loss,micro_foreground_IoU,micro_foreground_Dice.
% Figures describe the supplied smoke training set only, not generalization.

assert(isnumeric(X) && isreal(X) && size(X,3)==3 && size(X,4)==8, ...
    'plot_smoke_run:InputShape','Expected exactly eight RGB training inputs');
assert(all(isfinite(X(:))) && all(X(:)>=0 & X(:)<=1), ...
    'plot_smoke_run:InputRange','Input images must be finite [0,1] arrays');
[h,w,~,n]=size(X); ids=string(ids(:));
assert(numel(ids)==n,'plot_smoke_run:IDs','One fixed ID is required per image');
T=canonical_mask(T,h,w,n);V=canonical_mask(V,h,w,n);
assert(all(reshape(sum(V,[1 2]),[],1)>0),'plot_smoke_run:EmptyValid','An image has no valid pixels');
assert(isscalar(net.OutputNames),'plot_smoke_run:Outputs','Expected one two-class output');
probability=predict(net,dlarray(single(X),'SSCB'));
P=host_numeric(probability);
assert(isequal(size(P,[1 2 3 4]),[h w 2 n]) && all(isfinite(P(:))), ...
    'plot_smoke_run:Probabilities','Unexpected or nonfinite probabilities');
assert(min(P,[],'all')>=-1e-6 && max(P,[],'all')<=1+1e-6 && ...
    max(abs(sum(P,3)-1),[],'all')<=1e-4, ...
    'plot_smoke_run:ProbabilityRange','Expected normalized two-class probabilities');

plotDir=fullfile(runDir,'plots');
if ~exist(plotDir,'dir'),mkdir(plotDir);end
artifacts.training_panels_png=fullfile(plotDir,'smoke_training_panels.png');
artifacts.training_panels_fig=fullfile(plotDir,'smoke_training_panels.fig');
artifacts.diagnostics_png=fullfile(plotDir,'smoke_training_diagnostics.png');
artifacts.diagnostics_fig=fullfile(plotDir,'smoke_training_diagnostics.fig');

f=figure('Visible','off','Color','white','Position',[20 20 1300 2200]);
closeFigure=onCleanup(@()close(f));
tiledlayout(n,4,'TileSpacing','compact','Padding','compact');
for k=1:n
    valid=V(:,:,1,k)>0; target=T(:,:,1,k)>0;
    p=double(P(:,:,2,k)); prediction=p>=0.5;
    ax=nexttile;image(ax,X(:,:,:,k));axis(ax,'image','off');
    title(ax,{sprintf('Input %d: %s',k,char(ids(k))),'Training image; gray in label panels = excluded padding'}, ...
        'Interpreter','none','FontSize',8);
    ax=nexttile;image(ax,binary_panel(target,valid));axis(ax,'image','off');
    title(ax,'Ground truth','FontSize',10);
    ax=nexttile;imagesc(ax,p,[0 1]);set(ax,'Color',[0.3 0.3 0.3]);
    set(findobj(ax,'Type','image'),'AlphaData',double(valid));
    axis(ax,'image','off');colormap(ax,parula(256));
    title(ax,'Defect probability [0,1]','FontSize',10);
    ax=nexttile;image(ax,binary_panel(prediction,valid));axis(ax,'image','off');
    title(ax,'Prediction: p(defect) >= 0.5','FontSize',10);
end
sgtitle('Fixed eight-image smoke training diagnostics: NOT validation or test results', ...
    'FontSize',15,'FontWeight','bold');
savefig(f,artifacts.training_panels_fig);
exportgraphics(f,artifacts.training_panels_png,'Resolution',130);
clear closeFigure

historyFields={'global_update','total_loss','BCE_component','Dice_component', ...
    'unclipped_global_gradient_norm','clip_factor','update_time_s'};
evalFields={'global_update','total_loss','micro_foreground_IoU','micro_foreground_Dice'};
assert_fields(history,historyFields,'history');assert_fields(evals,evalFields,'evals');
updates=field_values(history,'global_update');evalUpdates=field_values(evals,'global_update');
assert(all(diff(updates)>0) && all(updates>=1 & updates==fix(updates)), ...
    'plot_smoke_run:Updates','history must list strictly increasing completed updates, excluding zero');
assert(all(diff(evalUpdates)>0) && all(evalUpdates>=0 & evalUpdates==fix(evalUpdates)), ...
    'plot_smoke_run:EvaluationUpdates','Evaluation updates must be increasing nonnegative integers');
for name=historyFields
    values=field_values(history,name{1});
    assert(all(isfinite(values)),'plot_smoke_run:HistoryNonfinite','History field %s is nonfinite',name{1});
end
assert(all(isfinite(field_values(evals,'total_loss'))), ...
    'plot_smoke_run:EvaluationLoss','Evaluation loss must be finite');
assert(all(field_values(history,'clip_factor')>0 & field_values(history,'clip_factor')<=1), ...
    'plot_smoke_run:ClipFactor','Gradient scale factors must lie in (0,1]');
assert(all(field_values(history,'unclipped_global_gradient_norm')>=0) && ...
    all(field_values(history,'update_time_s')>=0), ...
    'plot_smoke_run:HistorySign','Gradient norms and elapsed update times cannot be negative');

f=figure('Visible','off','Color','white','Position',[40 40 1200 850]);
closeFigure=onCleanup(@()close(f));
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
ax=nexttile;hold(ax,'on');
if ~isempty(history)
    plot(ax,updates,field_values(history,'total_loss'),'-o','DisplayName','Batch total loss');
    plot(ax,updates,field_values(history,'BCE_component'),'-','DisplayName','Batch BCE component');
    plot(ax,updates,field_values(history,'Dice_component'),'-','DisplayName','Batch Dice component');
end
if ~isempty(evals)
    plot(ax,evalUpdates,field_values(evals,'total_loss'),'k-s','LineWidth',1.3,'DisplayName','Fixed 8-image loss');
end
grid(ax,'on');xlabel(ax,'Completed optimizer updates');ylabel(ax,'Loss');
title(ax,'Training loss (batch and fixed subset)');
if ~isempty(history)||~isempty(evals),legend(ax,'Location','best');end

ax=nexttile;
if ~isempty(history)
    yyaxis(ax,'left');plot(ax,updates,field_values(history,'unclipped_global_gradient_norm'),'-o');
    ylabel(ax,'Unclipped global gradient L2 norm');
    yyaxis(ax,'right');plot(ax,updates,field_values(history,'clip_factor'),'-s');
    ylabel(ax,'Applied gradient scale');ylim(ax,[0 1.05]);
else
    text(ax,.5,.5,'No completed optimizer updates','HorizontalAlignment','center');
end
grid(ax,'on');xlabel(ax,'Completed optimizer updates');title(ax,'Gradient norm and clipping');

ax=nexttile;hold(ax,'on');
if ~isempty(evals)
    iou=field_values(evals,'micro_foreground_IoU'); dice=field_values(evals,'micro_foreground_Dice');
    assert(all(isnan(iou)|(isfinite(iou)&iou>=0&iou<=1)) && ...
        all(isnan(dice)|(isfinite(dice)&dice>=0&dice<=1)), ...
        'plot_smoke_run:Metrics','IoU and Dice must be [0,1] or undefined NaN');
    plot(ax,evalUpdates,iou,'-o','DisplayName','Micro defect IoU');
    plot(ax,evalUpdates,dice,'-s','DisplayName','Micro defect Dice');
    legend(ax,'Location','best');
    if all(isnan(iou)) && all(isnan(dice))
        text(ax,.5,.5,'Metrics undefined: zero denominators','Units','normalized','HorizontalAlignment','center');
    end
end
grid(ax,'on');ylim(ax,[0 1]);xlabel(ax,'Completed optimizer updates');ylabel(ax,'Training-subset metric');
title(ax,'Fixed eight training images; NaN shown as gaps');

ax=nexttile;
if ~isempty(history),plot(ax,updates,field_values(history,'update_time_s'),'-o');end
grid(ax,'on');xlabel(ax,'Completed optimizer updates');ylabel(ax,'Seconds per update');
title(ax,'Measured update time; NOT inference FPS');
sgtitle('Smoke training only: finite-chain diagnostics, no generalization claim','FontSize',14);
savefig(f,artifacts.diagnostics_fig);
exportgraphics(f,artifacts.diagnostics_png,'Resolution',140);
clear closeFigure
end

function panel=binary_panel(mask,valid)
gray=single(mask);gray(~valid)=0.3;panel=repmat(gray,1,1,3);
end

function M=canonical_mask(M,h,w,n)
assert((isnumeric(M)||islogical(M)) && isreal(M) && ndims(M)<=4, ...
    'plot_smoke_run:MaskType','Expected a real binary mask array');
if isequal(size(M,[1 2 3 4]),[h w 1 n]) || ...
        isequal(size(M,[1 2 3 4]),[h w n 1])
    M=reshape(single(M),h,w,1,n);
else
    error('plot_smoke_run:MaskShape','Mask dimensions do not match fixed inputs');
end
assert(all(isfinite(M(:))) && all(M(:)==0 | M(:)==1), ...
    'plot_smoke_run:MaskValues','Masks must contain binary 0/1 values');
end

function output=host_numeric(value)
if isa(value,'dlarray'),value=extractdata(value);end
if isa(value,'gpuArray'),value=gather(value);end
output=value;
end

function assert_fields(records,names,description)
assert(isstruct(records),'plot_smoke_run:RecordsType','%s must be a struct array',description);
if ~isempty(records)
    assert(all(isfield(records,names)),'plot_smoke_run:MissingField','Missing required %s fields',description);
end
end

function values=field_values(records,name)
if isempty(records),values=[];return;end
values=[records.(name)];
assert(numel(values)==numel(records) && isnumeric(values), ...
    'plot_smoke_run:ScalarField','Record field %s must contain numeric scalars',name);
values=double(values);
end
