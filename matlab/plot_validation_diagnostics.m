function plot_validation_diagnostics(runId,plotId)
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'runs',runId);
s=jsondecode(fileread(fullfile(out,'summary.json')));assert(strcmp(s.status,'PASS'));
if nargin<2,plotId='plots';end
assert(~isempty(regexp(plotId,'^[A-Za-z0-9_]+$','once')));
plots=fullfile(out,plotId);assert(~isfolder(plots));mkdir(plots);
models={'small_unet','mobilenet_deeplab'};seeds=[20261114 20261115 20261116];colors=lines(3);
f=figure('Visible','off','Color','w','Position',[30 30 1500 950]);tiledlayout(2,2,'TileSpacing','compact');
for model=1:2
    for panel=1:2
        nexttile((panel-1)*2+model);hold on;
        for i=1:numel(s.checkpoints)
            r=s.checkpoints(i);if ~strcmp(r.model,models{model}),continue;end
            a=readtable(fullfile(out,r.name,'threshold_curve.csv'));seedIndex=find(seeds==r.seed);style='-';if strcmp(r.checkpoint_kind,'last'),style='--';end
            if panel==1,x=a.threshold;y=a.IoU;else,x=a.recall;y=a.precision;end
            plot(x,y,style,'Color',colors(seedIndex,:),'LineWidth',1.3,'DisplayName',sprintf('%d %s',r.seed,strrep(r.checkpoint_kind,'_',' ')));
        end
        if panel==1,xlabel('Threshold');ylabel('Micro foreground IoU');xline(.5,':k','HandleVisibility','off');xlim([0 1]);ylim([0 .6]);else,xlabel('Recall');ylabel('Precision');xlim([0 1]);ylim([0 1]);end
        grid on;title(strrep(models{model},'_',' '));legend('Location','best','FontSize',8);set(gca,'FontName','Arial','FontSize',11);
    end
end
sgtitle('Validation diagnostics: 400 images, 15 proxy groups, 3 predefined seeds; not a final test');
exportgraphics(f,fullfile(plots,'threshold_overview.png'),'Resolution',150);savefig(f,fullfile(plots,'threshold_overview.fig'));print(f,fullfile(plots,'threshold_overview.svg'),'-dsvg');close(f);
for i=1:numel(s.checkpoints)
    r=s.checkpoints(i);if r.seed~=20261114,continue;end
    d=fullfile(out,r.name);q=load(fullfile(d,'qualitative_cases.mat'));a=readtable(fullfile(d,'per_image_primary.csv'),'TextType','string');
    [~,lo]=min(a.IoU);[~,fp]=max(a.FP_over_background);[~,fn]=max(a.FN_over_target);selected=[lo fp fn];labels={'Lowest IoU','Highest FP/background','Highest FN/target'};
    f=figure('Visible','off','Color','w','Position',[30 30 1400 1100]);tiledlayout(3,4,'TileSpacing','compact','Padding','compact');
    for row=1:3
        k=find(r.qualitative_case_indices==selected(row));assert(isscalar(k));
        gt=q.case_T(:,:,k);v=q.case_V(:,:,k);p=double(q.case_P(:,:,k))>=.5;
        error=zeros(256,256,3);error(:,:,1)=double(v&~gt&p);error(:,:,2)=double(v&gt&p);error(:,:,3)=double(v&gt&~p);
        for c=1:3,z=error(:,:,c);z(~v)=.3;error(:,:,c)=z;end
        nexttile;imshow(q.case_X(:,:,:,k));title(sprintf('%s\nIoU %.3f',labels{row},a.IoU(selected(row))));
        nexttile;imshow(gt);title('Ground truth');nexttile;imshow(p&v);title('Prediction at 0.5');nexttile;imshow(error);title('Red FP / blue FN / green TP');
    end
    sgtitle(sprintf('%s | first predefined seed | selected failures, not representative',strrep(r.name,'_',' ')));
    exportgraphics(f,fullfile(plots,[r.name '_failures.png']),'Resolution',140);savefig(f,fullfile(plots,[r.name '_failures.fig']));close(f);
end
end
