function plot_registration_results(runRoot,plotId)
% Numeric-only figure. Does not read any source images or masks.
if nargin<2,plotId='plots';end
assert(~isempty(regexp(plotId,'^[A-Za-z0-9_]+$','once')));
out=fullfile(runRoot,plotId);assert(~isfolder(out));mkdir(out);
t=readtable(fullfile(runRoot,'evaluation','aggregate_metrics.csv'),TextType='string');
profiles=sort(unique(t.subset(t.dimension=="profile")));methods=["odometry","visual_ncc","prior_ncc"];
e=zeros(numel(profiles),3);f=e;
for j=1:3
    for i=1:numel(profiles)
        row=t(t.dimension=="profile"&t.subset==profiles(i)&t.method==methods(j),:);assert(height(row)==1);
        e(i,j)=row.mean_epe_px;f(i,j)=100*row.failure_2px;
    end
end
fig=figure('Visible','off','Color','w','Position',[40 40 1460 850]);
tiledlayout(2,1,TileSpacing='compact',Padding='compact');
nexttile;bar(e);ylabel('Mean grid EPE (px), valid estimates');xticks(1:numel(profiles));xticklabels(profiles);xtickangle(20);set(gca,'TickLabelInterpreter','none');grid on;legend(methods,Interpreter='none',Location='northwest');
title('Translation registration: synthetic validation / 15 source groups');
nexttile;bar(f);ylabel('Unavailable or mean EPE > 2 px (%)');ylim([0 100]);xticks(1:numel(profiles));xticklabels(profiles);xtickangle(20);set(gca,'TickLabelInterpreter','none');grid on;
title('All pairs remain in failure denominator; no detector/fusion evaluation');
exportgraphics(fig,fullfile(out,'registration_overview.png'),Resolution=140);
print(fig,fullfile(out,'registration_overview.svg'),'-dsvg');savefig(fig,fullfile(out,'registration_overview.fig'));close(fig);
end
