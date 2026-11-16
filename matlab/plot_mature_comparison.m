function artifacts=plot_mature_comparison(comparisonDir,outDir,projectRoot)
%PLOT_MATURE_COMPARISON All-seed common-recipe validation comparison, no MAT.
% Call only after compare_mature_results.py emits a PASS comparison.json.
% Reads JSON/CSV metadata only. No images, checkpoint objects, new best
% selection, threshold changes, training, significance tests or tail filling.
% Example after all runs finish:
% addpath('runs/step11_preflight'); plot_mature_comparison;
% Default output: runs/step11_preflight/comparison_final/plots (must be new).
% Selected is the EXISTING IoU-selected checkpoint, for all four metrics.
% SD uses n-1 among defined runs and is never presented as a CI; n is shown.

if nargin<3||isempty(projectRoot),projectRoot=find_project_root;end
projectRoot=canonical_path(projectRoot);
if nargin<1||isempty(comparisonDir),comparisonDir='runs/step11_preflight/comparison_final';end
comparisonDir=resolve_path(projectRoot,comparisonDir);
if nargin<2||isempty(outDir),outDir=fullfile(comparisonDir,'plots');end
outDir=resolve_path(projectRoot,outDir);
safeRoot=fullfile(projectRoot,'runs','step11_preflight');
assert(is_inside(outDir,safeRoot)&&~strcmpi(outDir,canonical_path(safeRoot)), ...
    'plot_mature_comparison:OutputScope','Output must be a new subdirectory of runs/step11_preflight');
assert(~isfolder(outDir)&&~isfile(outDir), ...
    'plot_mature_comparison:Overwrite','Preserve prior plots; select a new output directory');

inputs=struct([]);
comparisonPath=fullfile(comparisonDir,'comparison.json');
comparison=read_json(comparisonPath);add_input(comparisonPath,'comparison JSON');
assert(isstruct(comparison)&&isscalar(comparison)&&comparison.schema_version==1&& ...
    strcmp(comparison.status,'PASS'), ...
    'plot_mature_comparison:Incomplete','Only complete PASS comparison JSON is accepted; no partial/baseline-only plot');
assert(comparison.shared_conditions.shared_recipe_verified&& ...
    comparison.shared_conditions.data_access_records_verified);
methods={'small_unet','mature_mobilenet_deeplab'};
labels={'Small U-Net',sprintf('MobileNetV2 +\nDeepLabv3+ project')};
roles={'selected','last'};roleLabels={'IoU-selected checkpoint','Actual last checkpoint'};
metrics={'micro_foreground_IoU','micro_foreground_Dice','micro_precision','micro_recall'};
metricLabels={'Foreground IoU','Foreground Dice','Pixel precision','Pixel recall'};
seeds=[20261114 20261115 20261116];
assert(isstruct(comparison.methods)&&numel(comparison.methods)==2&& ...
    isequal(sort(string({comparison.methods.method})),sort(string(methods))));
perSeed=comparison.per_seed(:);statistics=comparison.statistics(:);
assert(isstruct(perSeed)&&numel(perSeed)==12&&isstruct(statistics)&&numel(statistics)==16);
perSeedCsvPath=fullfile(comparisonDir,'per_seed.csv');
statisticsCsvPath=fullfile(comparisonDir,'statistics.csv');
perSeedCsv=read_csv(perSeedCsvPath);statisticsCsv=read_csv(statisticsCsvPath);
add_input(perSeedCsvPath,'comparison per-seed CSV');add_input(statisticsCsvPath,'comparison statistics CSV');
assert(height(perSeedCsv)==12&&height(statisticsCsv)==16);
values=nan(2,2,4,3);means=nan(2,2,4);deviations=nan(2,2,4);defined=zeros(2,2,4);
recordsByMethod=cell(2,3);curves=cell(2,3);sourceMethods=struct([]);
pointRows=struct([]);statRows=struct([]);curveRows=struct([]);

for m=1:2
    metadata=comparison.methods(strcmp({comparison.methods.method},methods{m}));
    assert(isscalar(metadata)&&metadata.complete&& ...
        isequal(double(metadata.expected_seeds(:)'),seeds)&& ...
        isequal(double(metadata.completed_seeds(:)'),seeds)&&isempty(metadata.pending_seeds));
    run=resolve_path(projectRoot,metadata.run);configPath=resolve_path(projectRoot,metadata.config_file);
    assert(is_inside(run,projectRoot)&&is_inside(configPath,projectRoot));
    require_hash(configPath,metadata.hashes.training_config_sha256);cfg=read_json(configPath);
    require_hash(resolve_path(projectRoot,cfg.model_config),metadata.hashes.model_config_sha256);
    require_hash(resolve_path(projectRoot,cfg.source_split),metadata.hashes.source_split_sha256);
    require_hash(resolve_path(projectRoot,cfg.batch_schedule),metadata.hashes.batch_schedule_sha256);
    require_hash(resolve_path(projectRoot,metadata.parameters.source),metadata.parameters.source_sha256);
    require_hash(fullfile(run,'input_access.json'),metadata.input_access_sha256);
    require_hash(fullfile(run,'summary.json'),metadata.group_summary_sha256);
    group=read_json(fullfile(run,'summary.json'));groupStatus=read_json(fullfile(run,'status.json'));
    add_input(fullfile(run,'status.json'),[methods{m} ' final status']);
    assert(strcmp(group.status,'PASS')&&strcmp(groupStatus.status,'PASS')&& ...
        group.training_images==1396&&group.validation_images==400&&group.heldout_test_images_opened==0&& ...
        isequal(double(group.initialization_seeds(:)'),seeds));
    source=read_json(fullfile(run,'source_version.json'));
    add_input(fullfile(run,'source_version.json'),[methods{m} ' source version']);
    assert(strcmp(source.git_commit,metadata.source_commit)&&~source.git_dirty);
    sourceRow.method=methods{m};sourceRow.run=relative_label(projectRoot,run);
    sourceRow.source_commit=metadata.source_commit;sourceRow.parameters=metadata.parameters;
    sourceRow.config_file=relative_label(projectRoot,configPath);sourceRow.hashes=metadata.hashes;
    if isempty(sourceMethods),sourceMethods=sourceRow;else,sourceMethods(end+1)=sourceRow;end %#ok<AGROW>
    for k=1:3
        seedDir=fullfile(run,'seeds',sprintf('seed_%d',seeds(k)));
        summaryPath=fullfile(seedDir,'summary.json');
        key=matlab.lang.makeValidName(sprintf('%d',seeds(k)));
        assert(isfield(metadata.per_seed_summary_sha256,key));
        require_hash(summaryPath,metadata.per_seed_summary_sha256.(key));
        summary=read_json(summaryPath);seedStatus=read_json(fullfile(seedDir,'status.json'));
        add_input(fullfile(seedDir,'status.json'),sprintf('%s seed%d final status',methods{m},seeds(k)));
        assert(strcmp(summary.status,'PASS')&&strcmp(seedStatus.status,'PASS')&& ...
            summary.initialization_seed==seeds(k)&&summary.heldout_test_images_opened==0&& ...
            summary.main_optimizer_updates==seedStatus.completed_updates);
        assert(summary.completed_epochs>=8&&summary.completed_epochs<=20&& ...
            summary.main_optimizer_updates==175*summary.completed_epochs);
        recordsByMethod{m,k}=summary;
        evaluationPath=fullfile(seedDir,'validation_evaluations.json');
        e=read_json(evaluationPath);e=e(:);add_input(evaluationPath,sprintf('%s seed%d actual validation curve',methods{m},seeds(k)));
        assert(isstruct(e)&&numel(e)==summary.completed_epochs+1&& ...
            isequal(double([e.epoch]),0:summary.completed_epochs)&& ...
            isequal(double([e.global_update]),175*(0:summary.completed_epochs))&& ...
            all([e.sample_count]==400));
        assert_scores(summary.final_validation,e(end),metrics,'actual last versus curve');
        curves{m,k}=e;
        for r=1:2
            selection=find(strcmp({perSeed.method},methods{m})& ...
                [perSeed.seed]==seeds(k)&strcmp({perSeed.checkpoint_role},roles{r}));
            assert(isscalar(selection),'plot_mature_comparison:Row','Missing or duplicate seed/role row');
            row=perSeed(selection);
            if r==1,chosen=summary.best_validation;else,chosen=summary.final_validation;end
            assert(row.parameters==metadata.parameters.count&&row.completed_epochs==summary.completed_epochs&& ...
                row.completed_updates==summary.main_optimizer_updates);
            same_number(number(row,'checkpoint_epoch'),number(chosen,'epoch'),'checkpoint epoch');
            same_number(number(row,'checkpoint_update'),number(chosen,'global_update'),'checkpoint update');
            assert(logical(row.is_untrained_reference)==(number(row,'checkpoint_epoch')==0));
            assert_scores(row,chosen,metrics,'per-seed comparison versus source summary');
            csvIndex=find(perSeedCsv.method==string(methods{m})& ...
                csv_numbers(perSeedCsv.seed)==seeds(k)&perSeedCsv.checkpoint_role==string(roles{r}));
            assert(isscalar(csvIndex));
            for field={'seed','parameters','completed_epochs','completed_updates','checkpoint_epoch','checkpoint_update'}
                same_number(csv_number(perSeedCsv,csvIndex,field{1}),number(row,field{1}),['per_seed.csv/' field{1}]);
            end
            csvFlag=lower(strtrim(perSeedCsv.is_untrained_reference(csvIndex)));
            assert(ismember(csvFlag,["true","false"])&&(csvFlag=="true")==logical(row.is_untrained_reference));
            for b=1:4
                v=number(row,metrics{b});assert_metric(v);values(m,r,b,k)=v;
                same_number(csv_number(perSeedCsv,csvIndex,metrics{b}),v,['per_seed.csv/' metrics{b}]);
                p.method=string(methods{m});p.model_label=string(strrep(labels{m},newline,' '));
                p.seed=seeds(k);p.checkpoint_role=string(roles{r});p.metric=string(metrics{b});
                p.value=v;p.value_is_defined=isfinite(v);p.checkpoint_epoch=number(row,'checkpoint_epoch');
                p.checkpoint_update=number(row,'checkpoint_update');p.is_untrained_reference=logical(row.is_untrained_reference);
                p.parameters=row.parameters;p.validation_images=400;
                if isempty(pointRows),pointRows=p;else,pointRows(end+1)=p;end %#ok<AGROW>
            end
            epoch=number(chosen,'epoch');update=number(chosen,'global_update');
            if isfinite(epoch)&&isfinite(update)
                selected=find([e.epoch]==epoch&[e.global_update]==update);assert(isscalar(selected));
                assert_scores(chosen,e(selected),metrics,'checkpoint versus actual evaluation');
            else
                assert(r==1&&all(isnan(arrayfun(@(b)number(chosen,metrics{b}),1:4))));
            end
        end
        for j=1:numel(e)
            c.method=string(methods{m});c.seed=seeds(k);c.epoch=e(j).epoch;c.global_update=e(j).global_update;
            c.total_loss=number(e(j),'total_loss');assert(isfinite(c.total_loss));
            for b=1:4,c.(metrics{b})=number(e(j),metrics{b});assert_metric(c.(metrics{b}));end
            c.is_untrained_reference=e(j).epoch==0;c.is_actual_last=j==numel(e);
            c.is_existing_IoU_selected=e(j).global_update==number(summary.best_validation,'global_update');
            c.validation_images=400;
            if isempty(curveRows),curveRows=c;else,curveRows(end+1)=c;end %#ok<AGROW>
        end
    end
end

for m=1:2
    for r=1:2
        for b=1:4
            v=reshape(values(m,r,b,:),1,3);valid=isfinite(v);n=nnz(valid);
            if n>0,mu=mean(v(valid));else,mu=NaN;end
            if n>=2,sd=std(v(valid),0);else,sd=NaN;end
            s=statistics(strcmp({statistics.method},methods{m})& ...
                strcmp({statistics.checkpoint_role},roles{r})&strcmp({statistics.metric},metrics{b}));
            assert(isscalar(s)&&s.expected_count==3&&s.record_count==3&&s.defined_count==n);
            supplied=numeric_vector(s.values);assert(numel(supplied)==3);
            for k=1:3,same_number(supplied(k),v(k),'statistics recorded seed values');end
            same_number(number(s,'mean'),mu,'statistics mean');same_number(number(s,'sample_std'),sd,'statistics sample SD');
            csvIndex=find(statisticsCsv.method==string(methods{m})& ...
                statisticsCsv.checkpoint_role==string(roles{r})&statisticsCsv.metric==string(metrics{b}));
            assert(isscalar(csvIndex));
            for field={'expected_count','record_count','defined_count','mean','sample_std'}
                same_number(csv_number(statisticsCsv,csvIndex,field{1}),number(s,field{1}),['statistics.csv/' field{1}]);
            end
            fromCsv=numeric_vector(jsondecode(char(statisticsCsv.values(csvIndex))));assert(numel(fromCsv)==3);
            for k=1:3,same_number(fromCsv(k),v(k),'statistics.csv seed values');end
            means(m,r,b)=mu;deviations(m,r,b)=sd;defined(m,r,b)=n;
            q.method=string(methods{m});q.checkpoint_role=string(roles{r});q.metric=string(metrics{b});
            q.expected_count=3;q.record_count=3;q.defined_count=n;q.mean=mu;q.sample_std=sd;
            q.sd_definition="Sample SD (n-1), among defined recorded values; undefined stays NaN";
            if isempty(statRows),statRows=q;else,statRows(end+1)=q;end %#ok<AGROW>
        end
    end
end

% All completion/provenance/JSON/CSV checks precede any output creation.
mkdir(outDir);artifacts.points_csv=fullfile(outDir,'model_comparison_points.csv');
artifacts.statistics_csv=fullfile(outDir,'model_comparison_statistics.csv');
artifacts.curves_csv=fullfile(outDir,'model_comparison_actual_epochs.csv');
artifacts.summary_png=fullfile(outDir,'model_comparison_metrics.png');
artifacts.summary_fig=fullfile(outDir,'model_comparison_metrics.fig');
artifacts.curves_png=fullfile(outDir,'model_comparison_actual_curves.png');
artifacts.curves_fig=fullfile(outDir,'model_comparison_actual_curves.fig');
artifacts.manifest_json=fullfile(outDir,'model_comparison_plot_manifest.json');
f=[];g=[];
try
    writetable(struct2table(pointRows),artifacts.points_csv);
    writetable(struct2table(statRows),artifacts.statistics_csv);
    writetable(struct2table(curveRows),artifacts.curves_csv);
    f=figure('Visible','off','Color','white','Position',[40 40 1680 980]);
    colors=[0.10 0.39 0.65;0.81 0.35 0.12];markers={'o','^','s'};offsets=[-.12 0 .12];
    for r=1:2
        for b=1:4
            ax=subplot(2,4,(r-1)*4+b,'Parent',f);hold(ax,'on');
            allMu=reshape(means(:,:,b),[],1);allSd=reshape(deviations(:,:,b),[],1);
            lowerBound=allMu-allSd;upperBound=allMu+allSd;
            lo=min([0;lowerBound(isfinite(lowerBound))]);hi=max([1;upperBound(isfinite(upperBound))]);
            span=hi-lo;ylim(ax,[lo-.04*span hi+.10*span]);xlim(ax,[.55 2.45]);
            for m=1:2
                for k=1:3
                    v=values(m,r,b,k);
                    if isfinite(v),plot(ax,m+offsets(k),v,markers{k},'Color',colors(m,:), ...
                        'MarkerFaceColor',colors(m,:),'MarkerSize',6,'LineStyle','none','LineWidth',1);end
                end
                if isfinite(means(m,r,b))
                    if isfinite(deviations(m,r,b))
                        errorbar(ax,m,means(m,r,b),deviations(m,r,b),'kd', ...
                            'LineStyle','none','MarkerFaceColor','white','MarkerSize',7,'LineWidth',1.4,'CapSize',9);
                    else
                        plot(ax,m,means(m,r,b),'kd','MarkerFaceColor','white','MarkerSize',7,'LineWidth',1.4);
                    end
                end
                text(ax,m,hi+.065*span,sprintf('n=%d/3',defined(m,r,b)), ...
                    'HorizontalAlignment','center','FontSize',9,'Color',[.30 .30 .30]);
            end
            set(ax,'XTick',[1 2],'XTickLabel',{'Small U-Net','MV2-DeepLabv3+'},'TickLabelInterpreter','none','YTick',0:.2:1, ...
                'FontName','Arial','FontSize',10,'Box','off','LineWidth',.8);
            grid(ax,'on');ax.GridAlpha=.14;ylabel(ax,metricLabels{b});
            title(ax,{roleLabels{r},metricLabels{b}},'FontSize',11,'Interpreter','none');
        end
    end
    sgtitle(f,{'Small U-Net versus MobileNetV2 + DeepLabv3+ project implementation', ...
        'Fixed val400, common recipe, three predefined seed runs per model; no significance or heldout-test claim'}, ...
        'Interpreter','none','FontName','Arial','FontSize',14);
    annotation(f,'textbox',[.06 .008 .90 .050],'String', ...
        {'Circles: seed20261114; triangles: seed20261115; squares: seed20261116. Horizontal offsets only separate coincident points.', ...
         'Open diamonds: mean; error bars: sample SD (not CI). n denotes defined runs; no point is drawn for an undefined metric.'}, ...
        'EdgeColor','none','HorizontalAlignment','center','FontName','Arial','FontSize',10,'Interpreter','none');
    exportgraphics(f,artifacts.summary_png,'Resolution',180);savefig(f,artifacts.summary_fig);close(f);f=[];

    g=figure('Visible','off','Color','white','Position',[40 40 1680 980]);
    seedColors=lines(3);styles={'-','--',':'};
    for m=1:2
        for b=1:4
            ax=subplot(2,4,(m-1)*4+b,'Parent',g);hold(ax,'on');
            for k=1:3
                e=curves{m,k};v=arrayfun(@(item)number(item,metrics{b}),e);epochs=[e.epoch];
                plot(ax,epochs,v,styles{k},'Color',seedColors(k,:),'LineWidth',1.3, ...
                    'DisplayName',sprintf('Seed%d; last%d',seeds(k),epochs(end)));
                bestEpoch=number(recordsByMethod{m,k}.best_validation,'epoch');
                if isfinite(bestEpoch)
                    at=find(epochs==bestEpoch);assert(isscalar(at));
                    if isfinite(v(at)),plot(ax,epochs(at),v(at),'*','Color',seedColors(k,:), ...
                        'MarkerSize',8,'LineStyle','none','HandleVisibility','off');end
                end
                if isfinite(v(end)),plot(ax,epochs(end),v(end),'s','Color',seedColors(k,:), ...
                    'MarkerFaceColor','white','MarkerSize',6,'LineStyle','none','HandleVisibility','off');end
            end
            set(ax,'FontName','Arial','FontSize',10,'Box','off','LineWidth',.8);
            xlim(ax,[0 20]);ylim(ax,[0 1]);xticks(ax,0:4:20);yticks(ax,0:.2:1);grid(ax,'on');ax.GridAlpha=.14;
            xlabel(ax,'Actual evaluated epoch (0 = untrained)');ylabel(ax,metricLabels{b});
            title(ax,{strrep(labels{m},newline,' '),metricLabels{b}},'FontSize',10,'Interpreter','none');
            if b==1,legend(ax,'Location','best','FontSize',8,'Interpreter','none','Box','off');end
        end
    end
    sgtitle(g,{'Actual per-seed validation curves; no filled early-stop tails or averaged curves', ...
        'Same fixed val400 and recipe; descriptive initialization/dropout variation, not significance or generalization'}, ...
        'Interpreter','none','FontName','Arial','FontSize',14);
    annotation(g,'textbox',[.06 .008 .90 .045],'String', ...
        'Stars: existing IoU-selected checkpoint (same epoch across all four metrics); open squares: actual last. Undefined values are gaps.', ...
        'EdgeColor','none','HorizontalAlignment','center','FontName','Arial','FontSize',10,'Interpreter','none');
    exportgraphics(g,artifacts.curves_png,'Resolution',180);savefig(g,artifacts.curves_fig);close(g);g=[];
    for k=1:numel(inputs),assert(strcmp(hash_file(resolve_path(projectRoot,inputs(k).file)),inputs(k).sha256), ...
            'plot_mature_comparison:ChangedInput','Input changed during plotting: %s',inputs(k).file);end
    outputFields=setdiff(fieldnames(artifacts),{'manifest_json'});outputHashes=struct([]);
    for k=1:numel(outputFields)
        p=artifacts.(outputFields{k});item.role=outputFields{k};item.file=relative_label(projectRoot,p);item.sha256=hash_file(p);
        if isempty(outputHashes),outputHashes=item;else,outputHashes(end+1)=item;end %#ok<AGROW>
    end
    manifest.schema_version='mature_comparison_plot_v1';manifest.status='PASS';
    manifest.created_utc=char(datetime('now','TimeZone','UTC','Format',"yyyy-MM-dd'T'HH:mm:ss'Z'"));
    manifest.generator.file=relative_label(projectRoot,[mfilename('fullpath') '.m']);
    manifest.generator.sha256=hash_file([mfilename('fullpath') '.m']);
    manifest.scope='Two completed models, three recorded runs each, fixed validation400; common recipe, not heldout test, CI or significance';
    manifest.inputs=inputs;manifest.methods=sourceMethods;manifest.outputs=outputHashes;
    manifest.points_policy='All three recorded seeds shown, deterministic horizontal offsets only; undefined remains absent, never zero-imputed';
    manifest.statistics_policy='Arithmetic mean and sample SD (n-1) of defined values, cross-checked against JSON and CSV; n/3 shown; SD undefined below2 defined runs';
    manifest.selection_policy='Use existing validation-IoU-selected checkpoint for every metric; no new selection or threshold tuning';
    manifest.curves_policy='Actual epoch0..actualLast only, no extrapolation, tail filling or across-seed mean curves';
    manifest.statistics=statRows;manifest.visual_qa='Pending actual PNG inspection after MATLAB execution';
    manifest.raw_images_or_test_data_opened=0;manifest.MAT_files_opened=0;manifest.training_started=false;
    manifest.formal_run_files_modified=false;
    write_json(artifacts.manifest_json,manifest);
catch err
    if ~isempty(f)&&isgraphics(f),close(f);end
    if ~isempty(g)&&isgraphics(g),close(g);end
    failure.status='FAILED';failure.message=err.message;failure.identifier=err.identifier;failure.stack=err.stack;
    write_json(fullfile(outDir,'plot_failure.json'),failure);rethrow(err);
end
fprintf('MATURE_COMPARISON_PLOTS_COMPLETE models=2 seeds_per_model=3 points=%d outputs=%d no_test=1\n',numel(pointRows),numel(outputHashes));

    function add_input(path,role)
        path=canonical_path(path);assert(is_inside(path,projectRoot));
        [~,~,extension]=fileparts(path);assert(ismember(lower(extension),{'.json','.csv'}),'Only JSON/CSV input metadata may be read');
        label=relative_label(projectRoot,path);
        if ~isempty(inputs)&&any(strcmp({inputs.file},label)),return;end
        entry.role=role;entry.file=label;entry.sha256=hash_file(path);
        if isempty(inputs),inputs=entry;else,inputs(end+1)=entry;end %#ok<AGROW>
    end
    function require_hash(path,expected)
        add_input(path,'Actual provenance source');assert(strcmp(hash_file(path),char(string(expected))), ...
            'plot_mature_comparison:SourceSHA','Source SHA changed since comparison was built: %s',path);
    end
end

function root=find_project_root
root=fileparts(mfilename('fullpath'));
while ~(isfile(fullfile(root,'WORK_PLAN.md'))&&isfolder(fullfile(root,'configs')))
    parent=fileparts(root);assert(~strcmp(parent,root),'Cannot locate project root; pass it explicitly');root=parent;
end
end
function path=canonical_path(path)
path=char(java.io.File(char(string(path))).getCanonicalPath());
end
function path=resolve_path(root,path)
path=char(string(path));
if isempty(regexp(path,'^([A-Za-z]:[\\/]|[\\/])','once')),path=fullfile(root,path);end
path=canonical_path(path);
end
function tf=is_inside(path,parent)
path=canonical_path(path);parent=canonical_path(parent);
tf=strcmpi(path,parent)||startsWith(lower(path),[lower(parent) filesep]);
end
function label=relative_label(root,path)
root=canonical_path(root);path=canonical_path(path);assert(is_inside(path,root));
if strcmpi(path,root),label='.';else,label=strrep(path(numel(root)+2:end),filesep,'/');end
end
function value=read_json(path)
assert(isfile(path),'Missing completed-source JSON: %s',path);
value=jsondecode(fileread(path));
end
function t=read_csv(path)
assert(isfile(path),'Missing comparison CSV: %s',path);
options=detectImportOptions(path,'Encoding','UTF-8','VariableNamingRule','preserve');
options=setvartype(options,options.VariableNames,'string');t=readtable(path,options);
end
function v=number(s,field)
assert(isfield(s,field),'Missing metric/scalar field: %s',field);v=s.(field);
if isempty(v),v=NaN;else,assert(isnumeric(v)&&isscalar(v)&&isreal(v));v=double(v);end
assert(isnan(v)||isfinite(v));
end
function v=numeric_vector(raw)
if iscell(raw),v=cellfun(@empty_to_nan,raw);else,v=double(raw);end
v=v(:)';assert(all(isnan(v)|isfinite(v)));
end
function v=empty_to_nan(value)
if isempty(value),v=NaN;else,assert(isnumeric(value)&&isscalar(value));v=double(value);end
end
function v=csv_numbers(raw)
v=str2double(raw);v(ismissing(raw)|strlength(strtrim(raw))==0)=NaN;
end
function v=csv_number(t,index,field)
assert(ismember(field,t.Properties.VariableNames));column=t.(field);v=csv_numbers(column(index));
end
function same_number(actual,expected,label)
assert((isnan(actual)&&isnan(expected))||(isfinite(actual)&&isfinite(expected)&& ...
    abs(actual-expected)<=1e-10*max(1,abs(expected))), ...
    'plot_mature_comparison:ValueMismatch','Comparison value differs: %s',label);
end
function assert_metric(value)
assert(isnan(value)||(isfinite(value)&&value>=0&&value<=1),'Invalid validation metric');
end
function assert_scores(left,right,metrics,label)
for b=1:numel(metrics),same_number(number(left,metrics{b}),number(right,metrics{b}),[label '/' metrics{b}]);end
end
function hash=hash_file(path)
fid=fopen(path,'rb');assert(fid>=0,'Cannot hash input: %s',path);cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
md=java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid)
    bytes=fread(fid,1024*1024,'*uint8');if ~isempty(bytes),md.update(typecast(bytes,'int8'));end
end
digest=typecast(md.digest(),'uint8');hash=lower(reshape(dec2hex(digest,2).',1,[]));
end
function write_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
