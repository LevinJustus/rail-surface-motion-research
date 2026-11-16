function artifacts = plot_repeat_summary(groupRunDir)
%PLOT_REPEAT_SUMMARY Descriptive comparison of the three completed full runs.
% Call only after the group AND all three seed summaries report PASS.
% Reads summary/validation JSON metadata only: no images, networks, training,
% checkpoints, heldout-test data, or new checkpoint/threshold selection.
% Epoch 0 is the untrained reference. Each seed curve ends at its actual last
% evaluated epoch; early-stopped tails are not filled or extrapolated.
% Selected means the checkpoint already selected by validation IoU, not the
% checkpoint with the largest Dice. Three-run SD describes seed variation
% (initialization and dropout) under this fixed validation protocol, not
% significance or testing. Data order is shared across the three seeds.

groupRunDir = char(string(groupRunDir));
assert(isfolder(groupRunDir),'plot_repeat_summary:MissingRun', ...
    'Group run directory does not exist: %s',groupRunDir);
group = read_json(fullfile(groupRunDir,'summary.json'));
assert_pass(group,'group summary');
assert_fields(group,{'initialization_seeds','training_images', ...
    'validation_images','heldout_test_images_opened'},'group summary');
seeds = double(group.initialization_seeds(:)');
assert(isequal(seeds,[20261114 20261115 20261116]), ...
    'plot_repeat_summary:Seeds','Expected the three predefined random seeds');
assert(scalar_number(group,'training_images')==1396 && ...
    scalar_number(group,'validation_images')==400 && ...
    scalar_number(group,'heldout_test_images_opened')==0, ...
    'plot_repeat_summary:Scope','Expected full1396/val400 with no heldout-test access');

n = numel(seeds);
reports = cell(1,n); evaluations = cell(1,n);
summaryPaths = strings(n,1); evaluationPaths = strings(n,1);
% Check all three PASS gates before reading curves or writing any artifacts.
for k = 1:n
    seedDir = fullfile(groupRunDir,'seeds',sprintf('seed_%d',seeds(k)));
    summaryPaths(k) = string(fullfile(seedDir,'summary.json'));
    reports{k} = read_json(summaryPaths(k));
    assert_pass(reports{k},sprintf('seed %d summary',seeds(k)));
end

seedRows = struct([]); epochRows = struct([]);
for k = 1:n
    r = reports{k};
    assert_fields(r,{'initialization_seed','completed_epochs','max_epochs', ...
        'main_optimizer_updates','stopped_early','stop_reason','best_validation', ...
        'final_validation','heldout_test_images_opened'},'seed summary');
    assert(scalar_number(r,'initialization_seed')==seeds(k) && ...
        scalar_number(r,'heldout_test_images_opened')==0, ...
        'plot_repeat_summary:SeedScope','Seed identity or heldout-test access mismatch');
    completed = scalar_number(r,'completed_epochs');
    maximum = scalar_number(r,'max_epochs');
    assert(isfinite(completed) && completed>=1 && completed==fix(completed) && ...
        isfinite(maximum) && maximum>=completed && maximum==fix(maximum), ...
        'plot_repeat_summary:Epochs','Invalid completed/max epoch counts');
    stopped = scalar_flag(r,'stopped_early');
    assert(stopped==(completed<maximum), ...
        'plot_repeat_summary:EarlyStop','Early-stop flag disagrees with actual epoch count');
    evaluationPaths(k) = string(fullfile(groupRunDir,'seeds', ...
        sprintf('seed_%d',seeds(k)),'validation_evaluations.json'));
    e = read_json(evaluationPaths(k));
    assert(isstruct(e) && ~isempty(e), ...
        'plot_repeat_summary:Evaluations','Validation evaluations must be a nonempty struct array');
    e = e(:);
    assert_fields(e,{'epoch','global_update','total_loss','micro_foreground_IoU', ...
        'micro_foreground_Dice','sample_count','probability_checks'},'validation evaluations');
    epochs = numeric_fields(e,'epoch'); updates = numeric_fields(e,'global_update');
    assert(isequal(epochs(:)',0:completed), ...
        'plot_repeat_summary:EpochCoverage', ...
        'Validation log must contain actual epoch 0 through actual last, once each');
    assert(updates(1)==0 && all(isfinite(updates)) && ...
        all(updates==fix(updates)) && all(diff(updates)>0) && ...
        updates(end)==scalar_number(r,'main_optimizer_updates'), ...
        'plot_repeat_summary:Updates','Validation update order or actual last update mismatch');
    assert(all(numeric_fields(e,'sample_count')==400), ...
        'plot_repeat_summary:ValidationCount','Every evaluation must use the same fixed val400');
    assert(all(isfinite(numeric_fields(e,'total_loss'))), ...
        'plot_repeat_summary:Loss','Validation loss must be finite');
    for metric = {'micro_foreground_IoU','micro_foreground_Dice'}
        values = numeric_fields(e,metric{1});
        assert(all(isnan(values) | (isfinite(values) & values>=0 & values<=1)), ...
            'plot_repeat_summary:MetricRange','Metrics must be defined in [0,1] or undefined NaN');
    end
    for j = 1:numel(e)
        assert_fields(e(j).probability_checks,{'all_finite'},'probability checks');
        assert(scalar_flag(e(j).probability_checks,'all_finite'), ...
            'plot_repeat_summary:Probabilities','Validation reports nonfinite probabilities');
    end
    assert_same_evaluation(r.final_validation,e(end),'actual last');
    best = r.best_validation;
    assert_fields(best,{'epoch','global_update','total_loss', ...
        'micro_foreground_IoU','micro_foreground_Dice'},'selected checkpoint');
    bestEpoch = scalar_number(best,'epoch'); bestUpdate = scalar_number(best,'global_update');
    if isfinite(bestEpoch) && isfinite(bestUpdate)
        chosen = find(epochs==bestEpoch & updates==bestUpdate);
        assert(isscalar(chosen),'plot_repeat_summary:SelectedPoint', ...
            'Selected checkpoint must correspond to an actual logged evaluation');
        assert_same_evaluation(best,e(chosen),'validation-selected checkpoint');
        selectedUntrained = double(bestUpdate==0);
    else
        assert(isnan(bestEpoch) && isnan(bestUpdate) && ...
            isnan(scalar_number(best,'micro_foreground_IoU')) && ...
            isnan(scalar_number(best,'micro_foreground_Dice')), ...
            'plot_repeat_summary:UnavailableBest','Unavailable selection must remain explicitly undefined');
        selectedUntrained = NaN;
    end
    evaluations{k} = e;
    row.initialization_seed = seeds(k);
    row.completed_epochs = completed; row.max_epochs = maximum;
    row.main_optimizer_updates = updates(end);
    row.selected_epoch = bestEpoch; row.selected_global_update = bestUpdate;
    row.selected_is_untrained_reference = selectedUntrained;
    row.selected_total_loss = scalar_number(best,'total_loss');
    row.selected_micro_foreground_IoU = scalar_number(best,'micro_foreground_IoU');
    row.selected_micro_foreground_Dice = scalar_number(best,'micro_foreground_Dice');
    row.actual_last_epoch = epochs(end); row.actual_last_global_update = updates(end);
    row.actual_last_total_loss = scalar_number(e(end),'total_loss');
    row.actual_last_micro_foreground_IoU = scalar_number(e(end),'micro_foreground_IoU');
    row.actual_last_micro_foreground_Dice = scalar_number(e(end),'micro_foreground_Dice');
    row.stopped_early = stopped; row.stop_reason = string(r.stop_reason);
    row.validation_images = 400; row.heldout_test_images_opened = 0;
    row.scope = "Fixed val400; descriptive seed variation (initialization and dropout); not heldout test";
    if isempty(seedRows), seedRows=row; else, seedRows(end+1)=row; end %#ok<AGROW>
    for j = 1:numel(e)
        point.initialization_seed = seeds(k); point.epoch = epochs(j);
        point.global_update = updates(j); point.total_loss = scalar_number(e(j),'total_loss');
        point.micro_foreground_IoU = scalar_number(e(j),'micro_foreground_IoU');
        point.micro_foreground_Dice = scalar_number(e(j),'micro_foreground_Dice');
        point.is_untrained_reference = updates(j)==0;
        point.is_validation_selected = epochs(j)==bestEpoch && updates(j)==bestUpdate;
        point.is_actual_last = j==numel(e); point.validation_images = 400;
        if isempty(epochRows), epochRows=point; else, epochRows(end+1)=point; end %#ok<AGROW>
    end
end

statistics = struct([]);
roles = {'selected','actual_last'};
metrics = {'micro_foreground_IoU','micro_foreground_Dice'};
for a = 1:numel(roles)
    for b = 1:numel(metrics)
        values = [seedRows.([roles{a} '_' metrics{b}])];
        s.checkpoint_role = string(roles{a}); s.metric = string(metrics{b});
        s.total_runs = n; s.defined_count = nnz(isfinite(values));
        % Never relabel a subset of defined values as a three-seed mean/SD.
        if s.defined_count==n
            s.mean = mean(values); s.sample_std = std(values,0);
        else
            s.mean = NaN; s.sample_std = NaN;
        end
        s.sd_definition = "Sample SD, denominator n-1; only populated when all three are defined";
        s.scope = "Three random-seed runs (initialization and dropout) on fixed validation; no significance or heldout-test claim";
        if isempty(statistics), statistics=s; else, statistics(end+1)=s; end %#ok<AGROW>
    end
end

% All input checks have passed. Only derived comparison artifacts are written.
plotDir = fullfile(groupRunDir,'plots');
if ~isfolder(plotDir), mkdir(plotDir); end
artifacts.seed_comparison_csv = fullfile(plotDir,'repeat_seed_comparison.csv');
artifacts.validation_epochs_csv = fullfile(plotDir,'repeat_validation_epochs.csv');
artifacts.descriptive_statistics_csv = fullfile(plotDir,'repeat_descriptive_statistics.csv');
artifacts.summary_png = fullfile(plotDir,'repeat_validation_summary.png');
artifacts.summary_fig = fullfile(plotDir,'repeat_validation_summary.fig');
artifacts.manifest_json = fullfile(plotDir,'repeat_summary_manifest.json');
writetable(struct2table(seedRows),artifacts.seed_comparison_csv);
writetable(struct2table(epochRows),artifacts.validation_epochs_csv);
writetable(struct2table(statistics),artifacts.descriptive_statistics_csv);

f = figure('Visible','off','Color','white','Position',[40 40 1350 920]);
closeFigure = onCleanup(@()close(f)); %#ok<NASGU>
tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
colors = lines(n); maximum = max([seedRows.max_epochs]);
for b = 1:numel(metrics)
    ax = nexttile; hold(ax,'on');
    for k = 1:n
        e = evaluations{k};
        plot(ax,numeric_fields(e,'epoch'),numeric_fields(e,metrics{b}),'-o', ...
            'Color',colors(k,:),'LineWidth',1.5,'MarkerSize',4, ...
            'DisplayName',sprintf('Seed %d (last epoch %d)',seeds(k),seedRows(k).completed_epochs));
    end
    grid(ax,'on'); xlim(ax,[0 maximum]); ylim(ax,[0 1]);
    xticks(ax,0:2:maximum); xlabel(ax,'Actual evaluated epoch (0 = untrained)');
    ylabel(ax,metric_label(metrics{b}));
    title(ax,sprintf('Val400 %s: curves end at actual last epoch',metric_label(metrics{b})));
    legend(ax,'Location','best','Interpreter','none');
end
for b = 1:numel(metrics)
    ax = nexttile;
    selected = [seedRows.(['selected_' metrics{b}])];
    last = [seedRows.(['actual_last_' metrics{b}])];
    bars = bar(ax,1:n,[selected(:) last(:)],'grouped');
    bars(1).FaceColor = [.2 .45 .7]; bars(2).FaceColor = [.85 .5 .2];
    xticks(ax,1:n); xticklabels(ax,string(seeds)); ylim(ax,[0 1]); grid(ax,'on');
    xlabel(ax,'Random seed'); ylabel(ax,metric_label(metrics{b}));
    title(ax,sprintf('Val400 %s: selected checkpoint versus actual last',metric_label(metrics{b})));
    legend(ax,{'Validation IoU-selected checkpoint','Actual last checkpoint'},'Location','best');
end
sgtitle(f,{'Full single-frame baseline: three predefined random seeds, fixed val400', ...
    'Validation only, not heldout test; no extrapolated tails, averaged curves, CI, or significance claim'}, ...
    'Interpreter','none','FontSize',13);
exportgraphics(f,artifacts.summary_png,'Resolution',180);
savefig(f,artifacts.summary_fig);

manifest.schema_version = 'repeat_summary_v1'; manifest.status = 'PASS';
manifest.initialization_seeds = seeds;
manifest.scope = 'Descriptive seed variation (initialization and dropout) on fixed validation only; not heldout test or significance';
manifest.inputs.group_summary = fullfile(groupRunDir,'summary.json');
manifest.inputs.seed_summaries = cellstr(summaryPaths);
manifest.inputs.validation_evaluations = cellstr(evaluationPaths);
manifest.curve_policy = 'Actual logged epoch0..actualLast per seed; no missing-tail imputation, extrapolation, or across-seed mean curves';
manifest.selection_policy = 'Use existing validation-selected checkpoint; no new selection, threshold tuning, or access to test';
manifest.statistics_policy = 'Mean and sample SD (n-1) only when all three metric values are finite; three runs describe seed variation (initialization and dropout), with shared data order';
manifest.statistics = statistics; manifest.artifacts = artifacts;
write_json(artifacts.manifest_json,manifest);
end

function value = read_json(path)
path = char(string(path));
assert(isfile(path),'plot_repeat_summary:MissingInput', ...
    'Required completed-run metadata is missing: %s',path);
value = jsondecode(fileread(path));
end

function assert_pass(value,label)
assert(isstruct(value) && isscalar(value) && isfield(value,'status') && ...
    strcmp(string(value.status),"PASS"), ...
    'plot_repeat_summary:NotComplete','%s must report PASS before this helper may run',label);
end

function assert_fields(value,names,label)
assert(isstruct(value) && all(isfield(value,names)), ...
    'plot_repeat_summary:Fields','%s is missing required fields',label);
end

function value = scalar_number(s,name)
assert(isstruct(s) && isscalar(s) && isfield(s,name), ...
    'plot_repeat_summary:ScalarField','Missing numeric scalar field: %s',name);
value = s.(name);
% JSON null represents undefined NaN metrics and decodes to empty double.
if isempty(value), value=NaN; end
assert(isnumeric(value) && isreal(value) && isscalar(value), ...
    'plot_repeat_summary:NumericField','Expected numeric scalar or null: %s',name);
value = double(value);
end

function value = scalar_flag(s,name)
assert(isstruct(s) && isscalar(s) && isfield(s,name), ...
    'plot_repeat_summary:FlagField','Missing Boolean scalar field: %s',name);
value = s.(name);
assert((islogical(value) || isnumeric(value)) && isscalar(value) && ...
    isfinite(value) && (value==0 || value==1), ...
    'plot_repeat_summary:BooleanField','Expected Boolean scalar: %s',name);
value = logical(value);
end

function values = numeric_fields(s,name)
values = arrayfun(@(x)scalar_number(x,name),s);
values = double(values(:)');
end

function assert_same_evaluation(a,b,label)
fields = {'epoch','global_update','total_loss', ...
    'micro_foreground_IoU','micro_foreground_Dice'};
for k = 1:numel(fields)
    x = scalar_number(a,fields{k}); y = scalar_number(b,fields{k});
    same = (isnan(x) && isnan(y)) || ...
        (isfinite(x) && isfinite(y) && abs(x-y)<=1e-10*max([1 abs(x) abs(y)]));
    assert(same,'plot_repeat_summary:SummaryMismatch', ...
        '%s does not match the logged evaluation field %s',label,fields{k});
end
end

function label = metric_label(name)
if strcmp(name,'micro_foreground_IoU'), label='foreground IoU';
else, label='foreground Dice'; end
end

function write_json(path,value)
fid = fopen(path,'w','n','UTF-8');
assert(fid>=0,'plot_repeat_summary:Output','Cannot write derived manifest');
closeFile = onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
