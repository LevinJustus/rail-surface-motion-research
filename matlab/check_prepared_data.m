function check_prepared_data(dataRoot)
% Exercise actual training data and the NEU logical-label reader, without training.
root=fileparts(fileparts(mfilename('fullpath')));
manifest=jsondecode(fileread(fullfile(root,'data','splits','conservative_group_v1.json')));
records=manifest.records;
index=find(strcmp({records.assigned_split},'training'),1);
assert(~isempty(index),'No training samples');
r=records(index);
[I,M,V,meta]=read_segmentation_sample(fullfile(dataRoot,r.image),fullfile(dataRoot,r.mask));
assert(all(isfinite(I),'all') && nnz(V)>0);
fprintf('ACTUAL_TRAIN_SAMPLE_OK %s valid=%d foreground=%d source=%dx%d\n', ...
    r.id,nnz(V),nnz(M),meta.source_hw);
[~,N,W]=read_segmentation_sample(fullfile(dataRoot,'neu_rsdds','extracted','2D_Image','INPUT_IMG','1.bmp'), ...
    fullfile(dataRoot,'neu_rsdds','extracted','Ground Truth','Ground Truth','1.png'));
assert(nnz(N)>0 && all(~N(~W)));
fprintf('ACTUAL_NEU_LOGICAL_LABEL_OK foreground=%d\n',nnz(N));
end
