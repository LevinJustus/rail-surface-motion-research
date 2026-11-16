function report=check_seed_initialization(outDir)
%CHECK_SEED_INITIALIZATION Same seed identical, distinct seeds distinct weights.
root=fileparts(fileparts(mfilename('fullpath')));
base=fullfile(root,'configs','baseline.json');before=file_sha256(base);
seeds=[20261114 20261115 20261116 20261114];hashes=cell(1,4);counts=zeros(1,4);
for k=1:4
    [net,~,~]=build_seeded_baseline(seeds(k),base,fullfile(outDir,sprintf('fixture_%d',k)));
    hashes{k}=network_parameter_sha256(net);
    for j=1:height(net.Learnables),counts(k)=counts(k)+numel(net.Learnables.Value{j});end
end
assert(strcmp(hashes{1},hashes{4})&&numel(unique(string(hashes(1:3))))==3);
assert(all(counts==482050)&&strcmp(before,file_sha256(base)));
report.status='PASS';report.synthetic_construction_only=true;report.optimizer_updates=0;
report.seeds=seeds;report.parameter_sha256=hashes;report.parameter_counts=counts;
report.parent_model_config_unchanged=true;
fid=fopen(fullfile(outDir,'seed_initialization_check.json'),'w','n','UTF-8');assert(fid>=0);
c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
fprintf('SEED_INITIALIZATION_CHECK_OK sameSeedIdentical distinctSeedsDifferent parentUnchanged\n');
end
