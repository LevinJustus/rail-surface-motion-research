function [net,modelCfg,path]=build_seeded_baseline(seed,baseConfigPath,outDir)
%BUILD_SEEDED_BASELINE Change initialization seed only, keeping parent immutable.
assert(isscalar(seed)&&seed==fix(seed)&&seed>=0);
if ~exist(outDir,'dir'),mkdir(outDir);end
modelCfg=jsondecode(fileread(baseConfigPath));modelCfg.random_seed=seed;
path=fullfile(outDir,'model_config_seed.json');
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);
fprintf(fid,'%s\n',jsonencode(modelCfg,PrettyPrint=true));fclose(fid);
[net,actual]=build_baseline(path);assert(actual.random_seed==seed);
end
