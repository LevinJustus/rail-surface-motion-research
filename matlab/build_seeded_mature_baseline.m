function [net,modelCfg,path,architecture]=build_seeded_mature_baseline(seed,baseConfigPath,outDir)
%BUILD_SEEDED_MATURE_BASELINE Fresh random MobileNetV2/DeepLab project baseline.
% Only the run-local random seed changes; parent configuration is immutable.
assert(isscalar(seed)&&seed==fix(seed)&&seed>=0);
if ~isfolder(outDir),mkdir(outDir);end
modelCfg=jsondecode(fileread(baseConfigPath));modelCfg.random_seed=seed;
path=fullfile(outDir,'model_config_seed.json');assert(~isfile(path),'Seed model config already exists');
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(modelCfg,PrettyPrint=true));
clear cleanup;
[net,actual,architecture]=build_mobilenet_deeplab(path);
assert(actual.random_seed==seed&&strcmp(actual.backbone_weights,'none'));
end
