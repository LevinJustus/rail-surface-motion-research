function check_baseline_environment()
% Validate construction, class probability layout and normalization on CPU.
% Random initial weights: output is not a trained defect prediction.
root=fileparts(fileparts(mfilename('fullpath')));
out=fullfile(root,'runs','environment_check');
if ~exist(out,'dir'),mkdir(out);end
[net,cfg]=build_baseline();
n=0;
for k=1:height(net.Learnables),n=n+numel(net.Learnables.Value{k});end
X=dlarray(zeros([cfg.input_size_hwc(:)' 1],'single'),'SSCB');
Y=predict(net,X);
prob=extractdata(Y);
assert(isequal(size(prob,[1 2 3]),[cfg.input_size_hwc(1:2)' cfg.num_classes]));
sumError=max(abs(sum(prob,3)-1),[],'all');
assert(all(isfinite(prob),'all') && sumError<1e-5);
report.matlab_version=version;
report.baseline_id=cfg.baseline_id;
report.parameters=n;
report.input_size_hwc=cfg.input_size_hwc;
report.output_size_hwc=size(prob,[1 2 3]);
report.max_probability_sum_error=sumError;
report.execution='CPU forward pass only; no training, accuracy or FPS evaluation';
report.channel_order=cfg.classes_in_channel_order;
report.gpu_count=0;
try
    report.gpu_count=gpuDeviceCount;
    if report.gpu_count>0
        g=gpuDevice;report.gpu_name=g.Name;report.gpu_available_memory_bytes=g.AvailableMemory;
    end
catch err
    report.gpu_query_status=err.message;
end
report.system_memory_query='not available';
try
    [~,s]=memory;report.physical_memory_bytes=s.PhysicalMemory.Total;
catch
end
fid=fopen(fullfile(out,'environment.json'),'w','n','UTF-8');
assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));
fprintf('BASELINE_FORWARD_OK parameters=%d input=%dx%dx%d output=%dx%dx%d GPUs=%d\n', ...
    n,cfg.input_size_hwc,report.output_size_hwc,report.gpu_count);
end
