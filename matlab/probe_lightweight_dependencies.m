function report=probe_lightweight_dependencies()
%PROBE_LIGHTWEIGHT_DEPENDENCIES Read installed APIs, construct only; no images/training.
root=fileparts(fileparts(mfilename('fullpath')));
out=fullfile(root,'runs','step11_preflight');if ~isfolder(out),mkdir(out);end
path=fullfile(out,'dependency_probe.json');assert(~isfile(path),'Probe already recorded');
report.matlab_version=version;report.computer=computer;
report.date_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
report.raw_images_opened=0;report.optimizer_updates=0;report.gpu_count=gpuDeviceCount;
names={'mobilenetv2','deeplabv3plus','deeplabv3plusLayers','enetLayers','enet','bisenetv2','resize2dLayer'};
for k=1:numel(names),report.available_functions.(names{k})=which(names{k});end
report.mobilenet_support_marker=which('nnet.internal.cnn.supportpackages.IsMobileNetV2Installed');
try
    graph=mobilenetv2('Weights','none');
    report.untrained_mobilenet.status='PASS';report.untrained_mobilenet.class=class(graph);
    report.untrained_mobilenet.layer_names={graph.Layers.Name};
    save(fullfile(out,'untrained_mobilenet_graph.mat'),'graph');
catch err
    report.untrained_mobilenet.status='UNAVAILABLE';report.untrained_mobilenet.identifier=err.identifier;
    report.untrained_mobilenet.message=err.message;
end
try
    net=deeplabv3plus([256 256 3],2,'mobilenetv2',DownsamplingFactor=16);
    report.official_deeplab.status='PASS';report.official_deeplab.class=class(net);
    report.official_deeplab.layer_count=numel(net.Layers);
    report.official_deeplab.note='Default constructor may use pretrained backbone; not a random-initialization comparison.';
    save(fullfile(out,'official_deeplab_initial.mat'),'net');
catch err
    report.official_deeplab.status='UNAVAILABLE';report.official_deeplab.identifier=err.identifier;
    report.official_deeplab.message=err.message;
end
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
disp(jsonencode(report,PrettyPrint=true));
end
