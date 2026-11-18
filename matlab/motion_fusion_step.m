function [state,out,diagnostic]=motion_fusion_step(state,frame,probability,cfg,constant,cachedEdge)
% Arrived-frame API. A cached edge is optional and must describe this pair.
allowed={'frame_hw','window_frames','lookahead_frames','timestamp_gap_reset_s','expected_interval_s',...
    'nonincreasing_timestamp','reverse_or_stop','sequence_boundary','primary_state','compute_type','output_type',...
    'range_tolerance','weight_rule','invalid_history','registration','warp','visual_gate','observed_gate','bounded_recursive','methods'};
assert(isequal(sort(string(fieldnames(cfg))),sort(string(allowed(:)))),'Unexpected inference config field');
assert(isequal(sort(string(fieldnames(frame))),sort(["image";"timestamp_s";"simulated_odometry_px";"odometry_available"])),'Unexpected frame field');
assert(isa(frame.image,'uint8')&&isequal(size(frame.image),[256 256 3])&&isscalar(frame.timestamp_s)&&isfinite(frame.timestamp_s));
assert(isscalar(frame.odometry_available)&&islogical(frame.odometry_available));
assert(isscalar(frame.simulated_odometry_px) && (~frame.odometry_available || isfinite(frame.simulated_odometry_px)));
assert(cfg.window_frames==3 && cfg.lookahead_frames==0);
if isempty(state),state=struct('frames',{{}},'probabilities',{{}},'edges',{{}});end
reset=false;edge=[];
if ~isempty(state.frames)
    previous=state.frames{end};dt=frame.timestamp_s-previous.timestamp_s;
    assert(dt>0,'Nonincreasing timestamp');
    if dt>cfg.timestamp_gap_reset_s
        state=struct('frames',{{}},'probabilities',{{}},'edges',{{}});reset=true;
    else
        delta=NaN;
        if previous.odometry_available&&frame.odometry_available,delta=frame.simulated_odometry_px-previous.simulated_odometry_px;end
        if nargin<6 || isempty(cachedEdge),edge=register_fusion_pair(previous.image,frame.image,delta,cfg);
        else
            edge=cachedEdge;assert(isequaln(edge.observed_delta,delta),'Cached edge does not match arrived odometry');
        end
    end
end
state.frames{end+1}=frame;state.probabilities{end+1}=probability;
if numel(state.frames)>1,state.edges{end+1}=edge;end
if numel(state.frames)>cfg.window_frames
    state.frames=state.frames(2:end);state.probabilities=state.probabilities(2:end);state.edges=state.edges(2:end);
end
images=cellfun(@(f)f.image,state.frames,'UniformOutput',false);
[out,paths]=fuse_motion_window(images,state.probabilities,state.edges,cfg,constant);
diagnostic=struct('reset_for_time_gap',reset,'window_size',numel(state.frames),'edge',edge,'paths',paths);
end
