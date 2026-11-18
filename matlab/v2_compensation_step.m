function [output,state,diagnostic]=v2_compensation_step(rgb,rawProbability,observation,state,cfg)
% Causal stream API. State stores raw observations only; [] resets a sequence.
% observation: timestamp_s, odometry_px, odometry_available. No labels/source IDs.
assert(isa(rgb,'uint8')&&isequal(size(rgb),[256 256 3]));
assert(isequal(size(rawProbability),[256 256])&&all(isfinite(rawProbability)&rawProbability>=0&rawProbability<=1,'all'));
assert(isfinite(observation.timestamp_s)&&isscalar(observation.odometry_available));
if isempty(state)
    state=struct('schema','motion_evidence_v2','images',{{}},'raw_probabilities',{{}},'observations',{{}},'edges',{{}},'frames_seen',0);
else
    assert(strcmp(state.schema,'motion_evidence_v2')&&observation.timestamp_s>state.observations{end}.timestamp_s);
end
if ~isempty(state.images)
    previous=state.observations{end};delta=NaN;
    if previous.odometry_available&&observation.odometry_available&&isscalar(previous.odometry_px)&&isscalar(observation.odometry_px)
        delta=observation.odometry_px-previous.odometry_px;
    end
    state.edges{end+1}=v2_motion_weights(state.images{end},rgb,delta,cfg);
end
state.images{end+1}=rgb;state.raw_probabilities{end+1}=single(rawProbability);state.observations{end+1}=observation;state.frames_seen=state.frames_seen+1;
if numel(state.images)>cfg.window_frames
    state.images=state.images(2:end);state.raw_probabilities=state.raw_probabilities(2:end);state.observations=state.observations(2:end);state.edges=state.edges(2:end);
end
[output,diagnostic]=v2_fuse_window(state.raw_probabilities,state.edges,cfg);
diagnostic.frames_seen=state.frames_seen;
end
