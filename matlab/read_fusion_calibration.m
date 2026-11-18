function freeze=read_fusion_calibration(path)
% MATLAB can decode identical object fields in differing orders as a cell.
% Normalize field order in memory; never rewrite frozen threshold bytes.
freeze=jsondecode(fileread(path));
if iscell(freeze.choices)
    choices=cellfun(@orderfields,freeze.choices,'UniformOutput',false);
    freeze.choices=vertcat(choices{:});
end
assert(isstruct(freeze.choices)&&numel(freeze.choices)==24);
end
