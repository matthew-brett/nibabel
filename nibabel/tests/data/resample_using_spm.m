% Script uses SPM to resample moved anatomical image
%
% Run from the directory containing this file
P = {'anatomical.nii', ...
     'func_moved.nii,1', 'func_moved.nii,2', 'func_moved.nii,3'};
% Resample without masking
flags = struct('mask', false, 'mean', false, ...
               'interp', 3, 'which', 1, ...
               'prefix', 'resliced_');
spm_reslice(P, flags);
% Reorient to canonical orientation at 1mm resolution
to_reorient = char(P(2:end));
reorient(to_reorient);
