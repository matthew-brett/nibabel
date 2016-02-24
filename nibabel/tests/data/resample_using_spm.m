% Script uses SPM to resample moved anatomical image
%
% Run from the directory containing this file
P = {'anatomical.nii', ...
     'functiontal.nii,1', 'functional.nii,2', 'functional.nii,3'};
% Resample without masking
flags = struct('mask', false, 'mean', false, ...
               'interp', 3, 'which', 1, ...
               'prefix', 'r');
spm_reslice(P, flags);
