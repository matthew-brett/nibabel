% Script uses SPM to resample moved anatomical image
%
% Run from the directory containing this file
P = {'functional.nii', 'anat_moved.nii'};
% Resample without masking
flags = struct('mask', false, 'mean', false, ...
               'interp', 3, 'which', 1, ...
               'prefix', 'r');
spm_reslice(P, flags);
