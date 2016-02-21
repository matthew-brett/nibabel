""" Make anatomical image with altered affine

* Add some rotations and translations to affine;
* Save as ``.nii`` file so SPM can read it.

See ``resample_using_spm.m`` for processing of this generated image by SPM.
"""

import nibabel as nib
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec

img = nib.load('anatomical.nii')
some_rotations = euler2mat(0.1, 0.2, 0.3)
extra_affine = from_matvec(some_rotations, [3, 4, 5])
mean_img_moved = nib.Nifti1Image(img.dataobj,
                                 extra_affine.dot(img.affine),
                                 img.header)
nib.save(mean_img_moved, 'anat_moved.nii')
