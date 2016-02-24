function reorient(PP)
% Re-orient images
% The function reslices the input images to a resolution of 1mm.
% Output images (with the prefix "r") are written in the transverse
% orientation (using information from the ".mat" files).
%_______________________________________________________________________
% %W% John Ashburner %E%

if nargin<1, PP = spm_select(Inf,'*.nii','Select file to reorient'); end;
VV = spm_vol(PP);

for V=VV',
	d = V.dim(1:3);
	c = [	1    1    1    1
		1    1    d(3) 1
		1    d(2) 1    1
		1    d(2) 1    1
		d(1) 1    1    1
		d(1) 1    d(3) 1
		d(1) d(2) 1    1
		d(1) d(2) 1    1]';

	tc = V.mat(1:3,1:4)*c;
	mx = round(max(tc,[],2)');
	mn = round(min(tc,[],2)');

	mat = spm_matrix([mn-1]);
	dim = (mat\[mx 1]')';

	VO               = V;
	[lpath,name,ext] = fileparts(V.fname);
	VO.fname         = fullfile(lpath,['r' name ext]);
	VO.dim(1:3)      = dim(1:3);
	VO.mat           = mat;


	spm_progress_bar('Init',dim(3),'reslicing...','planes completed');
	VO = spm_create_vol(VO);
	for i=1:dim(3),
		M   = inv(spm_matrix([0 0 -i])*inv(VO.mat)*V.mat);
		img = spm_slice_vol(V,M,dim(1:2),1);
		spm_write_plane(VO,img,i);
		spm_progress_bar('Set',i)
	end;
	spm_progress_bar('Clear');
end;
return;
