from os.path import join as pjoin
import glob
from warnings import warn
from distutils.version import LooseVersion

import numpy as np

from ..volumeutils import allopen
from .dicomwrappers import wrapper_from_data, wrapper_from_file
from ..optpkg import optional_package

dicom, HAVE_DICOM, _ = optional_package('dicom')

if HAVE_DICOM:
    # Earlier pydicom versions break when trying to reread deferred gz file
    # object dicoms
    REUSABLE_GZ_DCM = LooseVersion(dicom.__version__) >= LooseVersion('0.9.7')


class DicomReadError(Exception):
    pass


DPCS_TO_TAL = np.diag([-1, -1, 1, 1])


def mosaic_to_nii(dcm_data):
    ''' Get Nifti file from Siemens

    Parameters
    ----------
    dcm_data : ``dicom.DataSet``
       DICOM header / image as read by ``dicom`` package

    Returns
    -------
    img : ``Nifti1Image``
       Nifti image object
    '''
    import nibabel as nib
    dcm_w = wrapper_from_data(dcm_data)
    if not dcm_w.is_mosaic:
        raise DicomReadError('data does not appear to be in mosaic format')
    data = dcm_w.get_data()
    aff = np.dot(DPCS_TO_TAL, dcm_w.get_affine())
    return nib.Nifti1Image(data, aff)


def read_mosaic_dwi_dir(dicom_path, globber='*.dcm',
                        sort_func = None, ret_type='tuple',
                        dicom_args=None):
    return read_mosaic_dir(dicom_path, globber, check_is_dwi=True,
                           sort_func = sort_func, ret_type = ret_type,
                           dicom_args=dicom_args)


def read_mosaic_dir(dicom_path, globber='*.dcm', check_is_dwi=False,
                    sort_func = None, ret_type=None,
                    dicom_args=None):
    ''' Read all Siemens mosaic DICOMs in directory, return arrays, params

    Parameters
    ----------
    dicom_path : str
       path containing mosaic DICOM images
    globber : str, optional
       glob to apply within `dicom_path` to select DICOM files.  Default
       is ``*.dcm``
    check_is_dwi : bool, optional
       If True, raises an error if we don't find DWI information in the
       DICOM headers.
    sort_func : None or {'filename', 'glob', 'instance number'} or callable
       function to sort volumes.  Function should accept a single 2 element
       tuple as argument, where first value in tuple is filename and second is
       DICOM data object.  If None, sort by filename. If 'filename', sort by
       filename. If 'glob' use filename glob order.  If 'instance number' sort
       by DICOM InstanceNumber.
    ret_type : {None, 'tuple', 'dict'}
       If 'tuple', return results as tuple in order below.  If 'dict', return as
       dict with keys, values as listed below.  If None, return tuple (this
       default behavior will change to 'dict' in the next version of nibabel).
    dicom_args : dict, optional
       Optional parameters to pass to pydicom file reader

    Returns
    -------
    If `ret_type` == `tuple` return 4 element tuple with elements:

    data : 4D array
       data array with last dimension being acquisition. If there were N
       acquisitions, each of shape (X, Y, Z), `data` will be shape (X,
       Y, Z, N)
    affine : (4,4) array
       affine relating 3D voxel space in data to RAS world space
    b_values : (N,) array
       b values for each acquisition.  nan if we did not find diffusion
       information for these images.
    unit_gradients : (N, 3) array
       gradient directions of unit length for each acquisition.  (nan,
       nan, nan) if we did not find diffusion information.

    If `ret_type` == `dict` return dict with keys, values as above and:

    dcm0 : dicom data object
       dicom data object for first header (after sorting)
    '''
    if sort_func is None:
        warn('Default sort_func will change from "filename" to '
             '"instance number" in the next version of nibabel. '
             'Please change your code accurdingly',
             FutureWarning, stacklevel=2)
        sort_func = 'filename'
    if sort_func in ('filename', 'glob'):
        pass
    elif sort_func == 'instance number':
        sort_func = lambda x : x[1].InstanceNumber
    elif not callable(sort_func):
        raise TypeError('Curious sort_func %s' % sort_func)
    if ret_type is None:
        warn('Default `ret_type` will change from "tuple" to "dict" in the '
             'next version of nibabel. Please change your code accurdingly',
             FutureWarning, stacklevel=2)
        ret_type = 'tuple'
    elif ret_type not in ('tuple', 'dict'):
        raise ValueError('Expecting ret_type in ("tuple", "dict")')
    full_globber = pjoin(dicom_path, globber)
    filenames = glob.glob(full_globber)
    n_files = len(filenames)
    if n_files == 0:
        raise IOError('Found no files with "%s"' % full_globber)
    if dicom_args is None:
        dicom_args = {}
    datas = [None] * n_files
    if sort_func == 'glob':
        pass
    elif sort_func == 'filename':
        filenames = sorted(filenames)
    else: # Need to (partially) load the dicoms to do the sorting
        datas = [dicom.read_file(allopen(fname), defer_size=1000, **dicom_args)
                 for fname in filenames]
        # Sort filenames and data using sort func
        fns_data = sorted(zip(filenames, datas), key=sort_func)
        filenames, datas = [list(v) for v in zip(*fns_data)]
    # Initialize loop
    b_values = []
    gradients = []

    def dcm_generator():
        """ Iterate over dicoms, filling gradients, b_values
        """
        while len(datas) > 0:
            data = datas.pop(0)
            fname = filenames.pop(0)
            if (data is None or
                (not REUSABLE_GZ_DCM and
                 fname.endswith('.gz') or fname.endswith('bz2'))):
                # Older pydicom versions can't reread from deferred reads on
                # file objects
                dcm_w = wrapper_from_file(fname, **dicom_args)
            else:
                dcm_w = wrapper_from_data(data)
            # Confine to mosaic images for now.  Might work for slices with modified
            # sort function, but we haven't worked it out yet, in a general way
            if not dcm_w.is_mosaic:
                raise DicomReadError('data does not appear to be in mosaic '
                                     'format')
            q = dcm_w.q_vector
            if q is None:  # probably not diffusion
                if check_is_dwi:
                    raise DicomReadError('Could not find diffusion '
                                         'information reading file "%s"; is '
                                         'it possible this is not a _raw_ '
                                         'diffusion directory? Could it be a '
                                         'processed dataset like ADC etc?' %
                                         fname)
            if q is None:  # probably not diffusion
                b = np.nan
                g = np.ones((3,)) + np.nan
            else:
                b = np.sqrt(np.sum(q * q)) # vector norm
                g = q / b
            b_values.append(b)
            gradients.append(g)
            yield dcm_w

    # Read first, initialize output
    dcm_getter = dcm_generator()
    dcm_w0 = dcm_getter.next()
    affine = np.dot(DPCS_TO_TAL, dcm_w0.get_affine())
    arr = dcm_w0.get_data()
    out_shape = arr.shape + (n_files,)
    out_arr = np.empty(out_shape, arr.dtype)
    out_arr[..., 0] = arr
    # Loop over remaining
    for i in range(1, n_files):
        out_arr[..., i] = dcm_getter.next().get_data()
    b_values = np.array(b_values)
    gradients = np.array(gradients)
    if ret_type == 'tuple':
        return (out_arr,
                affine,
                b_values,
                gradients)
    return dict(data=out_arr,
                affine=affine,
                b_values=b_values,
                unit_gradients=gradients,
                dcm0 = dcm_w0.dcm_data)


def slices_to_series(wrappers):
    ''' Sort sequence of slice wrappers into series

    This follows the SPM model fairly closely

    Parameters
    ----------
    wrappers : sequence
       sequence of ``Wrapper`` objects for sorting into volumes

    Returns
    -------
    series : sequence
       sequence of sequences of wrapper objects, where each sequence is
       wrapper objects comprising a series, sorted into slice order
    '''
    # first pass
    volume_lists = [wrappers[0:1]]
    for dw in wrappers[1:]:
        for vol_list in volume_lists:
            if dw.is_same_series(vol_list[0]):
                vol_list.append(dw)
                break
        else: # no match in current volume lists
            volume_lists.append([dw])
    print 'We appear to have %d Series' % len(volume_lists)
    # second pass
    out_vol_lists = []
    for vol_list in volume_lists:
        if len(vol_list) > 1:
            vol_list.sort(_slice_sorter)
            zs = [s.slice_indicator for s in vol_list]
            if len(set(zs)) < len(zs): # not unique zs
                # third pass
                out_vol_lists += _third_pass(vol_list)
                continue
        out_vol_lists.append(vol_list)
    print 'We have %d volumes after second pass' % len(out_vol_lists)
    # final pass check
    for vol_list in out_vol_lists:
        zs = [s.slice_indicator for s in vol_list]
        diffs = np.diff(zs)
        if not np.allclose(diffs, np.mean(diffs)):
            raise DicomReadError('Largeish slice gaps - missing DICOMs?')
    return out_vol_lists


def _slice_sorter(s1, s2):
    return cmp(s1.slice_indicator, s2.slice_indicator)


def _instance_sorter(s1, s2):
    return cmp(s1.instance_number, s2.instance_number)


def _third_pass(wrappers):
    ''' What we do when there are not unique zs in a slice set '''
    inos = [s.instance_number for s in wrappers]
    msg_fmt = ('Plausibly matching slices, but where some have '
               'the same apparent slice location, and %s; '
               '- slices are probably unsortable')
    if None in inos:
        raise DicomReadError(msg_fmt % 'some or all slices with '
                             'missing InstanceNumber')
    if len(set(inos)) < len(inos):
        raise DicomReadError(msg_fmt % 'some or all slices with '
                             'the sane InstanceNumber')
    # sort by instance number
    wrappers.sort(_instance_sorter)
    # start loop, in which we start a new volume, each time we see a z
    # we've seen already in the current volume
    dw = wrappers[0]
    these_zs = [dw.slice_indicator]
    vol_list = [dw]
    out_vol_lists = [vol_list]
    for dw in wrappers[1:]:
        z = dw.slice_indicator
        if not z in these_zs:
            # same volume
            vol_list.append(dw)
            these_zs.append(z)
            continue
        # new volumne
        vol_list.sort(_slice_sorter)
        vol_list = [dw]
        these_zs = [z]
        out_vol_lists.append(vol_list)
    vol_list.sort(_slice_sorter)
    return out_vol_lists
