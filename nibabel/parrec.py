# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Humble attempt to read images in PAR/REC format.

This is yet another MRI image format generated by Philips scanners. It is an
ASCII header (PAR) plus a binary blob (REC).

This implementation aims to read version 4.2 of this format. Other versions
could probably be supported, but the author is lacking samples of them.

###############
PAR file format
###############

The PAR format appears to have two sections:

General information
###################

This is a set of lines each giving one key : value pair, examples::

    .    EPI factor        <0,1=no EPI>     :   39
    .    Dynamic scan      <0=no 1=yes> ?   :   1
    .    Diffusion         <0=no 1=yes> ?   :   0

(from nibabe/tests/data/phantom_EPI_asc_CLEAR_2_1.PAR)

Image information
#################

There is a ``#`` prefixed list of fields under the heading "IMAGE INFORMATION
DEFINITION".  From the same file, here is the start of this list::

    # === IMAGE INFORMATION DEFINITION =============================================
    #  The rest of this file contains ONE line per image, this line contains the following information:
    #
    #  slice number                             (integer)
    #  echo number                              (integer)
    #  dynamic scan number                      (integer)

There follows a space separated table with values for these fields, each row
containing all the named values. Here's the first few lines from the example
file above::

    # === IMAGE INFORMATION ==========================================================
    #  sl ec  dyn ph ty    idx pix scan% rec size                (re)scale              window        angulation              offcentre        thick   gap   info      spacing     echo     dtime   ttime    diff  avg  flip    freq   RR-int  turbo delay b grad cont anis         diffusion       L.ty

    1   1    1  1 0 2     0  16    62   64   64     0.00000   1.29035 4.28404e-003  1070  1860 -13.26  -0.00  -0.00    2.51   -0.81   -8.69  6.000  2.000 0 1 0 2  3.750  3.750  30.00    0.00     0.00    0.00   0   90.00     0    0    0    39   0.0  1   1    8    0   0.000    0.000    0.000  1
    2   1    1  1 0 2     1  16    62   64   64     0.00000   1.29035 4.28404e-003  1122  1951 -13.26  -0.00  -0.00    2.51    6.98  -10.53  6.000  2.000 0 1 0 2  3.750  3.750  30.00    0.00     0.00    0.00   0   90.00     0    0    0    39   0.0  1   1    8    0   0.000    0.000    0.000  1
    3   1    1  1 0 2     2  16    62   64   64     0.00000   1.29035 4.28404e-003  1137  1977 -13.26  -0.00  -0.00    2.51   14.77  -12.36  6.000  2.000 0 1 0 2  3.750  3.750  30.00    0.00     0.00    0.00   0   90.00     0    0    0    39   0.0  1   1    8    0   0.000    0.000    0.000  1

###########
Orientation
###########

PAR files refer to orientations "ap", "fh" and "rl".

Nibabel's required affine output axes are RAS (left to Right, posterior to
Anterior, inferior to Superior). The correspondence of the PAR file's axes to
RAS axes is:

* ap = anterior -> posterior = negative A in RAS
* fh = foot -> head = S in RAS
* rl = right -> left = negative R in RAS

The orientation of the PAR file axes corresponds to DICOM's LPS coordinate
system (right to Left, anterior to Posterior, inferior to Superior), but in a
different order.

We call the PAR file's axis system "PSL" (Posterior, Superior, Left)
"""
from __future__ import print_function, division

import warnings
import numpy as np
from copy import deepcopy
import re

from .externals.six import string_types
from .py3k import asbytes

from .spatialimages import SpatialImage, Header
from .eulerangles import euler2mat
from .volumeutils import Recoder, array_from_file, BinOpener
from .affines import from_matvec, dot_reduce, apply_affine
from .nifti1 import unit_codes

# PSL to RAS affine
PSL_TO_RAS = np.array([[0, 0, -1, 0],  # L -> R
                       [-1, 0, 0, 0],  # P -> A
                       [0, 1, 0, 0],   # S -> S
                       [0, 0, 0, 1]])

# Acquisition (tra/sag/cor) to PSL axes
# These come from looking at transverse, sagittal, coronal datasets where we
# can see the LR, PA, SI orientation of the slice axes from the scanned object
ACQ_TO_PSL = dict(
    transverse=np.array([[0,  1,  0, 0],  # P
                         [0,  0,  1, 0],  # S
                         [1,  0,  0, 0],  # L
                         [0,  0,  0, 1]]),
    sagittal=np.diag([1, -1, -1, 1]),
    coronal=np.array([[0,  0,  1, 0],  # P
                      [0, -1,  0, 0],  # S
                      [1,  0,  0, 0],  # L
                      [0,  0,  0, 1]])
)
# PAR header versions we claim to understand
supported_versions = ['V4.2']

# General information dict definitions
# assign props to PAR header entries
# values are: (shortname[, dtype[, shape]])
_hdr_key_dict = {
    'Patient name': ('patient_name',),
    'Examination name': ('exam_name',),
    'Protocol name': ('protocol_name',),
    'Examination date/time': ('exam_date',),
    'Series Type': ('series_type',),
    'Acquisition nr': ('acq_nr', int),
    'Reconstruction nr': ('recon_nr', int),
    'Scan Duration [sec]': ('scan_duration', float),
    'Max. number of cardiac phases': ('max_cardiac_phases', int),
    'Max. number of echoes': ('max_echoes', int),
    'Max. number of slices/locations': ('max_slices', int),
    'Max. number of dynamics': ('max_dynamics', int),
    'Max. number of mixes': ('max_mixes', int),
    'Patient position': ('patient_position',),
    'Preparation direction': ('prep_direction',),
    'Technique': ('tech',),
    'Scan resolution  (x, y)': ('scan_resolution', int, (2,)),
    'Scan mode': ('scan_mode',),
    'Repetition time [ms]': ('repetition_time', float),
    'FOV (ap,fh,rl) [mm]': ('fov', float, (3,)),
    'Water Fat shift [pixels]': ('water_fat_shift', float),
    'Angulation midslice(ap,fh,rl)[degr]': ('angulation', float, (3,)),
    'Off Centre midslice(ap,fh,rl) [mm]': ('off_center', float, (3,)),
    'Flow compensation <0=no 1=yes> ?': ('flow_compensation', int),
    'Presaturation     <0=no 1=yes> ?': ('presaturation', int),
    'Phase encoding velocity [cm/sec]': ('phase_enc_velocity', float, (3,)),
    'MTC               <0=no 1=yes> ?': ('mtc', int),
    'SPIR              <0=no 1=yes> ?': ('spir', int),
    'EPI factor        <0,1=no EPI>': ('epi_factor', int),
    'Dynamic scan      <0=no 1=yes> ?': ('dyn_scan', int),
    'Diffusion         <0=no 1=yes> ?': ('diffusion', int),
    'Diffusion echo time [ms]': ('diffusion_echo_time', float),
    'Max. number of diffusion values': ('max_diffusion_values', int),
    'Max. number of gradient orients': ('max_gradient_orient', int),
    'Number of label types   <0=no ASL>': ('nr_label_types', int),
    }

# Image information as coded into a numpy structured array
# header items order per image definition line
image_def_dtd = [
    ('slice number', int),
    ('echo number', int,),
    ('dynamic scan number', int,),
    ('cardiac phase number', int,),
    ('image_type_mr', int,),
    ('scanning sequence', int,),
    ('index in REC file', int,),
    ('image pixel size', int,),
    ('scan percentage', int,),
    ('recon resolution', int, (2,)),
    ('rescale intercept', float),
    ('rescale slope', float),
    ('scale slope', float),
    ('window center', int,),
    ('window width', int,),
    ('image angulation', float, (3,)),
    ('image offcentre', float, (3,)),
    ('slice thickness', float),
    ('slice gap', float),
    ('image_display_orientation', int,),
    ('slice orientation', int,),
    ('fmri_status_indication', int,),
    ('image_type_ed_es', int,),
    ('pixel spacing', float, (2,)),
    ('echo_time', float),
    ('dyn_scan_begin_time', float),
    ('trigger_time', float),
    ('diffusion_b_factor', float),
    ('number of averages', int,),
    ('image_flip_angle', float),
    ('cardiac frequency', int,),
    ('minimum RR-interval', int,),
    ('maximum RR-interval', int,),
    ('TURBO factor', int,),
    ('Inversion delay', float),
    ('diffusion b value number', int,),     # (imagekey!)
    ('gradient orientation number', int,),  # (imagekey!)
    ('contrast type', 'S30'),               # XXX might be too short?
    ('diffusion anisotropy type', 'S30'),   # XXX might be too short?
    ('diffusion', float, (3,)),
    ('label type', int,),                  # (imagekey!)
    ]
image_def_dtype = np.dtype(image_def_dtd)

# slice orientation codes
slice_orientation_codes = Recoder((  # code, label
    (1, 'transverse'),
    (2, 'sagittal'),
    (3, 'coronal')), fields=('code', 'label'))


class PARRECError(Exception):
    """Exception for PAR/REC format related problems.

    To be raised whenever PAR/REC is not happy, or we are not happy with
    PAR/REC.
    """
    pass


def _split_header(fobj):
    """ Split header into `version`, `gen_dict`, `image_lines` """
    version = None
    gen_dict = {}
    image_lines = []
    # Small state-machine
    state = 'top-header'
    for line in fobj:
        line = line.strip()
        if line == '':
            continue
        if state == 'top-header':
            if not line.startswith('#'):
                state = 'general-info'
            elif 'image export tool' in line:
                version = line.split()[-1]
        if state == 'general-info':
            if not line.startswith('.'):
                state = 'comment-block'
            else: # Let match raise error for unexpected field format
                key, value = GEN_RE.match(line).groups()
                gen_dict[key] = value
        if state == 'comment-block':
            if not line.startswith('#'):
                state = 'image-info'
        if state == 'image-info':
            if line.startswith('#'):
                break
            image_lines.append(line)
    return version, gen_dict, image_lines


GEN_RE = re.compile(r".\s+(.*?)\s*:\s+(.*)")


def _process_gen_dict(gen_dict):
    """ Process `gen_dict` key, values into `general_info`
    """
    general_info = {}
    for key, value in gen_dict.items():
        # get props for this hdr field
        props = _hdr_key_dict[key]
        # turn values into meaningful dtype
        if len(props) == 2:
            # only dtype spec and no shape
            value = props[1](value)
        elif len(props) == 3:
            # array with dtype and shape
            value = np.fromstring(value, props[1], sep=' ')
            value.shape = props[2]
        general_info[props[0]] = value
    return general_info


def _process_image_lines(image_lines):
    """ Process image information definition lines
    """
    # postproc image def props
    # create an array for all image defs
    image_defs = np.zeros(len(image_lines), dtype=image_def_dtype)
    # for every image definition
    for i, line in enumerate(image_lines):
        items = line.split()
        item_counter = 0
        # for all image properties we know about
        for props in image_def_dtd:
            if len(props) == 2:
                name, np_type = props
                value = items[item_counter]
                if not np.dtype(np_type).kind == 'S':
                    value = np_type(value)
                item_counter += 1
            elif len(props) == 3:
                name, np_type, shape = props
                nelements = np.prod(shape)
                value  = items[item_counter:item_counter + nelements]
                value  = [np_type(v) for v in value]
                item_counter += nelements
            image_defs[name][i] = value
    return image_defs


def vol_numbers(slice_nos):
    """ Calculate volume numbers inferred from slice numbers `slice_nos`

    The volume number for each slice is the number of times this slice has
    occurred previously in the `slice_nos` sequence

    Parameters
    ----------
    slice_nos : sequence
        Sequence of slice numbers, e.g. ``[1, 2, 3, 4, 1, 2, 3, 4]``.

    Returns
    -------
    vol_nos : list
        A list, the same length of `slice_nos` giving the volume number for
        each corresponding slice number.
    """
    counter = {}
    vol_nos = []
    for s_no in slice_nos:
        count = counter.setdefault(s_no, 0)
        vol_nos.append(count)
        counter[s_no] += 1
    return vol_nos


def vol_is_full(slice_nos, slice_max, slice_min=1):
    """ Vector with True for slices in complete volume, False otherwise

    Parameters
    ----------
    slice_nos : sequence
        Sequence of slice numbers, e.g. ``[1, 2, 3, 4, 1, 2, 3, 4]``.
    slice_max : int
        Highest slice number for a full slice set.  Slice set will be
        ``range(slice_min, slice_max+1)``.
    slice_min : int
        Lowest slice number for full slice set.

    Returns
    -------
    is_full : array
        Bool vector with True for slices in full volumes, False for slices in
        partial volumes.  A full volume is a volume with all slices in the
        ``slice set`` as defined above.

    Raises
    ------
    ValueError if any `slice_nos` value is outside slice set.
    """
    slice_set = set(range(slice_min, slice_max + 1))
    if not slice_set.issuperset(slice_nos):
        raise ValueError(
            'Slice numbers outside inclusive range {0} to {1}'.format(
                slice_min, slice_max))
    vol_nos = np.array(vol_numbers(slice_nos))
    slice_nos = np.asarray(slice_nos)
    is_full = np.ones(slice_nos.shape, dtype=bool)
    for vol_no in set(vol_nos):
        ours = vol_nos == vol_no
        if not set(slice_nos[ours]) == slice_set:
            is_full[ours] = False
    return is_full


def _truncation_checks(general_info, image_defs, permit_truncated):
    """ Check for presence of truncation in PAR file parameters

    Raise error if truncation present and `permit_truncated` is False.
    """
    def _err_or_warn(msg):
        if not permit_truncated:
            raise PARRECError(msg)
        warnings.warn(msg)

    def _chk_trunc(idef_name, gdef_max_name):
        id_values = image_defs[idef_name + ' number']
        n_have = len(set(id_values))
        n_expected = general_info[gdef_max_name]
        if n_have != n_expected:
            _err_or_warn(
                "Header inconsistency: Found {0} {1} values, "
                "but expected {2}".format(n_have, idef_name, n_expected))

    _chk_trunc('slice', 'max_slices')
    _chk_trunc('echo', 'max_echoes')
    _chk_trunc('dynamic scan', 'max_dynamics')
    _chk_trunc('diffusion b value', 'max_diffusion_values')
    _chk_trunc('gradient orientation', 'max_gradient_orient')

    # Final check for partial volumes
    if not np.all(vol_is_full(image_defs['slice number'],
                              general_info['max_slices'])):
        _err_or_warn("Found one or more partial volume(s)")


def one_line(long_str):
    """ Make maybe mutli-line `long_str` into one long line """
    return ' '.join(line.strip() for line in long_str.splitlines())


def parse_PAR_header(fobj):
    """Parse a PAR header and aggregate all information into useful containers.

    Parameters
    ----------
    fobj : file-object
        The PAR header file object.

    Returns
    -------
    general_info : dict
        Contains all "General Information" from the header file
    image_info : ndarray
        Structured array with fields giving all "Image information" in the
        header
    """
    # single pass through the header
    version, gen_dict, image_lines = _split_header(fobj)
    if version not in supported_versions:
        warnings.warn(one_line(
            """ PAR/REC version '{0}' is currently not supported -- making an
            attempt to read nevertheless. Please email the NiBabel mailing
            list, if you are interested in adding support for this version.
            """.format(version)))
    general_info = _process_gen_dict(gen_dict)
    image_defs = _process_image_lines(image_lines)
    return general_info, image_defs


def _data_from_rec(rec_fileobj, in_shape, dtype, slice_indices, out_shape,
                   scaling=None):
    """Get data from REC file

    Parameters
    ----------
    rec_fileobj : file-like
        The file to process.
    in_shape : tuple
        The input shape inferred from the PAR file.
    dtype : dtype
        The datatype.
    slice_indices : array of int
        The indices used to re-index the resulting array properly.
    out_shape : tuple
        The output shape.
    scaling : array | None
        Scaling to use.

    Returns
    -------
    data : array
        The scaled and sorted array.
    """
    rec_data = array_from_file(in_shape, dtype, rec_fileobj)
    rec_data = rec_data[..., slice_indices]
    rec_data = rec_data.reshape(out_shape, order='F')
    if not scaling is None:
         # Don't do in-place b/c this goes int16 -> float64
        rec_data = rec_data * scaling[0] + scaling[1]
    return rec_data


class PARRECArrayProxy(object):
    def __init__(self, file_like, header, scaling):
        self.file_like = file_like
        # Copies of values needed to read array
        self._shape = header.get_data_shape()
        self._dtype = header.get_data_dtype()
        self._slice_indices = header.get_sorted_slice_indices()
        self._slice_scaling = header.get_data_scaling(scaling)
        self._rec_shape = header.get_rec_shape()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_proxy(self):
        return True

    def get_unscaled(self):
        with BinOpener(self.file_like) as fileobj:
            return _data_from_rec(fileobj, self._rec_shape, self._dtype,
                                  self._slice_indices, self._shape)

    def __array__(self):
        with BinOpener(self.file_like) as fileobj:
            return _data_from_rec(fileobj, self._rec_shape, self._dtype,
                                  self._slice_indices, self._shape,
                                  scaling=self._slice_scaling)


class PARRECHeader(Header):
    """PAR/REC header"""
    def __init__(self, info, image_defs, permit_truncated=False):
        """
        Parameters
        ----------
        info : dict
            "General information" from the PAR file (as returned by
            `parse_PAR_header()`).
        image_defs : array
            Structured array with image definitions from the PAR file (as
            returned by `parse_PAR_header()`).
        permit_truncated : bool, optional
            If True, a warning is emitted instead of an error when a truncated
            recording is detected.
        """
        self.general_info = info.copy()
        self.image_defs = image_defs.copy()
        _truncation_checks(info, image_defs, permit_truncated)
        # charge with basic properties to be able to use base class
        # functionality
        # dtype
        bitpix = self._get_unique_image_prop('image pixel size')
        Header.__init__(self,
                        data_dtype=np.dtype('int' + str(bitpix)).type,
                        shape=self._calc_data_shape(),
                        zooms=self._calc_zooms())

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise PARRECError('Cannot create PARRECHeader from air.')
        if type(header) == klass:
            return header.copy()
        raise PARRECError('Cannot create PARREC header from '
                          'non-PARREC header.')

    @classmethod
    def from_fileobj(klass, fileobj, permit_truncated=False):
        info, image_defs = parse_PAR_header(fileobj)
        return klass(info, image_defs, permit_truncated)

    def copy(self):
        return PARRECHeader(deepcopy(self.general_info),
                            self.image_defs.copy())

    def as_analyze_map(self):
        """Convert PAR parameters to NIFTI1 format"""
        # Entries in the dict correspond to the parameters found in
        # the NIfTI1 header, specifically in nifti1.py `header_dtd` defs.
        # Here we set the parameters we can to simplify PAR/REC
        # to NIfTI conversion.
        descr = ("%s;%s;%s;%s"
                 % (self.general_info['exam_name'],
                    self.general_info['patient_name'],
                    self.general_info['exam_date'].replace(' ', ''),
                    self.general_info['protocol_name']))[:80]  # max len
        is_fmri = (self.general_info['max_dynamics'] > 1)
        t = 'msec' if is_fmri else 'unknown'
        xyzt_units = unit_codes['mm'] + unit_codes[t]
        return dict(descr=descr, xyzt_units=xyzt_units)  # , pixdim=pixdim)

    def get_water_fat_shift(self):
        """Water fat shift, in pixels"""
        return self.general_info['water_fat_shift']

    def get_echo_train_length(self):
        """Echo train length of the recording"""
        return self.general_info['epi_factor']

    def get_q_vectors(self):
        """Get Q vectors from the data

        Returns
        -------
        q_vectors : None or array
            Array of q vectors (bvals * bvecs), or None if not a diffusion
            acquisition.
        """
        bvals, bvecs = self.get_bvals_bvecs()
        if bvals is None and bvecs is None:
            return None
        return bvecs * bvals[:, np.newaxis]

    def get_bvals_bvecs(self):
        """Get bvals and bvecs from data

        Returns
        -------
        b_vals : None or array
            Array of b values, shape (n_directions,), or None if not a
            diffusion acquisition.
        b_vectors : None or array
            Array of b vectors, shape (n_directions, 3), or None if not a
            diffusion acquisition.
        """
        if self.general_info['diffusion'] == 0:
            return None, None
        reorder = self.get_sorted_slice_indices()
        n_slices, n_vols = self.get_data_shape()[-2:]
        bvals = self.image_defs['diffusion_b_factor'][reorder].reshape(
            (n_slices, n_vols), order='F')
        # All bvals within volume should be the same
        assert not np.any(np.diff(bvals, axis=0))
        bvals = bvals[0]
        bvecs = self.image_defs['diffusion'][reorder].reshape(
            (n_slices, n_vols, 3), order='F')
        # All 3 values of bvecs should be same within volume
        assert not np.any(np.diff(bvecs, axis=0))
        bvecs = bvecs[0]
        # rotate bvecs to match stored image orientation
        permute_to_psl = ACQ_TO_PSL[self.get_slice_orientation()]
        bvecs = apply_affine(np.linalg.inv(permute_to_psl), bvecs)
        return bvals, bvecs

    def _get_unique_image_prop(self, name):
        """ Scan image definitions and return unique value of a property.

        * Get array for named field of ``self.image_defs``;
        * Check that all rows in the array are the same and raise error
          otherwise;
        * Return the row.

        Parameters
        ----------
        name : str
            Name of the property in ``self.image_defs``

        Returns
        -------
        unique_value : scalar or array

        Raises
        ------
        PARRECError - if the rows of ``self.image_defs[name]`` do not all
        compare equal
        """
        props = self.image_defs[name]
        if np.any(np.diff(props, axis=0)):
            raise PARRECError('Varying {0} in image sequence ({1}). This is '
                              'not suppported.'.format(name, props))
        return props[0]

    def get_voxel_size(self):
        """Returns the spatial extent of a voxel.

        Does not include the slice gap in the slice extent.

        This function is deprecated and we will remove it in future versions of
        nibabel.  Please use ``get_zooms`` instead.  If you need the slice
        thickness not including the slice gap, use ``self.image_defs['slice
        thickness']``.

        Returns
        -------
        vox_size: shape (3,) ndarray
        """
        warnings.warn('Please use "get_zooms" instead of "get_voxel_size"',
                      DeprecationWarning,
                      stacklevel=2)
        # slice orientation for the whole image series
        slice_thickness = self._get_unique_image_prop('slice thickness')
        voxsize_inplane = self._get_unique_image_prop('pixel spacing')
        voxsize = np.array((voxsize_inplane[0],
                            voxsize_inplane[1],
                            slice_thickness))
        return voxsize

    def get_data_offset(self):
        """ PAR header always has 0 data offset (into REC file) """
        return 0

    def set_data_offset(self, offset):
        """ PAR header always has 0 data offset (into REC file) """
        if offset != 0:
            raise PARRECError("PAR header assumes offset 0")

    def _calc_zooms(self):
        """Compute image zooms from header data.

        Spatial axis are first three.

        Returns
        -------
        zooms : array
            Length 3 array for 3D image, length 4 array for 4D image.

        Notes
        -----
        This routine called in ``__init__``, so may not be able to use
        some attributes available in the fully initalized object.
        """
        # slice orientation for the whole image series
        slice_gap = self._get_unique_image_prop('slice gap')
        # scaling per image axis
        n_dim = 4 if self._get_n_vols() > 1 else 3
        zooms = np.ones(n_dim)
        # spatial sizes are inplane X mm, inplane Y mm + inter slice gap
        zooms[:2] = self._get_unique_image_prop('pixel spacing')
        slice_thickness = self._get_unique_image_prop('slice thickness')
        zooms[2] = slice_thickness + slice_gap
        # If 4D dynamic scan, convert time from milliseconds to seconds
        if len(zooms) > 3 and self.general_info['dyn_scan']:
            zooms[3] = self.general_info['repetition_time'] / 1000.
        return zooms

    def get_affine(self, origin='scanner'):
        """Compute affine transformation into scanner space.

        The method only considers global rotation and offset settings in the
        header and ignores potentially deviating information in the image
        definitions.

        Parameters
        ----------
        origin : {'scanner', 'fov'}
            Transformation origin. By default the transformation is computed
            relative to the scanner's iso center. If 'fov' is requested the
            transformation origin will be the center of the field of view
            instead.

        Returns
        -------
        aff : (4, 4) array
            4x4 array, with output axis order corresponding to RAS or (x,y,z)
            or (lr, pa, fh).

        Notes
        -----
        Transformations appear to be specified in (ap, fh, rl) axes.  The
        orientation of data is recorded in the "slice orientation" field of the
        PAR header "General Information".

        We need to:

        * translate to coordinates in terms of the center of the FOV
        * apply voxel size scaling
        * reorder / flip the data to Philips' PSL axes
        * apply the rotations
        * apply any isocenter scaling offset if `origin` == "scanner"
        * reorder and flip to RAS axes
        """
        # shape, zooms in original data ordering (ijk ordering)
        ijk_shape = np.array(self.get_data_shape()[:3])
        to_center = from_matvec(np.eye(3), -(ijk_shape - 1) / 2.)
        zoomer = np.diag(list(self.get_zooms()[:3]) + [1])
        slice_orientation = self.get_slice_orientation()
        permute_to_psl = ACQ_TO_PSL.get(slice_orientation)
        if permute_to_psl is None:
            raise PARRECError(
                "Unknown slice orientation ({0}).".format(slice_orientation))
        # hdr has deg, we need radians
        # Order is [ap, fh, rl]
        ang_rad = self.general_info['angulation'] * np.pi / 180.0
        # euler2mat accepts z, y, x angles and does rotation around z, y, x
        # axes in that order. It's possible that PAR assumes rotation in a
        # different order, we still need some relevant data to test this
        rot = from_matvec(euler2mat(*ang_rad[::-1]), [0, 0, 0])
        # compose the PSL affine
        psl_aff = dot_reduce(rot, permute_to_psl, zoomer, to_center)
        if origin == 'scanner':
            # offset to scanner's isocenter (in ap, fh, rl)
            iso_offset = self.general_info['off_center']
            psl_aff[:3, 3] += iso_offset
        # Currently in PSL; apply PSL -> RAS
        return np.dot(PSL_TO_RAS, psl_aff)

    def _get_n_slices(self):
        """ Get number of slices for output data """
        return len(set(self.image_defs['slice number']))

    def _get_n_vols(self):
        """ Get number of volumes for output data """
        slice_nos = self.image_defs['slice number']
        vol_nos = vol_numbers(slice_nos)
        is_full = vol_is_full(slice_nos, self.general_info['max_slices'])
        return len(set(np.array(vol_nos)[is_full]))

    def _calc_data_shape(self):
        """ Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        n_inplaneX : int
            number of voxels in X direction.
        n_inplaneY : int
            number of voxels in Y direction.
        n_slices : int
            number of slices.
        n_vols : int
            number of volumes or absent for 3D image.

        Notes
        -----
        This routine called in ``__init__``, so may not be able to use
        some attributes available in the fully initalized object.
        """
        inplane_shape = tuple(self._get_unique_image_prop('recon resolution'))
        shape = inplane_shape + (self._get_n_slices(),)
        n_vols = self._get_n_vols()
        return shape + (n_vols,) if n_vols > 1 else shape

    def get_data_scaling(self, method="dv"):
        """Returns scaling slope and intercept.

        Parameters
        ----------
        method : {'fp', 'dv'}
          Scaling settings to be reported -- see notes below.

        Returns
        -------
        slope : array
            scaling slope
        intercept : array
            scaling intercept

        Notes
        -----
        The PAR header contains two different scaling settings: 'dv' (value on
        console) and 'fp' (floating point value). Here is how they are defined:

        PV: value in REC
        RS: rescale slope
        RI: rescale intercept
        SS: scale slope

        DV = PV * RS + RI
        FP = DV / (RS * SS)
        """
        # These will be 3D or 4D
        scale_slope = self.image_defs['scale slope']
        rescale_slope = self.image_defs['rescale slope']
        rescale_intercept = self.image_defs['rescale intercept']
        if method == 'dv':
            slope, intercept = rescale_slope, rescale_intercept
        elif method == 'fp':
            slope = 1.0 / scale_slope
            intercept = rescale_intercept / (rescale_slope * scale_slope)
        else:
            raise ValueError("Unknown scling method '%s'." % method)
        reorder = self.get_sorted_slice_indices()
        slope = slope[reorder]
        intercept = intercept[reorder]
        shape = (1, 1) + self.get_data_shape()[2:]
        slope = slope.reshape(shape, order='F')
        intercept = intercept.reshape(shape, order='F')
        return slope, intercept

    def get_slice_orientation(self):
        """Returns the slice orientation label.

        Returns
        -------
        orientation : {'transverse', 'sagittal', 'coronal'}
        """
        lab = self._get_unique_image_prop('slice orientation')
        return slice_orientation_codes.label[lab]

    def get_rec_shape(self):
        inplane_shape = tuple(self._get_unique_image_prop('recon resolution'))
        return inplane_shape + (len(self.image_defs),)

    def get_sorted_slice_indices(self):
        """Indices to sort (and maybe discard) slices in REC file

        Returns list for indexing into the last (third) dimension of the REC
        data array, and (equivalently) the only dimension of
        ``self.image_defs``.

        If the recording is truncated, the returned indices take care of
        discarding any indices that are not meant to be used.
        """
        slice_nos = self.image_defs['slice number']
        is_full = vol_is_full(slice_nos, self.general_info['max_slices'])
        keys = (slice_nos, vol_numbers(slice_nos), np.logical_not(is_full))
        # Figure out how many we need to remove from the end, and trim them
        # Based on our sorting, they should always be last
        n_used = np.prod(self.get_data_shape()[2:])
        return np.lexsort(keys)[:n_used]


class PARRECImage(SpatialImage):
    """PAR/REC image"""
    header_class = PARRECHeader
    files_types = (('image', '.rec'), ('header', '.par'))

    ImageArrayProxy = PARRECArrayProxy

    @classmethod
    def from_file_map(klass, file_map, permit_truncated, scaling):
        pt = permit_truncated
        with file_map['header'].get_prepare_fileobj('rt') as hdr_fobj:
            hdr = klass.header_class.from_fileobj(hdr_fobj,
                                                  permit_truncated=pt)
        rec_fobj = file_map['image'].get_prepare_fileobj()
        data = klass.ImageArrayProxy(rec_fobj, hdr, scaling)
        return klass(data, hdr.get_affine(), header=hdr, extra=None,
                     file_map=file_map)


def load(filename, permit_truncated=False, scaling='dv'):
    file_map = PARRECImage.filespec_to_file_map(filename)
    return PARRECImage.from_file_map(file_map, permit_truncated, scaling)
load.__doc__ = PARRECImage.load.__doc__
