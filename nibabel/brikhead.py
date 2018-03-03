# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Class for reading AFNI BRIK/HEAD datasets

See https://afni.nimh.nih.gov/pub/dist/doc/program_help/README.attributes.html
for more information on required information to have a valid BRIK/HEAD dataset.
"""
from __future__ import print_function, division

from copy import deepcopy
import os
import re

import numpy as np

from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .keywordonly import kw_only_meth
from .spatialimages import (
    SpatialImage,
    SpatialHeader,
    HeaderDataError,
    ImageDataError
)
from .volumeutils import Recoder

# used for doc-tests
filepath = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.realpath(os.path.join(filepath, 'tests/data'))

_attr_dic = {
    'string': str,
    'integer': int,
    'float': float
}

_endian_dict = {
    'LSB_FIRST': '<',
    'MSB_FIRST': '>',
}

_dtype_dict = {
    0: 'B',
    1: 'h',
    3: 'f',
    5: 'D',
}

space_codes = Recoder((
    (0, 'unknown', ''),
    (1, 'scanner', 'ORIG'),
    (3, 'talairach', 'TLRC'),
    (4, 'mni', 'MNI')), fields=('code', 'label', 'space'))


class AFNIImageError(ImageDataError):
    """Error when reading AFNI BRIK files"""
    pass


class AFNIHeaderError(HeaderDataError):
    """Error when reading AFNI HEAD file"""
    pass


DATA_OFFSET = 0
TYPE_RE = re.compile('type\s*=\s*(string|integer|float)-attribute\s*\n')
NAME_RE = re.compile('name\s*=\s*(\w+)\s*\n')


def _unpack_var(var):
    """
    Parses key : value pair from `var`

    Parameters
    ----------
    var : str
        Entry from HEAD file

    Returns
    -------
    name : str
        Name of attribute
    value : object
        Value of attribute

    Examples
    --------
    >>> var = "type = integer-attribute\\nname = BRICK_TYPES\\ncount = 1\\n1\\n"
    >>> name, attr = _unpack_var(var)
    >>> print(name, attr)
    BRICK_TYPES 1
    >>> var = "type = string-attribute\\nname = TEMPLATE_SPACE\\ncount = 5\\n'ORIG~"
    >>> name, attr = _unpack_var(var)
    >>> print(name, attr)
    TEMPLATE_SPACE ORIG
    """

    err_msg = ('Please check HEAD file to ensure it is AFNI compliant. '
               'Offending attribute:\n%s' % var)

    atype, aname = TYPE_RE.findall(var), NAME_RE.findall(var)
    if len(atype) != 1:
        raise AFNIHeaderError('Invalid attribute type entry in HEAD file. '
                              '%s' % err_msg)
    if len(aname) != 1:
        raise AFNIHeaderError('Invalid attribute name entry in HEAD file. '
                              '%s' % err_msg)
    atype = _attr_dic.get(atype[0], str)
    attr = ' '.join(var.strip().split('\n')[3:])
    if atype is not str:
        try:
            attr = [atype(f) for f in attr.split()]
        except ValueError:
            raise AFNIHeaderError('Failed to read variable from HEAD file due '
                                  'to improper type casting. %s' % err_msg)
        if len(attr) == 1:
            attr = attr[0]
    else:
        # AFNI string attributes will always start with open single quote and
        # end with a tilde (NUL). These attributes CANNOT contain tildes (so
        # stripping is safe), but can contain single quotes (so we replace)
        attr = attr.replace('\'', '', 1).rstrip('~')

    return aname[0], attr


def _get_datatype(info):
    """
    Gets datatype of BRIK file associated with HEAD file yielding `info`

    Parameters
    ----------
    info : dict
        As obtained by :func:`parse_AFNI_header`

    Returns
    -------
    dt : np.dtype
        Datatype of BRIK file associated with HEAD
    """
    bo = info['BYTEORDER_STRING']
    bt = info['BRICK_TYPES']
    if isinstance(bt, list):
        if np.unique(bt).size > 1:
            raise AFNIImageError('Can\'t load file with multiple data types.')
        bt = bt[0]
    bo = _endian_dict.get(bo, '=')
    bt = _dtype_dict.get(bt, None)
    if bt is None:
        raise AFNIImageError('Can\'t deduce image data type.')

    return np.dtype(bo + bt)


def parse_AFNI_header(fobj):
    """
    Parses `fobj` to extract information from HEAD file

    Parameters
    ----------
    fobj : file-like object
        AFNI HEAD file object or filename. If file object, should
        implement at least ``read``

    Returns
    -------
    info : dict
        Dictionary containing AFNI-style key:value pairs from HEAD file

    Examples
    --------
    >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
    >>> info = parse_AFNI_header(fname)
    >>> print(info['BYTEORDER_STRING'])
    LSB_FIRST
    >>> print(info['BRICK_TYPES'])
    [1, 1, 1]
    """
    from six import string_types

    # edge case for being fed a filename instead of a file object
    if isinstance(fobj, string_types):
        with open(fobj, 'rt') as src:
            return parse_AFNI_header(src)
    # unpack variables in HEAD file
    head = fobj.read().split('\n\n')
    info = {key: value for key, value in map(_unpack_var, head)}

    return info


class AFNIArrayProxy(ArrayProxy):
    @kw_only_meth(2)
    def __init__(self, file_like, header, mmap=True, keep_file_open=None):
        """
        Initialize AFNI array proxy

        Parameters
        ----------
        file_like : file-like object
            File-like object or filename. If file-like object, should implement
            at least ``read`` and ``seek``.
        header : AFNIHeader object
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading data.
            If False, do not try numpy ``memmap`` for data array.  If one of
            {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
            True gives the same behavior as ``mmap='c'``.  If `file_like`
            cannot be memory-mapped, ignore `mmap` value and read array from
            file.
        keep_file_open : { None, 'auto', True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed. If
            ``'auto'``, and the optional ``indexed_gzip`` dependency is
            present, a single file handle is created and persisted. If
            ``indexed_gzip`` is not available, behaviour is the same as if
            ``keep_file_open is False``. If ``file_like`` refers to an open
            file handle, this setting has no effect. The default value
            (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT` being used.
        """
        super(AFNIArrayProxy, self).__init__(file_like,
                                             header,
                                             mmap=mmap,
                                             keep_file_open=keep_file_open)
        self._scaling = header.get_data_scaling()

    @property
    def scaling(self):
        return self._scaling

    def __array__(self):
        raw_data = self.get_unscaled()
        # apply volume specific scaling (may change datatype!)
        if self._scaling is not None:
            return raw_data * self._scaling

        return raw_data

    def __getitem__(self, slicer):
        raw_data = super(AFNIArrayProxy, self).__getitem__(slicer)
        # apply volume specific scaling (may change datatype!)
        if self._scaling is not None:
            scaling = self._scaling.copy()
            fake_data = strided_scalar(self._shape)
            _, scaling = np.broadcast_arrays(fake_data, scaling)
            raw_data = raw_data * scaling[slicer]

        return raw_data


class AFNIHeader(SpatialHeader):
    """Class for AFNI header"""

    def __init__(self, info):
        """
        Initialize AFNI header object

        Parameters
        ----------
        info : dict
            Information from HEAD file as obtained by :func:`parse_AFNI_header`

        Examples
        --------
        >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_data_dtype()
        dtype('int16')
        >>> header.get_zooms()
        (3.0, 3.0, 3.0, 3.0)
        >>> header.get_data_shape()
        (33, 41, 25, 3)
        """
        self.info = info
        dt = _get_datatype(self.info)
        super(AFNIHeader, self).__init__(data_dtype=dt,
                                         shape=self._calc_data_shape(),
                                         zooms=self._calc_zooms())

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            raise AFNIHeaderError('Cannot create AFNIHeader from nothing.')
        if type(header) == klass:
            return header.copy()
        raise AFNIHeaderError('Cannot create AFNIHeader from non-AFNIHeader.')

    @classmethod
    def from_fileobj(klass, fileobj):
        info = parse_AFNI_header(fileobj)
        return klass(info)

    def copy(self):
        return AFNIHeader(deepcopy(self.info))

    def _calc_data_shape(self):
        """
        Calculate the output shape of the image data

        Returns length 3 tuple for 3D image, length 4 tuple for 4D.

        Returns
        -------
        (x, y, z, t) : tuple of int
        """
        dset_rank = self.info['DATASET_RANK']
        shape = tuple(self.info['DATASET_DIMENSIONS'][:dset_rank[0]])
        n_vols = dset_rank[1]

        return shape + (n_vols,)

    def _calc_zooms(self):
        """
        Get image zooms from header data

        Spatial axes are first three indices, time axis is last index. If
        dataset is not a time series the last index will be zero.

        Returns
        -------
        zooms : tuple
        """
        xyz_step = tuple(np.abs(self.info['DELTA']))
        t_step = self.info.get('TAXIS_FLOATS', (0, 0,))
        if len(t_step) > 0:
            t_step = (t_step[1],)

        return xyz_step + t_step

    def get_space(self):
        """
        Returns space of dataset

        Returns
        -------
        space : str
            AFNI "space" designation; one of [ORIG, ANAT, TLRC, MNI]
        """
        listed_space = self.info.get('TEMPLATE_SPACE', 0)
        space = space_codes.space[listed_space]

        return space

    def get_affine(self):
        """
        Returns affine of dataset

        Examples
        --------
        >>> fname = os.path.join(datadir, 'example4d+orig.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_affine()
        array([[ -3.    ,  -0.    ,  -0.    ,  49.5   ],
               [ -0.    ,  -3.    ,  -0.    ,  82.312 ],
               [  0.    ,   0.    ,   3.    , -52.3511],
               [  0.    ,   0.    ,   0.    ,   1.    ]])
        """
        # AFNI default is RAI-/DICOM order (i.e., RAI are - axis)
        # need to flip RA sign to align with nibabel RAS+ system
        affine = np.asarray(self.info['IJK_TO_DICOM_REAL']).reshape(3, 4)
        affine = np.row_stack((affine * [[-1], [-1], [1]],
                               [0, 0, 0, 1]))

        return affine

    def get_data_scaling(self):
        """
        AFNI applies volume-specific data scaling

        Examples
        --------
        >>> fname = os.path.join(datadir, 'scaled+tlrc.HEAD')
        >>> header = AFNIHeader(parse_AFNI_header(fname))
        >>> header.get_data_scaling()
        array([  3.88336300e-08])
        """
        floatfacs = self.info.get('BRICK_FLOAT_FACS', None)
        if floatfacs is None or not np.any(floatfacs):
            return None
        scale = np.ones(self.info['DATASET_RANK'][1])
        floatfacs = np.atleast_1d(floatfacs)
        scale[floatfacs.nonzero()] = floatfacs[floatfacs.nonzero()]

        return scale

    def get_slope_inter(self):
        """Use `self.get_data_scaling()` instead"""
        return None, None

    def get_data_offset(self):
        """Data offset in BRIK file"""
        return DATA_OFFSET

    def get_volume_labels(self):
        """
        Returns volume labels

        Returns
        -------
        labels : list of str
            Labels for volumes along fourth dimension

        Examples
        --------
        >>> header = AFNIHeader(parse_AFNI_header(os.path.join(datadir, 'example4d+orig.HEAD')))
        >>> header.get_volume_labels()
        ['#0', '#1', '#2']
        """
        labels = self.info.get('BRICK_LABS', None)
        if labels is not None:
            labels = labels.split('~')

        return labels


class AFNIImage(SpatialImage):
    """
    AFNI Image file

    Can be loaded from either the BRIK or HEAD file (but MUST specify one!)

    Examples
    --------
    >>> brik = load(os.path.join(datadir, 'example4d+orig.BRIK.gz'))
    >>> brik.shape
    (33, 41, 25, 3)
    >>> brik.affine
    array([[ -3.    ,  -0.    ,  -0.    ,  49.5   ],
           [ -0.    ,  -3.    ,  -0.    ,  82.312 ],
           [  0.    ,   0.    ,   3.    , -52.3511],
           [  0.    ,   0.    ,   0.    ,   1.    ]])
    >>> head = load(os.path.join(datadir, 'example4d+orig.HEAD'))
    >>> np.array_equal(head.get_data(), brik.get_data())
    True
    """

    header_class = AFNIHeader
    valid_exts = ('.brik', '.head')
    files_types = (('image', '.brik'), ('header', '.head'))
    _compressed_suffixes = ('.gz', '.bz2')
    makeable = False
    rw = False
    ImageArrayProxy = AFNIArrayProxy

    @classmethod
    @kw_only_meth(1)
    def from_file_map(klass, file_map, mmap=True):
        """
        Creates an AFNIImage instance from `file_map`

        Parameters
        ----------
        file_map : dict
            dict with keys ``image, header`` and values being fileholder
            objects for the respective BRIK and HEAD files
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        """
        with file_map['header'].get_prepare_fileobj('rt') as hdr_fobj:
            hdr = klass.header_class.from_fileobj(hdr_fobj)
        imgf = file_map['image'].fileobj
        if imgf is None:
            imgf = file_map['image'].filename
        data = klass.ImageArrayProxy(imgf, hdr.copy(), mmap=mmap)
        return klass(data, hdr.get_affine(), header=hdr, extra=None,
                     file_map=file_map)

    @classmethod
    @kw_only_meth(1)
    def from_filename(klass, filename, mmap=True):
        """
        Creates an AFNIImage instance from `filename`

        Parameters
        ----------
        filename : str
            Path to BRIK or HEAD file to be loaded
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        """
        file_map = klass.filespec_to_file_map(filename)
        return klass.from_file_map(file_map, mmap=mmap)

    @classmethod
    def filespec_to_file_map(klass, filespec):
        """
        Make `file_map` from filename `filespec`

        Deals with idiosyncracies of AFNI BRIK / HEAD formats

        Parameters
        ----------
        filespec : str
            Filename that might be for this image file type.

        Returns
        -------
        file_map : dict
            dict with keys ``image`` and ``header`` where values are fileholder
            objects for the respective BRIK and HEAD files

        Raises
        ------
        ImageFileError
            If `filespec` is not recognizable as being a filename for this
            image type.
        """
        file_map = super(AFNIImage, klass).filespec_to_file_map(filespec)
        # only BRIK can be compressed; remove potential compression suffixes from HEAD
        head_fname = file_map['header'].filename
        if not os.path.exists(head_fname):
            for ext in klass._compressed_suffixes:
                head_fname = head_fname[:-len(ext)] if head_fname.endswith(ext) else head_fname
            file_map['header'].filename = head_fname
        # if HEAD is read in and BRIK is compressed, function won't detect the
        # compressed format; check for these cases
        if not os.path.exists(file_map['image'].filename):
            for ext in klass._compressed_suffixes:
                im_ext = file_map['image'].filename + ext
                if os.path.exists(im_ext):
                    file_map['image'].filename = im_ext
                    break
        return file_map

    load = from_filename


load = AFNIImage.load