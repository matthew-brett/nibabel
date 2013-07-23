# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Array proxy base class

The API is - at minimum:

* The object has an attribute ``shape``
* that the object returns the data array from ``np.asarray(obj)``
* that no changes to any object outside ``obj`` will affect the result of
  ``np.asarray(obj)``.  Specifically, if you pass a header into the the
  __init__, then modifying the original header will not affect the result of the
  array return.
"""

from .volumeutils import BinOpener, array_from_file, apply_read_scaling


class ArrayProxy(object):
    """
    The array proxy stores the passed fileobj and relevant header information so
    that the proxy can return the expected data array.

    This fairly generic implementation allows us to deal with Analyze and its
    variants, including Nifti1, and with the MGH format, apparently.

    It requires a ``header`` object with methods:
    * get_data_shape
    * get_data_dtype
    * get_slope_inter
    * get_data_offset

    Other image types might need to implement their own implementation of this
    API.  See :mod:`minc` for an example.
    """
    def __init__(self, file_like, header):
        self.file_like = file_like
        self.dtype = header.get_data_dtype()
        self.shape = header.get_data_shape()
        self.offset = header.get_data_offset()
        slope, inter = header.get_slope_inter()
        self.slope = 1.0 if slope is None else slope
        self.inter = 0.0 if inter is None else inter
        self._data = None

    def __array__(self):
        ''' Cached read of data from file '''
        if self._data is None:
            self._data = self._read_data()
        return self._data

    def _read_data(self):
        with BinOpener(self.file_like) as fileobj:
            data = array_from_file(self.shape,
                                   self.dtype,
                                   fileobj,
                                   self.offset)
            data = apply_read_scaling(data, self.slope, self.inter)
        return data
