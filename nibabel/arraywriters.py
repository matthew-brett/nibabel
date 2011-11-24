""" Array writer objects

Array writers have init signature::

    def __init__(self, array, out_dtype=None)

and methods

* to_fileobj(fileobj, offset=None, order='F')

They do have attributes:

* array
* out_dtype

They may have attributes:

* scale
* inter

They are designed to write arrays to a fileobj with reasonable memory
efficiency.

Subclasses of array writers may be able to scale the array or apply an
intercept, or do something else to make sense of conversions between float and
int, or between larger ints and smaller.
"""

import numpy as np


class WriterError(Exception):
    pass


class ArrayWriter(object):
    def __init__(self, array, out_dtype=None):
        """ Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.

        Examples
        --------
        >>> aw = ArrayWriter(np.arange(10))
        """
        self._array = np.asanyarray(array)
        arr_dtype = self._array.dtype
        if out_dtype is None:
            out_dtype = arr_dtype
        elif out_dtype not in (arr_dtype, arr_dtype.newbyteorder('S')):
            raise ValueError('out_dtype needs to be same kind as array dtype')
        self._out_dtype = out_dtype

    @property
    def array(self):
        """ Return array from arraywriter """
        return self._array

    @property
    def out_dtype(self):
        """ Return `out_dtype` from arraywriter """
        return self._out_dtype

    def to_fileobj(self, fileobj, order='F'):
        """ Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
        data = self._array
        jbs = data.dtype == self._out_dtype.newbyteorder('S') # just byte swapped
        if not order in 'FC':
            raise ValueError('Order should be one of F or C')
        if order == 'F':
            data = data.T
        if data.ndim < 2: # a little hack to allow 1D arrays in loop below
            data = [data]
        for dslice in data: # cycle over largest dimension to save memory
            if jbs:
                dslice = dslice.byteswap()
            fileobj.write(dslice.tostring())
