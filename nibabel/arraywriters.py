""" Array writer objects

Array writers have init signature::

    def __init__(self, array, out_dtype=None)

and methods

* to_fileobj(fileobj, offset=None, order='F')

They do have attributes:

* array
* out_dtype
* scale
* inter

They are designed to write arrays to a fileobj with reasonable memory
efficiency.

Array writers may be able to scale the array or apply an
intercept, or do something else to make sense of conversions between float and
int, or between larger ints and smaller.
"""

import numpy as np


class WriterError(Exception):
    pass


class ScaleInterArrayWriter(object):
    # Working precision of scaling
    scale_inter_type = np.float32

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
        self._scale = 1.0
        self._inter = 0.0
        if out_dtype is None:
            out_dtype = arr_dtype
        else:
            out_dtype = np.dtype(out_dtype)
        self._out_dtype = out_dtype
        if np.can_cast(arr_dtype, out_dtype):
            return
        if 'V' in (arr_dtype.kind, out_dtype.kind):
            raise WriterError('Cannot cast to or from non-numeric types')
        if out_dtype.kind == 'c':
            return
        if arr_dtype.kind == 'c':
            raise WriterError('Cannot cast complex types to non-complex')
        if out_dtype.kind == 'f':
            return
        # Try to scale, raise error if we fail
        self._scale_fi2i()
        if np.all(np.isfinite([self.scale, self.inter])):
            raise WriterError('Non-finite scaling or intercept')

    @property
    def array(self):
        """ Return array from arraywriter """
        return self._array

    @property
    def out_dtype(self):
        """ Return `out_dtype` from arraywriter """
        return self._out_dtype

    def _get_scale(self):
        return self._scale
    def _set_scale(self, val):
        self._scale = self.scale_inter_type(val)
    scale = property(_get_scale, _set_scale, None, 'get/set scale')

    def _get_inter(self):
        return self._inter
    def _set_inter(self, val):
        self._inter = self.scale_inter_type(val)
    inter = property(_get_inter, _set_inter, None, 'get/set inter')

    def _scale_fi2i(self):
        """ Calculate / set scaling for floats/(u)ints to (u)ints
        """
        arr = self._array
        arr_dtype = arr.dtype
        out_dtype = self._out_dtype
        assert out_dtype.kind in 'iu'
        mn, mx = self.finite_range()
        if mn == np.inf : # No valid data
            return
        if (mn, mx) == (0.0, 0.0): # Data all zero
            return
        if arr_dtype.kind == 'f':
            # Float to (u)int scaling
            self._scale_f2i()
            return
        # (u)int to (u)int
        out_max, out_min = np.iinfo(out_dtype)
        if mx <= out_max and mn >= out_min: # already in range
            return
        if out_min == 0 and mx < 0 and abs(mn) <= out_max: # sign flip?
            # Careful: -1.0 * arr will be in scale_type precision
            self.scale = -1.0
            return
        # (u)int to (u)int scaling
        self._scale_i2i()

    def _scale_f2i(self):
        # with intercept
        mn, mx = self.finite_range() # These will be floats
        out_max, out_min = np.iinfo(self._out_dtype) # These integers
        if out_min == 0: # uint
            if mn < 0 and mx > 0:
                # Maybe an intercept of the min will be enough
                # NNB - overflow?
                if (mx - mn) <= out_max:
                    self.inter = mn
                    return
        # Overflow?
        data_range = mx-mn
        if data_range == 0:
            self.inter = mn
            return
        type_range = out_max - out_min
        self.scale = data_range / type_range
        self.inter = mn - out_min * self.scale

    def _scale_i2i(self):
        # with intercept
        mn, mx = self.finite_range() # These will be integers
        out_max, out_min = np.iinfo(self._out_dtype) # These too
        # Overflow?
        data_range = mx-mn
        if data_range == 0:
            self.inter = mn
            return
        type_range = out_max - out_min
        self.scale = data_range / type_range
        self.inter = mn - out_min * self.scale

    def finite_range(self):
        """ Return (maybe cached) finite range of data array """
        try:
            return self._finite_range
        except AttributeError:
            pass
        from .volumeutils import finite_range
        self._finite_range = finite_range(self._array)
        return self._finite_range

    def to_fileobj(self, fileobj, order='F'):
        """ Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
        data = self._array
        out_dtype = self._out_dtype
        scale, inter = self._scale, self._inter
        if not order in 'FC':
            raise ValueError('Order should be one of F or C')
        if order == 'F':
            data = data.T
        if data.ndim < 2: # a little hack to allow 1D arrays in loop below
            data = [data]
        for dslice in data: # cycle over largest dimension to save memory
            if inter != 0.0:
                dslice = dslice - inter
            if scale != 1.0:
                dslice = dslice / scale
            if dslice.dtype != out_dtype:
                dslice = dslice.astype(out_dtype)
            fileobj.write(dslice.tostring())


class ScaleArrayWriter(ScaleInterArrayWriter):
    @property
    def inter(self):
        """ Intercept read only for slope writer """
        return self._inter

    def _scale_f2i(self):
        # without intercept
        mn, mx = self.finite_range() # These will be floats
        out_max, out_min = np.iinfo(self._out_dtype) # These integers
        if out_min == 0: # uint
            if mn < 0 and mx > 0:
                raise ValueError('Cannot scale negative and positive '
                                'numbers to uint without intercept')
            if mx < 0: # All input numbers < 0
                self.scale = mn / out_max
            else: # All input numbers > 0
                self.scale = mx / out_max
        else: # int
            if abs(mx) >= abs(mn):
                self.scale = mx / out_max
            else: # Is this right?  Why?
                self.scale = mn / out_min

    def _scale_i2i(self):
        # without intercept
        mn, mx = self.finite_range() # These will be integers
        out_max, out_min = np.iinfo(self._out_dtype) # These too
        scale_type = self.scale_inter_type
        if out_min == 0: # uint
            if mn < 0 and mx > 0:
                raise ValueError('Cannot scale negative and positive '
                                'numbers to uint without intercept')
            if mx < 0: # All input numbers < 0
                self.scale = scale_type(mn) / out_max
            else: # All input numbers > 0
                self.scale = scale_type(mx) / out_max
        else: # int
            if abs(mx) >= abs(mn):
                self.scale = scale_type(mx) / out_max
            else: # Is this right?  Why?
                self.scale = scale_type(mn) / out_min
