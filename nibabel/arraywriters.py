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

from .casting import shared_range, int_to_float


class WriterError(Exception):
    pass


class ScaleInterArrayWriter(object):
    # Working precision of scaling
    scale_inter_type = np.float32

    def __init__(self, array, out_dtype=None, calc_scale=True):
        """ Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        calc_scale : {True, False}
            Whether to calculate scaling for writing `array` on initialization.
            If False, then you can calculate this scaling with
            ``obj.calc_scale()`` - see examples

        Examples
        --------
        >>> arr = np.array([0, 255], np.uint8)
        >>> aw = ScaleInterArrayWriter(arr)
        >>> aw.scale, aw.inter
        (1.0, 0.0)
        >>> aw = ScaleInterArrayWriter(arr, np.int8)
        >>> (aw.scale, aw.inter) == (1.0, 128)
        True
        """
        self._array = np.asanyarray(array)
        arr_dtype = self._array.dtype
        # Use raw set to bypass properties.  Not needed for this class but for
        # subclasses overriding properties but wanting to keep init
        self._scale = self.scale_inter_type(1.0)
        self._inter = self.scale_inter_type(0.0)
        if out_dtype is None:
            out_dtype = arr_dtype
        else:
            out_dtype = np.dtype(out_dtype)
        self._out_dtype = out_dtype
        if calc_scale:
            self.calc_scale()

    def calc_scale(self):
        """ Calculate scaling (if any) for contained array """
        arr_dtype = self._array.dtype
        out_dtype = self._out_dtype
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
        if not np.all(np.isfinite([self.scale, self.inter])):
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
        self._scale = np.squeeze(self.scale_inter_type(val))
    scale = property(_get_scale, _set_scale, None, 'get/set scale')

    def _get_inter(self):
        return self._inter
    def _set_inter(self, val):
        self._inter = np.squeeze(self.scale_inter_type(val))
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
            self._range_scale()
            return
        # (u)int to (u)int
        info = np.iinfo(out_dtype)
        out_max, out_min = info.max, info.min
        if mx <= out_max and mn >= out_min: # already in range
            return
        # From here we will be using scaling, so we need to take into account
        # the precision of the type to which we will scale
        shared_min, shared_max = shared_range(self.scale_inter_type, out_dtype)
        if out_dtype.kind == 'u':
            if mx < 0 and abs(mn) <= shared_max: # sign flip enough?
                # -1.0 * arr will be in scale_inter_type precision
                self.scale = -1.0
                return
        # (u)int to (u)int scaling
        self._ui2i()

    def _ui2i(self):
        """ (u)int to (u)int scaling """
        if self._out_dtype.kind == 'u':
            shared_min, shared_max = shared_range(self.scale_inter_type,
                                                  self._out_dtype)
            mn, mx = self.finite_range()
            if (mx - mn) <= shared_max: # offset enough?
                self.inter = mn
                return
        self._range_scale()

    def _range_scale(self):
        """ Calculate scaling, intercept based on data range and output type """
        mn, mx = self.finite_range() # Values of self.array.dtype type
        if mx == mn: # Only one number in array
            self.inter = mn
            return
        # We need to allow for precision of the type to which we will scale
        # These will be floats of type scale_inter_type
        shared_min, shared_max = shared_range(self.scale_inter_type,
                                              self._out_dtype)
        scaled_mn2mx = np.diff(np.array([shared_min, shared_max],
                                        dtype=np.longdouble))
        # Straight mx-mn can overflow.
        if mn.dtype.kind == 'f': # Already floats
            # float64 and below cast correctly to longdouble.  Longdouble needs
            # no casting
            mn2mx = np.diff(np.array([mn, mx], dtype=np.longdouble))
        else: # max possible (u)int range is 2**64-1 (int64, uint64)
            # int_to_float covers this range.  On windows longdouble is the same
            # as double so mn2mx will be 2**64 - thus overestimating scale
            # slightly.  Casting to int here is probably not necessary
            mn2mx = int_to_float(int(mx) - int(mn), np.longdouble)
        scale = mn2mx / scaled_mn2mx
        self.inter = mn - shared_min * scale
        self.scale = scale

    def finite_range(self):
        """ Return (maybe cached) finite range of data array """
        try:
            return self._finite_range
        except AttributeError:
            pass
        from .volumeutils import finite_range
        self._finite_range = finite_range(self._array)
        return self._finite_range

    def to_fileobj(self, fileobj, order='F', nan2zero=True):
        """ Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        nan2zero : {True, False, None}, optional
            Whether to set NaN values to 0 when writing integer output.
            Defaults to True.  If False, NaNs get converted with numpy
            ``astype``, and the behavior is undefined.  Ignored for floating
            point output.
        """
        data = self._array
        out_dtype = self._out_dtype
        need_int = out_dtype.kind in 'iu'
        nan2zero = nan2zero and data.dtype.kind == 'f'
        scale, inter = self._scale, self._inter
        if not order in 'FC':
            raise ValueError('Order should be one of F or C')
        data = np.atleast_2d(data)
        if order == 'F' or (data.ndim == 2 and data.shape[1] == 1):
            data = data.T
        for dslice in data: # cycle over first dimension to save memory
            if inter != 0.0:
                dslice = dslice - inter
            if scale != 1.0:
                dslice = dslice / scale
            if need_int and dslice.dtype.kind == 'f':
                both_mn, both_mx = shared_range(dslice.dtype, out_dtype)
                dslice = np.clip(np.rint(dslice), both_mn, both_mx)
                if nan2zero:
                    dslice[np.isnan(dslice)] = 0
                dslice = dslice.astype(out_dtype)
            elif dslice.dtype != out_dtype:
                dslice = dslice.astype(out_dtype)
            fileobj.write(dslice.tostring())


class ScaleArrayWriter(ScaleInterArrayWriter):
    @property
    def inter(self):
        """ Intercept read only for slope writer """
        return self._inter

    def _ui2i(self):
        """ (u)int to (u)int scaling """
        self._range_scale()

    def _range_scale(self):
        """ Calculate scaling based on data range and output type """
        mn, mx = self.finite_range() # These can be floats or integers
        # We need to allow for precision of the type to which we will scale
        # These will be floats of type scale_inter_type
        shared_min, shared_max = shared_range(self.scale_inter_type,
                                              self._out_dtype)
        # But we want maximum precision for the calculations
        shared_min, shared_max = np.array([shared_min, shared_max],
                                          dtype = np.longdouble)
        if self._out_dtype.kind == 'u':
            if mn < 0 and mx > 0:
                raise WriterError('Cannot scale negative and positive '
                                  'numbers to uint without intercept')
            if mx < 0: # All input numbers < 0
                self.scale = mn / shared_max
            else: # All input numbers > 0
                self.scale = mx / shared_max
            return
        # Scaling to int
        if abs(mx) >= abs(mn):
            self.scale = mx / shared_max
        else: # Is this right?  Why?
            self.scale = mn / shared_min
