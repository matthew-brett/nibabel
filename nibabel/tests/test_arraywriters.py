""" Testing array writer objects

Array writers have init signature::

    def __init__(self, array, out_dtype=None, order='F')

and methods

* to_fileobj(fileobj, offset=None)

They do have attributes:

* array
* out_dtype
* order

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

from ..py3k import BytesIO

from ..arraywriters import ScaleInterArrayWriter, ScaleArrayWriter, WriterError

from ..volumeutils import array_from_file

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


FLOAT_TYPES = np.sctypes['float']
COMPLEX_TYPES = np.sctypes['complex']
INT_TYPES = np.sctypes['int']
UINT_TYPES = np.sctypes['uint']
CFLOAT_TYPES = FLOAT_TYPES + COMPLEX_TYPES
IUINT_TYPES = INT_TYPES + UINT_TYPES
NUMERIC_TYPES = CFLOAT_TYPES + IUINT_TYPES

def round_trip(writer, order='F'):
    sio = BytesIO()
    arr = writer.array
    writer.to_fileobj(sio, order)
    return array_from_file(arr.shape, writer.out_dtype, sio, order=order)


def test_arraywriters():
    # Test initialize
    # Simple cases
    for klass in (ScaleInterArrayWriter, ScaleArrayWriter):
        for type in NUMERIC_TYPES:
            arr = np.arange(10, dtype=type)
            aw = klass(arr)
            assert_true(aw.array is arr)
            assert_equal(aw.out_dtype, arr.dtype)
            assert_array_equal(arr, round_trip(aw))
            # Byteswapped is OK
            bs_arr = arr.byteswap().newbyteorder('S')
            bs_aw = klass(bs_arr)
            assert_array_equal(bs_arr, round_trip(bs_aw))
            bs_aw2 = klass(bs_arr, arr.dtype)
            assert_array_equal(bs_arr, round_trip(bs_aw2))
            # 2D array
            arr2 = np.reshape(arr, (2, 5))
            a2w = klass(arr2)
            # Default out - in order is Fortran
            arr_back = round_trip(a2w)
            assert_array_equal(arr2, arr_back)
            arr_back = round_trip(a2w, 'F')
            assert_array_equal(arr2, arr_back)
            # C order works as well
            arr_back = round_trip(a2w, 'C')
            assert_array_equal(arr2, arr_back)
            assert_true(arr_back.flags.c_contiguous)


def test_to_float():
    for in_type in NUMERIC_TYPES:
        if in_type in IUINT_TYPES:
            info = np.iinfo(in_type)
            if info.min == 0: # uint
                mn, mx, start, stop, step = info.min, info.max, 0, 100, 1
            else: # int
                mn, mx, start, stop, step = info.min, info.max, 0, 100, 1
        else:
            info = np.finfo(in_type)
            mn, mx, start, stop, step = info.min, info.max, 0, 100, 0.5
        arr = np.arange(start, stop, step, dtype=in_type)
        arr[0] = mn
        arr[-1] = mx
        for out_type in CFLOAT_TYPES:
            for klass in (ScaleInterArrayWriter, ScaleArrayWriter):
                if in_type in COMPLEX_TYPES and out_type in FLOAT_TYPES:
                    assert_raises(WriterError, klass, arr, out_type)
                    continue
                aw = klass(arr, out_type)
                assert_true(aw.array is arr)
                assert_equal(aw.out_dtype, out_type)
                arr_back = round_trip(aw)
                assert_array_equal(arr.astype(out_type), arr_back)
                # Check too-big values overflowed correctly
                out_min = np.finfo(out_type).min
                out_max = np.finfo(out_type).max
                assert_true(np.all(arr_back[arr > out_max] == np.inf))
                assert_true(np.all(arr_back[arr < out_min] == -np.inf))
