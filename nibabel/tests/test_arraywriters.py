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

from ..arraywriters import ArrayWriter
from ..volumeutils import array_from_file

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def check_round_trip(writer, order='F'):
    sio = BytesIO()
    arr = writer.array
    writer.to_fileobj(sio, order)
    arr_back = array_from_file(arr.shape, writer.out_dtype, sio, order=order)
    assert_array_equal(arr_back, arr)
    return arr_back


def test_arraywriter():
    # Test initialize
    # Simple cases
    arr = np.arange(10)
    aw = ArrayWriter(arr)
    assert_true(aw.array is arr)
    assert_equal(aw.out_dtype, arr.dtype)
    check_round_trip(aw)
    # Byteswapped is OK
    bs_arr = arr.byteswap().newbyteorder('S')
    bs_aw = ArrayWriter(bs_arr)
    check_round_trip(bs_aw)
    bs_aw2 = ArrayWriter(bs_arr, arr.dtype)
    check_round_trip(bs_aw2)
    # Different type is not
    assert_raises(ValueError, ArrayWriter, arr, np.float32)
    # 2D array
    arr2 = np.reshape(arr, (2, 5))
    a2w = ArrayWriter(arr2)
    arr_back = check_round_trip(a2w)
    # Array comes back in written order
    assert_true(arr_back.flags.f_contiguous)
    # Or given order
    arr_back = check_round_trip(a2w, 'C')
    assert_true(arr_back.flags.c_contiguous)
