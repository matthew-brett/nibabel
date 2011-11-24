# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for scaling / rounding in volumeutils module '''
from __future__ import with_statement

from functools import partial

import numpy as np

from ..py3k import BytesIO
from ..casting import CastingError
from ..volumeutils import (calculate_scale, scale_min_max, finite_range,
                           int_scinter_ftype, array_to_file, array_from_file)

from numpy.testing import (assert_array_almost_equal, assert_array_equal)

from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_not_equal)

def test_scale_min_max():
    mx_dt = np.maximum_sctype(np.float)
    for tp in np.sctypes['uint'] + np.sctypes['int']:
        info = np.iinfo(tp)
        # Need to pump up to max fp type to contain python longs
        imin = np.array(info.min, dtype=mx_dt)
        imax = np.array(info.max, dtype=mx_dt)
        value_pairs = (
            (0, imax),
            (imin, 0),
            (imin, imax),
            (1, 10),
            (-1, -1),
            (1, 1),
            (-10, -1),
            (-100, 10))
        for mn, mx in value_pairs:
            # with intercept
            scale, inter = scale_min_max(mn, mx, tp, True)
            if mx-mn:
                assert_array_almost_equal, (mx-inter) / scale, imax
                assert_array_almost_equal, (mn-inter) / scale, imin
            else:
                assert_equal, (scale, inter), (1.0, mn)
            # without intercept
            if imin == 0 and mn < 0 and mx > 0:
                (assert_raises, ValueError,
                       scale_min_max, mn, mx, tp, False)
                continue
            scale, inter = scale_min_max(mn, mx, tp, False)
            assert_equal, inter, 0.0
            if mn == 0 and mx == 0:
                assert_equal, scale, 1.0
                continue
            sc_mn = mn / scale
            sc_mx = mx / scale
            assert_true, sc_mn >= imin
            assert_true, sc_mx <= imax
            if imin == 0:
                if mx > 0: # numbers all +ve
                    assert_array_almost_equal, mx / scale, imax
                else: # numbers all -ve
                    assert_array_almost_equal, mn / scale, imax
                continue
            if abs(mx) >= abs(mn):
                assert_array_almost_equal, mx / scale, imax
            else:
                assert_array_almost_equal, mn / scale, imin


def test_finite_range():
    # Finite range utility function
    a = np.array([[-1, 0, 1],[np.inf, np.nan, -np.inf]])
    assert_equal(finite_range(a), (-1.0, 1.0))
    a = np.array([[np.nan],[np.nan]])
    assert_equal(finite_range(a), (np.inf, -np.inf))
    a = np.array([[-3, 0, 1],[2,-1,4]], dtype=np.int)
    assert_equal(finite_range(a), (-3, 4))
    a = np.array([[1, 0, 1],[2,3,4]], dtype=np.uint)
    assert_equal(finite_range(a), (0, 4))
    a = a + 1j
    assert_raises(TypeError, finite_range, a)
    # 1D case
    a = np.array([0., 1, 2, 3])
    assert_equal(finite_range(a), (0,3))


def test_calculate_scale():
    # Test for special cases in scale calculation
    npa = np.array
    # Case where sign flip handles scaling
    res = calculate_scale(npa([-2, -1], dtype=np.int8), np.uint8, 1)
    assert_equal(res, (-1.0, 0.0, None, None))
    # Not having offset not a problem obviously
    res = calculate_scale(npa([-2, -1], dtype=np.int8), np.uint8, 0)
    assert_equal(res, (-1.0, 0.0, None, None))
    # Case where offset handles scaling
    res = calculate_scale(npa([-1, 1], dtype=np.int8), np.uint8, 1)
    assert_equal(res, (1.0, -1.0, None, None))
    # Can't work for no offset case
    assert_raises(ValueError,
                  calculate_scale, npa([-1, 1], dtype=np.int8), np.uint8, 0)
    # Offset trick can't work when max is out of range
    res = calculate_scale(npa([-1, 255], dtype=np.int16), np.uint8, 1)
    assert_not_equal(res, (1.0, -1.0, None, None))


def test_int_scinter():
    # Finding float type needed for applying scale, offset to ints
    assert_equal(int_scinter_ftype(np.int8, 1.0, 0.0), np.float32)
    assert_equal(int_scinter_ftype(np.int8, -1.0, 0.0), np.float32)
    assert_equal(int_scinter_ftype(np.int8, 1e38, 0.0), np.float64)
    assert_equal(int_scinter_ftype(np.int8, -1e38, 0.0), np.float64)


def test_a2f_mn_mx():
    # Test array to file mn, mx handling
    str_io = BytesIO()
    for out_type in (np.int16, np.float32):
        arr = np.arange(6, dtype=out_type)
        arr_orig = arr.copy() # safe backup for testing against
        # Basic round trip to warm up
        array_to_file(arr, str_io)
        data_back = array_from_file(arr.shape, out_type, str_io)
        assert_array_equal(arr, data_back)
        # Clip low
        array_to_file(arr, str_io, mn=2)
        data_back = array_from_file(arr.shape, out_type, str_io)
        # arr unchanged
        assert_array_equal(arr, arr_orig)
        # returned value clipped low
        assert_array_equal(data_back, [2,2,2,3,4,5])
        # Clip high
        array_to_file(arr, str_io, mx=4)
        data_back = array_from_file(arr.shape, out_type, str_io)
        # arr unchanged
        assert_array_equal(arr, arr_orig)
        # returned value clipped high
        assert_array_equal(data_back, [0,1,2,3,4,4])
        # Clip both
        array_to_file(arr, str_io, mn=2, mx=4)
        data_back = array_from_file(arr.shape, out_type, str_io)
        # arr unchanged
        assert_array_equal(arr, arr_orig)
        # returned value clipped high
        assert_array_equal(data_back, [2,2,2,3,4,4])


def test_a2f_nan2zero():
    # Test conditions under which nans written to zero
    arr = np.array([np.nan, 99.], dtype=np.float32)
    str_io = BytesIO()
    array_to_file(arr, str_io)
    data_back = array_from_file(arr.shape, np.float32, str_io)
    assert_array_equal(np.isnan(data_back), [True, False])
    # nan2zero ignored for floats
    array_to_file(arr, str_io, nan2zero=True)
    data_back = array_from_file(arr.shape, np.float32, str_io)
    assert_array_equal(np.isnan(data_back), [True, False])
    # Integer output without nan2zero should raise error
    a2f_no_n2z = partial(array_to_file, nan2zero=False)
    assert_raises(CastingError, a2f_no_n2z, arr, str_io, np.int32)
    # Integer output with gives zero
    array_to_file(arr, str_io, np.int32, nan2zero=True)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, [0, 99])


def type_min_max(dtype_type):
    # Utility routine to return min, max for dtype dtype
    if dtype_type in (np.sctypes['complex'] + np.sctypes['float']):
        info = np.finfo(dtype_type)
    elif dtype_type in (np.sctypes['int'] + np.sctypes['uint']):
        info = np.iinfo(dtype_type)
    else:
        raise ValueError('Sorry, out of my range')
    return info.min, info.max


def test_array_file_scales():
    # Test scaling works for max, min when going from larger to smaller type,
    # and from float to integer.
    bio = BytesIO()
    for in_type, out_type, err in ((np.int16, np.int16, None),
                                   (np.int16, np.int8, None),
                                   (np.uint16, np.uint8, None),
                                   (np.int32, np.int8, None),
                                   (np.float32, np.uint8, None),
                                   (np.float32, np.int16, None)):
        out_dtype = np.dtype(out_type)
        arr = np.zeros((3,), dtype=in_type)
        arr[0], arr[1] = type_min_max(in_type)
        if not err is None:
            assert_raises(err, calculate_scale, arr, out_dtype, True)
            continue
        slope, inter, mn, mx = calculate_scale(arr, out_dtype, True)
        array_to_file(arr, bio, out_type, 0, inter, slope, mn, mx)
        bio.seek(0)
        arr2 = array_from_file(arr.shape, out_dtype, bio)
        arr3 = arr2 * slope + inter
        # Max rounding error for integer type
        max_miss = slope / 2.
        assert_true(np.all(np.abs(arr - arr3) <= max_miss))
        bio.truncate(0)
        bio.seek(0)


def test_scaling_in_abstract():
    # Confirm that, for all ints and uints as input, and all possible outputs,
    # for any simple way of doing the calculation, the result is near enough
    for category0, category1 in (('int', 'int'),
                                 ('uint', 'int'),
                                ):
        for in_type in np.sctypes[category0]:
            for out_type in np.sctypes[category1]:
                check_int_a2f(in_type, out_type)
    # Converting floats to integer
    for category0, category1 in (('float', 'int'),
                                 ('float', 'uint'),
                                 ('complex', 'int'),
                                 ('complex', 'uint'),
                                ):
        for in_type in np.sctypes[category0]:
            for out_type in np.sctypes[category1]:
                check_int_a2f(in_type, out_type)


def check_int_a2f(in_type, out_type):
    # Check that array to / from file returns roughly the same as input
    big_floater = np.maximum_sctype(np.float)
    this_min, this_max = type_min_max(in_type)
    if not in_type in np.sctypes['complex']:
        data = np.array([this_min, this_max], in_type)
    else: # Funny behavior with complex256
        data = np.zeros((2,), in_type)
        data[0] = this_min + 0j
        data[1] = this_max + 0j
    str_io = BytesIO()
    scale, inter, mn, mx = calculate_scale(data, out_type, True)
    if scale == np.inf or inter == np.inf:
        return
    array_to_file(data, str_io, out_type, 0, inter, scale, mn, mx)
    data_back = array_from_file(data.shape, out_type, str_io)
    if not scale is None and scale !=1.0:
        data_back = data_back * scale
    if not inter is None and inter !=0:
        data_back = data_back + inter
    assert_true(np.allclose(big_floater(data), big_floater(data_back)))
    # Try with analyze-size scale and inter
    scale32 = np.float32(scale)
    inter32 = np.float32(inter)
    if scale32 == np.inf or inter32 == np.inf:
        return
    data_back = array_from_file(data.shape, out_type, str_io)
    if not scale32 is None and scale32 !=1.0:
        data_back = data_back * scale32
    if not inter32 is None and inter32 !=0:
        data_back = data_back + inter32
    # Clip at extremes to remove inf
    out_min, out_max = type_min_max(in_type)
    assert_true(np.allclose(big_floater(data),
                            big_floater(np.clip(data_back, out_min, out_max))))