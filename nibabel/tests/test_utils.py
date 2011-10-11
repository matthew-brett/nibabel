# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for volumeutils module '''
from __future__ import with_statement
from ..py3k import BytesIO, asbytes
import tempfile

import numpy as np

from ..tmpdirs import InTemporaryDirectory

from ..floating import floor_exact
from ..volumeutils import (array_from_file,
                           array_to_file,
                           ScalingError,
                           calculate_scale,
                           scale_min_max,
                           finite_range,
                           can_cast, allopen,
                           make_dt_codes,
                           native_code,
                           shape_zoom_affine,
                           rec2dict)

from numpy.testing import (assert_array_almost_equal,
                           assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_not_equal)


def test_array_from_file():
    shape = (2,3,4)
    dtype = np.dtype(np.float32)
    in_arr = np.arange(24, dtype=dtype).reshape(shape)
    # Check on string buffers
    offset = 0
    assert_true(buf_chk(in_arr, BytesIO(), None, offset))
    offset = 10
    assert_true(buf_chk(in_arr, BytesIO(), None, offset))
    # check on real file
    fname = 'test.bin'
    with InTemporaryDirectory() as tmpdir:
        # fortran ordered
        out_buf = open(fname, 'wb')
        in_buf = open(fname, 'rb')
        assert_true(buf_chk(in_arr, out_buf, in_buf, offset))
        # Drop offset to check that shape's not coming from file length
        out_buf.seek(0)
        in_buf.seek(0)
        offset = 5
        assert_true(buf_chk(in_arr, out_buf, in_buf, offset))
        del out_buf, in_buf
    # Make sure empty shape, and zero length, give empty arrays
    arr = array_from_file((), np.dtype('f8'), BytesIO())
    assert_equal(len(arr), 0)
    arr = array_from_file((0,), np.dtype('f8'), BytesIO())
    assert_equal(len(arr), 0)
    # Check error from small file
    assert_raises(IOError, array_from_file,
                        shape, dtype, BytesIO())
    # check on real file
    fd, fname = tempfile.mkstemp()
    with InTemporaryDirectory():
        open(fname, 'wb').write(asbytes('1'))
        in_buf = open(fname, 'rb')
        # For windows this will raise a WindowsError from mmap, Unices
        # appear to raise an IOError
        assert_raises(Exception, array_from_file,
                            shape, dtype, in_buf)
        del in_buf


def buf_chk(in_arr, out_buf, in_buf, offset):
    ''' Write contents of in_arr into fileobj, read back, check same '''
    instr = asbytes(' ') * offset + in_arr.tostring(order='F')
    out_buf.write(instr)
    out_buf.flush()
    if in_buf is None: # we're using in_buf from out_buf
        out_buf.seek(0)
        in_buf = out_buf
    arr = array_from_file(
        in_arr.shape,
        in_arr.dtype,
        in_buf,
        offset)
    return np.allclose(in_arr, arr)


def test_array_to_file():
    arr = np.arange(10).reshape(5,2)
    str_io = BytesIO()
    for tp in (np.uint64, np.float, np.complex):
        dt = np.dtype(tp)
        for code in '<>':
            ndt = dt.newbyteorder(code)
            for allow_intercept in (True, False):
                scale, intercept, mn, mx = calculate_scale(arr,
                                                           ndt,
                                                           allow_intercept)
                data_back = write_return(arr, str_io, ndt,
                                         0, intercept, scale)
                assert_array_almost_equal(arr, data_back)
    ndt = np.dtype(np.float)
    arr = np.array([0.0, 1.0, 2.0])
    # intercept
    data_back = write_return(arr, str_io, ndt, 0, 1.0)
    assert_array_equal(data_back, arr-1)
    # scaling
    data_back = write_return(arr, str_io, ndt, 0, 1.0, 2.0)
    assert_array_equal(data_back, (arr-1) / 2.0)
    # min thresholding
    data_back = write_return(arr, str_io, ndt, 0, 0.0, 1.0, 1.0)
    assert_array_equal(data_back, [1.0, 1.0, 2.0])
    # max thresholding
    data_back = write_return(arr, str_io, ndt, 0, 0.0, 1.0, 0.0, 1.0)
    assert_array_equal(data_back, [0.0, 1.0, 1.0])
    # order makes not difference in 1D case
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, [0.0, 1.0, 2.0])
    # but does in the 2D case
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    data_back = write_return(arr, str_io, ndt, order='F')
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, arr.T)
    # nans set to 0 for integer output case, not float
    arr = np.array([[np.nan, 0],[0, np.nan]])
    data_back = write_return(arr, str_io, ndt) # float, thus no effect
    assert_array_equal(data_back, arr)
    # True is the default, but just to show its possible
    data_back = write_return(arr, str_io, ndt, nan2zero=True)
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io,
                             np.dtype(np.int64), nan2zero=True)
    assert_array_equal(data_back, [[0, 0],[0, 0]])
    # otherwise things get a bit weird; tidied here
    # How weird?  Look at arr.astype(np.int64)
    data_back = write_return(arr, str_io,
                             np.dtype(np.int64), nan2zero=False)
    assert_array_equal(data_back, arr.astype(np.int64))
    # check that non-zero file offset works
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    str_io = BytesIO()
    str_io.write(asbytes('a') * 42)
    array_to_file(arr, str_io, np.float, 42)
    data_back = array_from_file(arr.shape, np.float, str_io, 42)
    assert_array_equal(data_back, arr.astype(np.float))
    # that default dtype is input dtype
    str_io = BytesIO()
    array_to_file(arr.astype(np.int16), str_io)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    assert_array_equal(data_back, arr.astype(np.int16))
    # that, if there is no valid data, we get zeros
    str_io = BytesIO()
    array_to_file(arr + np.inf, str_io, np.int32, 0, 0.0, None)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, np.zeros(arr.shape))


def test_a2f_clipping():
    # Check scaled range clipping
    arr = np.arange(10).astype(np.int32)
    arr_orig = arr.copy() # check for overwriting
    str_io = BytesIO()
    array_to_file(arr, str_io, sc_mn=2, sc_mx=8)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(arr, arr_orig)
    assert_array_equal(data_back, [2,2,2,3,4,5,6,7,8,8])
    array_to_file(arr, str_io, np.int16, sc_mn=2, sc_mx=8)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    assert_array_equal(data_back, [2,2,2,3,4,5,6,7,8,8])
    array_to_file(arr, str_io, np.int16, intercept=-1.0, sc_mn=2, sc_mx=8)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    assert_array_equal(data_back, [2,2,3,4,5,6,7,8,8,8])
    assert_array_equal(arr, arr_orig)


def test_a2f_mn_mx():
    # Test array to file mn, mx handling
    arr = np.arange(6, dtype=np.int16)
    arr_orig = arr.copy() # safe backup for testing against
    str_io = BytesIO()
    # Basic round trip to warm up
    array_to_file(arr, str_io)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    assert_array_equal(arr, data_back)
    # Clip low
    array_to_file(arr, str_io, mn=2)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    # arr unchanged
    assert_array_equal(arr, arr_orig)
    # returned value clipped low
    assert_array_equal(data_back, [2,2,2,3,4,5])
    # Clip high
    array_to_file(arr, str_io, mx=4)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    # arr unchanged
    assert_array_equal(arr, arr_orig)
    # returned value clipped high
    assert_array_equal(data_back, [0,1,2,3,4,4])
    # Clip both
    array_to_file(arr, str_io, mn=2, mx=4)
    data_back = array_from_file(arr.shape, np.int16, str_io)
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
    # Integer output without
    array_to_file(arr, str_io, np.int32, nan2zero=False)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    crude_conv = np.array([np.nan, 99]).astype(np.int32)
    assert_true(crude_conv[0] != 0)
    assert_array_equal(data_back, crude_conv)
    # Integer output with
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


def test_scaling_in_abstract():
    # Confirm that, for all ints and uints as input, and all possible outputs,
    # for any simple way of doing the calculation, the result is near enough
    for category0, category1 in (('int', 'int'),
                                 ('uint', 'int'),
                                ):
        for in_type in np.sctypes[category0]:
            for out_type in np.sctypes[category1]:
                check_int_conv(in_type, out_type)
                check_int_a2f(in_type, out_type)
    # Converting floats to integer
    for category0, category1 in (('float', 'int'),
                                 ('float', 'uint'),
                                 ('complex', 'int'),
                                 ('complex', 'uint'),
                                ):
        for in_type in np.sctypes[category0]:
            for out_type in np.sctypes[category1]:
                try:
                    check_int_conv(in_type, out_type)
                    check_int_a2f(in_type, out_type)
                except ScalingError:
                    print 'Scaling failed for %s to %s' % (in_type, out_type)


def check_int_conv(in_type, out_type):
    # Test that input and output are the same with scaling, intercept
    this_min, this_max = type_min_max(in_type)
    out_min, out_max = type_min_max(out_type)
    if not in_type in np.sctypes['complex']:
        data = np.array([this_min, this_max], in_type)
    else: # Funny behavior with complex256
        data = np.zeros((2,), in_type)
        data[0] = this_min + 0j
        data[1] = this_max + 0j
    out_dtype = np.dtype(out_type)
    scale, inter, mn, mx, sc_mn, sc_mx = calculate_scale(data, out_dtype,
                                                         True, True)
    if scale == np.inf or inter == np.inf:
        return
    if scale == 1.0 and inter == 0.0:
        return
    # Simulate float casting
    cast_dtype = np.array(inter / scale).dtype
    if in_type in np.sctypes['float'] + np.sctypes['complex']:
        res = data.astype(cast_dtype)
    else:
        res = data.copy()
    # Thresholding; assume mx not None if mn is not None
    if not mn is None:
        np.clip(res, mn, mx, res)
    res = res / scale - inter / scale
    # Simulate scaled thresholding
    if not sc_mn is None:
        np.clip(res, sc_mn, sc_mx, res)
    # Simulate rinting
    rinted = np.rint(res)
    raw_back = rinted.astype(out_type)
    scaled_back = raw_back * scale + inter
    assert_true(np.allclose(data, scaled_back))
    # Check actual scaling as it will be applied
    scale32 = np.float32(scale)
    inter32 = np.float32(inter)
    if scale32 == np.inf or inter32 == np.inf:
        return
    scaled_back = raw_back * scale32 + inter32
    assert_true(np.allclose(data, clip_infs(scaled_back)))


def check_int_a2f(in_type, out_type):
    # Check that array to file returns roughly the same
    big_floater = np.maximum_sctype(np.float)
    this_min, this_max = type_min_max(in_type)
    if not in_type in np.sctypes['complex']:
        data = np.array([this_min, this_max], in_type)
    else: # Funny behavior with complex256
        data = np.zeros((2,), in_type)
        data[0] = this_min + 0j
        data[1] = this_max + 0j
    str_io = BytesIO()
    scale, inter, mn, mx, sc_mn, sc_mx = calculate_scale(data, out_type,
                                                         True, True)
    if scale == np.inf or inter == np.inf:
        return
    array_to_file(data, str_io, out_type, 0, inter, scale, mn, mx,
                  sc_mn=sc_mn, sc_mx=sc_mx)
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
    if not scale is None and scale !=1.0:
        data_back = data_back * scale
    if not inter is None and inter !=0:
        data_back = data_back + inter
    assert_true(np.allclose(big_floater(data),
                            big_floater(clip_infs(data_back))))


def clip_infs(arr):
    # Clips arrays at maximum value and thus removes infs
    dtt = arr.dtype.type
    try:
        info = np.finfo(dtt)
    except ValueError:
        return arr
    return np.clip(arr, info.min, info.max)


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


def write_return(data, fileobj, out_dtype, *args, **kwargs):
    fileobj.truncate(0)
    array_to_file(data, fileobj, out_dtype, *args, **kwargs)
    data = array_from_file(data.shape, out_dtype, fileobj)
    return data


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


def test_calc_scale():
    # Test calculate scale routine
    i16i = np.iinfo(np.int16)
    i64i = np.iinfo(np.int64)
    u64i = np.iinfo(np.uint64)
    # scaling should scale to full range
    scale, inter = scale_min_max(0, i64i.max, np.uint64, False)
    assert_equal((scale, inter), (0.5, 0.0))
    # calc scale should shortcut this because we gain no precision
    data = np.array([0, i64i.max], dtype=np.int64)
    scale, inter, mn, mx = calculate_scale(data, np.uint64, False)
    assert_equal((scale, inter, mn, mx), (1.0, 0.0, None, None))
    # Even if we allow an intercept
    scale, inter, mn, mx = calculate_scale(data, np.uint64, True)
    assert_equal((scale, inter), (1.0, 0.0))
    # Trying to scale neg, pos without intercept kicks error
    assert_raises(ScalingError, scale_min_max, -1, 1, np.uint64, False)
    data = np.array([-1, 1], dtype=np.int64)
    assert_raises(ScalingError, calculate_scale, data, np.uint64, False)
    # But it's OK with an intercept
    scale, inter = scale_min_max(i64i.min, i64i.max, np.uint64, True)
    assert_equal((scale, inter), (1.0, i64i.min))
    i64_data = np.array([i64i.min, i64i.max], dtype=np.int64)
    scale, inter, mn, mx = calculate_scale(i64_data, np.uint64, True)
    assert_equal((scale, inter, mn, mx), (1.0, i64i.min, None, None))
    # Check in_type works as passed parameter
    scale, inter = scale_min_max(-1.5, 1.5, np.int8, True)
    assert_almost_equal((scale, inter), (3.0/255, 128./255*3.0-1.5))
    # Now floats should be truncated
    scale, inter = scale_min_max(-1.5, 1.5, np.int8, True, in_type=np.int8)
    assert_almost_equal((scale, inter), (2.0/255, 128./255*2.0-1))
    # Check range checking output
    # This first shouldn't be controversial
    scale, inter, sc_mn, sc_mx = scale_min_max(
        i16i.min, i16i.max, np.int16, False, True)
    assert_equal((scale, inter, sc_mn, sc_mx), (1.0, 0.0, None, None))
    i16_data = np.array([i16i.min, i16i.max], dtype=np.int16)
    scale, inter, mn, mx, sc_mn, sc_mx = calculate_scale(i16_data, np.int16,
                                                         False, True)
    assert_equal((scale, inter, sc_mn, sc_mx), (1.0, 0.0, None, None))
    # This will (at least on my laptop) run into range errors with rounding We
    # need to pass in the in_type to calc_scale, because it appears that
    # u64i.max becomes an object by default when cast into an array.
    scale, inter, out_mn, out_mx = scale_min_max(0, u64i.max, np.int64,
                                                 True, True, np.uint64)
    # Assert we do need the range check (float rounding above max for dtype)
    work_type = np.array(scale).dtype.type
    default_scaled = work_type(u64i.max) / scale - inter / scale
    floored_scaled = floor_exact(i64i.max, work_type)
    assert_true(int(default_scaled) > i64i.max)
    # Check the floored check worked
    assert_not_equal(floored_scaled, default_scaled)
    assert_equal((scale, inter, out_mn, out_mx),
                 (1.0, -i64i.min, i64i.min, floored_scaled))
    # Calc scale version.
    u64_data = np.array([0, u64i.max], dtype=np.uint64)
    scale, inter, mn, mx, sc_mn, sc_mx = calculate_scale(u64_data, np.int64,
                                                         True, True)
    assert_equal((scale, inter, mn, mx, sc_mn, sc_mx),
                 (1.0, -i64i.min, None, None, i64i.min, floored_scaled))
    # Simple no-scale cases
    # in-range
    vals = calculate_scale(np.array([0, 1], dtype=np.int64), np.uint8,
                           False, False)
    assert_equal(vals, [1.0, 0.0, None, None])
    vals = calculate_scale(np.array([0, 1], dtype=np.int64), np.uint8,
                           False, True)
    assert_equal(vals, [1.0, 0.0, None, None, None, None])
    # Sign flip
    vals = calculate_scale(np.array([0, -1], dtype=np.int64), np.uint8,
                           False, False)
    assert_equal(vals, [-1.0, 0.0, None, None])
    vals = calculate_scale(np.array([0, -1], dtype=np.int64), np.uint8,
                           False, True)
    assert_equal(vals, [-1.0, 0.0, None, None, None, None])
    # Just intercept
    vals = calculate_scale(np.array([-1, 1], dtype=np.int64), np.uint8,
                           True, False)
    assert_equal(vals, [1.0, -1.0, None, None])
    vals = calculate_scale(np.array([-1, 1], dtype=np.int64), np.uint8,
                           True, True)
    assert_equal(vals, [1.0, -1.0, None, None, None, None])
    # No valid data
    vals = calculate_scale(np.array([np.nan, np.nan], dtype=np.float),
                           np.uint8, True, False)
    assert_equal(vals, [None, None, None, None])


def test_can_cast():
    tests = ((np.float32, np.float32, True, True, True),
             (np.float64, np.float32, True, True, True),
             (np.complex128, np.float32, False, False, False),
             (np.float32, np.complex128, True, True, True),
             (np.uint32, np.complex128, True, True, True),
             (np.int64, np.float32, True, True, True),
             (np.complex128, np.int16, False, False, False),
             (np.float32, np.int16, False, True, True),
             (np.uint8, np.int16, True, True, True),
             (np.uint16, np.int16, False, True, True),
             (np.int16, np.uint16, False, False, True),
             (np.int8, np.uint16, False, False, True),
             (np.uint16, np.uint8, False, True, True),
             )
    for intype, outtype, def_res, scale_res, all_res in tests:
        assert_equal, def_res, can_cast(intype, outtype)
        assert_equal, scale_res, can_cast(intype, outtype, False, True)
        assert_equal, all_res, can_cast(intype, outtype, True, True)


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
    assert_equal(finite_range(a), (0+1j, 4+1j))
    # 1D case
    a = np.array([0., 1, 2, 3])
    assert_equal(finite_range(a), (0,3))


def test_allopen():
    # Test default mode is 'rb'
    fobj = allopen(__file__)
    assert_equal(fobj.mode, 'rb')
    # That we can set it
    fobj = allopen(__file__, 'r')
    assert_equal(fobj.mode, 'r')
    # with keyword arguments
    fobj = allopen(__file__, mode='r')
    assert_equal(fobj.mode, 'r')
    # fileobj returns fileobj
    sobj = BytesIO()
    fobj = allopen(sobj)
    assert_true(fobj is sobj)
    # mode is gently ignored
    fobj = allopen(sobj, mode='r')


def test_shape_zoom_affine():
    shape = (3, 5, 7)
    zooms = (3, 2, 1)
    res = shape_zoom_affine((3, 5, 7), (3, 2, 1))
    exp = np.array([[-3.,  0.,  0.,  3.],
                    [ 0.,  2.,  0., -4.],
                    [ 0.,  0.,  1., -3.],
                    [ 0.,  0.,  0.,  1.]])
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine((3, 5), (3, 2))
    exp = np.array([[-3.,  0.,  0.,  3.],
                    [ 0.,  2.,  0., -4.],
                    [ 0.,  0.,  1., -0.],
                    [ 0.,  0.,  0.,  1.]])
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine((3, 5, 7), (3, 2, 1), False)
    exp = np.array([[ 3.,  0.,  0., -3.],
                    [ 0.,  2.,  0., -4.],
                    [ 0.,  0.,  1., -3.],
                    [ 0.,  0.,  0.,  1.]])
    assert_array_almost_equal(res, exp)


def test_rec2dict():
    r = np.zeros((), dtype = [('x', 'i4'), ('s', 'S10')])
    d = rec2dict(r)
    assert_equal(d, {'x': 0, 's': asbytes('')})


def test_dtypes():
    # numpy - at least up to 1.5.1 - has odd behavior for hashing -
    # specifically:
    # In [9]: hash(dtype('<f4')) == hash(dtype('<f4').newbyteorder('<'))
    # Out[9]: False
    # In [10]: dtype('<f4') == dtype('<f4').newbyteorder('<')
    # Out[10]: True
    # where '<' is the native byte order
    dt_defs = ((16, 'float32', np.float32),)
    dtr = make_dt_codes(dt_defs)
    # check we have the fields we were expecting
    assert_equal(dtr.value_set(), set((16,)))
    assert_equal(dtr.fields, ('code', 'label', 'type',
                              'dtype', 'sw_dtype'))
    # These of course should pass regardless of dtype
    assert_equal(dtr[np.float32], 16)
    assert_equal(dtr['float32'], 16)
    # These also pass despite dtype issue
    assert_equal(dtr[np.dtype(np.float32)], 16)
    assert_equal(dtr[np.dtype('f4')], 16)
    assert_equal(dtr[np.dtype('f4').newbyteorder('S')], 16)
    # But this one used to fail
    assert_equal(dtr[np.dtype('f4').newbyteorder(native_code)], 16)
    # Check we can pass in niistring as well
    dt_defs = ((16, 'float32', np.float32, 'ASTRING'),)
    dtr = make_dt_codes(dt_defs)
    assert_equal(dtr[np.dtype('f4').newbyteorder('S')], 16)
    assert_equal(dtr.value_set(), set((16,)))
    assert_equal(dtr.fields, ('code', 'label', 'type', 'niistring',
                              'dtype', 'sw_dtype'))
    assert_equal(dtr.niistring[16], 'ASTRING')
    # And that unequal elements raises error
    dt_defs = ((16, 'float32', np.float32, 'ASTRING'),
               (16, 'float32', np.float32))
    assert_raises(ValueError, make_dt_codes, dt_defs)
    # And that 2 or 5 elements raises error
    dt_defs = ((16, 'float32'),)
    assert_raises(ValueError, make_dt_codes, dt_defs)
    dt_defs = ((16, 'float32', np.float32, 'ASTRING', 'ANOTHERSTRING'),)
    assert_raises(ValueError, make_dt_codes, dt_defs)
