""" Test slicing of file-like objects """

from io import BytesIO
from itertools import product

import numpy as np

from ..fileslice import (is_fancy, canonical_slicers, fileslice,
                         predict_shape, _read_segments, _positive_slice,
                         _space_heuristic, _analyze_slice, slice2len,
                         fill_slicer, _get_segments)

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from numpy.testing import assert_array_equal


def _check_slice(sliceobj):
    # Fancy indexing always returns a copy, basic indexing returns a view
    a = np.arange(100).reshape((10, 10))
    b = a[sliceobj]
    if np.isscalar(b):
        return # Can't check
    # Check if this is a view
    a[:] = 99
    b_is_view = np.all(b == 99)
    assert_equal(not is_fancy(sliceobj), b_is_view)


def test_is_fancy():
    slices = (2, [2], [2, 3], Ellipsis, np.array(2), np.array((2, 3)))
    for slice0 in slices:
        _check_slice(slice0)
        _check_slice((slice0,)) # tuple is same
        for slice1 in slices:
            _check_slice((slice0, slice1))
    assert_false(is_fancy((None,)))
    assert_false(is_fancy((None, 1)))
    assert_false(is_fancy((1, None)))
    # Chack that actual False returned (rather than falsey)
    assert_equal(is_fancy(1), False)


def test_canonical_slicers():
    # Check transformation of sliceobj into canonical form
    slicers = (slice(None),
               slice(9),
               slice(0, 9),
               slice(1, 10),
               slice(1, 10, 2),
               2)
    shape = (10, 10)
    for slice0 in slicers:
        assert_equal(canonical_slicers((slice0,), shape), (slice0, slice(None)))
        for slice1 in slicers:
            sliceobj = (slice0, slice1)
            assert_equal(canonical_slicers(sliceobj, shape), sliceobj)
            assert_equal(canonical_slicers(sliceobj, shape + (2, 3, 4)),
                         sliceobj + (slice(None),) * 3)
            assert_equal(canonical_slicers(sliceobj * 3, shape * 3),
                         sliceobj * 3)
            # Check None passes through
            assert_equal(canonical_slicers(sliceobj + (None,), shape),
                         sliceobj + (None,))
            assert_equal(canonical_slicers((None,) + sliceobj, shape),
                         (None,) + sliceobj)
            assert_equal(canonical_slicers((None,) + sliceobj + (None,), shape),
                         (None,) + sliceobj + (None,))
    # Check Ellipsis
    assert_equal(canonical_slicers((Ellipsis,), shape),
                 (slice(None), slice(None)))
    assert_equal(canonical_slicers((Ellipsis, 1), shape),
                 (slice(None), 1))
    assert_equal(canonical_slicers((1, Ellipsis), shape),
                 (1, slice(None)))
    # Ellipsis at end does nothing
    assert_equal(canonical_slicers((1, 1, Ellipsis), shape),
                 (1, 1))
    assert_equal(canonical_slicers((1, Ellipsis, 2), (10, 1, 2, 3, 11)),
                 (1, slice(None), slice(None), slice(None), 2))
    assert_raises(ValueError,
                  canonical_slicers, (Ellipsis, 1, Ellipsis), (2, 3, 4, 5))
    # Check full slices get expanded
    for slice0 in (slice(10), slice(0, 10), slice(0, 10, 1)):
        assert_equal(canonical_slicers((slice0, 1), shape),
                     (slice(None), 1))
    for slice0 in (slice(10), slice(0, 10), slice(0, 10, 1)):
        assert_equal(canonical_slicers((slice0, 1), shape),
                     (slice(None), 1))
        assert_equal(canonical_slicers((1, slice0), shape),
                     (1, slice(None)))
    # Check ints etc get parsed through to tuples
    assert_equal(canonical_slicers(1, shape), (1, slice(None)))
    assert_equal(canonical_slicers(slice(None), shape),
                 (slice(None), slice(None)))
    # Check fancy indexing raises error
    assert_raises(ValueError, canonical_slicers, (np.array(1), 1), shape)
    assert_raises(ValueError, canonical_slicers, (1, np.array(1)), shape)
    # Check out of range integer raises error
    assert_raises(ValueError, canonical_slicers, (10,), shape)
    assert_raises(ValueError, canonical_slicers, (1, 10), shape)
    # Check negative -> positive
    assert_equal(canonical_slicers(-1, shape), (9, slice(None)))
    assert_equal(canonical_slicers((slice(None), -1), shape), (slice(None), 9))


def _slices_for_len(L):
    # Example slices for a dimension of length L
    return (
        0,
        L // 2,
        L - 1,
        -1,
        -2,
        slice(None),
        slice(L-1),
        slice(1, L-1),
        slice(1, L-1, 2),
        slice(L-1, 1, -1),
        slice(L-1, 1, -2))


def _check_slicer(sliceobj, arr, fobj, offset, order):
    new_slice = fileslice(fobj, sliceobj, arr.shape, arr.dtype, offset, order)
    assert_array_equal(arr[sliceobj], new_slice)


def test_slice2len():
    # Test slice length calculation
    assert_equal(slice2len(slice(None), 10), 10)
    assert_equal(slice2len(slice(11), 10), 10)
    assert_equal(slice2len(slice(1, 11), 10), 9)
    assert_equal(slice2len(slice(1, 1), 10), 0)
    assert_equal(slice2len(slice(1, 11, 2), 10), 5)
    assert_equal(slice2len(slice(0, 11, 3), 10), 4)
    assert_equal(slice2len(slice(1, 11, 3), 10), 3)
    assert_equal(slice2len(slice(None, None, -1), 10), 10)
    assert_equal(slice2len(slice(11, None, -1), 10), 10)
    assert_equal(slice2len(slice(None, 1, -1), 10), 8)
    assert_equal(slice2len(slice(None, None, -2), 10), 5)
    assert_equal(slice2len(slice(None, None, -3), 10), 4)
    assert_equal(slice2len(slice(None, 0, -3), 10), 3)
    # Start, end are always taken to be relative if negative
    assert_equal(slice2len(slice(None, -4, -1), 10), 3)
    assert_equal(slice2len(slice(-4, -2, 1), 10), 2)
    # start after stop
    assert_equal(slice2len(slice(3, 2, 1), 10), 0)
    assert_equal(slice2len(slice(2, 3, -1), 10), 0)


def test_fill_slicer():
    # Test slice length calculation
    assert_equal(fill_slicer(slice(None), 10), slice(0, 10, 1))
    assert_equal(fill_slicer(slice(11), 10), slice(0, 10, 1))
    assert_equal(fill_slicer(slice(1, 11), 10), slice(1, 10, 1))
    assert_equal(fill_slicer(slice(1, 1), 10), slice(1, 1, 1))
    assert_equal(fill_slicer(slice(1, 11, 2), 10), slice(1, 10, 2))
    assert_equal(fill_slicer(slice(0, 11, 3), 10), slice(0, 10, 3))
    assert_equal(fill_slicer(slice(1, 11, 3), 10), slice(1, 10, 3))
    assert_equal(fill_slicer(slice(None, None, -1), 10),
                 slice(9, None, -1))
    assert_equal(fill_slicer(slice(11, None, -1), 10),
                 slice(9, None, -1))
    assert_equal(fill_slicer(slice(None, 1, -1), 10),
                 slice(9, 1, -1))
    assert_equal(fill_slicer(slice(None, None, -2), 10),
                 slice(9, None, -2))
    assert_equal(fill_slicer(slice(None, None, -3), 10),
                 slice(9, None, -3))
    assert_equal(fill_slicer(slice(None, 0, -3), 10),
                 slice(9, 0, -3))
    # Start, end are always taken to be relative if negative
    assert_equal(fill_slicer(slice(None, -4, -1), 10),
                 slice(9, 6, -1))
    assert_equal(fill_slicer(slice(-4, -2, 1), 10),
                 slice(6, 8, 1))
    # start after stop
    assert_equal(fill_slicer(slice(3, 2, 1), 10),
                 slice(3, 2, 1))
    assert_equal(fill_slicer(slice(2, 3, -1), 10),
                 slice(2, 3, -1))


def test__positive_slice():
    # Reverse slice direction to be positive
    assert_equal(_positive_slice(slice(0, 5, 1)), slice(0, 5, 1))
    assert_equal(_positive_slice(slice(1, 5, 3)), slice(1, 5, 3))
    assert_equal(_positive_slice(slice(4, None, -2)), slice(0, 5, 2))
    assert_equal(_positive_slice(slice(4, None, -1)), slice(0, 5, 1))
    assert_equal(_positive_slice(slice(4, 1, -1)), slice(2, 5, 1))
    assert_equal(_positive_slice(slice(4, 1, -2)), slice(2, 5, 2))


def test__space_heuristic():
    # Test for default skip / read heuristic
    # int
    assert_equal(_space_heuristic(1, 9, 1, skip_thresh=8), 'full')
    assert_equal(_space_heuristic(1, 9, 1, skip_thresh=7), None)
    assert_equal(_space_heuristic(1, 9, 2, skip_thresh=16), 'full')
    assert_equal(_space_heuristic(1, 9, 2, skip_thresh=15), None)
    # full slice, smallest step size
    assert_equal(_space_heuristic(
        slice(0, 9, 1), 9, 2, skip_thresh=2),
        'full')
    # Dropping skip thresh below step size gives None
    assert_equal(_space_heuristic(
        slice(0, 9, 1), 9, 2, skip_thresh=1),
        None)
    # As does increasing step size
    assert_equal(_space_heuristic(
        slice(0, 9, 2), 9, 2, skip_thresh=3),
        None)
    # Negative step size same as positive
    assert_equal(_space_heuristic(
        slice(9, None, -1), 9, 2, skip_thresh=2),
        'full')
    # Add a gap between start and end. Now contiguous because of step size
    assert_equal(_space_heuristic(
        slice(2, 9, 1), 9, 2, skip_thresh=2),
        'contiguous')
    # To not-contiguous, even with step size 1
    assert_equal(_space_heuristic(
        slice(2, 9, 1), 9, 2, skip_thresh=1),
        None)
    # Back to full when skip covers gap
    assert_equal(_space_heuristic(
        slice(2, 9, 1), 9, 2, skip_thresh=4),
        'full')
    # Until it doesn't cover the gap
    assert_equal(_space_heuristic(
        slice(2, 9, 1), 9, 2, skip_thresh=3),
        'contiguous')


# Some dummy heuristics for _analyze_slice
def _always(slicer, dim_len, stride):
    return 'full'
def _partial(slicer, dim_len, stride):
    return 'contiguous'
def _never(slicer, dim_len, stride):
    return None


def test__analyze_slice():
    # Analyze slice for fullness, contiguity, direction
    #
    # If all_full:
    # - make positive slicer
    # - decide if worth reading continuous block
    # - if so, modify as_read, as_returned accordingly, set contiguous / full
    # - if not, fill as_read for non-contiguous case
    # If not all_full
    # - make positive slicer
    for all_full in (True, False):
        for heuristic in (_always, _never, _partial):
            for is_slowest in (True, False):
                # following tests not affected by all_full or optimization
                # full - always passes through
                assert_equal(
                    _analyze_slice(slice(None), 10, all_full, 4, heuristic),
                    (slice(None), slice(None)))
                # Even if full specified with explicit values
                assert_equal(
                    _analyze_slice(slice(10), 10, all_full, 4, heuristic),
                    (slice(None), slice(None)))
                assert_equal(
                    _analyze_slice(slice(0, 10), 10, all_full, 4, heuristic),
                    (slice(None), slice(None)))
                assert_equal(
                    _analyze_slice(slice(0, 10, 1), 10, all_full, 4, heuristic),
                    (slice(None), slice(None)))
                # Reversed full is still full, but with reversed post_slice
                assert_equal(
                    _analyze_slice(
                        slice(None, None, -1), 10, all_full, 4, heuristic),
                    (slice(None), slice(None, None, -1)))
    # Contiguous is contiguous unless heuristic kicks in, in which case it may
    # be 'full'
    assert_equal(
        _analyze_slice(slice(9), 10, False, False, 4, _always),
        (slice(0, 9, 1), slice(None)))
    assert_equal(
        _analyze_slice(slice(9), 10, True, False, 4, _always),
        (slice(None), slice(0, 9, 1)))
    # Unless this is the slowest dimenion, and all_true is True, in which case
    # we don't update to full
    assert_equal(
        _analyze_slice(slice(9), 10, True, True, 4, _always),
        (slice(0, 9, 1), slice(None)))
    # Nor if the heuristic won't update
    assert_equal(
        _analyze_slice(slice(9), 10, True, False, 4, _never),
        (slice(0, 9, 1), slice(None)))
    assert_equal(
        _analyze_slice(slice(1, 10), 10, True, False, 4, _never),
        (slice(1, 10, 1), slice(None)))
    # Reversed contiguous still contiguous
    assert_equal(
        _analyze_slice(slice(8, None, -1), 10, False, False, 4, _never),
        (slice(0, 9, 1), slice(None, None, -1)))
    assert_equal(
        _analyze_slice(slice(8, None, -1), 10, True, False, 4, _always),
        (slice(None), slice(8, None, -1)))
    assert_equal(
        _analyze_slice(slice(8, None, -1), 10, False, False, 4, _never),
        (slice(0, 9, 1), slice(None, None, -1)))
    assert_equal(
        _analyze_slice(slice(9, 0, -1), 10, False, False, 4, _never),
        (slice(1, 10, 1), slice(None, None, -1)))
    # Non-contiguous
    assert_equal(
        _analyze_slice(slice(0, 10, 2), 10, False, False, 4, _never),
        (slice(0, 10, 2), slice(None)))
    # all_full triggers optimization, but optimization does nothing
    assert_equal(
        _analyze_slice(slice(0, 10, 2), 10, True, False, 4, _never),
        (slice(0, 10, 2), slice(None)))
    # all_full triggers optimization, optimization does something
    assert_equal(
        _analyze_slice(slice(0, 10, 2), 10, True, False, 4, _always),
        (slice(None), slice(0, 10, 2)))
    # all_full disables optimization, optimization does something
    assert_equal(
        _analyze_slice(slice(0, 10, 2), 10, False, False, 4, _always),
        (slice(0, 10, 2), slice(None)))
    # Non contiguous, reversed
    assert_equal(
        _analyze_slice(slice(10, None, -2), 10, False, False, 4, _never),
        (slice(1, 10, 2), slice(None, None, -1)))
    assert_equal(
        _analyze_slice(slice(10, None, -2), 10, True, False, 4, _always),
        (slice(None), slice(9, None, -2)))
    # Short non-contiguous
    assert_equal(
        _analyze_slice(slice(2, 8, 2), 10, False, False, 4, _never),
        (slice(2, 8, 2), slice(None)))
    # with partial read
    assert_equal(
        _analyze_slice(slice(2, 8, 2), 10, True, False, 4, _partial),
        (slice(2, 8, 1), slice(None, None, 2)))
    # If this is the slowest changing dimension, heuristic can upgrade None to
    # contiguous, but not (None, contiguous) to full
    assert_equal( # we've done this one already
        _analyze_slice(slice(0, 10, 2), 10, True, False, 4, _always),
        (slice(None), slice(0, 10, 2)))
    assert_equal( # if slowest, just upgrade to contiguous
        _analyze_slice(slice(0, 10, 2), 10, True, True, 4, _always),
        (slice(0, 10, 1), slice(None, None, 2)))
    assert_equal( # contiguous does not upgrade to full
        _analyze_slice(slice(9), 10, True, True, 4, _always),
        (slice(0, 9, 1), slice(None)))
    # integer
    assert_equal(
        _analyze_slice(0, 10, True, False, 4, _never),
        (0, 'dropped'))
    assert_equal( # can be negative
        _analyze_slice(-1, 10, True, False, 4, _never),
        (9, 'dropped'))
    assert_equal( # or float
        _analyze_slice(0.9, 10, True, False, 4, _never),
        (0, 'dropped'))
    assert_raises(ValueError, # should never get 'contiguous'
        _analyze_slice, 0, 10, True, False, 4, _partial)
    assert_equal( # full can be forced with heuristic
        _analyze_slice(0, 10, True, False, 4, _always),
        (slice(None), 0))
    assert_equal( # but disabled for slowest changing dimension
        _analyze_slice(0, 10, True, True, 4, _always),
        (0, 'dropped'))


def test__get_segments():
    # Check get_segments routine.  This thing is huge and a bear to test
    segments, out_shape, new_slicing = _get_segments(
        (1,), (10,), 4, 7, 'F', _never)
    assert_equal(segments, [[11, 4]])
    assert_equal(new_slicing, ())
    assert_equal(out_shape, ())
    assert_equal(
        _get_segments((slice(None),), (10,), 4, 7, 'F', _never),
        ([[7, 40]],
         (10,),
         (),
        ))
    assert_equal(
        _get_segments((slice(9),), (10,), 4, 7, 'F', _never),
        ([[7, 36]],
         (9,),
         (),
        ))
    assert_equal(
        _get_segments((slice(1, 9),), (10,), 4, 7, 'F', _never),
        ([[11, 32]],
         (8,),
         (),
        ))
    # Two dimensions, single slice
    assert_equal(
        _get_segments((0,), (10, 6), 4, 7, 'F', _never),
        ([[7, 4], [47, 4], [87, 4], [127, 4], [167, 4], [207, 4]],
         (6,),
         (),
        ))
    assert_equal(
        _get_segments((0,), (10, 6), 4, 7, 'C', _never),
        ([[7, 6 * 4]],
         (6,),
         (),
        ))
    # Two dimensions, contiguous not full
    assert_equal(
        _get_segments((1, slice(1, 5)), (10, 6), 4, 7, 'F', _never),
        ([[51, 4], [91, 4], [131, 4], [171, 4]],
         (4,),
         (),
        ))
    assert_equal(
        _get_segments((1, slice(1, 5)), (10, 6), 4, 7, 'C', _never),
        ([[7 + 7*4, 16]],
         (4,),
         (),
        ))
    # With full slice first
    assert_equal(
        _get_segments((slice(None), slice(1, 5)), (10, 6), 4, 7, 'F', _never),
        ([[47, 160]],
         (10, 4),
         (),
        ))
    # Check effect of heuristic on _get_segments
    # Even integer slices can generate full when heuristic says so
    assert_equal(
        _get_segments((1, slice(None)), (10, 6), 4, 7, 'F', _always),
        ([[7, 10 * 6 * 4]],
         (10, 6),
         (1, slice(None)),
        ))
    # Except when last
    assert_equal(
        _get_segments((slice(None), 1), (10, 6), 4, 7, 'F', _always),
        ([[7 + 10 * 4, 10 * 4]],
         (10,),
         (),
        ))


def test_predict_shape():
    shapes = (15, 16, 17, 18)
    for n_dim in range(len(shapes)):
        shape = shapes[:n_dim + 1]
        arr = np.arange(np.prod(shape)).reshape(shape)
        slicers_list = []
        for i in range(n_dim):
            slicers_list.append(_slices_for_len(shape[i]))
            for sliceobj in product(*slicers_list):
                assert_equal(predict_shape(sliceobj, shape),
                             arr[sliceobj].shape)
    # Try some Nones and ellipses
    assert_equal(predict_shape((Ellipsis,), (2, 3)), (2, 3))
    assert_equal(predict_shape((Ellipsis, 1), (2, 3)), (2,))
    assert_equal(predict_shape((1, Ellipsis), (2, 3)), (3,))
    assert_equal(predict_shape((1, slice(None), Ellipsis), (2, 3)), (3,))
    assert_equal(predict_shape((None,), (2, 3)), (1, 2, 3))
    assert_equal(predict_shape((None, 1), (2, 3)), (1, 3))
    assert_equal(predict_shape((1, None, slice(None)), (2, 3)), (1, 3))
    assert_equal(predict_shape((1, slice(None), None), (2, 3)), (3, 1))


def _check_bytes(bytes, arr):
    barr = np.ndarray(arr.shape, arr.dtype, buffer=bytes)
    assert_array_equal(barr, arr)


def test__read_segments():
    # Test segment reading
    fobj = BytesIO()
    arr = np.arange(100, dtype=np.int16)
    fobj.write(arr.tostring())
    _check_bytes(_read_segments(fobj, [(0, 200)], 200), arr)
    _check_bytes(_read_segments(fobj, [(0, 100), (100, 100)], 200), arr)
    _check_bytes(_read_segments(fobj, [(0, 50), (100, 50)], 100),
                 np.r_[arr[:25], arr[50:75]])
    _check_bytes(_read_segments(fobj, [(10, 40), (100, 50)], 90),
                 np.r_[arr[5:25], arr[50:75]])
    _check_bytes(_read_segments(fobj, [], 0), arr[0:0])
    # Error conditions
    assert_raises(ValueError, _read_segments, fobj, [], 1)
    assert_raises(ValueError, _read_segments, fobj, [(0, 200)], 199)
    assert_raises(Exception, _read_segments, fobj, [(0, 100), (100, 200)], 199)


def test_fileslice():
    shapes = (15, 16, 17)
    for n_dim in range(1, len(shapes) + 1):
        shape = shapes[:n_dim]
        arr = np.arange(np.prod(shape)).reshape(shape)
        for order in 'FC':
            for offset in (0, 20):
                fobj = BytesIO()
                fobj.write(b'\0' * offset)
                fobj.write(arr.tostring(order=order))
                slicers_list = []
                for i in range(n_dim):
                    slicers_list.append(_slices_for_len(shape[i]))
                    for sliceobj in product(*slicers_list):
                        _check_slicer(sliceobj, arr, fobj, offset, order)
    # Try some Nones and Ellipses
    for order in 'FC':
        arr = np.arange(24).reshape((2, 3, 4))
        fobj = BytesIO()
        fobj.write(b'\0' * offset)
        fobj.write(arr.tostring(order=order))
        _check_slicer((None,), arr, fobj, offset, order)
        _check_slicer((None, 1), arr, fobj, offset, order)
        _check_slicer((1, None, slice(None)), arr, fobj, offset, order)
        _check_slicer((slice(None), 1, 2, None), arr, fobj, offset, order)
        _check_slicer((Ellipsis, 2, None), arr, fobj, offset, order)
        _check_slicer((1, Ellipsis, None), arr, fobj, offset, order)


def test_fileslice_errors():
    # Test fileslice causes error on fancy indexing
    arr = np.arange(24).reshape((2, 3, 4))
    fobj = BytesIO(arr.tostring())
    _check_slicer((1,), arr, fobj, 0, 'C')
    # Fancy indexing raises error
    assert_raises(ValueError,
                  fileslice, fobj, (np.array(1),), (2, 3, 4), arr.dtype)
