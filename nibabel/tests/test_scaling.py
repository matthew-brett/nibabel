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

import numpy as np

from ..volumeutils import (calculate_scale, scale_min_max, finite_range)

from numpy.testing import (assert_array_almost_equal, assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises

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


