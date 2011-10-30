""" Test casting utilities
"""

import numpy as np

from ..casting import nice_round, _clippers

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_equal, assert_raises)


def test_casting():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            ii = np.iinfo(it)
            arr = [ii.min-1, ii.max+1, -np.inf, np.inf, np.nan, 0.2, 10.6]
            farr = np.array(arr, dtype=ft)
            iarr = nice_round(farr, ii)
            mn, mx = _clippers(ft, it)
            assert_array_equal(iarr, [mn, mx, mn, mx, 0, 0, 11])
