""" Test floating point deconstructions and floor methods
"""
import numpy as np

from ..floating import (parts_from_val, val_from_parts, step_towards_zero,
                        floor_exact, maskfor, maskat)

from nose.tools import assert_equal

def test_parts_vals():
    for t in (np.float16, np.float32, np.float64):
        fi = np.finfo(t)
        for i in range(fi.nmant+1):
            v = 2**i
            g, s, e = parts_from_val(t(v))
            assert_equal((g, s, e), (False, 0, i))
            assert_equal(val_from_parts(t, g, s, e), v)
        top_1 = maskat(fi.nmant-1)
        for v, eg, es, ee in ((3, False, top_1, 1),
                              (-3, True, top_1, 1),
                              (fi.max, False, maskfor(fi.nmant), fi.maxexp-1),
                              (fi.min, True, maskfor(fi.nmant), fi.maxexp-1),
                              (fi.tiny, False, 0, fi.minexp),
                             ):
            g, s, e = parts_from_val(t(v))
            assert_equal((g, s, e), (eg, es, ee))
            assert_equal(val_from_parts(t, g, s, e), v)


def test_floor_exact():
    for t in (np.float16, np.float32, np.float64, np.float):
        fi = np.finfo(t)
        nmant = fi.nmant
        for i in range(nmant+1):
            iv = 2**i
            # up to 2**nmant should be exactly representable
            assert_equal(step_towards_zero(t(iv)), iv-1)
            assert_equal(floor_exact(iv, t), iv)
            assert_equal(step_towards_zero(t(-iv)), -iv+1)
            assert_equal(floor_exact(-iv, t), -iv)
        # Not so 2**(nmant+1)
        iv = 2**(nmant+1)
        assert_equal(step_towards_zero(t(iv+2)), iv)
        assert_equal(floor_exact(iv+1, t), iv)
        # negatives
        assert_equal(step_towards_zero(t(-iv-2)), -iv)
        assert_equal(floor_exact(-iv-1, t), -iv)
        # When we get to exponents nmant+2, gap between integers is 4
        iv = 2**(nmant+2)
        assert_equal(t(iv+4), iv+4)
        assert_equal(step_towards_zero(t(iv+4)), iv)
        assert_equal(floor_exact(iv+3, t), iv)
        # negatives
        assert_equal(step_towards_zero(t(-iv-4)), -iv)
        assert_equal(floor_exact(-iv-3, t), -iv)

