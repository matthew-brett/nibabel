""" Test floating point deconstructions and floor methods
"""
import numpy as np

from ..floating import floor_exact, flt2nmant

from nose.tools import assert_equal

IEEE_floats = [np.float32, np.float64]
try:
    IEEE_floats.append(np.float16)
except AttributeError: # float16 not present in np < 1.6
    pass

LD_INFO = np.finfo(np.longdouble)


def test_flt2nmant():
    for t in IEEE_floats:
        assert_equal(flt2nmant(t), np.finfo(t).nmant)
    if (LD_INFO.nmant, LD_INFO.nexp) == (63, 15):
        assert_equal(flt2nmant(np.longdouble), 63)


def test_floor_exact():
    to_test = IEEE_floats + [float]
    if (LD_INFO.nmant, LD_INFO.nexp) == (63, 15):
        to_test.append(np.longdouble)
    # When numbers go above int64 - I believe, numpy comparisons break down,
    # so we have to cast to int before comparison
    int_flex = lambda x, t : int(floor_exact(x, t))
    for t in to_test:
        nmant = flt2nmant(t)
        for i in range(nmant+1):
            iv = 2**i
            # up to 2**nmant should be exactly representable
            assert_equal(int_flex(iv, t), iv)
            assert_equal(int_flex(-iv, t), -iv)
        # Not so 2**(nmant+1)
        iv = 2**(nmant+1)
        assert_equal(int_flex(iv+1, t), iv)
        # negatives
        assert_equal(int_flex(-iv-1, t), -iv)
        # The gap in representable numbers is 2 above 2**(nmant+1), 4 above
        # 2**(nmant+2), and so on
        for i in range(5):
            iv = 2**(nmant+1+i)
            gap = 2**(i+1)
            assert_equal(int(t(iv+gap)), iv+gap)
            for j in range(1,gap):
                assert_equal(int_flex(iv+j, t), iv)
                assert_equal(int_flex(iv+gap+j, t), iv+gap)
            # negatives
            for j in range(1,gap):
                assert_equal(int_flex(-iv-j, t), -iv)
                assert_equal(int_flex(-iv-gap-j, t), -iv-gap)
