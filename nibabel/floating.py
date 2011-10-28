""" Working with IEEE floating point values

Getting nearest exact integers in particular floating point types
"""
from math import log

import numpy as np

flts2uints = {
    np.float32: np.uint32,
    np.float64: np.uint64,
    }
try:
    flt16 = np.float16
except AttributeError: # float16 not present in np < 1.6
    pass
else:
    flts2uints[flt16] = np.uint16


class FloatingError(Exception):
    pass


def floor_exact(val, flt_type):
    """ Get nearest exact integer to `val`, towards 0, in float type `flt_type`

    Parameters
    ----------
    val : int
        We have to pass val as an int rather than the floating point type
        because large integers cast as floating point may be rounded by the
        casting process.
    flt_type : numpy type
        numpy float type.  Only IEEE types supported (np.float16, np.float32,
        np.float64)

    Returns
    -------
    floor_val : object
        value of same floating point type as `val`, that is the next excat
        integer in this type, towards zero, or == `val` if val is exactly
        representable.

    Examples
    --------
    Obviously 2 is within the range of representable integers for float32

    >>> floor_exact(2, np.float32)
    2.0

    As is 2**24-1 (the number of significand digits is 23 + 1 implicit)

    >>> floor_exact(2**24-1, np.float32) == 2**24-1
    True

    But 2**24+1 gives a number that float32 can't represent exactly

    >>> floor_exact(2**24+1, np.float32) == 2**24
    True
    """
    val = int(val)
    flt_type = np.dtype(flt_type).type
    sign = val > 0 and 1 or -1
    aval = abs(val)
    faval = flt_type(aval)
    if int(faval) <= aval:
        # Float casting has made the value go down or stay the same
        return sign * faval
    # Float casting made the value go up
    fi = np.finfo(flt_type)
    biggest_gap = 2**(int(log(aval, 2)) - fi.nmant)
    assert biggest_gap > 1
    faval -= flt_type(biggest_gap)
    if biggest_gap > 2:
        assert int(faval) == faval + flt_type(1)
    return sign * faval
    # return step_towards_zero(tval)


def step_towards_zero(val):
    """ Value `fv` giving the smallest ``d=val-fv`` for positive ``d``

    Floating point values are exact values on a continuous scale of the real
    numbers.  This routine analyzes the parts of the floating point
    representation to return the next exact value nearer to zero, that the
    floating point type of `val` can represent.

    Parameters
    ----------
    val : object
        value of numpy float type.  Only IEEE types supported (np.float16,
        np.float32, np.float64)

    Returns
    -------
    fv : object
        value of same floating point type as `val`, that is the next excat
        integer in this type, towards zero.

    Examples
    --------
    Obviously 2 is within the range of representable integers for float32

    >>> step_towards_zero(np.float32(3))
    2.0

    As is 2**24-1 (the number of significand digits is 23 + 1 implicit)

    >>> step_towards_zero(np.float32(2**24)) == 2**24-1
    True

    But 2**24+1 gives a number that float32 can't represent exactly

    >>> step_towards_zero(np.float32(2**24+2)) == 2**24
    True
    """
    if val == 0:
        return val
    # If the resolution allows it already, our job is not hard
    flt_type = np.asarray(val).dtype.type
    step = val >= 0 and 1 or -1
    valm1 = flt_type(val - step)
    if valm1 != val:
        return valm1
    if val == 0:
        return val
    # val-1 not exactly represented
    info = np.finfo(flt_type)
    s_bits = info.nmant
    n, s, e = parts_from_val(val)
    if s == 0:
        s = maskfor(s_bits)
        e -=1
    else:
        s-=1
    return val_from_parts(flt_type, n, s, e)


def parts_from_val(val):
    """ Return sign, significand, exponent from floating point value `val`

    Parameters
    ----------
    val : object
        value of numpy float type.  Only IEEE types supported (np.float16,
        np.float32, np.float64)

    Returns
    -------
    g : bool
        Sign.  True for negative, False for positive
    s : significand
        Significand as unsigned integer, and without implied leading 1.  For
        example, float32 has 23 real binary digits for the significand, and one
        implied, so the significand for 3 == 2**22 (1 at the first of 23 bits).
    e : exponent
        as signed integer

    Examples
    --------
    >>> parts_from_val(np.float32(-3))
    (True, 4194304L, 1L)
    >>> val_from_parts(np.float32, True, 4194304, 1)
    -3.0
    """
    val = np.asarray(val)
    flt_type = val.dtype.type
    fi = np.finfo(flt_type)
    try:
        utype = flts2uints[flt_type]
    except KeyError:
        if (fi.nexp, fi.nmant) == (15, 63): # Intel 80 bit
            return _parts_80(val)
        raise FloatingError("We don't support type %s" % flt_type)
    uint = int(val.view(utype))
    s = uint & maskfor(fi.nmant)
    rest = uint >> fi.nmant
    u = rest & maskfor(fi.nexp)
    e = u + fi.minexp - 1
    g = bool(rest >> fi.nexp)
    return g, s, e


def _parts_80(val):
    """ Return sign, significand, exponent from 80-bit fp value `val`

    Parameters
    ----------
    val : object
        value of float96 or float128 - Intel 80-bit extended precision

    Returns
    -------
    g : bool
        Sign.  True for negative, False for positive
    s : significand
        Significand as unsigned integer, and without explicit leading 1.
    e : exponent
        as signed integer
    """
    val = np.asarray(val)
    flt_type = val.dtype.type
    flt_size = val.dtype.itemsize
    n_pad = flt_size - 10
    fi = np.finfo(flt_type)
    assert fi.nexp == 15
    assert fi.nmant == 63
    uints = val.view([('s','u8'),('gu', 'u2'),('padding','u1', n_pad)])
    s = int(uints['s']) & maskfor(fi.nmant)
    gu = int(uints['gu'])
    u = gu & maskfor(fi.nexp)
    e = u + fi.minexp - 1
    g = bool(gu >> fi.nexp)
    return g, s, e


def maskfor(nbits):
    return 2**nbits-1


def maskat(bitn):
    return 2**bitn


def val_from_parts(flt_type, g, s, e):
    """ Return value of `flt_type` from sign `g`, significand `g`, exponent `e`

    Parameters
    ----------
    flt_type : numpy type
        numpy float type.  Only IEEE types supported (np.float16, np.float32,
        np.float64)
    g : bool
        Sign.  True for negative, False for positive
    s : significand
        Significand as unsigned integer, and without implied leading 1.  For
        example, float32 has 23 real binary digits for the significand, and one
        implied, so the significand for 3 == 2**22 (1 at the first of 23 bits).
    e : exponent
        as signed integer

    Returns
    -------
    v : object
        floating point value of type `flt_type`

    Examples
    --------
    >>> parts_from_val(np.float32(-3))
    (True, 4194304L, 1L)
    >>> val_from_parts(np.float32, True, 4194304, 1)
    -3.0
    """
    fi = np.finfo(flt_type)
    try:
        utype = flts2uints[flt_type]
    except KeyError:
        if (fi.nexp, fi.nmant) == (15, 63): # Intel 80 bit
            return _val_80(flt_type, g, s, e)
        raise FloatingError("We don't support type %s" % flt_type)
    uint = e - fi.minexp + 1
    uint = (uint << fi.nmant) + s
    if g:
        uint += maskat(fi.nmant + fi.nexp)
    return utype(uint).view(flt_type)


def _val_80(flt_type, g, s, e):
    """ Return value of `flt_type` from sign `g`, significand `g`, exponent `e`

    For Intel 80 bit type

    Parameters
    ----------
    flt_type : numpy type
        numpy float type - either float96 or float128
    g : bool
        Sign.  True for negative, False for positive
    s : significand
        Significand as unsigned integer, and without leading 1.
    e : exponent
        as signed integer

    Returns
    -------
    v : object
        floating point value of type `flt_type`
    """
    fi = np.finfo(flt_type)
    assert fi.nexp == 15
    assert fi.nmant == 63
    flt_size = np.dtype(flt_type).itemsize
    n_pad = flt_size - 10
    uints = np.zeros((1,), [('s','u8'),('gu', 'u2'),('padding','u1', n_pad)])
    gu = e - fi.minexp + 1
    if g:
        gu += maskat(fi.nexp)
    # Add back explicit leading 1
    s += maskat(fi.nmant)
    uints['s'] = s
    uints['gu'] = gu
    return uints.view(flt_type)
