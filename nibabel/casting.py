""" Utilties for casting floats to integers
"""

import numpy as np

from .floating import floor_exact

def _clippers(flt_type, int_type):
    """ Min and max in float type giving min max in integer type

    This is not as easy as it sounds, as you see from the code

    Parameters
    ----------
    flt_type : object
        numpy floating point type
    int_type : object
        numpy integer type

    Returns
    -------
    mn : object
        Number of type `flt_type` that is the minumum value in the range of
        `int_type`, such that ``mn.astype(int_type)`` >= min of `int_type`
    mx : object
        Number of type `flt_type` that is the maximum value in the range of
        `int_type`, such that ``mx.astype(int_type)`` <= max of `int_type`
    """
    ii = np.iinfo(int_type)
    return floor_exact(ii.min, flt_type), floor_exact(ii.max, flt_type)


class RoundingError(Exception):
    pass


def nice_round(arr, int_type, nan2zero=True):
    """ Round floating point array `arr` to type `int_type`

    Parameters
    ----------
    arr : array-like
        Array of floating point type
    int_type : object
        Numpy integer type
    nan2zero : {True, False}
        Whether to convert NaN value to zero.  Default is True.  If False, and
        NaNs are present, raise RoundingError

    Returns
    -------
    iarr : ndarray
        of type `int_type`

    Examples
    --------
    >>> nice_round([np.nan, np.inf, -np.inf, 1.1, 6.6], np.int16)
    array([     0,  32767, -32768,      1,      7], dtype=int16)

    Notes
    -----
    We always set +-inf to be the min / max of the integer type.  If you want
    something different you'll need to filter them before passing to this
    routine.
    """
    arr = np.asarray(arr)
    flt_type = arr.dtype.type
    mn, mx = _clippers(flt_type, int_type)
    nans = np.isnan(arr)
    have_nans = np.any(nans)
    if not nan2zero and have_nans:
        raise RoundingError('NaNs in array, nan2zero not True')
    iarr = np.clip(np.rint(arr), mn, mx).astype(int_type)
    if have_nans:
        iarr[nans] = 0
    return iarr
