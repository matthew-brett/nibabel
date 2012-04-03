""" Simulate effect of scaling floats on write / read roundtrip errors

I have some vector of numbers ``V`` in some floating point format ``ffmt``.

I am going to take V, calculate some scaling factor and intercept ``scale`` and
``inter``, and convert to vector ``I`` in some integer format ``ifmt``.

My task should I choose to accept it is to minimize error in the round trip from
V to I to Vdash.

What is error?  For the moment let's take this as relative error, that is::

    (V-Vdash)/V

The parameters I'm exploring are:

    The working precision of the applied scaling
"""

from StringIO import StringIO

import numpy as np

from nibabel import Nifti1Image
from nibabel.spatialimages import HeaderDataError
import nibabel.volumeutils as nvu
try:
    from nibabel.arraywriters import ScalingError
except ImportError:
    class ScalingError(object): pass
from nibabel.casting import type_info, floor_log2

def round_trip(arr, out_dtype):
    img = Nifti1Image(arr, np.eye(4))
    img.file_map['image'].fileobj = StringIO()
    img.set_data_dtype(out_dtype)
    img.to_file_map()
    back = Nifti1Image.from_file_map(img.file_map)
    hdr = back.get_header()
    return (back.get_data(),) + hdr.get_slope_inter()


def raw_round_trip(arr, out_dtype, slope, inter, mn, mx):
    out_dtype = np.dtype(out_dtype)
    str_io = StringIO()
    nvu.array_to_file(arr, str_io, out_dtype, intercept=inter, divslope=slope,
                      mn=mn, mx = mx)
    back = nvu.array_from_file(arr.shape, out_dtype, str_io)
    return nvu.apply_read_scaling(back, slope, inter)


def check_params(in_arr, in_type, out_type):
    arr = in_arr.astype(in_type)
    # clip infs that can arise from downcasting
    if arr.dtype.kind == 'f':
        info = np.finfo(in_type)
        arr = np.clip(arr, info.min, info.max)
    try:
        arr_dash, slope, inter = round_trip(arr, out_type)
    except (ScalingError, HeaderDataError):
        return arr, None, None
    return arr, arr_dash, slope, inter


def biggest_gap(val):
    aval = np.abs(val)
    info = type_info(val.dtype)
    return 2**(floor_log2(aval) - info['nmant'])


if __name__ == '__main__':
    rng = np.random.RandomState(20111121)
    N = 10000
    sds = [10.0**n for n in range(-20, 51, 5)]
    out_types = np.sctypes['int'] + np.sctypes['uint']
    # Remove intp types, which cannot be set into nifti header datatype
    out_types.remove(np.intp)
    out_types.remove(np.uintp)
    f_types = [np.float32, np.float64]
    in_types = f_types + out_types
    outs = np.zeros((len(sds), len(in_types), len(out_types)), dtype=np.float64)
    tests = np.ones_like(outs)
    for i, sd in enumerate(sds):
        V_in = rng.normal(0, sd, size=(N,1))
        for j, in_type in enumerate(in_types):
            for k, out_type in enumerate(out_types):
                print sd, out_type, in_type
                arr, arr_dash, slope, inter = check_params(V_in, in_type, out_type)
                if arr_dash is None:
                    outs[i, j, k] = np.nan
                    tests[i, j, k] = np.nan
                    continue
                nzs = arr != 0 # avoid divide by zero error
                arr = arr[nzs]
                arr_dash_L = arr_dash.astype(np.longdouble)[nzs]
                top = arr - arr_dash_L
                if not np.any(nzs) or not np.any(top != 0):
                    continue
                rel_err = np.abs(top / arr)
                abs_err = np.abs(top)
                outs[i, j, k] = np.max(rel_err)
                abs_thresh = np.abs(slope / 2)
                rel_thresh = 0.1
                test_vals = (abs_err <= abs_thresh) | (rel_err <= rel_thresh)
                tests[i, j, k] = np.all(test_vals)
                if tests[i, j, k] == 0:
                    1/0
