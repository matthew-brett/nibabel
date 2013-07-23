# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Tests for arrayproxy module
"""
from __future__ import division, print_function, absolute_import

from ..externals.six import BytesIO
from ..tmpdirs import InTemporaryDirectory

import numpy as np

from ..arrayproxy import ArrayProxy
from ..nifti1 import Nifti1Header

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)


class FunkyHeader(object):
    """ Minimal API for ArrayProxy """
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def get_data_shape(self):
        return self.shape[:]

    def get_data_dtype(self):
        return self.dtype

    def get_slope_inter(self):
        return 1, 0

    def get_data_offset(self):
        return 0


def test_init():
    shape = [2,3,4]
    dtype = np.dtype(np.int16)
    arr = np.arange(24, dtype=dtype).reshape((2,3,4))
    bio = BytesIO()
    bio.write(arr.tostring(order='F'))
    hdr = FunkyHeader(shape, dtype)
    ap = ArrayProxy(bio, hdr)
    assert_true(ap.file_like is bio)
    assert_equal(ap.shape, shape)
    # Check there has been a copy of the header information
    hdr.shape = [4, 3, 2]
    hdr.dtype = np.dtype(np.int32)
    assert_equal(ap.shape, shape) # shape not changed
    assert_equal(ap.dtype, dtype) # dtype not changed
    # Get the data
    assert_array_equal(np.asarray(ap), np.arange(24).reshape((2,3,4)))


def write_raw_data(arr, hdr, fileobj):
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    fileobj.write(b'\x00' * hdr.get_data_offset())
    fileobj.write(arr.tostring(order='F'))


def test_nifti1_init():
    # Check a nifti1 header works as expected
    bio = BytesIO()
    shape = (2,3,4)
    dtype = np.dtype(np.int16)
    hdr = Nifti1Header()
    arr = np.arange(24, dtype=dtype).reshape(shape)
    write_raw_data(arr, hdr, bio)
    hdr.set_slope_inter(2, 10)
    ap = ArrayProxy(bio, hdr)
    assert_true(ap.file_like == bio)
    assert_equal(ap.shape, shape)
    # Modifying the header had no effect on the proxy
    hdr.set_data_shape([4, 3, 2])
    hdr.set_data_dtype(np.int32)
    assert_equal(ap.shape, shape) # shape not changed
    assert_equal(ap.dtype, dtype) # dtype not changed
    # Get the data
    assert_array_equal(np.asarray(ap), arr * 2.0 + 10)
    with InTemporaryDirectory():
        f = open('test.nii', 'wb')
        write_raw_data(arr, hdr, f)
        f.close()
        ap = ArrayProxy('test.nii', hdr)
        assert_true(ap.file_like == 'test.nii')
        assert_equal(ap.shape, shape)
        assert_array_equal(np.asarray(ap), arr * 2.0 + 10)
