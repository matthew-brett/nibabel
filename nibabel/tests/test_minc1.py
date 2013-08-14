# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

import os
import gzip
import bz2

import numpy as np

from .. import load, Nifti1Image
from ..externals.netcdf import netcdf_file
from .. import minc1
from ..minc1 import Minc1File, Minc1Image

from nose.tools import (assert_true, assert_equal, assert_false, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..tmpdirs import InTemporaryDirectory
from ..testing import data_path

from . import test_spatialimages as tsi

def test_old_namespace():
    # Check old names are defined in minc1 module and top level
    from ..minc1 import MincFile, MincImage
    assert_true(MincFile is Minc1File)
    assert_true(MincImage is Minc1Image)
    from .. import MincImage, Minc1Image as M1
    assert_true(MincImage is M1)


class _TestMincFile(object):
    module = minc1
    file_class = Minc1File
    opener = netcdf_file
    example_params = dict(
        fname = os.path.join(data_path, 'tiny.mnc'),
        shape = (10,20,20),
        type = np.uint8,
        affine = np.array([[0, 0, 2.0, -20],
                           [0, 2.0, 0, -20],
                           [2.0, 0, 0, -10],
                           [0, 0, 0, 1]]),
        zooms = (2., 2., 2.),
        # These values from SPM2
        min = 0.20784314,
        max = 0.74901961,
        mean = 0.60602819)

    def test_mincfile(self):
        mnc_obj = self.opener(self.example_params['fname'], 'r')
        mnc = self.file_class(mnc_obj)
        assert_equal(mnc.get_data_dtype().type, self.example_params['type'])
        assert_equal(mnc.get_data_shape(), self.example_params['shape'])
        assert_equal(mnc.get_zooms(), self.example_params['zooms'])
        assert_array_equal(mnc.get_affine(), self.example_params['affine'])
        data = mnc.get_scaled_data()
        assert_equal(data.shape, self.example_params['shape'])

    def test_load(self):
        # Check highest level load of minc works
        img = load(self.example_params['fname'])
        data = img.get_data()
        assert_equal(data.shape, self.example_params['shape'])
        # min, max, mean values from read in SPM2
        assert_array_almost_equal(data.min(), self.example_params['min'])
        assert_array_almost_equal(data.max(), self.example_params['max'])
        assert_array_almost_equal(data.mean(), self.example_params['mean'])
        # check if mnc can be converted to nifti
        ni_img = Nifti1Image.from_image(img)
        assert_array_equal(ni_img.get_affine(), self.example_params['affine'])
        assert_array_equal(ni_img.get_data(), data)


class TestMinc1File(_TestMincFile):

    def test_compressed(self):
        # we can read minc compressed
        # Not so for MINC2; hence this small sub-class
        content = open(self.example_params['fname'], 'rb').read()
        openers_exts = ((gzip.open, '.gz'), (bz2.BZ2File, '.bz2'))
        with InTemporaryDirectory():
            for opener, ext in openers_exts:
                fname = 'test.mnc' + ext
                fobj = opener(fname, 'wb')
                fobj.write(content)
                fobj.close()
                img = self.module.load(fname)
                data = img.get_data()
                assert_array_almost_equal(data.mean(), self.example_params['mean'])
                del img


class TestMinc1Image(tsi.TestSpatialImage):
    image_class = Minc1Image
