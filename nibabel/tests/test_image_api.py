""" Validate image API """
from __future__ import division, print_function, absolute_import

import numpy as np

try:
    import scipy
except ImportError:
    have_scipy = False
else:
    have_scipy = True

from nibabel import (AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage,
                     Nifti1Pair, Nifti1Image, MGHImage, Minc1Image,
                     Minc2Image)
from nibabel.spatialimages import SpatialImage
from nibabel.ecat import EcatImage

from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from numpy.testing import (assert_almost_equal, assert_array_equal)

from ..tmpdirs import InTemporaryDirectory

from .test_api_validators import ValidateAPI
from .test_helpers import bytesio_round_trip


class TestAnalyzeAPI(ValidateAPI):
    """ General image validation API instantiated for Analyze images
    """
    image_class = AnalyzeImage
    shapes = ((2,), (2, 3), (2, 3, 4), (2, 3, 4, 5))
    has_scaling = False
    can_save = True
    standard_extension = '.img'

    def img_from_arr_aff(self, arr, aff, header=None):
        return self.image_class(arr, aff, header)

    def obj_params(self):
        """ Return (``img_creator``, ``img_params``) pairs

        ``img_creator`` is a function taking no arguments and returning a fresh
        image.  We need to pass this function rather than an image instance so
        we can recreate the images fresh for each of multiple tests run from the
        ``validate_xxx`` autogenerated test methods.  This allows the tests to
        modify the image without having an effect on the later tests in the same
        function.
        """
        aff = np.diag([1, 2, 3, 1])
        def make_imaker(arr, aff, header=None):
            return lambda : self.image_class(arr, aff, header)
        for shape in self.shapes:
            for dtype in (np.uint8, np.int16, np.float32):
                arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                hdr = self.image_class.header_class()
                hdr.set_data_dtype(dtype)
                func = make_imaker(arr.copy(), aff, hdr)
                params = dict(
                    dtype = dtype,
                    affine = aff,
                    data = arr,
                    is_proxy = False)
                yield func, params
        if not self.can_save:
            return
        # Add a proxy image
        # We assume that loading from a fileobj creates a proxy image
        params['is_proxy'] = True
        def prox_imaker():
            img = self.image_class(arr, aff, hdr)
            rt_img = bytesio_round_trip(img)
            return self.image_class(rt_img.dataobj, aff, rt_img.header)
        yield prox_imaker, params

    def validate_affine(self, imaker, params):
        # Check affine API
        img = imaker()
        assert_array_equal(img.affine, params['affine'])
        assert_equal(img.affine.dtype, np.float64)
        img.affine[0, 0] = 1.5
        assert_equal(img.affine[0, 0], 1.5)
        # Read only
        assert_raises(AttributeError, setattr, img, 'affine', np.eye(4))

    def validate_affine_deprecated(self, imaker, params):
        # Check deprecated affine API
        img = imaker()
        assert_array_equal(img.get_affine(), params['affine'])
        assert_equal(img.get_affine().dtype, np.float64)
        aff = img.get_affine()
        aff[0, 0] = 1.5
        assert_true(aff is img.get_affine())

    def validate_header(self, imaker, params):
        # Check header API
        img = imaker()
        hdr = img.header # we can fetch it
        # Change shape in header, check this changes img.header
        shape = hdr.get_data_shape()
        new_shape = (shape[0] + 1,) + shape[1:]
        hdr.set_data_shape(new_shape)
        assert_true(img.header is hdr)
        assert_equal(img.header.get_data_shape(), new_shape)
        # Read only
        assert_raises(AttributeError, setattr, img, 'header', hdr)

    def validate_header_deprecated(self, imaker, params):
        # Check deprecated header API
        img = imaker()
        hdr = img.get_header()
        assert_true(hdr is img.get_header())

    def validate_shape(self, imaker, params):
        # Validate shape
        img = imaker()
        # Same as array shape
        exp_shape = params['data'].shape
        assert_equal(img.shape, exp_shape)
        assert_equal(img.shape, img.get_data().shape)
        # Read only
        assert_raises(AttributeError, setattr, img, 'shape', np.eye(4))

    def validate_dtype(self, imaker, params):
        # data / storage dtype
        img = imaker()
        # Need to rename this one
        assert_equal(img.get_data_dtype().type, params['dtype'])
        # dtype survives round trip
        if self.has_scaling and self.can_save:
            rt_img = bytesio_round_trip(img)
            assert_equal(rt_img.get_data_dtype().type, params['dtype'])
        # Setting to a different dtype
        img.set_data_dtype(np.float32) # assumed supported for all formats
        assert_equal(img.get_data_dtype().type, np.float32)
        # dtype survives round trip
        if self.can_save:
            rt_img = bytesio_round_trip(img)
            assert_equal(rt_img.get_data_dtype().type, np.float32)

    def validate_data(self, imaker, params):
        # Check get data returns array, and caches
        img = imaker()
        assert_array_equal(img.dataobj, params['data'])
        if params['is_proxy']:
            assert_false(isinstance(img.dataobj, np.ndarray))
            proxy_data = np.asarray(img.dataobj)
            proxy_copy = proxy_data.copy()
            data = img.get_data()
            assert_false(proxy_data is data)
            # changing array data does not change proxy data, or reloaded data
            data[:] = 42
            assert_array_equal(proxy_data, proxy_copy)
            assert_array_equal(np.asarray(img.dataobj), proxy_copy)
            # It does change the result of get_data
            assert_array_equal(img.get_data(), 42)
            # until we uncache
            img.uncache()
            assert_array_equal(img.get_data(), proxy_copy)
        else: # not proxy
            assert_true(isinstance(img.dataobj, np.ndarray))
            non_proxy_data = np.asarray(img.dataobj)
            data = img.get_data()
            assert_true(non_proxy_data is data)
            # changing array data does change proxy data, and reloaded data
            data[:] = 42
            assert_array_equal(np.asarray(img.dataobj), 42)
            # It does change the result of get_data
            assert_array_equal(img.get_data(), 42)
            # Unache has no effect
            img.uncache()
            assert_array_equal(img.get_data(), 42)
        # Read only
        assert_raises(AttributeError, setattr, img, 'dataobj', np.eye(4))

    def validate_filenames(self, imaker, params):
        # Validate the filename, file_map interface
        if not self.can_save:
            raise SkipTest
        img = imaker()
        img.set_data_dtype(np.float32) # to avoid rounding in load / save
        # The bytesio_round_trip helper tests bytesio load / save via file_map
        rt_img = bytesio_round_trip(img)
        assert_array_equal(img.shape, rt_img.shape)
        assert_almost_equal(img.get_data(), rt_img.get_data())
        # get_ / set_ filename
        fname = 'an_image' + self.standard_extension
        img.set_filename(fname)
        assert_equal(img.get_filename(), fname)
        assert_equal(img.file_map['image'].filename, fname)
        # to_ / from_ filename
        fname = 'another_image' + self.standard_extension
        with InTemporaryDirectory():
            img.to_filename(fname)
            rt_img = img.__class__.from_filename(fname)
            assert_array_equal(img.shape, rt_img.shape)
            assert_almost_equal(img.get_data(), rt_img.get_data())


class TestSpatialImageAPI(TestAnalyzeAPI):
    image_class = SpatialImage
    can_save = False


class TestSpm99AnalyzeAPI(TestAnalyzeAPI):
    # SPM-type analyze need scipy for mat file IO
    image_class = Spm99AnalyzeImage
    has_scaling = True
    can_save = have_scipy


class TestSpm2AnalyzeAPI(TestSpm99AnalyzeAPI):
    image_class = Spm2AnalyzeImage


class TestNifti1PairAPI(TestSpm99AnalyzeAPI):
    image_class = Nifti1Pair
    can_save = True


class TestNifti1API(TestNifti1PairAPI):
    image_class = Nifti1Image
    standard_extension = '.nii'


class TestMinc1API(TestAnalyzeAPI):
    image_class = Minc1Image
    can_save = False


class TestMinc2API(TestMinc1API):
    image_class = Minc2Image



# ECAT is a special case and needs more thought
# class TestEcatAPI(TestAnalyzeAPI):
#     image_class = EcatImage
#     has_scaling = True
#     can_save = True
#    standard_extension = '.v'


class TestMGHAPI(TestAnalyzeAPI):
    image_class = MGHImage
    shapes = ((2, 3, 4), (2, 3, 4, 5)) # MGH can only do >= 3D
    has_scaling = True
    can_save = True
    standard_extension = '.mgh'
