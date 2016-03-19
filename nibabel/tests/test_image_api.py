""" Validate image API

What is the image API?

* ``img.dataobj``

    * Returns ``np.ndarray`` from ``np.array(img.databj)``
    * Has attribute ``shape``

* ``img.header`` (image metadata) (changes in the image metadata should not
  change any of ``dataobj``, ``affine``, ``shape``)
* ``img.affine`` (4x4 float ``np.ndarray`` relating spatial voxel coordinates
  to world space)
* ``img.shape`` (shape of data as read with ``np.array(img.dataobj)``
* ``img.get_data()`` (returns data as read with ``np.array(img.dataobj)``)
* ``img.uncache()`` (``img.get_data()`` is allowed to cache the result of the
  array creation.  If it does, this call empties that cache.  Implement this
  as a no-op if ``get_data()`` does not cache.
* ``img[something]`` generates an informative TypeError
* ``img.in_memory`` is True for an array image, and for a proxy image that is
  cached, but False otherwise.
"""
from __future__ import division, print_function, absolute_import

import warnings
from functools import partial
from ..externals.six import string_types

import numpy as np

from ..optpkg import optional_package
_, have_scipy, _ = optional_package('scipy')
_, have_h5py, _ = optional_package('h5py')

from .. import (AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage,
                Nifti1Pair, Nifti1Image, Nifti2Pair, Nifti2Image,
                MGHImage, Minc1Image, Minc2Image)
from ..spatialimages import SpatialImage
from .. import minc1, minc2, parrec

from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from numpy.testing import (assert_almost_equal, assert_array_equal)
from ..testing import clear_and_catch_warnings
from ..tmpdirs import InTemporaryDirectory

from .test_api_validators import ValidateAPI
from .test_helpers import bytesio_round_trip, assert_data_similar
from .test_minc1 import EXAMPLE_IMAGES as MINC1_EXAMPLE_IMAGES
from .test_minc2 import EXAMPLE_IMAGES as MINC2_EXAMPLE_IMAGES
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLE_IMAGES


class GenericImageAPI(ValidateAPI):
    """ General image validation API """
    # Whether this image type can do scaling of data
    has_scaling = False
    # Whether the image can be saved to disk / file objects
    can_save = False
    # Filename extension to which to save image; only used if `can_save` is
    # True
    standard_extension = '.img'

    def obj_params(self):
        """ Return generator returning (`img_creator`, `img_params`) tuples

        ``img_creator`` is a function taking no arguments and returning a fresh
        image.  We need to return this ``img_creator`` function rather than an
        image instance so we can recreate the images fresh for each of multiple
        tests run from the ``validate_xxx`` autogenerated test methods.  This
        allows the tests to modify the image without having an effect on the
        later tests in the same function, because each test will create a fresh
        image with ``img_creator``.

        Returns
        -------
        func_params_gen : generator
            Generator returning tuples with:

            * img_creator : callable
              Callable returning a fresh image for testing
            * img_params : mapping
              Expected properties of image returned from ``img_creator``
              callable.  Key, value pairs should include:

              * ``data`` : array returned from ``get_data()`` on image - OR -
                ``data_summary`` : dict with data ``min``, ``max``, ``mean``;
              * ``shape`` : shape of image;
              * ``affine`` : shape (4, 4) affine array for image;
              * ``dtype`` : dtype of data returned from ``get_data()``;
              * ``is_proxy`` : bool, True if image data is proxied;

        Notes
        -----
        Passing ``data_summary`` instead of ``data`` allows you gentle user to
        avoid having to have a saved copy of the entire data array from example
        images for testing.
        """
        raise NotImplementedError

    def validate_affine(self, imaker, params):
        # Check affine API
        img = imaker()
        assert_almost_equal(img.affine, params['affine'], 6)
        assert_equal(img.affine.dtype, np.float64)
        img.affine[0, 0] = 1.5
        assert_equal(img.affine[0, 0], 1.5)
        # Read only
        assert_raises(AttributeError, setattr, img, 'affine', np.eye(4))

    def validate_affine_deprecated(self, imaker, params):
        # Check deprecated affine API
        img = imaker()
        assert_almost_equal(img.get_affine(), params['affine'], 6)
        assert_equal(img.get_affine().dtype, np.float64)
        aff = img.get_affine()
        aff[0, 0] = 1.5
        assert_true(aff is img.get_affine())

    def validate_header(self, imaker, params):
        # Check header API
        img = imaker()
        hdr = img.header  # we can fetch it
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
        with clear_and_catch_warnings() as w:
            warnings.simplefilter('always', DeprecationWarning)
            img = imaker()
            hdr = img.get_header()
            assert_equal(len(w), 1)
            assert_true(hdr is img.header)

    def validate_shape(self, imaker, params):
        # Validate shape
        img = imaker()
        # Same as expected shape
        assert_equal(img.shape, params['shape'])
        # Same as array shape if passed
        if 'data' in params:
            assert_equal(img.shape, params['data'].shape)
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
            with np.errstate(invalid='ignore'):
                rt_img = bytesio_round_trip(img)
            assert_equal(rt_img.get_data_dtype().type, params['dtype'])
        # Setting to a different dtype
        img.set_data_dtype(np.float32)  # assumed supported for all formats
        assert_equal(img.get_data_dtype().type, np.float32)
        # dtype survives round trip
        if self.can_save:
            rt_img = bytesio_round_trip(img)
            assert_equal(rt_img.get_data_dtype().type, np.float32)

    def validate_data(self, imaker, params):
        # Check get data returns array, and caches
        img = imaker()
        assert_data_similar(img.dataobj, params)
        if params['is_proxy']:
            assert_false(isinstance(img.dataobj, np.ndarray))
            proxy_data = np.asarray(img.dataobj)
            proxy_copy = proxy_data.copy()
            # Not yet cached, proxy image: in_memory is False
            assert_false(img.in_memory)
            # Load with caching='unchanged'
            data = img.get_data(caching='unchanged')
            # Still not cached
            assert_false(img.in_memory)
            # Default load, does caching
            data = img.get_data()
            # Data now cached
            assert_true(img.in_memory)
            assert_false(proxy_data is data)
            # Now caching='unchanged' does nothing, returns cached version
            data_again = img.get_data(caching='unchanged')
            assert_true(data is data_again)
            # caching='fill' does nothing because the cache is already full
            data_yet_again = img.get_data(caching='fill')
            assert_true(data is data_yet_again)
            # changing array data does not change proxy data, or reloaded data
            data[:] = 42
            assert_array_equal(proxy_data, proxy_copy)
            assert_array_equal(np.asarray(img.dataobj), proxy_copy)
            # It does change the result of get_data
            assert_array_equal(img.get_data(), 42)
            # until we uncache
            img.uncache()
            # Which unsets in_memory
            assert_false(img.in_memory)
            assert_array_equal(img.get_data(), proxy_copy)
            # Check caching='fill' does cache data
            img = imaker()
            assert_false(img.in_memory)
            data = img.get_data(caching='fill')
            assert_true(img.in_memory)
            data_again = img.get_data()
            assert_true(data is data_again)
        else:  # not proxy
            for caching in (None, 'fill', 'unchanged'):
                img = imaker()
                get_data_func = (img.get_data if caching is None else
                                 partial(img.get_data, caching=caching))
                assert_true(isinstance(img.dataobj, np.ndarray))
                assert_true(img.in_memory)
                data = get_data_func()
                assert_true(data is img.dataobj)
                # changing array data does change proxy data, and reloaded data
                data[:] = 42
                assert_array_equal(np.asarray(img.dataobj), 42)
                # It does change the result of get_data
                assert_array_equal(get_data_func(), 42)
                # Unache has no effect
                img.uncache()
                assert_array_equal(get_data_func(), 42)
                assert_true(img.in_memory)
        # dataobj is read only
        fake_data = np.zeros(img.shape).astype(img.get_data_dtype())
        assert_raises(AttributeError, setattr, img, 'dataobj', fake_data)
        # So is in_memory
        assert_raises(AttributeError, setattr, img, 'in_memory', False)
        # Values to get_data caching parameter must be 'fill' or 'unchanged'
        assert_raises(ValueError, img.get_data, caching='something')

    def validate_data_deprecated(self, imaker, params):
        # Check _data property still exists, but raises warning
        img = imaker()
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            assert_data_similar(img._data, params)
            assert_equal(warns.pop(0).category, FutureWarning)
        # Check setting _data raises error
        fake_data = np.zeros(img.shape).astype(img.get_data_dtype())
        assert_raises(AttributeError, setattr, img, '_data', fake_data)

    def validate_filenames(self, imaker, params):
        # Validate the filename, file_map interface
        if not self.can_save:
            raise SkipTest
        img = imaker()
        img.set_data_dtype(np.float32)  # to avoid rounding in load / save
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
            del rt_img  # to allow windows to delete the directory

    def validate_no_slicing(self, imaker, params):
        img = imaker()
        assert_raises(TypeError, img.__getitem__, 'string')
        assert_raises(TypeError, img.__getitem__, slice(None))


class LoadImageAPI(GenericImageAPI):
    # Callable returning an image from a filename
    loader = None
    # Sequence of dictionaries, where dictionaries have keys
    # 'fname" in addition to keys for ``params`` (see obj_params docstring)
    example_images = ()
    # Class of images to be tested
    klass = None

    def obj_params(self):
        for img_params in self.example_images:
            yield lambda: self.loader(img_params['fname']), img_params

    def validate_path_maybe_image(self, imaker, params):
        for img_params in self.example_images:
            test, sniff = self.klass.path_maybe_image(img_params['fname'])
            assert_true(isinstance(test, bool))
            if sniff is not None:
                assert isinstance(sniff[0], bytes)
                assert isinstance(sniff[1], string_types)


class MakeImageAPI(LoadImageAPI):
    """ Validation for images we can make with ``func(data, affine, header)``
    """
    # A callable returning an image from ``image_maker(data, affine, header)``
    image_maker = None
    # A callable returning a header from ``header_maker()``
    header_maker = None
    # Example shapes for created images
    example_shapes = ((2,), (2, 3), (2, 3, 4), (2, 3, 4, 5))

    def img_from_arr_aff(self, arr, aff, header=None):
        return self.image_maker(arr, aff, header)

    def obj_params(self):
        # Return any obj_params from superclass
        for func, params in super(MakeImageAPI, self).obj_params():
            yield func, params
        # Create a new images
        aff = np.diag([1, 2, 3, 1])

        def make_imaker(arr, aff, header=None):
            return lambda: self.image_maker(arr, aff, header)
        for shape in self.example_shapes:
            for dtype in (np.uint8, np.int16, np.float32):
                arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                hdr = self.header_maker()
                hdr.set_data_dtype(dtype)
                func = make_imaker(arr.copy(), aff, hdr)
                params = dict(
                    dtype=dtype,
                    affine=aff,
                    data=arr,
                    shape=shape,
                    is_proxy=False)
                yield func, params
        if not self.can_save:
            return
        # Add a proxy image
        # We assume that loading from a fileobj creates a proxy image
        params['is_proxy'] = True

        def prox_imaker():
            img = self.image_maker(arr, aff, hdr)
            rt_img = bytesio_round_trip(img)
            return self.image_maker(rt_img.dataobj, aff, rt_img.header)
        yield prox_imaker, params


class ImageHeaderAPI(MakeImageAPI):
    """ When ``self.image_maker`` is an image class, make header from class
    """

    def header_maker(self):
        return self.image_maker.header_class()


class TestAnalyzeAPI(ImageHeaderAPI):
    """ General image validation API instantiated for Analyze images
    """
    klass = image_maker = AnalyzeImage
    has_scaling = False
    can_save = True
    standard_extension = '.img'


class TestSpatialImageAPI(TestAnalyzeAPI):
    klass = image_maker = SpatialImage
    can_save = False


class TestSpm99AnalyzeAPI(TestAnalyzeAPI):
    # SPM-type analyze need scipy for mat file IO
    klass = image_maker = Spm99AnalyzeImage
    has_scaling = True
    can_save = have_scipy


class TestSpm2AnalyzeAPI(TestSpm99AnalyzeAPI):
    klass = image_maker = Spm2AnalyzeImage


class TestNifti1PairAPI(TestSpm99AnalyzeAPI):
    klass = image_maker = Nifti1Pair
    can_save = True


class TestNifti1API(TestNifti1PairAPI):
    klass = image_maker = Nifti1Image
    standard_extension = '.nii'


class TestNifti2PairAPI(TestNifti1PairAPI):
    klass = image_maker = Nifti2Pair


class TestNifti2API(TestNifti1API):
    klass = image_maker = Nifti2Image


class TestMinc1API(ImageHeaderAPI):
    klass = image_maker = Minc1Image
    loader = minc1.load
    example_images = MINC1_EXAMPLE_IMAGES


class TestMinc2API(TestMinc1API):

    def __init__(self):
        if not have_h5py:
            raise SkipTest('Need h5py for these tests')

    klass = image_maker = Minc2Image
    loader = minc2.load
    example_images = MINC2_EXAMPLE_IMAGES


class TestPARRECAPI(LoadImageAPI):

    def loader(self, fname):
        return parrec.load(fname)

    klass = parrec.PARRECImage
    example_images = PARREC_EXAMPLE_IMAGES


# ECAT is a special case and needs more thought
# class TestEcatAPI(TestAnalyzeAPI):
#     image_maker = ecat.EcatImage
#     has_scaling = True
#     can_save = True
#    standard_extension = '.v'


class TestMGHAPI(ImageHeaderAPI):
    klass = image_maker = MGHImage
    example_shapes = ((2, 3, 4), (2, 3, 4, 5))  # MGH can only do >= 3D
    has_scaling = True
    can_save = True
    standard_extension = '.mgh'
