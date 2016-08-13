""" Testing dataobj_images module
"""

import numpy as np

from nibabel.filebasedimages import FileBasedHeader
from nibabel.dataobj_images import DataobjImage

from nibabel.tests.test_image_api import DataInterfaceMixin
from nibabel.tests.test_filebasedimages import TestFBImageAPI as _TFI


class DoNumpyImage(DataobjImage):
    header_class = FileBasedHeader
    valid_exts = ('.npy',)
    files_types = (('image', '.npy'),)

    @classmethod
    def from_file_map(klass, file_map):
        with file_map['image'].get_prepare_fileobj('rb') as fobj:
            arr = np.load(fobj)
        return klass(arr)

    def to_file_map(self, file_map=None):
        file_map = self.file_map if file_map is None else file_map
        with file_map['image'].get_prepare_fileobj('wb') as fobj:
            np.save(fobj, self.dataobj)

    def get_data_dtype(self):
        return self.dataobj.dtype

    def set_data_dtype(self, dtype):
        self._dataobj = self._dataobj.astype(dtype)


class TestDataobjAPI(_TFI, DataInterfaceMixin):
    """ Validation for DataobjImage instances
    """
    # A callable returning an image from ``image_maker(data, header)``
    image_maker = DoNumpyImage
