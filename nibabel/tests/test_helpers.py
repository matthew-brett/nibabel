""" Helper functions for tests
"""

from io import BytesIO

import numpy as np


def bytesio_filemap(klass):
    """ Return bytes io filemap for this image class `klass` """
    file_map = klass.make_file_map()
    for name, fileholder in file_map.items():
        fileholder.fileobj = BytesIO()
        fileholder.pos = 0
    return file_map


def bytesio_round_trip(img):
    """ Save then load image from bytesio
    """
    klass = img.__class__
    bytes_map = bytesio_filemap(klass)
    img.to_file_map(bytes_map)
    return klass.from_file_map(bytes_map)


def nan_equal(first, second):
    """ Return True if two sequences are the same accounting for NaN values """
    first_arr = np.asarray(first)
    second_arr = np.asarray(second)
    if first_arr.ndim > 1 or second_arr.ndim != first_arr.ndim:
        raise ValueError('Can only compare 1D sequences')
    for a, b in zip(first, second):
        if (a, b) == (None, None):
            continue
        if np.isnan(a) and np.isnan(b):
            continue
        if a == b:
            continue
        return False
    return True
