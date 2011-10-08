""" Load and save benchmarks
"""
import sys

import numpy as np

from ..py3k import BytesIO
from .. import Nifti1Image

from numpy.testing import measure

def bench_load_save():
    np.random.seed(20111001)
    repeat = 4
    img_shape = (128, 128, 64)
    arr = np.random.normal(size=img_shape)
    img = Nifti1Image(arr, np.eye(4))
    sio = BytesIO()
    img.file_map['image'].fileobj = sio
    hdr = img.get_header()
    sys.stdout.flush()
    print "\nImage load save"
    print "----------------"
    hdr.set_data_dtype(np.float32)
    mtime = measure('img.to_file_map()', repeat)
    print '%20s %6.2f' % ('Out to float32', mtime)
    mtime = measure('img.from_file_map(img.file_map)', repeat)
    print '%20s %6.2f' % ('In from float32', mtime)
    hdr.set_data_dtype(np.int16)
    mtime = measure('img.to_file_map()', repeat)
    print '%20s %6.2f' % ('Out to int16', mtime)
    mtime = measure('img.from_file_map(img.file_map)', repeat)
    print '%20s %6.2f' % ('In from int16', mtime)
    arr = np.random.random_integers(low=-1000,high=-1000, size=img_shape)
    arr = arr.astype(np.int16)
    img = Nifti1Image(arr, np.eye(4))
    sio = BytesIO()
    img.file_map['image'].fileobj = sio
    hdr = img.get_header()
    hdr.set_data_dtype(np.float32)
    mtime = measure('img.to_file_map()', repeat)
    print '%20s %6.2f' % ('Int16 to float32', mtime)
    sys.stdout.flush()
