""" Benchmarks for fileslicing

    import nibabel as nib
    nib.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_fileslice.py
"""
from __future__ import division, print_function

import sys
from timeit import timeit
from functools import partial

import numpy as np

from io import BytesIO
# from ..externals.six import BytesIO
from ..openers import Opener
from ..fileslice import fileslice
from ..tmpdirs import InTemporaryDirectory

SHAPE = (64, 64, 32, 100)
ROW_NAMES = ['axis {0}, len {1}'.format(i, SHAPE[i])
             for i in range(len(SHAPE))]
COL_NAMES = ['mid int',
             'step 1',
             'step mid int']


def _slices_for_len(L):
    # Example slices for a dimension of length L
    return (
        L // 2,
        slice(None, None, 1),
        slice(None, None, L // 2))


def run_slices(file_like, repeat=3, offset=0, order='F'):
    arr = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    n_dim = len(SHAPE)
    n_slicers = len(_slices_for_len(1))
    times_arr = np.zeros((n_dim, n_slicers))
    with Opener(file_like, 'wb') as fobj:
        fobj.write(b'\0' * offset)
        fobj.write(arr.tostring(order=order))
    with Opener(file_like, 'rb') as fobj:
        for i, L in enumerate(SHAPE):
            for j, slicer in enumerate(_slices_for_len(L)):
                sliceobj = [slice(None)] * n_dim
                sliceobj[i] = slicer
                def f():
                    fileslice(fobj,
                              tuple(sliceobj),
                              arr.shape,
                              arr.dtype,
                              offset,
                              order)
                times_arr[i, j] = timeit(f, number=repeat)
        def g():
            fobj.seek(offset)
            data = fobj.read()
            _ = np.ndarray(SHAPE, arr.dtype, buffer=data, order=order)
        base_time = timeit(g, number=repeat)
    return times_arr, base_time


def result_table(title, times, row_names, col_names):
    row_len = max(len(name) for name in row_names)
    col_len = max(len(name) for name in col_names)
    joiner = ' | '
    ender = ' |'
    row_fmt = "{0:<" + str(row_len) + "}"
    row_names = [row_fmt.format(name) for name in row_names]
    col_fmt = "{0:^" + str(col_len) + "}"
    col_names = [col_fmt.format(name) for name in col_names]
    col_header = joiner.join([' ' * row_len] + col_names) + ender
    row_val_fmt = '{0:' + str(col_len) + '.2f}'
    print()
    print('=' * len(col_header))
    print(title)
    print('=' * len(col_header))
    print(col_header)
    print('-' * len(col_header))
    for row_no, row_name in enumerate(row_names):
        row_vals = [row_val_fmt.format(val) for val in times[row_no]]
        print(joiner.join([row_name] + row_vals) + ender)
    print('-' * len(col_header))


def bench_fileslice():
    sys.stdout.flush()
    repeat = 2
    def my_table(title, times):
        result_table(title, times, ROW_NAMES, COL_NAMES)
    fobj = BytesIO()
    times, base = run_slices(fobj, repeat)
    my_table('Bytes slice raw', times)
    my_table('Bytes slice ratio', times / base)
    print('Base time: {0:3.2f}'.format(base))
    with InTemporaryDirectory():
        file_times, file_base = run_slices('data.bin', repeat)
    my_table('File slice raw', file_times)
    my_table('File slice ratio', file_times / file_base)
    print('Base time: {0:3.2f}'.format(file_base))
    with InTemporaryDirectory():
        gz_times, gz_base = run_slices('data.gz', repeat)
    my_table('gz slice raw', gz_times)
    my_table('gz slice ratio', gz_times / gz_base)
    print('Base time: {0:3.2f}'.format(gz_base))
    return
    with InTemporaryDirectory():
        bz2_times, bz2_base = run_slices('data.bz2', repeat)
    my_table('bz2 slice raw', bz2_times)
    my_table('bz2 slice ratio', bz2_times / bz2_base)
    print('Base time: {0:3.2f}'.format(bz2_base))
    sys.stdout.flush()
