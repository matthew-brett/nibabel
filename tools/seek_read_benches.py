""" Benchmarks to look for optimization of fileslice skip / read choice

I found myself doing iterations of:

* ``ipython --pylab``
* ``file_likes, labels = make_files()``
* ``times = get_times(file_like)`` : where file_like was one of the returned
  file_likes from ``make_files``
* ``plot_times('some label', times)``

Then I'd look to see if there was an obvious cut-off GAP size beyond which it
did not make much difference to prefer SKIP to READ (see ``_analyze_slice``
docstring in ``fileslice.py``.

See ``main()`` for an automation of this procedure.
"""
from __future__ import print_function, division
from io import BytesIO
from timeit import timeit

import numpy as np

from nibabel.openers import Opener
from nibabel.fileslice import _read_segments
from nibabel.tmpdirs import InTemporaryDirectory


OFFSET = 100
ITEM_SIZES = np.array([1, 2, 4, 8])
SLICE_SIZES = [1, 64, 128]
SIZES = np.outer(SLICE_SIZES, ITEM_SIZES).ravel()
GAPS = np.sort(np.outer(SIZES, np.arange(1, 5) ** 2).ravel())
N = OFFSET + (max(GAPS) + max(SIZES)) * 5
N_TIMES = 10


def make_files():
    arr = np.random.normal(size=(N,))
    bytes = arr.tostring()
    bytes_obj = BytesIO(bytes)
    fnames = ('data.bin', 'data.gz', 'data.bz2')
    for fname in fnames:
        with Opener(fname, 'w') as fobj:
            fobj.write(bytes)
    return (bytes_obj,) + fnames, ('bytes io', 'file', 'gzip file', 'bz2 file')


def do_read_segments(file_like, segments, n_bytes):
    with Opener(file_like, 'rb') as fobj:
        data = _read_segments(fobj, segments, n_bytes)


def make_segments(offset, gap, size, axis_len):
    segments = []
    element_size = gap + size
    usable_len = axis_len - offset
    n_elements = usable_len // element_size
    start = offset
    for i in range(n_elements):
        segments.append([start, size])
        start += element_size
    return segments, size * n_elements


def get_times(file_like,
              gaps = GAPS,
              sizes = SIZES,
              offset = OFFSET,
              axis_len = N,
              n_times=N_TIMES):
    base_time = timeit(
        lambda : do_read_segments(file_like,
                                 [[offset, axis_len - offset]],
                                  axis_len - offset),
        number = n_times
    )
    times = np.zeros((len(gaps), len(sizes)))
    n_segments = np.zeros((len(gaps), len(sizes)))
    for i, gap in enumerate(gaps):
        for j, size in enumerate(sizes):
            segments, n_bytes = make_segments(offset, gap, size, axis_len)
            the_time = timeit(
                lambda : do_read_segments(file_like, segments, n_bytes),
                number = n_times
            )
            times[i, j] = the_time
            n_segments[i, j] = len(segments)
    return times - base_time


def plot_times(title, times, ax = None, gaps = GAPS, sizes = SIZES):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    for i, sz in enumerate(sizes):
        ax.plot(gaps, times[:, i], label=str(sz))
    ax.legend()
    ax.set_xlabel('gap (bytes)')
    ax.set_ylabel('time (s)')
    ax.set_title(title)


def main():
    with InTemporaryDirectory():
        file_likes, labels = make_files()
        times_list = [get_times(f) for f in file_likes]
    import matplotlib.pyplot as plt
    n_files = len(file_likes)
    n_cols = np.floor(np.sqrt(n_files))
    n_rows = np.ceil(n_files / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols)
    for i, ax in enumerate(axes.ravel()):
        plot_times(labels[i], times_list[i], ax)
    plt.show()


if __name__ == '__main__':
    main()
