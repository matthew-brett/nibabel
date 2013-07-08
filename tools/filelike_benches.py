from __future__ import print_function, division
import time
from io import BytesIO

import numpy as np

from nibabel.openers import Opener

N_SEEK = 1000
SEEK_MAX = 1000
N = 64 * 64 * 32 * 32
MEGABYTE = 2 ** 20


def make_files():
    arr = np.random.normal(size=(N,))
    bytes = arr.tostring()
    bytes_obj = BytesIO(bytes)
    for fname in ('data.bin', 'data.gz'):
        with Opener(fname, 'w') as fobj:
            fobj.write(bytes)
    return bytes_obj, 'data.bin', 'data.gz'


def time_things(file_likes):
    # Time things
    # Seek.  Seek to random position in file; seek random bytes forward
    starts = np.random.randint(1, SEEK_MAX, size=(N_SEEK,))
    seeks = np.random.randint(SEEK_MAX, N-1, size=(N_SEEK,))
    distances = seeks - starts
    n_fls = len(file_likes)
    results = np.empty((n_fls, N_SEEK, 2))
    X = np.c_[np.ones((N_SEEK,)), distances / float(MEGABYTE)]
    betas = np.empty((n_fls, 2, 2))
    for f_no, file_like in enumerate(file_likes):
        print('file like', file_like)
        times = np.empty((N_SEEK, 2))
        for i in range(N_SEEK):
            start = starts[i]
            seek = seeks[i]
            distance = distances[i]
            with Opener(file_like) as fobj:
                fobj.seek(start)
                start_time = time.time()
                fobj.seek(seek)
                stop_time = time.time()
                times[i, 0] = stop_time - start_time
            with Opener(file_like) as fobj:
                fobj.seek(start)
                start_time = time.time()
                fobj.read(distance)
                stop_time = time.time()
                times[i, 1] = stop_time * 1000 - start_time * 1000
        results[f_no, :, :] = times
        for col in 0, 1:
            Y = times[:, col]
            B = np.linalg.pinv(X).dot(Y)
            # Refit with distant observations removed
            E = Y - X.dot(B)
            # long residuals probably system related
            msk = E < np.percentile(E, 75)
            B = np.linalg.pinv(X[msk, :]).dot(Y[msk, :])
            betas[f_no, :, col] = B
    return distances, results, betas


def plot_results(distances, results, betas):
    import matplotlib.pyplot as plt
    colors = 'brkg'
    n_fobj, _, n_seek_read = results.shape
    max_distance = np.max(distances)
    max_for_beta = max_distance / MEGABYTE
    for f_no in range(n_fobj):
        fig, axes = plt.subplots(1, n_seek_read)
        for i in range(n_seek_read):
            axes[i].plot(distances, results[f_no, :, i], 'x' + colors[i])
            axes[i].hold = True
            axes[i].plot([0, max_distance],
                         [betas[f_no, 0, i], betas[f_no, 1, i] * max_for_beta],
                         'k:')


def calculate_betas(distances, results, pct_thresh=75):
    n_fobj, _, n_seek_read = results.shape
    X = np.c_[np.ones((N_SEEK,)), distances / float(MEGABYTE)]
    betas = np.empty((n_fobj, 2, 2))
    for f_no in range(n_fobj):
        times = results[f_no]
        for col in range(n_seek_read):
            Y = times[:, col]
            B = np.linalg.pinv(X).dot(Y)
            if pct_thresh != 100:
                # Refit with distant observations removed
                E = Y - X.dot(B)
                # long residuals probably system related
                msk = E < np.percentile(E, pct_thresh)
                B = np.linalg.pinv(X[msk, :]).dot(Y[msk, :])
            betas[f_no, :, col] = B
    return betas


def main():
    file_likes = make_files()
    distances, results, betas = time_things(file_likes)
    print(betas)


if __name__ == '__main__':
    main()
