""" Utilities for getting array-like slices out of files
"""
from __future__ import division

import operator
import ctypes

from .externals.six.moves import reduce

import numpy as np

# Intercept, slope of time costs for different file types in ms, ms / byte
_TIME_COSTS = dict(
    bytesio=dict(seek = (2e-6, 0),
                 read = (0, 6.3e-7)),
    file=dict(seek = (1.4e-5, 0),
              read = (0, 1.2e-6)),
    gzip=dict(seek = (3.7e-5, 6.9e-8),
              read = (5.7, 4.57e-5)))

# Threshold for memory gap beyond which we always skip, to save memory
_MEGABYTE = 2**20
_BYTES_THRESHOLD = _MEGABYTE * 100
# Price in bytes to pay to save 1 ms
_BYTES_PER_MS = _BYTES_THRESHOLD

def make_skip_coster(times, bytes_per_ms):
    """
    Let's say we are in the middle of reading a file at the start of some memory
    length $B$ bytes.  We don't need the memory, and we are considering whether
    to read it anyway (then throw it away) (READ) or stop reading, skip $B$
    bytes and restart reading from there (SKIP).

    To decide between READ and SKIP we compare time costs.

    SKIP time cost is (time to skip $B$ bytes) + (time to restart read)

    READ time cost is (time to read $B$ bytes)

    The READ option also causes a memory cost of $B$ bytes.  Maybe we can
    quantify how many bytes $W$ we are prepared to waste in order to save 1
    millisecond.

    We then add $B / W$ to the READ cost.

    We will (elsewhere) choose SKIP when (SKIP cost - READ cost) < 0
    """
    seek_inter, seek_slope = times['seek']
    read_inter, read_slope = times['read']
    def skip_coster(gap):
        if gap > _BYTES_THRESHOLD:
            return -np.inf # Prefer skip
        skip_cost = seek_inter + seek_slope * gap + read_inter
        read_cost = read_slope * gap + gap / bytes_per_ms
        return skip_cost - read_cost
    return skip_coster


skip_coster_bytesio = make_skip_coster(_TIME_COSTS['bytesio'], _BYTES_PER_MS)
skip_coster_file = make_skip_coster(_TIME_COSTS['file'], _BYTES_PER_MS)
skip_coster_gzip = make_skip_coster(_TIME_COSTS['gzip'], _BYTES_PER_MS)


def is_fancy(sliceobj):
    """ Returns True if sliceobj is attempting fancy indexing

    Parameters
    ----------
    sliceobj : object
        soomething that can be used to slice an array as in ``arr[sliceobj]``

    Returns
    -------
    tf: bool
        True if sliceobj represents fancy indexing, False for basic indexing
    """
    if type(sliceobj) != type(()):
        sliceobj = (sliceobj,)
    for slicer in sliceobj:
        if hasattr(slicer, 'dtype'): # nodarray always fancy
            return True
        # slice or Ellipsis or None OK for  basic
        if isinstance(slicer, slice) or slicer in (None, Ellipsis):
            continue
        try:
            int(slicer)
        except TypeError:
            return True
    return False


def canonical_slicers(sliceobj, shape):
    """ Return canical version of sliceobj expanding missing axes

    Does not handle fancy indexing (indexing with arrays or array-like indices)

    Parameters
    ----------
    sliceobj : object
        soomething that can be used to slice an array as in ``arr[sliceobj]``
    shape : sequence
        shape of array that will be indexed by `sliceobj`

    Returns
    -------
    can_slicers : tuple
        version of `sliceobj` for which Ellipses have been expanded, missing
        (implied) dimensions have been appended, and slice objects equivalent to
        ``slice(None)`` have been replaced by ``slice(None)``, integer axes have
        been checked, and negative indices set to positive equivalent
    """
    if type(sliceobj) != type(()):
        sliceobj = (sliceobj,)
    if is_fancy(sliceobj):
        raise ValueError("Cannot handle fancy indexing")
    can_slicers = []
    n_dim = len(shape)
    n_real = 0
    for i, slicer in enumerate(sliceobj):
        if slicer is None:
            can_slicers.append(None)
            continue
        if slicer == Ellipsis:
            remaining = sliceobj[i+1:]
            if Ellipsis in remaining:
                raise ValueError("More than one Ellipsis in slicing expression")
            real_remaining = [r for r in remaining if not r is None]
            n_ellided = n_dim - n_real - len(real_remaining)
            can_slicers.extend((slice(None),) * n_ellided)
            n_real += n_ellided
            continue
        # int / slice indexing cases
        dim_len = shape[n_real]
        n_real += 1
        try: # test for integer indexing
            slicer = int(slicer)
        except TypeError: # should be slice object
            if slicer != slice(None):
                # Could this be full slice?
                if (slicer.stop == dim_len and
                    slicer.start in (None, 0) and
                    slicer.step in (None, 1)):
                    slicer = slice(None)
        else:
            if slicer < 0:
                slicer = dim_len + slicer
            elif slicer >= dim_len:
                raise ValueError('Integer index %d to large' % slicer)
        can_slicers.append(slicer)
    # Fill out any missing dimensions
    if n_real < n_dim:
        can_slicers.extend((slice(None),) * (n_dim - n_real))
    return tuple(can_slicers)


def slice2len(slicer, in_len):
    """ Output length after slicing original length `in_len` with `slicer`

    Parameters
    ----------
    slicer : slice object
    in_len : int

    Returns
    -------
    out_len : int
        Length after slicing

    Notes
    -----
    Returns same as ``len(np.arange(in_len)[slicer])``
    """
    if slicer == slice(None):
        return in_len
    full_slicer = fill_slicer(slicer, in_len)
    return _full_slicer_len(full_slicer)


def _full_slicer_len(full_slicer):
    """ Return length of slicer processed by ``fill_slicer``
    """
    start, stop, step = full_slicer.start, full_slicer.stop, full_slicer.step
    if stop == None: # case of negative step
        stop = -1
    gap = stop - start
    if (step > 0 and gap <= 0) or (step < 0 and gap >= 0):
        return 0
    return int(np.ceil(gap / step))


def fill_slicer(slicer, in_len):
    """ Return slice object with Nones filled out to match `in_len`

    Also fixes too large stop / start values according to slice() slicing rules.

    Parameters
    ----------
    slicer : slice object
    in_len : int

    Returns
    -------
    can_slicer : slice object
        slice with start, stop, step set to explicit values, with the exception
        of ``stop`` for negative step, which is None for the case of slicing
        down through the first element
    """
    start, stop, step = slicer.start, slicer.stop, slicer.step
    if step is None:
        step = 1
    if not start is None and start < 0:
        start = in_len + start
    if not stop is None and stop < 0:
        stop = in_len + stop
    if step > 0:
        if start is None:
            start = 0
        if stop is None:
            stop = in_len
        else:
            stop = min(stop, in_len)
    else: # step < 0
        if start is None:
            start = in_len - 1
        else:
            start = min(start, in_len - 1)
    return slice(start, stop, step)


def predict_shape(sliceobj, in_shape):
    """ Predict shape given slicing by `sliceobj` in array shape `in_shape`

    Parameters
    ----------
    sliceobj : object
        soomething that can be used to slice an array as in ``arr[sliceobj]``
    in_shape : sequence
        shape of array that could be sliced by `sliceobj`

    Returns
    -------
    out_shape : tuple
        predicted shape arising from slicing array shape `in_shape` with
        `sliceobj`
    """
    if type(sliceobj) != type(()):
        sliceobj = (sliceobj,)
    sliceobj = canonical_slicers(sliceobj, in_shape)
    out_shape = []
    real_no = 0
    for slicer in sliceobj:
        if slicer is None:
            out_shape.append(1)
            continue
        real_no += 1
        try: # if int - we drop a dim (no append)
            slicer = int(slicer)
        except TypeError:
            out_shape.append(slice2len(slicer, in_shape[real_no - 1]))
    return tuple(out_shape)


def _positive_slice(slicer):
    """ Make full slice (``fill_slicer``) positive
    """
    start, stop, step = slicer.start, slicer.stop, slicer.step
    if step > 0:
        return slicer
    if stop is None:
        stop = -1
    gap =  stop - start
    n = gap / step
    n = int(n) - 1 if int(n) == n else int(n)
    end = start + n * step
    return slice(end, start+1, -step)


def _space_heuristic(slicer,
                     dim_len,
                     stride,
                     skip_coster=None):
    """ Whether to force full axis read or contiguous read of stepped slice

    Allows ``fileslice`` to sometimes read memory that it will throw away in
    order to get maximum speed.  In other words, trade memory for fewer disk
    reads.

    Parameters
    ----------
    slicer : slice object
        Can be assumed to be full as in ``fill_slicer``
    dim_len : int
        length of axis being sliced
    stride : int
        memory distance between elements on this axis
    skip_coster : None or callable
        function returning cost for seek as opposed to read for a given memory
        gap in bytes.  Positive values mean seek is more expensive than read.
        None gives default for ordinary python files.

    Returns
    -------
    action : {'full', 'contiguous', None}
        * 'full' - read whole axis
        * 'contiguous' - read between start and stop
        * None - read only memory needed for output

    Notes
    -----
    See docstring for the ``make_skip_coster`` function
    """
    if skip_coster is None:
        skip_coster = skip_coster_file
    step_size = abs(slicer.step) * stride
    if skip_coster(step_size) <= 0:
        return None # Prefer skip
    slicer = _positive_slice(slicer)
    start, stop, step = slicer.start, slicer.stop, slicer.step
    gap_size = (stop - start) * stride
    return 'contiguous' if skip_coster(gap_size) <= 0 else 'full'


def _analyze_slice(slicer, dim_len, all_full, stride,
                   heuristic=_space_heuristic):
    """ Return maybe modified slice and post-slice slicing for `slicer`

    Parameters
    ----------
    slicer : slice object
    dim_len : int
        length of axis along which to slice
    all_full : bool
        Whether dimensions up until now have been full (all elements)
    stride : int
        size of one step along this axis
    heuristic : callable, optional
        function taking slice object, dim_len, stride length as arguments,
        returning one of 'full', 'contiguous', None.

    Returns
    -------
    to_read : slice object
        maybe modified slice based on `slicer` expressing what data should be
        read from an underlying file or buffer. `to_read` must always have
        positive ``step`` (because we don't want to go backwards in the buffer /
        file)
    post_slice : slice object
        slice to be applied after array has been read.  Applies any
        transformations in `slicer` that have not been applied in `to_read`
    typestr : {'full', 'contiguous', None}
        whether data as read (after applying `to_read` slicing) is full (full
        axis has been read); contiguous (step in (1, -1)), or neither

    Notes
    -----
    This is the heart of the algorithm for making segments from slice objects.

    A continuous slice is a slice with ``slice.step in (1, -1)``

    A full slice is a continuous slice returning all elements.

    The main question we have to ask is whether we should transform `to_read`,
    `post_slice` to prefer a full read and partial slice.  We only do this in
    the case of all_full==True.  In this case we might benefit from reading a
    continuous chunk of data even if the slice is not continuous, or reading all
    the data even if the slice is not full. Apply a heuristic to decide whether
    to do this, and adapt `to_read` and `post_slice` slice accordingly.

    Otherwise (apart from constraint to be positive) return `to_read` unaltered
    and `post_slice` as ``slice(None)``
    """
    # Deal with full cases first
    if slicer == slice(None):
        return slicer, slicer, 'full'
    slicer = fill_slicer(slicer, dim_len)
    # actually equivalent to slice(None)
    if slicer == slice(0, dim_len, 1):
        return slice(None), slice(None), 'full'
    # full, but reversed
    if slicer == slice(dim_len-1, None, -1):
        return slice(None), slice(None, None, -1), 'full'
    # Not full, mabye continuous
    typestr = 'contiguous' if slicer.step in (1, -1) else None
    if all_full:
        action = heuristic(slicer, dim_len, stride)
        if action == 'full':
            return slice(None), slicer, 'full'
        elif action == 'contiguous':
            # If this is already contiguous, default None behavior handles it
            if typestr != 'contiguous':
                step = slicer.step
                if step < 0:
                    slicer = _positive_slice(slicer)
                return (slice(slicer.start, slicer.stop, 1),
                        slice(None, None, step),
                        'contiguous')
        elif action != None:
            raise ValueError('Unexpected return %s from heuristic' % action)
    # We only need to be positive
    if slicer.step > 0:
        return slicer, slice(None), typestr
    return _positive_slice(slicer), slice(None, None, -1), typestr


def _get_segments(sliceobj, in_shape, itemsize, offset, order,
                 heuristic=_space_heuristic):
    """ Get segments, output shape, added slicing from `sliceobj`, memory info

    Parameters
    ----------
    sliceobj : object
        soomething that can be used to slice an array as in ``arr[sliceobj]``
    in_shape : sequence
        shape of underlying array to be sliced
    itemsize : int
        element size in array (in bytes)
    offset : int
        offset of array data in underlying file or memory buffer
    order : {'C', 'F'}
        memory layout of underlying array
    heuristic : callable, optional
        function taking slice object, dim_len, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        ``_analyze_slice``.

    Returns
    -------
    segments : list
        list of 2 element lists where lists are (offset, length), giving
        absolute memory offset in bytes and number of bytes to read
    out_shape : tuple
        shape with which to interpret memory as read from `segments`
    new_slicing : tuple
        Any new slicing to be applied to the array after reading.  In terms of
        `out_shape`.  If empty, no new slicing to apply
    """
    if not order in "CF":
        raise ValueError("order should be one of 'CF'")
    sliceobj = canonical_slicers(sliceobj, in_shape)
    # order fastest changing forst (record reordering)
    if order == 'C':
        sliceobj = sliceobj[::-1]
        in_shape = in_shape[::-1]
    # Analyze sliceobj
    # Compile out shape
    # Compile new slicing
    all_full = True
    out_shape = []
    all_segments = [[offset, itemsize]]
    new_slicing = []
    need_new_slicing = False
    real_no = 0
    stride = itemsize
    for slicer in sliceobj:
        if slicer is None:
            out_shape.append(1)
            new_slicing.append(slice(None))
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        # int or slice
        try: # if int - we drop a dim (no append)
            slicer = int(slicer)
        except TypeError:
            pass # We deal with slices later
        else: # integer
            if slicer < 0:
                slicer = dim_len + slicer
            for segment in all_segments:
                segment[0] += stride * slicer
            all_full = False
            stride *= dim_len
            continue
        # slices
        # make modified sliceobj (to_read, post_slice)
        to_read_slicer, post_slice_slicer, typestr = _analyze_slice(
            slicer, dim_len, all_full, stride, heuristic)
        to_read_slicer = fill_slicer(to_read_slicer, dim_len)
        slice_len = _full_slicer_len(to_read_slicer)
        out_shape.append(slice_len)
        if post_slice_slicer != slice(None):
            need_new_slicing = True
        new_slicing.append(post_slice_slicer)
        if all_full and typestr in ('full', 'contiguous'):
            if to_read_slicer.start != 0:
                all_segments[0][0] += stride * to_read_slicer.start
            all_segments[0][1] *= slice_len
            all_full = typestr == 'full'
        else: # Previous or current stuff is not contiguous
            segments = all_segments
            all_segments = []
            for i in range(to_read_slicer.start,
                           to_read_slicer.stop,
                           to_read_slicer.step):
                for s in segments:
                    all_segments.append([s[0] + stride * i, s[1]])
            all_full = False
        stride *= dim_len
    if not need_new_slicing:
        new_slicing = []
    # If reordered, order shape, new_slicing
    if order == 'C':
        out_shape = out_shape[::-1]
        new_slicing = new_slicing[::-1]
    return all_segments, tuple(out_shape), tuple(new_slicing)


def _read_segments(fileobj, segments, n_bytes):
    """ Read `n_bytes` byte data implied by `segments` from `fileobj`

    Parameters
    ----------
    fileobj : file-like object
        Implements `seek` and `read`
    segments : sequence
        list of 2 tuples where tuples are (offset, length), giving absolute
        file offset in bytes and number of bytes to read
    n_bytes : int
        number of bytes that will be read

    Returns
    -------
    buffer : buffer object
        object implementing buffer protocol.  Can by byte string or ndarray or
        ctypes ``c_char_array``
    """
    if len(segments) == 0:
        if n_bytes != 0:
            raise ValueError("No segments, but non-zero n_bytes")
        return b''
    if len(segments) == 1:
        offset, length = segments[0]
        fileobj.seek(offset)
        bytes = fileobj.read(length)
        if len(bytes) != n_bytes:
            raise ValueError("Whoops, not enough data in file")
        return bytes
    # More than one segment
    bytes = ctypes.create_string_buffer(n_bytes)
    bytes_offset = 0
    for offset, length in segments:
        fileobj.seek(offset)
        these_bytes = fileobj.read(length)
        bytes_next = bytes_offset + length
        bytes[bytes_offset:bytes_next] = these_bytes
        bytes_offset = bytes_next
    if bytes_offset != n_bytes:
        raise ValueError("Whoops, n_bytes does not look right")
    return bytes


def _simple_fileslice(fileobj, sliceobj, shape, dtype, offset=0, order='C'):
    fileobj.seek(offset)
    nbytes = reduce(operator.mul, shape) * dtype.itemsize
    bytes = fileobj.read(nbytes)
    new_arr = np.ndarray(shape, dtype, buffer=bytes, order=order)
    return new_arr[sliceobj]


def fileslice(fileobj, sliceobj, shape, dtype, offset=0, order='C'):
    """ Slice array inside `fileobj` using `sliceobj` slicer, array definitions

    Parameters
    ----------
    fileobj : file-like object
        implements ``read`` and ``seek``
    sliceobj : object
        soomething that can be used to slice an array as in ``arr[sliceobj]``
    shape : sequence
        shape of full array inside `fileobj`
    dtype : dtype object
        dtype of array inside `fileobj`
    offset : int, optional
        offset of array data within `fileobj`
    order : {'C', 'F'}, optional
        memory layout of array in `fileobj`

    Returns
    -------
    sliced_arr : array
        Array in `fileobj` as sliced with `sliceobj`
    """
    if is_fancy(sliceobj):
        raise ValueError("Cannot handle fancy indexing")
    # Temporary hack to make tests pass
    # return _simple_fileslice(
    #   fileobj, sliceobj, shape, dtype, offset=offset, order=order)
    # Code that may work someday
    itemsize = dtype.itemsize
    segments, sliced_shape, post_slicers = _get_segments(
        sliceobj, shape, itemsize, offset, order)
    n_bytes = reduce(operator.mul, sliced_shape, 1) * itemsize
    bytes = _read_segments(fileobj, segments, n_bytes)
    sliced = np.ndarray(sliced_shape, dtype, buffer=bytes, order=order)
    return sliced[post_slicers]
