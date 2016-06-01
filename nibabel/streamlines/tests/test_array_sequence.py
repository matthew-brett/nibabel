import os
import sys
import unittest
import tempfile
import numpy as np

from nose.tools import assert_equal, assert_raises, assert_true
from nibabel.testing import assert_arrays_equal
from numpy.testing import assert_array_equal

from ..array_sequence import ArraySequence, is_array_sequence


SEQ_DATA = {}


def setup():
    global SEQ_DATA
    rng = np.random.RandomState(42)
    SEQ_DATA['rng'] = rng
    SEQ_DATA['data'] = generate_data(nb_arrays=5, common_shape=(3,), rng=rng)
    SEQ_DATA['seq'] = ArraySequence(SEQ_DATA['data'])


def generate_data(nb_arrays, common_shape, rng):
    data = [rng.rand(*(rng.randint(3, 20),) + common_shape)
            for _ in range(nb_arrays)]
    return data


def check_empty_arr_seq(seq):
    assert_equal(len(seq), 0)
    assert_equal(len(seq._offsets), 0)
    assert_equal(len(seq._lengths), 0)
    # assert_equal(seq._data.ndim, 0)
    assert_equal(seq._data.ndim, 1)
    assert_true(seq.common_shape == ())


def check_arr_seq(seq, arrays):
    lengths = list(map(len, arrays))
    assert_true(is_array_sequence(seq))
    assert_equal(len(seq), len(arrays))
    assert_equal(len(seq._offsets), len(arrays))
    assert_equal(len(seq._lengths), len(arrays))
    assert_equal(seq._data.shape[1:], arrays[0].shape[1:])
    assert_equal(seq.common_shape, arrays[0].shape[1:])
    assert_arrays_equal(seq, arrays)

    # If seq is a view, then order of internal data is not guaranteed.
    if seq._is_view:
        # The only thing we can check is the _lengths.
        assert_array_equal(sorted(seq._lengths), sorted(lengths))
    else:
        seq.shrink_data()
        assert_equal(seq._data.shape[0], sum(lengths))
        assert_array_equal(seq._data, np.concatenate(arrays, axis=0))
        assert_array_equal(seq._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(seq._lengths, lengths)


def check_arr_seq_view(seq_view, seq):
    assert_true(seq_view._is_view)
    assert_true(seq_view is not seq)
    assert_true(np.may_share_memory(seq_view._data, seq._data))
    assert_true(seq_view._offsets is not seq._offsets)
    assert_true(seq_view._lengths is not seq._lengths)


class TestArraySequence(unittest.TestCase):

    def test_creating_empty_arraysequence(self):
        check_empty_arr_seq(ArraySequence())

    def test_creating_arraysequence_from_list(self):
        # Empty list
        check_empty_arr_seq(ArraySequence([]))

        # List of ndarrays.
        N = 5
        for ndim in range(1, N+1):
            common_shape = tuple([SEQ_DATA['rng'].randint(1, 10)
                                 for _ in range(ndim-1)])
            data = generate_data(nb_arrays=5, common_shape=common_shape,
                                 rng=SEQ_DATA['rng'])
            check_arr_seq(ArraySequence(data), data)

        # Force ArraySequence constructor to use buffering.
        buffer_size = 1. / 1024**2  # 1 bytes
        check_arr_seq(ArraySequence(iter(SEQ_DATA['data']), buffer_size),
                      SEQ_DATA['data'])

    def test_creating_arraysequence_from_generator(self):
        gen = (e for e in SEQ_DATA['data'])
        check_arr_seq(ArraySequence(gen), SEQ_DATA['data'])

        # Already consumed generator
        check_empty_arr_seq(ArraySequence(gen))

    def test_creating_arraysequence_from_arraysequence(self):
        seq = ArraySequence(SEQ_DATA['data'])
        check_arr_seq(ArraySequence(seq), SEQ_DATA['data'])

        # From an empty ArraySequence
        seq = ArraySequence()
        check_empty_arr_seq(ArraySequence(seq))

    def test_arraysequence_iter(self):
        assert_arrays_equal(SEQ_DATA['seq'], SEQ_DATA['data'])

        # Try iterating through a corrupted ArraySequence object.
        seq = SEQ_DATA['seq'].copy()
        seq._lengths = seq._lengths[::2]
        assert_raises(ValueError, list, seq)

    def test_arraysequence_copy(self):
        orig = SEQ_DATA['seq']
        seq = orig.copy()
        n_rows = seq.nb_elements
        assert_equal(n_rows, orig.nb_elements)
        assert_array_equal(seq._data, orig._data[:n_rows])
        assert_true(seq._data is not orig._data)
        assert_array_equal(seq._offsets, orig._offsets)
        assert_true(seq._offsets is not orig._offsets)
        assert_array_equal(seq._lengths, orig._lengths)
        assert_true(seq._lengths is not orig._lengths)
        assert_equal(seq.common_shape, orig.common_shape)

        # Taking a copy of an `ArraySequence` generated by slicing.
        # Only keep needed data.
        seq = orig[::2].copy()
        check_arr_seq(seq, SEQ_DATA['data'][::2])
        assert_true(seq._data is not orig._data)

    def test_arraysequence_append(self):
        element = generate_data(nb_arrays=1,
                                common_shape=SEQ_DATA['seq'].common_shape,
                                rng=SEQ_DATA['rng'])[0]

        # Append a new element.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        seq.append(element)
        check_arr_seq(seq, SEQ_DATA['data'] + [element])

        # Append a list of list.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        seq.append(element.tolist())
        check_arr_seq(seq, SEQ_DATA['data'] + [element])

        # Append to an empty ArraySequence.
        seq = ArraySequence()
        seq.append(element)
        check_arr_seq(seq, [element])

        # Append an element with different shape.
        element = generate_data(nb_arrays=1,
                                common_shape=SEQ_DATA['seq'].common_shape*2,
                                rng=SEQ_DATA['rng'])[0]
        assert_raises(ValueError, seq.append, element)

    def test_arraysequence_extend(self):
        new_data = generate_data(nb_arrays=10,
                                 common_shape=SEQ_DATA['seq'].common_shape,
                                 rng=SEQ_DATA['rng'])

        # Extend with an empty list.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        seq.extend([])
        check_arr_seq(seq, SEQ_DATA['data'])

        # Extend with a list of ndarrays.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        seq.extend(new_data)
        check_arr_seq(seq, SEQ_DATA['data'] + new_data)

        # Extend with a generator.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        seq.extend((d for d in new_data))
        check_arr_seq(seq, SEQ_DATA['data'] + new_data)

        # Extend with another `ArraySequence` object.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        seq.extend(ArraySequence(new_data))
        check_arr_seq(seq, SEQ_DATA['data'] + new_data)

        # Extend with an `ArraySequence` view (e.g. been sliced).
        # Need to make sure we extend only the data we need.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        seq.extend(ArraySequence(new_data)[::2])
        check_arr_seq(seq, SEQ_DATA['data'] + new_data[::2])

        # Test extending an empty ArraySequence
        seq = ArraySequence()
        seq.extend(ArraySequence())
        check_empty_arr_seq(seq)

        seq.extend(SEQ_DATA['seq'])
        check_arr_seq(seq, SEQ_DATA['data'])

        # Extend with elements of different shape.
        data = generate_data(nb_arrays=10,
                             common_shape=SEQ_DATA['seq'].common_shape*2,
                             rng=SEQ_DATA['rng'])
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        assert_raises(ValueError, seq.extend, data)

    def test_arraysequence_extend_using_coroutine(self):
        new_data = generate_data(nb_arrays=10,
                                 common_shape=SEQ_DATA['seq'].common_shape,
                                 rng=SEQ_DATA['rng'])

        # Extend with an empty list.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        coroutine = seq._extend_using_coroutine()
        coroutine.send(None)
        coroutine.close()
        check_arr_seq(seq, SEQ_DATA['data'])

        # Extend with a list of ndarrays.
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.
        coroutine = seq._extend_using_coroutine()
        coroutine.send(None)
        for e in new_data:
            coroutine.send(e)
        coroutine.close()
        check_arr_seq(seq, SEQ_DATA['data'] + new_data)

        # Extend with elements of different shape.
        data = generate_data(nb_arrays=10,
                             common_shape=SEQ_DATA['seq'].common_shape*2,
                             rng=SEQ_DATA['rng'])
        seq = SEQ_DATA['seq'].copy()  # Copy because of in-place modification.

        coroutine = seq._extend_using_coroutine()
        coroutine.send(None)
        assert_raises(ValueError, coroutine.send, data[0])

    def test_arraysequence_getitem(self):
        # Get one item
        for i, e in enumerate(SEQ_DATA['seq']):
            assert_array_equal(SEQ_DATA['seq'][i], e)

            if sys.version_info < (3,):
                assert_array_equal(SEQ_DATA['seq'][long(i)], e)

        # Get all items using indexing (creates a view).
        indices = list(range(len(SEQ_DATA['seq'])))
        seq_view = SEQ_DATA['seq'][indices]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        # We took all elements so the view should match the original.
        check_arr_seq(seq_view, SEQ_DATA['seq'])

        # Get multiple items using ndarray of dtype integer.
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            seq_view = SEQ_DATA['seq'][np.array(indices, dtype=dtype)]
            check_arr_seq_view(seq_view, SEQ_DATA['seq'])
            # We took all elements so the view should match the original.
            check_arr_seq(seq_view, SEQ_DATA['seq'])

        # Get multiple items out of order (creates a view).
        SEQ_DATA['rng'].shuffle(indices)
        seq_view = SEQ_DATA['seq'][indices]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, [SEQ_DATA['data'][i] for i in indices])

        # Get slice (this will create a view).
        seq_view = SEQ_DATA['seq'][::2]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, SEQ_DATA['data'][::2])

        # Use advanced indexing with ndarray of data type bool.
        selection = np.array([False, True, True, False, True])
        seq_view = SEQ_DATA['seq'][selection]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view,
                      [SEQ_DATA['data'][i]
                       for i, keep in enumerate(selection) if keep])

        # Test invalid indexing
        assert_raises(TypeError, SEQ_DATA['seq'].__getitem__, 'abc')

        # Get specific columns.
        seq_view = SEQ_DATA['seq'][:, 2]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, [d[:, 2] for d in SEQ_DATA['data']])

    def test_arraysequence_repr(self):
        # Test that calling repr on a ArraySequence object is not falling.
        repr(SEQ_DATA['seq'])

        # Test calling repr when the number of arrays is bigger dans Numpy's
        # print option threshold.
        nb_arrays = 50
        seq = ArraySequence(generate_data(nb_arrays, common_shape=(1,),
                                          rng=SEQ_DATA['rng']))

        bkp_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=nb_arrays*2)
        txt1 = repr(seq)
        np.set_printoptions(threshold=nb_arrays//2)
        txt2 = repr(seq)
        assert_true(len(txt2) < len(txt1))
        np.set_printoptions(threshold=bkp_threshold)

    def test_save_and_load_arraysequence(self):
        # Test saving and loading an empty ArraySequence.
        with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
            seq = ArraySequence()
            seq.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_seq = ArraySequence.load(f)
            assert_array_equal(loaded_seq._data, seq._data)
            assert_array_equal(loaded_seq._offsets, seq._offsets)
            assert_array_equal(loaded_seq._lengths, seq._lengths)

        # Test saving and loading a ArraySequence.
        with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
            seq = SEQ_DATA['seq']
            seq.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_seq = ArraySequence.load(f)
            assert_array_equal(loaded_seq._data, seq._data)
            assert_array_equal(loaded_seq._offsets, seq._offsets)
            assert_array_equal(loaded_seq._lengths, seq._lengths)
