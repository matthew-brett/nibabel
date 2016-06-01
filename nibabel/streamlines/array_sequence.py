import numbers
import numpy as np


def is_array_sequence(obj):
    """ Return True if `obj` is an array sequence. """
    try:
        return obj.is_array_sequence
    except AttributeError:
        return False


def is_ndarray_of_int_or_bool(obj):
    return (isinstance(obj, np.ndarray) and
            (np.issubdtype(obj.dtype, np.integer) or
            np.issubdtype(obj.dtype, np.bool)))


class ArraySequence(object):
    """ Sequence of ndarrays having variable first dimension sizes.

    This is a container that can store multiple ndarrays where each ndarray
    might have a different first dimension size but a *common* size for the
    remaining dimensions.

    More generally, an instance of :class:`ArraySequence` of length $N$ is
    composed of $N$ ndarrays of shape $(d_1, d_2, ... d_D)$ where $d_1$
    can vary in length between arrays but $(d_2, ..., d_D)$ have to be the
    same for every ndarray.
    """

    def __init__(self, iterable=None, buffer_size=4):
        """ Initialize array sequence instance

        Parameters
        ----------
        iterable : None or iterable or :class:`ArraySequence`, optional
            If None, create an empty :class:`ArraySequence` object.
            If iterable, create a :class:`ArraySequence` object initialized
            from array-like objects yielded by the iterable.
            If :class:`ArraySequence`, create a view (no memory is allocated).
            For an actual copy use :meth:`.copy` instead.
        buffer_size : float, optional
            Size (in Mb) for memory allocation when `iterable` is a generator.
        """
        # Create new empty `ArraySequence` object.
        self._is_view = False
        self._data = np.array([])
        self._offsets = []
        self._lengths = []

        if iterable is None:
            return

        if is_array_sequence(iterable):
            # Create a view.
            self._data = iterable._data
            self._offsets = iterable._offsets
            self._lengths = iterable._lengths
            self._is_view = True
            return

        try:
            # If possible try pre-allocating memory.
            if len(iterable) > 0:
                first_element = np.asarray(iterable[0])
                n_elements = np.sum([len(iterable[i])
                                     for i in range(len(iterable))])
                new_shape = (n_elements,) + first_element.shape[1:]
                self._data = np.empty(new_shape, dtype=first_element.dtype)
        except TypeError:
            pass

        # Initialize the `ArraySequence` object from iterable's item.
        coroutine = self._extend_using_coroutine()
        coroutine.send(None)  # Run until the first yield.

        for e in iterable:
            coroutine.send(e)

        coroutine.close()  # Terminate coroutine.

    @property
    def is_array_sequence(self):
        return True

    @property
    def common_shape(self):
        """ Matching shape of the elements in this array sequence. """
        return self._data.shape[1:]

    @property
    def nb_elements(self):
        """ Total number of elements in this array sequence. """
        return np.sum(self._lengths)

    @property
    def data(self):
        """ Elements in this array sequence. """
        return self._data

    @property
    def _next_offset(self):
        if len(self._offsets) == 0:
            return 0
        imax = np.argmax(self._offsets)
        return self._offsets[imax] + self._lengths[imax]

    def append(self, element):
        """ Appends `element` to this array sequence.

        Parameters
        ----------
        element : ndarray
            Element to append. The shape must match already inserted elements
            shape except for the first dimension.

        Returns
        -------
        None

        Notes
        -----
        If you need to add multiple elements you should consider
        `ArraySequence.extend`.
        """
        element = np.asarray(element)

        if self.common_shape != () and element.shape[1:] != self.common_shape:
            msg = "All dimensions, except the first one, must match exactly"
            raise ValueError(msg)

        next_offset = self._next_offset
        size = (next_offset + element.shape[0],) + element.shape[1:]
        self._data.resize(size)
        self._data[next_offset:] = element
        self._offsets.append(next_offset)
        self._lengths.append(element.shape[0])

    def extend(self, elements):
        """ Appends all `elements` to this array sequence.

        Parameters
        ----------
        elements : iterable of ndarrays or :class:`ArraySequence` object
            If iterable of ndarrays, each ndarray will be concatenated along
            the first dimension then appended to the data of this
            ArraySequence.
            If :class:`ArraySequence` object, its data are simply appended to
            the data of this ArraySequence.

        Returns
        -------
        None

        Notes
        -----
        The shape of the elements to be added must match the one of the data of
        this :class:`ArraySequence` except for the first dimension.
        """
        if not is_array_sequence(elements):
            self.extend(self.__class__(elements))
            return

        if len(elements) == 0:
            return

        if (self.common_shape != () and
                elements.common_shape != self.common_shape):
            msg = "All dimensions, except the first one, must match exactly"
            raise ValueError(msg)

        next_offset = self._next_offset
        self._data.resize((next_offset + sum(elements._lengths),
                           elements._data.shape[1]))

        offsets = []
        for offset, length in zip(elements._offsets, elements._lengths):
            offsets.append(next_offset)
            chunk = elements._data[offset:offset + length]
            self._data[next_offset:next_offset + length] = chunk
            next_offset += length

        self._lengths += elements._lengths
        self._offsets += offsets

    def _extend_using_coroutine(self, buffer_size=4):
        """ Creates a coroutine allowing to append elements.

        Parameters
        ----------
        buffer_size : float, optional
            Size (in Mb) for memory pre-allocation.

        Returns
        -------
        coroutine
            Coroutine object which expects the values to be appended to this
            array sequence.

        Notes
        -----
        This method is essential for
        :func:`create_arraysequences_from_generator` as it allows for an
        efficient way of creating multiple array sequences in a hyperthreaded
        fashion and still benefit from the memory buffering. Whitout this
        method the alternative would be to use :meth:`append` which does
        not have such buffering mechanism and thus is at least one order of
        magnitude slower.
        """
        offsets = []
        lengths = []

        offset = 0 if len(self) == 0 else self._offsets[-1] + self._lengths[-1]
        try:
            first_element = True
            while True:
                e = (yield)
                e = np.asarray(e)
                if first_element:
                    first_element = False
                    n_rows_buffer = int(buffer_size * 1024**2 // e.nbytes)
                    new_shape = (n_rows_buffer,) + e.shape[1:]
                    if len(self) == 0:
                        self._data = np.empty(new_shape, dtype=e.dtype)

                end = offset + len(e)
                if end > len(self._data):
                    # Resize needed, adding `len(e)` items plus some buffer.
                    nb_points = len(self._data)
                    nb_points += len(e) + n_rows_buffer
                    self._data.resize((nb_points,) + self.common_shape)

                offsets.append(offset)
                lengths.append(len(e))
                self._data[offset:offset + len(e)] = e
                offset += len(e)

        except GeneratorExit:
            pass

        self._offsets += offsets
        self._lengths += lengths

        # Clear unused memory.
        self._data.resize((offset,) + self.common_shape)

    def copy(self):
        """ Creates a copy of this :class:`ArraySequence` object.

        Returns
        -------
        seq_copy : :class:`ArraySequence` instance
            Copy of `self`.

        Notes
        -----
        We do not simply deepcopy this object because we have a chance to use
        less memory. For example, if the array sequence being copied is the
        result of a slicing operation on an array sequence.
        """
        seq = self.__class__()
        total_lengths = np.sum(self._lengths)
        seq._data = np.empty((total_lengths,) + self._data.shape[1:],
                             dtype=self._data.dtype)

        next_offset = 0
        offsets = []
        for offset, length in zip(self._offsets, self._lengths):
            offsets.append(next_offset)
            chunk = self._data[offset:offset + length]
            seq._data[next_offset:next_offset + length] = chunk
            next_offset += length

        seq._offsets = offsets
        seq._lengths = self._lengths[:]

        return seq

    def __getitem__(self, idx):
        """ Get sequence(s) through standard or advanced numpy indexing.

        Parameters
        ----------
        idx : int or slice or list or ndarray
            If int, index of the element to retrieve.
            If slice, use slicing to retrieve elements.
            If list, indices of the elements to retrieve.
            If ndarray with dtype int, indices of the elements to retrieve.
            If ndarray with dtype bool, only retrieve selected elements.

        Returns
        -------
        ndarray or :class:`ArraySequence`
            If `idx` is an int, returns the selected sequence.
            Otherwise, returns a :class:`ArraySequence` object which is a view
            of the selected sequences.
        """
        if isinstance(idx, (numbers.Integral, np.integer)):
            start = self._offsets[idx]
            return self._data[start:start + self._lengths[idx]]

        seq = self.__class__()
        seq._is_view = True
        if isinstance(idx, tuple):
            off_idx = idx[0]
            seq._data = self._data.__getitem__((slice(None),) + idx[1:])
        else:
            off_idx = idx
            seq._data = self._data

        if isinstance(off_idx, slice):  # Standard list slicing
            seq._offsets = self._offsets[off_idx]
            seq._lengths = self._lengths[off_idx]
            return seq

        if isinstance(off_idx, list) or is_ndarray_of_int_or_bool(off_idx):
            # Fancy indexing
            seq._offsets = list(np.array(self._offsets)[off_idx])
            seq._lengths = list(np.array(self._lengths)[off_idx])
            return seq

        raise TypeError("Index must be either an int, a slice, a list of int"
                        " or a ndarray of bool! Not " + str(type(idx)))

    def __iter__(self):
        if len(self._lengths) != len(self._offsets):
            raise ValueError("ArraySequence object corrupted:"
                             " len(self._lengths) != len(self._offsets)")

        for offset, lengths in zip(self._offsets, self._lengths):
            yield self._data[offset: offset + lengths]

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        if len(self) > np.get_printoptions()['threshold']:
            # Show only the first and last edgeitems.
            edgeitems = np.get_printoptions()['edgeitems']
            data = str(list(self[:edgeitems]))[:-1]
            data += ", ..., "
            data += str(list(self[-edgeitems:]))[1:]
        else:
            data = str(list(self))

        return "{name}({data})".format(name=self.__class__.__name__,
                                       data=data)

    def save(self, filename):
        """ Saves this :class:`ArraySequence` object to a .npz file. """
        np.savez(filename,
                 data=self._data,
                 offsets=self._offsets,
                 lengths=self._lengths)

    @classmethod
    def load(cls, filename):
        """ Loads a :class:`ArraySequence` object from a .npz file. """
        content = np.load(filename)
        seq = cls()
        seq._data = content["data"]
        seq._offsets = content["offsets"]
        seq._lengths = content["lengths"]
        return seq


def create_arraysequences_from_generator(gen, n):
    """ Creates :class:`ArraySequence` objects from a generator yielding tuples

    Parameters
    ----------
    gen : generator
        Generator yielding a size `n` tuple containing the values to put in the
        array sequences.
    n : int
        Number of :class:`ArraySequences` object to create.
    """
    seqs = [ArraySequence() for _ in range(n)]
    coroutines = [seq._extend_using_coroutine() for seq in seqs]

    for coroutine in coroutines:
        coroutine.send(None)

    for data in gen:
        for i, coroutine in enumerate(coroutines):
            if data[i].nbytes > 0:
                coroutine.send(data[i])

    for coroutine in coroutines:
        coroutine.close()

    return seqs
