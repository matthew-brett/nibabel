#################
Images and memory
#################

Nibabel images are the association of three things:

* An *array* of image data;
* An *affine* expressing the relationship between array coordinates and an
  RAS+ reference space (see :doc:`coordinate_systems`);
* Metadata (data about the image).

The affine and header are attached ``affine`` and ``header`` properties of the
image respectively.

The array storage is a little more complicated, because the image array can
be an in-memory numpy array or it can be an *array proxy*.  The next section
describes the simpler case, of the in-memory array.

*******************************
Array data from in-memory array
*******************************

If you create an image from an array in memory, the image data is just this
array.  The array data is attached to the ``dataobj`` property of the image:

>>> import numpy as np
>>> import nibabel as nib
>>> data = np.arange(24).reshape((2, 3, 4))
>>> affine = np.diag([1, 2, 3, 1])
>>> in_memory_img = nib.Nifti1Image(data, affine)
>>> type(in_memory_img.dataobj)
<type 'numpy.ndarray'>
>>> in_memory_img.dataobj is data
True

*********************
Array data from files
*********************

Sometimes the image array data is large, so we may want to defer loading the
image array data into memory until we need it.

In fact, this is what most nibabel image types do; when you load an image from
the filesystem, you do not load the whole array into memory, and the image
initially contains only a proxy_ for the image data - that is, something that
can load the array data on request.

In this case the ``dataobj`` property points to a proxy to the array data;
something that can return the array, but it not the array:

An image where ``dataobj`` is a proxy object is a proxy image.

We load an example image from file:

>>> import os
>>> from nibabel.testing import data_path
>>> fname = os.path.join(data_path, 'example4d.nii.gz')
>>> proxy_img = nib.load(fname)
>>> type(proxy_img.dataobj)
<class 'nibabel.arrayproxy.ArrayProxy'>

You can also test if the image has a array proxy like this:

>>> nib.is_proxy(proxy_img.dataobj)
True

The proxy will fetch the data from disk when we ask it to.  We ask it to
return the data by passing it to the numpy ``asarray`` function:

>>> data_array = np.asarray(proxy_img.dataobj)
>>> type(data_array)
<type 'numpy.ndarray'>

***********************************************
``get_data`` - a unified interface with caching
***********************************************

The ``get_data()`` method of images always returns the image array data as an
array:

>>> in_mem_arr = in_memory_img.get_data()
>>> type(in_mem_arr)
<type 'numpy.ndarray'>
>>> from_proxy_arr = proxy_img.get_data()
>>> type(from_proxy_arr)
<type 'numpy.ndarray'>

In the case of the in-memory image, ``get_data`` just gave us a reference to
the original array we passed in when creating the image object:

>>> in_mem_arr is data
True

For the proxy image, ``get_data`` has retrieved the array data by calling
``np.asarray`` on the ``dataobj`` proxy object.

Loading the array from disk is a relatively expensive operation, so a call to
``get_data`` on a proxy image also stores an copy of the array in a cache.
Storing the array in the cache means that, on the next call to ``get_data``
you get the cached copy instead of loading the array from disk again:

>>> from_proxy_again = proxy_img.get_data()
>>> from_proxy_again is from_proxy_arr
True

Caching the array might not be what you want; if you aren't using the array
elsewhere, then the array reference in the image cache might be the only thing
holding onto a large amount of memory, and you may not mind the cost of
reloading the array from disk.  In that case you can dump the array from the
cache after you have used it:

>>> proxy_img.uncache()

.. _proxy: http://en.wikipedia.org/wiki/Proxy_pattern

.. include:: links_names.txt
