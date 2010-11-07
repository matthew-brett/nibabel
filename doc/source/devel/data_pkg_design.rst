.. _data-package-design:

Design of data packages for the nipy suite
==========================================

When developing or using nipy, many data files can be useful. We divide
the data files nipy uses into four vague categories

#. *small test data* - data files required for routine code testing
#. *large test data* - larger data files for optional tests
#. *template data* - data files required for algorithms to function,
   such as templates or atlases
#. *example data* - data files for running examples, or optional tests

Files used for routine testing are typically very small data files. They are
shipped with the software, and live in the code repository. For example, in the
case of ``nipy`` itself, there are some test files that live in the module path
``nipy.testing.data``.

*large test data*, *template data* and *example data* are example of *data
packages*.  What follows is a discussion of the design and use of data packages.

The general idea
++++++++++++++++

Data packages are just directories containing files.  The interface here is just
that data pacakges have a:

#. *name*  A unique string identifying the package
#. *path*  The file path where the package directory can be found

There is no versioning to data packages, other than that you might imply in the
package *name*.  For example, we imagine that the ``0.3`` version of
``nipy-data`` will be called ``nipy-data-0.3``.

Then we need to work out how a program *using* a data package with ``name`` can
find the path associated with ``name``.  Similarly we need to define how to
install a data package so that it can be found.

Using the data package
``````````````````````

The programmer may want to use the data by *name*:

.. testcode::

   from nibabel.data import make_datasource

   templates = make_datasource('nipy-templates-0.3')
   fname = templates.get_filename('ICBM152', '2mm', 'T1.nii.gz')

where ``fname`` will be the absolute path to the template image
``ICBM152/2mm/T1.nii.gz``.

If the repository cannot find the data, then:

>>> make_datasource('nipy-implausible')
Traceback
 ...
nibabel.data.DataError

where ``DataError`` gives a helpful warning about why the data was not
found, and how it should be installed.

Warnings during installation
````````````````````````````

The example data and template data may be important, and it would be
useful to warn the user if NIPY cannot find either of the two sets of
data when installing the package.  Thus::

   python setup.py install

will import nipy after installation to check whether these raise an error:

>>> from nibabel.data import make_datasource
>>> template = make_datasource('nipy-templates-0.2')
>>> example_data = make_datasource('nipy-data-0.3')

and warn the user accordingly, with some basic instructions for how to
install the data.

.. _find-data:

Finding the data
````````````````

The routine ``make_datasource`` will need to be able to find the data
that has been installed.  Thus ``make_datasource`` needs to be able to get the
package *path* given the package *name*.  When we call this:

>>> templates = make_datasource('nipy-templates-0.2')

``make_datasource`` needs to be able to find a path for ``nipy-templates-0.2``.

For this we need somewhere to store and find the name / path pairs.  We store
these in ini-type files.  The ini files have a section ``PACKAGE PATHS`` with
name / path pairs like this::

    [PACKAGE PATHS]
    nipy-data-0.3 = /home/me/data/nipy-data-0.3
    nipy-templates-0.2 = /usr/local/share/data/nipy/nipy-templates-0.2

We search for ini files in the following places:

#. File named by the ``NIPY_PKG_INI`` environment variable, if set
#. Files matching "*.pkgpth" in the nipy home directory (the result of
   ``get_nipy_user_dir()``)
#. Files matching "*.pkgpth" in the *nipy system directory* where the *nipy system
   directory* is ``/etc/nipy`` on Unix, and ``C:\etc\nipy`` on Windows.

We search these in reverse order, that is, any package name / path pairs found in a file pointed
to by the ``NIPY_PKG_INI`` environment variable will overwrite name / path pairs
with the same name in the nipy user directory or the nipy system directory.

Requirements for a data package
```````````````````````````````

To be a valid NIPY project data package, you only need to provide a directory
tree containing the data, and add the ``my-package = /some/path`` name / path
indictor in one of the ini files above.

We recommend that you install data in a standard location such as
``<prefix>/share/nipy`` where ``<prefix>`` is the standard Python prefix
obtained by ``>>> import sys; print sys.prefix``

Installing data is just unpacking an archive into a directory, and then pointing
the ini files (above) at the resulting directory.

Remember that there is a distinction between the NIPY project - the
umbrella of neuroimaging in python - and the NIPY package - the main
code package in the NIPY project.  Thus, if you want to install data
under the NIPY *package* umbrella, your data might go to
``/usr/share/nipy/nipy/my-package`` (on Unix).  Note ``nipy`` twice -
once for the project, once for the package.  If you want to install data
under - say - the ``pbrain`` package umbrella, that would go in
``/usr/share/nipy/pbrain/my-package``.

Current implementation
``````````````````````

This section describes how we (the nipy community) implement data packages at
the moment.

The data in the data packages will not usually be under source control.  This is
because images don't compress very well, and any change in the data will result
in a large extra storage cost in the repository.  If you're pretty clear that
the data files aren't going to change, then a repository could work OK.

The data packages will be available at a central release location.  For
now this will be: http://nipy.sourceforge.net/data-packages/ .

A package, such as ``nipy-templates-0.2.tar.gz`` will have the following
sort of structure::

  <ROOT>
    `-- nipy-templates-0.2
        |-- README.txt
        |-- COPYING.txt
        |-- ICBM152
        |   |-- 1mm
        |   |   `-- T1_brain.nii.gz
        |   `-- 2mm
        |       `-- T1.nii.gz
        |-- colin27
        |   `-- 2mm
        |       `-- T1.nii.gz
        `-- config.ini

Where the ``config.ini`` has optional metadata.

Making a new package tarball is simply:

#. Downloading and unpacking e.g ``nipy-templates-0.2.tar.gz`` to form
   the directory structure above.
#. Making any changes to the directory
#. Packing up the directory with (e.g.) ``tar cvf nipy-templates-0.2.tar.gz *``

The process of making a release should be:

#. Increment the major or minor version number in the package name by renaming
   the directory - e.g from ``nipy-templates-0.2`` to ``nipy-templates-0.3``.
#. Make a package tarball as above - with the new tarball name - e.g
   ``nipy-templates-0.3.tar.gz``.
#. Upload to distribution site

