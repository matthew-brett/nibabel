# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os
import shutil
from os.path import abspath, realpath, join as pjoin

from ..sysutils import bt
from ..gitter import have_git

from nose.tools import assert_equal

import numpy.testing as npt

from ..tmpdirs import InTemporaryDirectory, InGivenDirectory

HAVE_GIT = have_git()

