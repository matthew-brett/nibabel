# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Test utility functions for interacting with the system
"""

from ..sysutils import bt, bt_or

from nose.tools import assert_true, assert_equal, assert_raises

import re


def test_bt():
    # Test backtick utility
    assert_equal(bt("echo Nice!"), 'Nice!')
    assert_true(re.match('Nice![\n\r]+', bt('echo Nice!', strip=False)))
    assert_raises(RuntimeError, bt, "unlikely_to_be_a_command")
    # bt_or
    assert_equal(bt_or("echo Nice!", 10), 'Nice!')
    assert_true(re.match('Nice![\n\r]+', bt_or('echo Nice!', 10, strip=False)))
    assert_equal(bt_or("unlikely_to_be_a_command"), None)
    assert_equal(bt_or("unlikely_to_be_a_command", 10), 10)
    assert_equal(bt_or("unlikely_to_be_a_command", 'Nasty!'), 'Nasty!')
