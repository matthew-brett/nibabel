# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Test utilities to work with git commands from Python
"""
import os
import shutil
from os.path import abspath, realpath, join as pjoin

from ..gitter import (have_git, get_git_dir, get_toplevel, get_config,
                      _parse_config, parse_attributes, parse_attrfile)

from ..sysutils import bt, bt_or

from nose.tools import assert_equal

import numpy.testing as npt

from ..tmpdirs import InTemporaryDirectory, InGivenDirectory

HAVE_GIT = have_git()

git_test = npt.dec.skipif(not HAVE_GIT, 'git not available on path')

def realabs(path):
    return realpath(abspath(path))


@git_test
def test_git_dir_base():
    # Test git_dir, git_base_dir
    with InTemporaryDirectory() as tmpdir:
        bt('git init another_dir')
        with InGivenDirectory('another_dir'):
            assert_equal(get_git_dir(), '.git')
            os.mkdir('subdirectory')
            with InGivenDirectory('subdirectory'):
                assert_equal(realabs(get_git_dir()),
                             realabs(pjoin(tmpdir, 'another_dir', '.git')))
                assert_equal(realabs(get_toplevel()),
                             realabs(pjoin(tmpdir, 'another_dir')))
        # Back in containing directory
        assert_equal(get_git_dir(), None)
        shutil.rmtree('another_dir')
        bt('git init --bare another_dir')
        with InGivenDirectory('another_dir'):
            assert_equal(get_git_dir(), '.')
            assert_equal(get_toplevel(), None)
        assert_equal(get_toplevel(), None)


def test__parse_config():
    # Test _parse_config (indirectly test get_config
    assert_equal(_parse_config(
"""core.editor=vim
user.name=Daffy leCanard
user.email=daffylecanard@gmail.com
alias.st=status
"""),
        {'core.editor': 'vim',
         'user.name': 'Daffy leCanard',
         'user.email': 'daffylecanard@gmail.com',
         'alias.st': 'status'})


@git_test
def test_git_config():
    # Test git config directly
    with InTemporaryDirectory():
        bt('git init')
        bt('git config core.editor not_an_editor')
        bt('git config user.name "Chat qui creve"')
        config = get_config()
        assert_equal(config['core.editor'], 'not_an_editor')
        assert_equal(config['user.name'], 'Chat qui creve')


@git_test
def test_git_describe():
    # Test git describe
    with InTemporaryDirectory():
        bt('git init')
        assert_equal(bt_or('git_describe'), None)
        with open('test.txt', 'wt') as fobj:
            fobj.write('Elementary my dear Watson')
        bt('git add test.txt')
        bt('git commit -m "Interesting text"')
        assert_equal(bt_or('git describe'), None)
        bt('git tag not_annotated')
        assert_equal(bt_or('git describe'), None)
        bt('git tag -a annotated -m "Important point"')
        assert_equal(bt_or('git describe'), 'annotated')


def test_parse_attributes():
    # Parse contents of attribute file
    assert_equal(parse_attributes('test* anattr'),
                 ({'test*': {'anattr': True}}, {}))
    assert_equal(parse_attributes('test*   anattr'),
                 ({'test*': {'anattr': True}}, {}))
    assert_equal(parse_attributes('test* \t \t anattr'),
                 ({'test*': {'anattr': True}}, {}))
    assert_equal(parse_attributes('test* anattr=true'),
                 ({'test*': {'anattr': "true"}}, {}))
    assert_equal(parse_attributes('test* anattr=blah'),
                 ({'test*': {'anattr': 'blah'}}, {}))
    assert_equal(parse_attributes('test* anattr=false'),
                 ({'test*': {'anattr': "false"}}, {}))
    assert_equal(parse_attributes('test* +anattr'),
                 ({'test*': {'anattr': True}}, {}))
    assert_equal(parse_attributes('test* !anattr'),
                 ({'test*': {'anattr': None}}, {}))
    # White space at beginning of line does not work
    assert_equal(parse_attributes(' test* !anattr'),
                 ({}, {}))
    # Comments seem to be OK
    assert_equal(parse_attributes('#test* !anattr'),
                 ({}, {}))
    # Macros
    assert_equal(parse_attributes('[attr]binary -diff +text'),
                 ({}, {'binary': {'diff': False, 'text': True}}))
    # Several lines, including blanks and comments
    in_str = """[attr]binary +diff -text
ab*     merge=filfre

abc     -foo -bar
*.c     frotz

# Another macro
[attr]ducks -baz +bosh

"""
    defs, macros = parse_attributes(in_str)
    assert_equal(defs,
                 {
                     'ab*': {'merge': 'filfre'},
                     'abc': {'foo': False, 'bar': False},
                     '*.c': {'frotz': True}
                 })
    assert_equal(macros,
                 {
                     'binary': {'diff': True, 'text': False},
                     'ducks': {'baz': False, 'bosh': True},
                 })
    # Edge cases
    assert_equal(parse_attributes('test anattr='),
                 ({}, {}))
    assert_equal(parse_attributes('test =anattr'),
                 ({}, {}))
    assert_equal(parse_attributes('test anattr=blah='),
                 ({}, {}))
    assert_equal(parse_attributes('test anattr=blah=true'),
                 ({}, {}))
    assert_equal(parse_attributes('test !'),
                 ({}, {}))
    assert_equal(parse_attributes('test -'),
                 ({}, {}))
    assert_equal(parse_attributes('test +'),
                 ({}, {}))


def test_parse_attrfile():
    # Test parsing of attribute file
    with InTemporaryDirectory():
        in_str = """[attr]binary +diff -text
ab*     merge=filfre

abc     -foo -bar
*.c     frotz

# Another macro
[attr]ducks -baz +bosh

"""
        with open('.gitattributes', 'wt') as fobj:
            fobj.write(in_str)
        defs, macros = parse_attrfile('.gitattributes')
        assert_equal(defs,
                    {
                        'ab*': {'merge': 'filfre'},
                        'abc': {'foo': False, 'bar': False},
                        '*.c': {'frotz': True}
                    })
        assert_equal(macros,
                    {
                        'binary': {'diff': True, 'text': False},
                        'ducks': {'baz': False, 'bosh': True},
                    })


@git_test
def test_write_attribute():
    # Test writing of git attributes
    with InTemporaryDirectory():
        bt('git init adir')
