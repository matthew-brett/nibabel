# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Tests for dpm utilties
"""
from ..py3k import StringIO

import os
from os.path import (join as pjoin, isfile, isdir, splitdrive)
import json

from ..dpmutils import (write_dpm_meta, enversion, deversion,
                        set_version_filter, _get_init_toplevel, git_init,
                        get_meta, write_meta, EMPTY_META, _guess_name,
                        _guess_urls)
from ..gitter import get_config
from ..sysutils import bt

from nose.tools import assert_equal, assert_true, assert_false

from ..tmpdirs import InTemporaryDirectory, InGivenDirectory
from .test_gitter import git_test, realabs


def test_write_dpm_meta():
    # Writing JSON in dpm preferred format
    meta = {'name': 'daffy',
            'author': 'daffo',
            'email': 'daffum'}
    sio = StringIO()
    write_dpm_meta(meta, sio)
    assert_equal(sio.getvalue(), # leave the whitespace!
"""{
  "author": "daffo", 
  "email": "daffum", 
  "name": "daffy"
}""")
    assert_equal(json.loads(sio.getvalue()), meta)


@git_test
def test_get_init_toplevel():
    # Test routine to get toplevel or init repo here to get toplevel
    with InTemporaryDirectory():
        assert_false(isdir('.git'))
        assert_equal(realabs('.'), realabs(_get_init_toplevel()))
        assert_true(isdir('.git'))
        os.mkdir('another_dir')
        with InGivenDirectory('another_dir'):
            assert_equal(realabs('..'), realabs(_get_init_toplevel()))
            assert_false(isdir('.git'))
    with InTemporaryDirectory():
        bt('git init --bare arepo')
        with InGivenDirectory('arepo'):
            assert_equal(_get_init_toplevel(), None)


def test__guess_name():
    # Test guess of name from directory
    a_drive, _ = splitdrive(realabs('.'))
    meta = {}
    _guess_name(meta, pjoin(a_drive, 'path1'))
    assert_equal(meta['name'], 'path1')
    # Name already set, keep it
    _guess_name(meta, pjoin(a_drive, 'path2'))
    assert_equal(meta['name'], 'path1')
    meta = {}
    _guess_name(meta, pjoin(a_drive, 'path2'))
    assert_equal(meta['name'], 'path2')
    meta = {}
    _guess_name(meta, 'path2')
    assert_equal(meta['name'], 'path2')
    meta = {}
    _guess_name(meta, '')
    assert_equal(meta['name'], '')
    meta = {}
    _guess_name(meta, a_drive)
    assert_equal(meta['name'], '')


def test__guess_urls():
    # Test algorithm for guessing URLs
    meta = {}
    config = {}
    _guess_urls(meta, config)
    assert_equal(meta, {})
    config['remote.somewhere.url'] = 'http://somewhere.on.net/repo'
    _guess_urls(meta, config)
    assert_equal(meta, {})
    url = 'http://another.place.on.net/here'
    config['remote.origin.url'] = url
    _guess_urls(meta, config)
    assert_equal(meta['download_url'], url)
    assert_equal(meta['url'], url)


def _json_proc(meta):
    # Replicate unicoding by json system.  Otherwise python 2 and python 3 can
    # disagree about bytes, unicode
    return json.loads(json.dumps(meta))


@git_test
def test_en_de_version():
    # Test filter to put version into file and remove it again
    meta = dict(name = 'donald')
    def _run_ver(meta, func):
        in_stream = StringIO(json.dumps(meta))
        out_stream = StringIO()
        res = func(in_stream, out_stream)
        return res, json.loads(out_stream.getvalue())
    def _add_ver(meta):
        return _run_ver(meta, enversion)
    def _rm_ver(meta):
        return _run_ver(meta, deversion)
    with InTemporaryDirectory():
        # No git - no git describe
        assert_equal(_add_ver({'name': 'donald'}),
                     (1, _json_proc(meta)))
        bt('git init')
        # No tags - no git describe result
        assert_equal(_add_ver({'name': 'donald'}),
                     (1, _json_proc(meta)))
        # Removing is still happy
        assert_equal(_rm_ver({'name': 'donald'}),
                     (0, _json_proc(meta)))
        # Add commit and tag
        with open('test', 'wt') as fobj:
            fobj.write('Some text')
        bt('git add test')
        bt('git commit -m "Some text"')
        bt('git tag -a first_tag -m "The first tag"')
        tag_hash = bt('git show-ref --hash=7 master').decode('ascii')
        expected_version = 'first_tag-0-g' + tag_hash
        # Adding now works
        code, back_meta = _add_ver({'name': 'donald'})
        assert_equal(code, 0)
        assert_equal(back_meta,
                     _json_proc({'name': 'donald',
                                 'version': expected_version}))
        # Removing reverses
        assert_equal(_rm_ver(back_meta),
                     (0, _json_proc({'name': 'donald', 'version': ''})))
        # Another commit
        with open('test', 'wt') as fobj:
            fobj.write('Some more text')
        bt('git add test')
        bt('git commit -m "Some text"')
        # Different describe
        tag_hash = bt('git show-ref --hash=7 master').decode('ascii')
        expected_version = 'first_tag-1-g' + tag_hash
        code, back_meta = _add_ver({'name': 'donald'})
        assert_equal(code, 0)
        assert_equal(back_meta,
                     _json_proc({'name': 'donald',
                                 'version': expected_version}))
        # Removing reverses
        assert_equal(_rm_ver(back_meta),
                     (0, _json_proc({'name': 'donald', 'version': ''})))
        # Already a version - adding does nothing
        meta_with = {'name': 'donald', 'version': '0.0.1'}
        assert_equal(_add_ver(meta_with),
                     (0, _json_proc(meta_with)))
        # Nor does removing
        assert_equal(_rm_ver(meta_with),
                     (0, _json_proc(meta_with)))


@git_test
def test_set_version_filter():
    # Test setting of filter into config
    orig_config = get_config()
    sio = StringIO()
    with InTemporaryDirectory():
        # Can't set config if not in git directory
        assert_equal(set_version_filter('a_filter', 'a_cmd'), 1)
        # Check stderr stream output
        assert_equal(set_version_filter('a_filter', 'a_cmd', sio), 1)
        assert_equal(sio.getvalue(),
                     'Cannot find git top level for working directory here\n')
        # Make a repo and retest
        bt('git init')
        # Now we can set config
        assert_equal(set_version_filter('a_filter', 'a_cmd'), 0)
        new_config = get_config()
        assert_equal(new_config['filter.a_filter.smudge'],
                     'a_cmd enversion')
        assert_equal(new_config['filter.a_filter.clean'],
                     'a_cmd deversion')
    # Back to normality when out of git directory
    assert_equal(get_config(), orig_config)


def test_get_write_meta():
    # Test routine to load json file and fill into defaults
    with InTemporaryDirectory():
        assert_equal(get_meta('afile', None), EMPTY_META)
        def_meta = _json_proc(dict(name='', version='', other=''))
        assert_equal(get_meta('afile', def_meta), def_meta)
        new_meta = dict(name='I. M. Awesome')
        write_meta(new_meta, 'afile')
        exp_meta = def_meta.copy()
        exp_meta.update(new_meta)
        assert_equal(get_meta('afile', def_meta), _json_proc(exp_meta))
        assert_equal(def_meta['name'], '')
        new_meta['left field'] = 'this is not a ball'
        write_meta(new_meta, 'afile')
        exp_meta['left field'] = 'this is not a ball'
        assert_equal(get_meta('afile', def_meta), _json_proc(exp_meta))
        assert_false('left field' in def_meta)


@git_test
def test_init():
    # Test whole init process
    def _read_meta():
        with open('datapackage.json', 'rt') as fobj:
            meta = json.load(fobj)
        return meta
    with InTemporaryDirectory():
        # Check fails in bare repo
        bt('git init --bare bare_repo')
        with InGivenDirectory('bare_repo'):
            assert_equal(git_init('my_command'), 1)
            assert_false(isfile('datapackage.json'))
        # Works in a non-bare repo
        os.mkdir('my_package')
        with InGivenDirectory('my_package'):
            assert_equal(git_init('my_command'), 0)
            assert_true(isdir('.git'))
            config = get_config()
            assert_equal(config['filter.gitversion.smudge'],
                        'my_command enversion')
            assert_equal(config['filter.gitversion.clean'],
                        'my_command deversion')
            meta = _read_meta()
            assert_equal(meta['name'], 'my_package')
            for key in ('author', 'author_email',
                        'maintainer',
                        'maintainer_email',
                        'url',
                        'download_url'):
                assert_equal(meta[key], '')
            # Rerun after adding a non-origin remote
            bt('git remote add some-remote http://some/repo.git')
            assert_equal(git_init('my_command'), 0)
            meta = _read_meta()
            assert_equal(meta['download_url'], '')
            assert_equal(meta['url'], '')
            # After adding origin remote
            bt('git remote add origin http://another/repo.git')
            assert_equal(git_init('my_command'), 0)
            meta = _read_meta()
            assert_equal(meta['download_url'], 'http://another/repo.git')
            assert_equal(meta['url'], 'http://another/repo.git')
            # Do greedy
            bt('git config user.name "Daffy LeCanard"')
            bt('git config user.email daffylecanard@yahoo.com')
            reconfig = _json_proc({'name': 'Daffy LeCanard',
                                   'email': 'daffylecanard@yahoo.com'})
            assert_equal(git_init('my_command', greedy=True), 0)
            meta = _read_meta()
            for key in ('author', 'maintainer'):
                assert_equal(meta[key], reconfig['name'])
            for key in ('author_email', 'maintainer_email'):
                assert_equal(meta[key], reconfig['email'])
            # Put real values in there and make sure they don't get overwritten
            os.unlink('datapackage.json')
            new_meta = {
                "author": "Pato Loco",
                "author_email": "patoloco@mymail.org",
                "url": "http://patosblancos.org",
                "download_url": "http://patosblancos.org/download",
                "maintainer": "Donald",
                "maintainer_email": "someduck@ducksville.com",
                "version": "3.1415etc",
                "name": "a_sensible_name"}
            full_meta = EMPTY_META.copy()
            full_meta.update(new_meta)
            with open('datapackage.json', 'wt') as fobj:
                write_dpm_meta(full_meta, fobj)
            assert_equal(git_init('my_command', greedy=True), 0)
            assert_equal(_read_meta(), _json_proc(full_meta))
