# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Commnds for interacting with data package management
"""
import os
from os.path import join as pjoin, isfile, split as psplit
import sys
import json
import re
import optparse

from .sysutils import bt, bt_or

from .gitter import (have_git, get_git_dir, get_config, write_attr_line,
                     get_toplevel, set_filter)

GIT_DESCRIBE_RE = re.compile("^(.+)-(\d+)-g([0-9a-f]+)$")

JSON_FNAME = 'datapackage.json'

FILTER_NAME = 'gitversion'

EMPTY_META = {
  "author": "",
  "author_email": "",
  "download_url": "",
  "extras": {},
  "id": "",
  "license": "",
  "maintainer": "",
  "maintainer_email": "",
  "name": "",
  "notes": "",
  "relationships": "",
  "resources": [],
  "tags": [],
  "title": "",
  "url": "",
  "version": ""
}


def write_dpm_meta(meta, fobj):
    """ Write `meta` in dpm favored JSON format into file-like `fobj`

    Parameters
    ----------
    meta : dict
        dictionary to write to JSON format
    fobj : file-like
        stream to write to
    """
    json.dump(meta, fobj, indent=2, sort_keys=True)


def enversion(stdin=None, stdout=None):
    """ Add ``git describe`` version to stream `stdin`, write to `stdout`

    Parameters
    ----------
    stdin : None or file-like, optional
        Stream to read JSON string data from.  Default to `sys.stdin`.
    stdout : None or file-like, optional
        Stream to write changed JSON string data to.  Default to `sys.stdout`.

    Returns
    -------
    retcode : int
        0 if successful, non-zero if failed
    """
    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout
    contents = stdin.read()
    json_contents = json.loads(contents)
    if json_contents.get('version', '') != '':
        stdout.write(contents)
        return 0
    git_version = bt_or('git describe --long')
    if git_version is None:
        stdout.write(contents)
        return 1
    git_version = git_version.decode('ascii')
    json_contents['version'] = git_version
    write_dpm_meta(json_contents, stdout)
    return 0


def deversion(stdin=None, stdout=None):
    """ Remove ``git describe`` version from stream `stdin`, write to `stdout`

    Parameters
    ----------
    stdin : None or file-like, optional
        Stream to read JSON string data from.  Default to `sys.stdin`.
    stdout : None or file-like, optional
        Stream to write changed JSON string data to.  Default to `sys.stdout`.

    Returns
    -------
    retcode : int
        0 if successful, non-zero if failed
    """
    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout
    contents = stdin.read()
    json_contents = json.loads(contents)
    version = json_contents.get('version', '')
    if not GIT_DESCRIBE_RE.match(version):
        stdout.write(contents)
        return 0
    json_contents['version'] = ''
    write_dpm_meta(json_contents, stdout)
    return 0


def _get_init_toplevel():
    """ Return toplevel directory, make new repo in cwd if none found
    """
    base_dir = get_toplevel()
    if not base_dir is None:
        return base_dir
    if get_git_dir() != None: # Oops - in a bare repo?
        return None
    print("Initializing new git repo in working directory")
    bt('git init')
    return get_toplevel()


def get_meta(json_path, defaults=None):
    """ Get metadata from file at `json_path` or defaults `defaults`

    Parameters
    ----------
    json_path : str
        filename of JSON file from which to load metadata
    defaults : None or dict, optional
        dict containing default values. If None, defaults to module level
        EMPTY_META

    Returns
    -------
    meta : dict
        dict containing metadata, seeded from `defaults` and filled (updated)
        from any information in `json_path`.  Can be modified without changing
        original defaults.
    """
    if defaults is None:
        defaults = EMPTY_META
    meta = defaults.copy()
    if not isfile(json_path):
        return meta
    with open(json_path, 'rt') as fobj:
        existing_meta = json.load(fobj)
    meta.update(existing_meta)
    return meta


def _guess_name(meta, base_dir):
    if meta.get('name', '') != '':
        return
    _, contain_name = psplit(base_dir)
    print('Guessed package name as "%s"' % contain_name)
    meta['name'] = contain_name
    return


def _claim_credit(meta, config):
    # Guess author / maintainer name and email
    print("Trying to claim credit as author / maintainer")
    for meta_key, config_key in (('author', 'user.name'),
                                 ('author_email', 'user.email'),
                                 ('maintainer', 'user.name'),
                                 ('maintainer_email', 'user.email')):
        if meta.get(meta_key, '') == '':
            meta[meta_key] = config.get(config_key, '')


def _guess_urls(meta, config):
    # Guess url
    origin_url = config.get('remote.origin.url', '')
    if origin_url == '':
        return
    if meta.get('download_url', '') == '':
        meta['download_url'] = origin_url
    if meta.get('url', '') == '':
        meta['url'] = origin_url


def write_meta(meta, json_path):
    """ Write metadata dictionary `meta` to JSON as filename `json_path`

    Parameters
    ----------
    meta : dict
        metadata for package
    json_path : str
        filename to which to write
    """
    with open(json_path, 'wt') as fobj:
        write_dpm_meta(meta, fobj)
    return


def _check_describe():
    # Check whether git describe works, warn about annotated tag
    if bt_or('git describe') is None:
        print("Check output of git describe; have you made an annotated "
              "tag yet (git tag -a)?")


def _write_attr(json_fname, filter_name):
    # write filter attributes to .gitattributes
    filter_line = "%s filter=%s" % (json_fname, filter_name)
    if write_attr_line(filter_line):
        print("Written new .gitattributes file")


def set_version_filter(filter_name, cmd_name, stderr=None):
    """ Set filter `filtername` with command `cmd_name` into local .git/config

    Parameters
    ----------
    filter_name : str
        name for the filter
    cmd_name : str
        command taking sub-commands ``enversion`` and ``deversion`` used for
        smudge and clean respectively
    stderr : None or file-like
        Stream for standard error.  None means use ``sys.stderr``

    Returns
    -------
    ret_code : int
        0 if successful, 1 otherwise
    """
    if stderr is None:
        stderr = sys.stderr
    if get_toplevel() is None:
        stderr.write(
            'Cannot find git top level for working directory here\n')
        return 1
    set_filter(filter_name,
               '"%s enversion"' % cmd_name,
               '"%s deversion"' % cmd_name)
    print("Set filter to insert / remove git version from JSON file")
    return 0


def _commit_tag(base_dir, json_fname, version):
    """ Commit `json_fname` and ``.gitattributes``; add annotated tag `version`
    """
    json_path = pjoin(base_dir, json_fname)
    git_attrs = pjoin(base_dir, '.gitattributes')
    if isfile(json_path):
        bt('git add %s' % json_path)
    if isfile(git_attrs):
        bt('git add ' + git_attrs)
    bt('git commit -m "AUTO: add git dpm setup"')
    bt('git tag -a %s -m "Add version tag %s"' % (version, version))
    os.unlink(json_path)
    # Checkout json file to apply filters
    bt('git checkout ' + json_path)
    print("Committed git setup and added annotated version tag")


def git_init(cmd_name, version=None, greedy=False):
    """ Initialize a directory with setup for git dpm

    Parameters
    ----------
    cmd_name : str
        Command name for in/out git filter
    version : None or str, optional
        Version to set.  If not None, commit datapackage.json and add an
        annotated tag with this version.
    greedy : bool, optional
        Whether to give credit to current git user if credit not yet given.

    Returns
    -------
    ret_code : int
        0 if successful, non-zero otherwise
    """
    base_dir = _get_init_toplevel()
    if base_dir is None:
        print("Oops - is this a bare repo?")
        return 1
    json_path = pjoin(base_dir, JSON_FNAME)
    base_meta = get_meta(json_path, EMPTY_META)
    meta = base_meta.copy() # to compare against later
    _guess_name(meta, base_dir)
    config = get_config()
    if greedy:
        _claim_credit(meta, config)
    _guess_urls(meta, config)
    if not meta == base_meta:
        write_meta(meta, json_path)
        print("Written metadata to " + json_path)
    _write_attr(JSON_FNAME, FILTER_NAME)
    set_version_filter(FILTER_NAME, cmd_name)
    if not version is None and not version == '':
        _commit_tag(base_dir, JSON_FNAME, version)
    else:
        _check_describe()
    return 0
