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

from .gitter import (have_git, get_toplevel, get_config)

GIT_DESCRIBE_RE = re.compile("^(.+)-(\d+)-g([0-9a-f]+)$")

JSON_FNAME = 'datapackage.json'

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


def write_json(meta, fobj):
    json.dump(meta, fobj, indent=2, sort_keys=True)


def write_attribute(new_line):
    """ Write attribute line `new_line` into .gitattributes file

    Parameters
    ----------
    new_line : str
        line to write into .gitattributes file
    """
    base_dir = get_toplevel()
    if base_dir is None:
        return 1
    attribute_fname = pjoin(base_dir, '.gitattributes')
    if isfile(attribute_fname):
        with open(attribute_fname, 'rt') as fobj:
            attributes = fobj.readlines()
        attributes = [attr.strip() for attr in attributes]
    else:
        attributes = []
    if not new_line in attributes:
        attributes.append(new_line)
        with open(attribute_fname, 'wt') as fobj:
            for line in attributes:
                fobj.write(line + '\n')
        print("Wrote %s file" % attribute_fname)
    return 0


def enversion(opts, stdin=None, stdout=None):
    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout
    contents = stdin.read()
    json_contents = json.loads(contents)
    if json_contents['version'] != '':
        stdout.write(contents)
        return 0
    git_version = bt_or('git describe --long')
    if git_version is None:
        stdout.write(contents)
        return 1
    git_version = git_version.decode('ascii')
    json_contents['version'] = git_version
    write_json(json_contents, stdout)
    return 0


def deversion(opts):
    contents = sys.stdin.read()
    json_contents = json.loads(contents)
    version = json_contents['version']
    if not GIT_DESCRIBE_RE.match(version):
        sys.stdout.write(contents)
        return 0
    json_contents['version'] = ''
    write_json(json_contents, sys.stdout)
    return 0


def set_filter(opts, progname=None):
    if progname is None:
        progname = sys.argv[0]
    # Set config
    config = get_config()
    if config.get('filter.gitversion.smudge'):
        bt('git config --unset-all filter.gitversion.smudge')
    bt('git config filter.gitversion.smudge "%s enversion"' % progname)
    if config.get('filter.gitversion.clean'):
        bt('git config --unset-all filter.gitversion.clean')
    bt('git config filter.gitversion.clean "%s deversion"' % progname)
    print("Set filter")
    return 0


def init(opts):
    base_dir = get_toplevel()
    if base_dir is None:
        bt('git init')
        base_dir = get_toplevel()
    # Read meta, or use default
    json_path = pjoin(base_dir, JSON_FNAME)
    if isfile(json_path):
        with open(json_path, 'rt') as fobj:
            base_meta = json.load(fobj)
    else:
        base_meta = EMPTY_META
    meta = base_meta.copy()
    # Guess name if empty
    if meta['name'] == '':
        _, contain_name = psplit(base_dir)
        meta['name'] = contain_name
    # Fill stuff from git config
    config = get_config()
    if opts.greedy:
        # Guess author / maintainer name and email
        if meta.get('author', '') == '':
            meta['author'] = config.get('user.name', '')
        if meta.get('author_email', '') == '':
            meta['author_email'] = config.get('user.email', '')
        if meta.get('maintainer', '') == '':
            meta['maintainer'] = config.get('user.name', '')
        if meta.get('maintainer_email', '') == '':
            meta['maintainer_email'] = config.get('user.email', '')
    # Guess url
    if meta.get('download_url', '') == '':
        meta['download_url'] = config.get('remote.origin.url', '')
    # Write new meta if we changed anything
    if not meta == base_meta:
        print("Filling %s with guessed defaults" % json_path)
        with open(json_path, 'wt') as fobj:
            write_json(meta, fobj)
    # Check whether git describe works, warn about annotated tag
    if bt_or('git describe') is None:
        print("Check output of git describe; have you made an annotated "
              "tag yet (git tag -a)?")
    # write filter attributes to .gitattributes
    filter_line = "%s filter=gitversion" % JSON_FNAME
    ret = write_attribute(filter_line)
    if ret !=0:
        return ret
    # Set the filter into the local .git/config
    return set_filter(opts)


def register(args, opts, stderr=None):
    """ Register directory as nibabel data package
    """
    if stderr == None:
        stderr = sys.stderr
    if len(args) == 0:
        path = os.getcwd()
    elif len(args) == 1:
        path = args[0]
    else:
        stderr.write('Too many arguments "%s"\n' % ' '.join(args))
        return 2
    if not isfile(pjoin(path, JSON_FNAME)):
        stderr.write('No %s file at %s\n' % (JSON_FNAME, path))
        return 2


def cli():
    """ Command line interface to dpmutils
    """
    parser = optparse.OptionParser(
        usage = "usage: %prog <enversion|deversion|init|set-filter>")
    parser.add_option("-g", "--greedy", action='store_true',
                      help = "(init command) - let git author claim all "
                      "unclaimed credit")
    (opts, args) = parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    cmd = args[0]
    if (cmd in ('enversion',
                'deversion',
                'init',
                'set-filter')
        and not have_git()):
        sys.stderr.write("You need git on the path for these utilities")
        sys.exit(128)
    if cmd == 'enversion':
        sys.exit(enversion(opts))
    elif cmd == "deversion":
        sys.exit(deversion(opts))
    elif cmd == "init":
        sys.exit(init(opts))
    elif cmd == "set-filter":
        sys.exit(set_filter(opts))
    elif cmd == "register":
        sys.exit(register(args[1:], opts))
    parser.print_help()
    sys.exit(1)


