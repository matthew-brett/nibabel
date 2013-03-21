# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Utilities to work with git commands from Python

Uses very simple pipe wrapper to call git commands.
"""

from .sysutils import bt, bt_or


def have_git():
    """ Return True if we can run ``git``
    """
    return not bt_or('git --version') is None


def get_git_dir():
    """ Return git directory or None if not found
    """
    return bt_or('git rev-parse --git-dir')


def get_toplevel():
    """ Return working tree directory containing git dir or None

    Returns
    -------
    toplevel_dir : None or str
        Return None if git directory not found, or the directory below the git
        directory is not in the working tree. Otherwise return the path of the
        directory below the git directory.
    """
    top_level = bt_or('git rev-parse --show-toplevel')
    if top_level in ('', None):
        return None
    return top_level


def _parse_config(config_str):
    config = {}
    for line in config_str.split('\n'):
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=', 1)
        config[key] = value
    return config


def get_config():
    """ Parse git config, return as a dictionary
    """
    return _parse_config(bt('git config -l'))
