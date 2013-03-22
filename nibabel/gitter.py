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

from os.path import join as pjoin, isfile

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


def _get_init_toplevel():
    """ Return toplevel directory, make new repo in cwd if none found
    """
    base_dir = get_toplevel()
    if not base_dir is None:
        return base_dir
    if get_git_dir() != None: # Oops - in a bare repo?
        return None
    bt('git init')
    return get_toplevel()


def write_attr_line(new_line):
    """ Write attribute line `new_line` into top level .gitattributes file

    Parameters
    ----------
    new_line : str
        line to write into top level .gitattributes file

    Returns
    -------
    changed_file : bool
        True if .gitattributes file changed, False otherwise

    Raises
    ------
    RuntimeError - if cannot fine the top level directory
    """
    base_dir = get_toplevel()
    if base_dir is None:
        raise RuntimeError("Cannot find top level directory")
    attribute_fname = pjoin(base_dir, '.gitattributes')
    if isfile(attribute_fname):
        with open(attribute_fname, 'rt') as fobj:
            attributes = fobj.readlines()
        attributes = [attr.strip() for attr in attributes]
    else:
        attributes = []
    if new_line in attributes:
        return False
    attributes.append(new_line)
    with open(attribute_fname, 'wt') as fobj:
        fobj.write('\n'.join(attributes))
    return True


def set_filter(filtername, smudge_cmd=None, clean_cmd=None):
    """ Set filter `filtername` smudge and clean to repository config

    Parameters
    ----------
    filtername : str
        Name of filter
    smudge_cmd : None or str, optional
        command line of program to smudge
    clean_cmd : None or str
        command line of program to smudge
    """
    # Set config
    config = get_config()
    if config.get('filter.%s.smudge' % filtername):
        bt('git config --unset-all filter.%s.smudge' % filtername)
    if not smudge_cmd is None:
        bt('git config filter.%s.smudge %s' % (filtername, smudge_cmd))
    if config.get('filter.%s.clean' % filtername):
        bt('git config --unset-all filter.%s.clean' % filtername)
    if not clean_cmd is None:
        bt('git config filter.%s.clean %s' % (filtername, clean_cmd))
