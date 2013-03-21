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

import re

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


def parse_attributes(attr_str):
    """ Parse string of attributes in .gitattribute format

    Parameters
    ----------
    attr_str : str
        Possibly multiline string containing attribute text

    Returns
    -------
    attr_defs : dict
        dict with (key, value) pairs of (path_expr, attrs), where ``path_expr``
        is the path expression to which the attributes apply, and ``attrs`` is a
        dict with (key, value) pairs of (``attr_name`` and ``attr_value``)
    macros : dict
        dict with (key, value) pairs of (macro_name, attrs), where ``macro_name``
        is the name of the macro , and ``attrs`` is a dict with (key, value)
        pairs of (``attr_name`` and ``attr_value``)
    """
    attr_defs = {}
    macros = {}
    for line in attr_str.split('\n'):
        line = line.rstrip()
        if line == '':
            continue
        if line[0] in ('# \t'): # Comment or leading blank
            continue
        elements = re.split('\s+', line)
        if len(elements) < 2:
            continue
        is_macro = elements[0].startswith('[attr]')
        value = _parse_elements(elements[1:])
        if len(value) == 0:
            continue
        if is_macro:
            key = elements[0][6:]
            macros[key] = value
        else:
            attr_defs[elements[0]] = value
    return attr_defs, macros


def _parse_elements(elements):
    # Parse elements from an attribute line
    attrs = {}
    for element in elements:
        key, value = _parse_element(element)
        if not key is None:
            attrs[key] = value
    return attrs


_CODE_VALS = {'!': None,
             '-': False,
             '+': True}


def _parse_element(element):
    # Parse element from attribute list
    if len(element) == 0:
        return None, None
    keyval = element.split('=')
    if len(keyval) == 2:
        key, val = keyval
        if len(key) == 0 or len(val) == 0:
            return None, None
        return key, val
    if len(keyval) > 2:
        return None, None
    if element[0] in _CODE_VALS:
        code, key = element[0], element[1:]
        if len(key) == 0:
            return None, None
        return key, _CODE_VALS[code]
    return element, True


def parse_attrfile(fname):
    """ Read and parse gitattributes file with filename `fname`

    Returns `attr_defs`, `macros` as for ``parse_attributes``
    """
    with open(fname, 'rt') as fobj:
        contents = fobj.read()
    return parse_attributes(contents)
