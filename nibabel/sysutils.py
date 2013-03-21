# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Utility functions for interacting with the system
"""

from subprocess import Popen, PIPE

NEED_SHELL = True

def bt(cmd_str, strip=True):
    r""" Backtick equivalent; run `cmd_str`, return stdout or raise error

    Parameters
    ----------
    cmd_str : str
        Command to run.  Can include spaces
    strip : bool, optional
        whether to strip returned stdout

    Returns
    -------
    stdout_str : str
        Result of stdout from command

    Raises
    ------
    RuntimeError - if command returns non-zero exit code

    Exmples
    -------
    >>> bt("echo Nice!").strip()
    'Nice!'
    >>> bt("unlikely_to_be_a_command") #doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    RuntimeError: ...

    >>> import re # windows / unix line endings
    >>> res = bt("echo Nice!", strip=False)
    >>> re.match('Nice![\r\n]+', res) is not None
    True
    """
    proc = Popen(cmd_str,
                 stdout = PIPE,
                 stderr = PIPE,
                 shell = NEED_SHELL)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError('Error running %s\n%s' % (cmd_str, stderr))
    if strip:
        return stdout.strip()
    return stdout


def bt_or(cmd_str, otherwise=None, strip=True):
    r""" Return stdout of `cmd_str` or `otherwise` if `cmd_str` fails

    Parameters
    ----------
    cmd_str : str
        Command to run.  Can include spaces
    otherwise : object, optional
        Value to return if running `cmd_str` returns non-zero return code
    strip : bool, optional
        whether to strip returned stdout

    Returns
    -------
    res : str or object
        Result of stdout from `cmd_str` or `otherwise` if `cmd_str` had non-zero
        return code

    Exmples
    -------
    >>> bt_or("echo Nice!", 10)
    'Nice!'

    >>> import re # windows / unix line endings
    >>> res = bt_or("echo Nice!", 10, strip=False)
    >>> re.match('Nice![\r\n]+', res) is not None
    True

    >>> bt_or("unlikely_to_be_a_command") is None
    True
    >>> bt_or("unlikely_to_be_a_command", 10)
    10
    """
    proc = Popen(cmd_str,
                 stdout = PIPE,
                 stderr = PIPE,
                 shell = NEED_SHELL)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        return otherwise
    if strip:
        return stdout.strip()
    return stdout
