#!/usr/bin/env python
""" Utility for git-bisecting nose failures
"""
DESCRIP = 'Check nose output for given text, set sys exit for git bisect'
EPILOG = \
"""
Imagine you've just detected a nose test failure.  The failure is in a
particular test or test module - here 'test_analyze.py'.  The failure *is* in
git branch ``main-master`` but it *is not* in tag ``v1.6.1``. Then you can
bisect with something like::

    git co main-master
    git bisect start HEAD v1.6.1 --
    git bisect run /path/to/bisect_nose.py nibabel/tests/test_analyze.py:TestAnalyzeImage.test_str

You might well want to test that::

    nosetests nibabel/tests/test_analyze.py:TestAnalyzeImage.test_str

works as you expect first.

Let's say instead that you prefer to recognize the failure with an output
string.  Maybe this is because there are lots of errors but you are only
interested in one of them, or because you are looking for a Segmentation fault
instead of a test failure. Then::

    git co main-master
    git bisect start HEAD v1.6.1 --
    git bisect run /path/to/bisect_nose.py --error-txt='HeaderDataError: data dtype "int64" not recognized'  nibabel/tests/test_analyze.py

where ``error-txt`` is in fact a regular expression.

You will need 'argparse' installed somewhere. This is in the system libraries
for python 2.7 and python 3.2 onwards.

We run the tests in a temporary directory, so the code you are testing must be
on the python path.

When debugging, you can run this script and echo the return value in bash with
``echo $?``.

Here is an actual example from a complicated numpy virtualenv run::

    ~/dev_trees/nibabel/tools/bisect_nose.py ~/dev_trees/numpy/numpy/lib/tests/test_io.py --build-cmd="rm -rf $VIRTUAL_ENV/lib/python2.7/site-packages/numpy; python setup.py install" --clean-before --clean-after --error-txt="Bus error"
"""
import os
import sys
import shutil
import tempfile
import re
from functools import partial
from subprocess import check_call, Popen, PIPE, CalledProcessError

from argparse import ArgumentParser, RawDescriptionHelpFormatter

caller = partial(check_call, shell=True)
popener = partial(Popen, stdout=PIPE, stderr=PIPE, shell=True)

# git bisect exit codes
UNTESTABLE = 125
GOOD = 0
BAD = 1

def call_or_untestable(cmd):
    try:
        caller(cmd)
    except CalledProcessError:
        sys.exit(UNTESTABLE)


def main():
    parser = ArgumentParser(description=DESCRIP,
                            epilog=EPILOG,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('test_path',  type=str,
                        help='Path to test')
    parser.add_argument('--error-txt', type=str,
                        help='regular expression for error of interest')
    parser.add_argument('--error-untest', action='store_true',
                        help='Commit is untestable if returns non-zero exit '
                        'code; (ignored without ``error-txt`` option)')
    parser.add_argument('--clean-before', action='store_true',
                        help='Clean git tree before running tests')
    parser.add_argument('--clean-after', action='store_true',
                        help='Clean git tree after running tests')
    parser.add_argument('--build-cmd', type=str,
                        help='Command to build package')
    # parse the command line
    args = parser.parse_args()
    path = os.path.abspath(args.test_path)
    if not args.error_txt and args.error_untest:
        raise RuntimeError('ERROR_UNTEST option only used with ERROR_TXT')
    if args.clean_before:
        call_or_untestable('git clean -fxd')
    if args.build_cmd:
        print "Building"
        try:
            caller(args.build_cmd)
        except CalledProcessError:
            if args.clean_after:
                call_or_untestable('git clean -fxd')
                call_or_untestable('git reset --hard')
            sys.exit(UNTESTABLE)
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        os.chdir(tmpdir)
        print "Testing"
        proc = popener('nosetests ' + path)
        stdout, stderr = proc.communicate()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)
    if args.clean_after: # to make certain a checkout can work
        call_or_untestable('git clean -fxd')
        call_or_untestable('git reset --hard')
    if args.error_txt:
        regex = re.compile(args.error_txt)
        if regex.search(stderr): # the error we were expecting
            sys.exit(BAD)
        if args.error_untest and proc.returncode != 0: # an unexpected error
            sys.exit(UNTESTABLE)
        sys.exit(GOOD) # no error
    sys.exit(proc.returncode)


if __name__ == '__main__':
    main()
