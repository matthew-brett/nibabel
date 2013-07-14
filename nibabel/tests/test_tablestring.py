""" Test printable table
"""
from __future__ import division, print_function

import numpy as np

from ..tablestring import printable_table

from nose.tools import assert_equal, assert_raises

def test_printable_table():
    # Tests for printable table function
    R, C = 3, 4
    cell_values = np.arange(R * C).reshape((R, C))
    assert_equal(printable_table(cell_values),
"""============================================
       | col[0] | col[1] | col[2] | col[3] |
--------------------------------------------
row[0] |   0.00 |   1.00 |   2.00 |   3.00 |
row[1] |   4.00 |   5.00 |   6.00 |   7.00 |
row[2] |   8.00 |   9.00 |  10.00 |  11.00 |
--------------------------------------------""")
    assert_equal(printable_table(cell_values, ['a', 'b', 'c']),
"""=======================================
  | col[0] | col[1] | col[2] | col[3] |
---------------------------------------
a |   0.00 |   1.00 |   2.00 |   3.00 |
b |   4.00 |   5.00 |   6.00 |   7.00 |
c |   8.00 |   9.00 |  10.00 |  11.00 |
---------------------------------------""")
    assert_raises(ValueError,
                  printable_table, cell_values, ['a', 'b'])
    assert_raises(ValueError,
                  printable_table, cell_values, ['a', 'b', 'c', 'd'])
    assert_equal(printable_table(cell_values, None, ['1', '2', '3', '4']),
"""========================================
       |   1   |   2   |   3   |   4   |
----------------------------------------
row[0] |  0.00 |  1.00 |  2.00 |  3.00 |
row[1] |  4.00 |  5.00 |  6.00 |  7.00 |
row[2] |  8.00 |  9.00 | 10.00 | 11.00 |
----------------------------------------""")
    assert_raises(ValueError,
                  printable_table, cell_values, None, ['1', '2', '3'])
    assert_raises(ValueError,
                  printable_table, cell_values, None, list('12345'))
    assert_equal(printable_table(cell_values, title='A title'),
"""============================================
A title
============================================
       | col[0] | col[1] | col[2] | col[3] |
--------------------------------------------
row[0] |   0.00 |   1.00 |   2.00 |   3.00 |
row[1] |   4.00 |   5.00 |   6.00 |   7.00 |
row[2] |   8.00 |   9.00 |  10.00 |  11.00 |
--------------------------------------------""")
    assert_equal(printable_table(cell_values, val_fmt = '{0}'),
"""============================================
       | col[0] | col[1] | col[2] | col[3] |
--------------------------------------------
row[0] |      0 |      1 |      2 |      3 |
row[1] |      4 |      5 |      6 |      7 |
row[2] |      8 |      9 |     10 |     11 |
--------------------------------------------""")
    # Doing a fancy cell format
    cell_values_back = np.arange(R * C)[::-1].reshape((R, C))
    cell_3d = np.dstack((cell_values, cell_values_back))
    assert_equal(printable_table(cell_3d, val_fmt = '{0[0]}-{0[1]}'),
"""============================================
       | col[0] | col[1] | col[2] | col[3] |
--------------------------------------------
row[0] |   0-11 |   1-10 |    2-9 |    3-8 |
row[1] |    4-7 |    5-6 |    6-5 |    7-4 |
row[2] |    8-3 |    9-2 |   10-1 |   11-0 |
--------------------------------------------""")
    return


def null():
    row_names = ['one', 'two', 'three']
    col_names = ['first', 'second', 'third', 'fourth']
    result_table('A title', times, row_names, col_names)
    bfs.result_table('A title', times, row_names, col_names)
    reload(bfs)
    bfs.result_table('A title', times, row_names, col_names)
    print(bfs.result_table('A title', times, row_names, col_names))
    col_names = ['first', 'second', 'third', 'fourth']
    print(bfs.result_table('A title', times, row_names, col_names))
    times2 = np.random.normal(size=(3, 4, 2))
    val_fmt = '{0[0]:3.2f} ({1[1]:3.2f}'
    print(bfs.result_table('A title', times2, row_names, col_names, val_fmt))
    history
    val_fmt = '{0[0]:3.2f} ({1[1]:3.2f})'
    print(bfs.result_table('A title', times2, row_names, col_names, val_fmt))
    times2[0]
    times2[0][0]
    val_fmt = '{0[0]:3.2f} ({1[1]:3.2f})'
    val_fmt = '{0[0]:3.2f} ({0[1]:3.2f})'
    print(bfs.result_table('A title', times2, row_names, col_names, val_fmt))
    print(bfs.result_table('A title', times2, row_names, col_names, val_fmt))
    reload(bfs)
    print(bfs.result_table('A title', times2, row_names, col_names, val_fmt))
    debug
    reload(bfs)
    print(bfs.result_table('A title', times2, row_names, col_names, val_fmt))
    history
    np.concatenate
    reload(bfs)
    print(bfs.result_table(times2, val_fmt = val_fmt))
    reload(bfs)
    reload(bfs)
    reload(bfs)
    print(bfs.result_table(times2, val_fmt = val_fmt))
    reload(bfs)
    print(bfs.result_table(times2, val_fmt = val_fmt))
