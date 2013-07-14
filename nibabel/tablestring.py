""" Make printable tables from arrays of results
"""
from __future__ import division

import numpy as np


def printable_table(cell_values,
                    row_names = None,
                    col_names = None,
                    title='',
                    val_fmt = '{0:5.2f}'):
    """ Return table with entries `cell_values`, and optional formatting

    Parameters
    ----------
    cell_values : (R, C) array-like
        At least 2D.  Can be greater than 2D, in which case you should adapt the
        `val_fmt` to deal with the multiple entries that will go in each cell
    row_names : None or (R,) length sequence, optional
        Row names.  If None, use ``row[0]`` etc.
    col_names : None or (C,) length sequence, optional
        Column names.  If None, use ``col[0]`` etc.
    title : str, optional
        Title for table
    val_fmt : str, optional
        Format string using string ``format`` method mini-language. Converts the
        result of ``cell_values[r, c]`` to a string to make the cell contents.
        Default assumes a floating point value in a 2D ``cell_values``.

    Returns
    -------
    table_str : str
        Multiline string with ascii table, suitable for printing
    """
    cell_values = np.asarray(cell_values)
    R, C = cell_values.shape[:2]
    if row_names is None:
        row_names = ['row[{0}]'.format(r) for r in range(R)]
    elif len(row_names) != R:
        raise ValueError('len(row_names) != number of rows')
    if col_names is None:
        col_names = ['col[{0}]'.format(c) for c in range(C)]
    elif len(col_names) != C:
        raise ValueError('len(col_names) != number of columns')
    row_len = max(len(name) for name in row_names)
    col_len = max(len(name) for name in col_names)
    # Compile row value strings, find longest, extend col length to match
    row_str_list = []
    for row_no in range(R):
        row_strs = [val_fmt.format(val) for val in cell_values[row_no]]
        max_len = max(len(s) for s in row_strs)
        if max_len > col_len:
            col_len = max_len
        row_str_list.append(row_strs)
    joiner = ' | '
    ender = ' |'
    row_name_fmt = "{0:<" + str(row_len) + "}"
    row_names = [row_name_fmt.format(name) for name in row_names]
    col_name_fmt = "{0:^" + str(col_len) + "}"
    col_names = [col_name_fmt.format(name) for name in col_names]
    col_header = joiner.join([' ' * row_len] + col_names) + ender
    row_val_fmt = '{0:>' + str(col_len) + '}'
    table_strs = []
    if title != '':
        table_strs += ['=' * len(col_header),
                       title]
    table_strs += ['=' * len(col_header),
                   col_header,
                   '-' * len(col_header)]
    for row_no, row_name in enumerate(row_names):
        row_vals = [row_val_fmt.format(row_str)
                    for row_str in row_str_list[row_no]]
        table_strs.append(joiner.join([row_name] + row_vals) + ender)
    table_strs.append('-' * len(col_header))
    return '\n'.join(table_strs)
