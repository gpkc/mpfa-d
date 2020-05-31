"""calls for using kuzmin method"""
# import numpy as np


def check_if_matrix_is_diagonal_dominant(index_column, np_row):
    diagonal_sum = np_row[index_column]
    off_diagonal_sum = sum(np_row) - diagonal_sum
    return diagonal_sum - off_diagonal_sum > 0


def check_off_diagonal_nonpositiviness(index_column, np_row, source_term):
    import pdb

    pdb.set_trace()
    if source_term <= 0:
        is_antidiffusive = [
            off_diagonal <= 0
            for i, off_diagonal in enumerate(np_row)
            if i != index_column
        ]
    else:
        is_antidiffusive = [
            off_diagonal > 0
            for i, off_diagonal in enumerate(np_row)
            if i != index_column
        ]
    return all(is_antidiffusive)
