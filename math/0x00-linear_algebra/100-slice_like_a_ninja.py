#!/usr/bin/env python3

def np_slice(matrix, axes={}):
    #https://docs.python.org/3/library/functions.html#slice
    slice_mat = [slice(None, None, None)] * matrix.ndim

    for k, v in sorted(axes.items()):
        slice_val = slice(*v)
        print('*****')
        print(slice_val)
        print('======')
        slice_mat[k] = slice_val
        print(slice_mat[k])
        print("@@@@@")
        print(tuple(slice_mat))
    matrix = matrix[tuple(slice_mat)]
    return matrix