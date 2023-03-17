#!/usr/bin/env python3
def matrix_shape(matrix):
    ### Try 1
    # if type(matrix[0]) != list:
    #     return [len(matrix)]
    # else:
    #     return [len(matrix)] + matrix_shape(matrix[0])
    matrix_shape = []
    while (type(matrix) is list):
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape