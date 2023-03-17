#!/usr/bin/env python3

def matrix_shape(matrix):
    #shape of matrix
    matrix_shape = []
    while type(matrix) is list:
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape


def add_matrices(mat1, mat2):
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        if len(matrix_shape(mat1)) == 1:
            #not iterable 
            return [mat1[i] + mat2[i] for i in range(len(mat1))]
        else:
            #Recursively construct a new sum of two matrices
            return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]



