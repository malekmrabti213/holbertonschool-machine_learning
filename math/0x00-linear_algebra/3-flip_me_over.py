import numpy as np 
def matrix_transpose(matrix):

    matrix_transpose = []
    for index, row in enumerate(matrix):
        if index is 0:
            for i in row:
                matrix_transpose.append([])
        for idx, i in enumerate(row):
            matrix_transpose[idx].append(i)
    return matrix_transpose
    #np.transpose(matrix)
