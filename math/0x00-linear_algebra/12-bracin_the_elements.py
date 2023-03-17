import numpy as np

def np_elementwise(mat1, mat2):
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    # add = np.add(mat1, mat2)
    # sub = np.subtract(mat1, mat2)
    # mul = np.multiply(mat1, mat2)
    # div = np.divide(mat1, mat2)


    return (add, sub, mul, div)
    