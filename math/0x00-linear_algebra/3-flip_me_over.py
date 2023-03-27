#!/usr/bin/env python3
def matrix_transpose(matrix):

    """returns the transpose of a 2d matrix"""
    transpose = []
    for i in range(len(matrix[0])):
        transpose.append([])
        for j in range(len(matrix)):
            transpose[i].append(matrix[j][i])
            print(matrix[j][i])
            #we start with the first row when i = 0
            # [0,0][1,0] i= row and j column position first loop
            #then [0,1][1,1]


    return transpose
 
