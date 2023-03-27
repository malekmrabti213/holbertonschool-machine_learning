#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):

    add=[] 
    #to do addition or subtraction we should have the same number of elements in each row or column
    for i in range(len(mat1)):       
        if (len(mat1[i])!=len(mat2[i])):
            return None
        else:                  
            sumadd=[]
            for j in range(len(mat1[i])):   
                sumadd.append(mat1[i][j]+mat2[i][j])
            add.append(sumadd) 
    return add   

