#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    import copy
    if axis == 0 :
        for i in range(len(mat1)):
             for j in range(len(mat2)):       
                if (len(mat1[i])==len(mat2[j])):
                 
                    mat1copy=copy.deepcopy(mat1)
                    mat2copy=copy.deepcopy(mat2)

                    axis0 = mat1copy+mat2copy
                    return axis0
                              
    elif (axis == 1) and (len(mat1)==len(mat2)):
            
            axis1=[]
            for i in range(len(mat1)):                     
                axis1.append(mat1[i]+mat2[i])
            return axis1
    
    else:
            return None
    
            
  