#!/usr/bin/env python3

def poly_derivative(poly):
    if (type(poly) is not list) or (poly == []):
        return None 
    
    #check the constant
    elif len(poly)==1:
        return [0]
    #The derivative of a polynomial is obtained by multiplying 
    # each coefficient by its power and reducing the power by 1
    else:
       return [poly[i] * i for i in range(1, len(poly))] 

    
     
    