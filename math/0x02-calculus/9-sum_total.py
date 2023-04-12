#!/usr/bin/env python3

def summation_i_squared(n):
    if (type(n) is not int) or (n<0):
        return None
    
    elif n==1:
        return 1
    else:
        # the function recursively calls itself with the argument 
        # n-1 to calculate the sum from 1 to n-1, and adds n^2 to the result
        #decrementing until n = 1
        return n**2+summation_i_squared(n-1)