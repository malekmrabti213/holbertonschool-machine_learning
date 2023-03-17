#!/usr/bin/env python3
def cat_arrays(arr1, arr2):
    add=arr1.copy()
    #try1
    # for x in arr2:
    #     add.append(x)

    #try2
    #add.extend(arr2)

    #try3
    add=add+arr2
    return add



    