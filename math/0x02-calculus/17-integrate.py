#!/usr/bin/env python3

def poly_integral(poly, C=0):
    if type(poly) is not list or not len(poly) or type(C) is not int:
        return None
    integral = [C]
    if poly == [0]:
        return integral
    for i, val in enumerate(poly):
        co = val/(i + 1)
        co = int(co) if co == int(co) else co
        integral.append(co)
    return integral