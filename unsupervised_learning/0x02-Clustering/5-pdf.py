#!/usr/bin/env python3

import numpy as np

def pdf(X, m, S):
    """calculate the pdf of the multinormal distribution"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    d = X.shape[1]
    if type(m) is not np.ndarray or m.ndim != 1 or m.shape[0] != d:
        return None
    if type(S) is not np.ndarray or S.ndim != 2 or S.shape != (d, d):
        return None
    Xm = X - m
    e = - 0.5 * np.sum(Xm * np.matmul(np.linalg.inv(S), Xm.T).T, axis=1)
    num = np.exp(e)
    det = np.linalg.det(S)
    prob = num / np.sqrt(((2 * np.pi) ** d) * det)
    return  np.maximum(prob, 1e-300)