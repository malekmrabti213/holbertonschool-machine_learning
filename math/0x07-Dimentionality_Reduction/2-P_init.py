#!/usr/bin/env python3

import numpy as np


def P_init(X, perplexity):
    """initializes all of the variables needed to calculate p affinities"""
    n = X.shape[0]
    X1 = X[np.newaxis, :, :]
    X2 = X[:, np.newaxis, :]
    D = np.sum(np.square(X1 - X2), axis=2)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H