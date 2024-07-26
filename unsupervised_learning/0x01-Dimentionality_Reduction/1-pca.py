#!/usr/bin/env python3

import numpy as np


def pca(X, ndim):
    """calculates pca for a specific number of dims"""
    X = X - np.mean(X, axis=0)
    _, _, Vh = np.linalg.svd(X)
    W = Vh[:ndim].T
    return np.matmul(X, W)