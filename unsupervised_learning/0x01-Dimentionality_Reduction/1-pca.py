#!/usr/bin/env python3
"""
Project Dimensionality reduction
By Ced+
"""
import numpy as np


def pca(X, ndim):
    """
    Takes X a matrix,
    return Tr( reduction score)
    """
    n = X.shape[0]  # number of data points
    d = X.shape[1]  # number of dimensions

    # normalize X
    X = X - np.mean(X, axis=0)

    # decompose X into SVD
    U, S, V = np.linalg.svd(X)

    # reduction dimension r
    if ndim >= d:
        ndim = d
    # Use V to project the data onto the principal components
    W = V[:ndim].T  # Transpose to get the correct shape

    # Apply the transformation to X
    T = X @ W

    return T