#!/usr/bin/env python3

import numpy as np

def initialize(X, k):
    """initializes centroids for kmeans"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(k) is not int or int(k) != k or k < 1:
        return None
    _, d = X.shape
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    return np.random.uniform(mins, maxs, size=(k, d))