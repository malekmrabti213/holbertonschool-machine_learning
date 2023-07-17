#!/usr/bin/env python3

import numpy as np

def variance(X, C):
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(C) is not np.ndarray or C.ndim != 2 or C.shape[1] != X.shape[1]:
        return None
    Xe = np.expand_dims(X, axis=1)
    Ce = np.expand_dims(C, axis=0)
    pairwise_dist = np.sum(np.square(Xe - Ce), axis=2)
    centroid_dist = np.min(pairwise_dist, axis=1)
    var = np.sum(centroid_dist)
    return var