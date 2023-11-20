#!/usr/bin/env python3

import numpy as np


def pca(X, var=0.95):
    """calculates the PCA loadings
    import
    """
    _, s, vh = np.linalg.svd(X)
    total_var = np.cumsum(s) / np.sum(s)
    r = np.argwhere(total_var >= var)[0, 0]
    return vh[:r + 1].T