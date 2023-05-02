#!/usr/bin/env python3

import numpy as np

def shuffle_data(X, Y):
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]