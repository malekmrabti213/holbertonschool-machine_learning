#!/usr/bin/env python3

import numpy as np


def Q_affinities(Y):
    """Calculates the Q affinities"""
    n = Y.shape[0]
    sY = np.sum(np.square(Y), 1)
    D = np.add(np.add(-2. * np.matmul(Y, Y.T), sY).T, sY)
    num = 1. / (1. + D)
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)
    return Q, num