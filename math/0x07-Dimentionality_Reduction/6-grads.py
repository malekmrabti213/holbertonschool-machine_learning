#!/usr/bin/env python3

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


# def grads(Y, P):
#     """calculates the gradients of Y"""
#     n, ndims = Y.shape
#     dY = np.zeros((n, ndims))
#     Q, num = Q_affinities(Y)
#     PQ = P - Q
#     for i in range(n):
#         part = np.expand_dims((PQ[:, i] * num[:, i]).T, axis=-1)
#         dY[i, :] = np.sum(part * (Y[i, :] - Y), 0)
#     return dY, Q
def grads(Y, Pij):
    """
    Compute the grads
    :param Y: Containing the low dimensional transformation of X
    :param Pij: The Pij for all ij affinities
    :return: The grads
    """
    n, ndims = Y.shape

    Qij, numerator = Q_affinities(Y)
    PQij = Pij - Qij

    dY = np.dot(PQij * numerator.T, Y - np.tile(Y, (n, 1)))

    return -dY, Qij

