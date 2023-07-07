#!/usr/bin/env python3

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """calculates P affinities so each Gaussian has the same perplexity"""
    n = X.shape[0]
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        low = None
        high = None
        Di = np.append(D[i, :i], D[i, i+1:])
        (Hi, Pi) = HP(Di, betas[i])
        Hdiff = Hi - H
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i] = betas[i] * 2.
                else:
                    betas[i] = (betas[i] + high) / 2.
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i] = betas[i] / 2.
                else:
                    betas[i] = (betas[i] + low) / 2.
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H
        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]

    P = P + P.T
    P = P / np.sum(P)
    return P