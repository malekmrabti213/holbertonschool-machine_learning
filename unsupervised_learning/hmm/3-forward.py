#!/usr/bin/env python3

import numpy as np

def forward(Observation, Emission, Transition, Initial):
    # add checks here
    T = Observation.shape[0]
    N, _ = Emission.shape

    F = np.zeros((N, T))
    for s in range(N):
        F[s, 0] = Initial[s, 0] * Emission[s, Observation[0]]
    for t, o in enumerate(Observation):
        if t == 0:
            continue
        for s in range(N):
            F[s, t] = np.sum(F[:, t - 1] * Transition[:, s] * Emission[s, o])
    return np.sum(F[:, T - 1]), F