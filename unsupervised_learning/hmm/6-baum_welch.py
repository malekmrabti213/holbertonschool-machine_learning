#!/usr/bin/env python3

import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observation, Transition, Emission, Initial, iterations=1000):
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    T = Observation.shape[0]
    if T < 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    hidden_states = Transition.shape[0]
    if Transition.shape != (hidden_states, hidden_states) or not np.allclose(np.sum(Transition, axis=1), 1):
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    output_states = Emission.shape[1]
    if Emission.shape != (hidden_states, output_states) or not np.allclose(np.sum(Emission, axis=1), 1):
        return None, None
    if type(Initial) is not np.ndarray or Initial.shape != (hidden_states, 1) or not np.allclose(np.sum(Initial), 1):
        return None, None

    for _ in range(iterations):
        _, F = forward(Observation, Emission, Transition, Initial)
        _, B = backward(Observation, Emission, Transition, Initial)
        xsi = np.expand_dims(F[:, :-1], axis=1) * np.expand_dims(B[:, 1:], axis=0) * np.expand_dims(Transition, axis=2) * np.expand_dims(Emission[:, Observation[1:]], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            xsi = np.divide(xsi, np.sum(xsi, axis=(0,1)))
        xsi = np.where(~np.isfinite(xsi), 0., xsi)
        gamma = np.sum(xsi, axis=1)
        #Initial = gamma[:, 0:1]
        with np.errstate(divide='ignore', invalid='ignore'):
            Transition = np.divide(np.sum(xsi, axis=2), np.sum(gamma, axis=1).reshape((-1, 1)))
        Transition = np.where(~np.isfinite(Transition), 0., Transition)
        gamma = np.hstack((gamma, np.sum(xsi[:, :, T - 2], axis=0).reshape((-1, 1))))
        Emission = np.zeros((hidden_states, output_states))
        denom = np.sum(gamma, axis=1).reshape((-1, 1))
        for o in range(output_states):
            Emission[:, o] = np.sum(gamma[:, np.where(Observation == o)[0]], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Emission = np.divide(Emission, denom)
        Emission = np.where(~np.isfinite(Emission), 0., Emission)
    return Transition, Emission
