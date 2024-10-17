#!/usr/bin/env python3
"""
module containing function epsilon_greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    function that uses epsilon-greedy to determine the next action
    Args:
        Q: numpy.ndarray containing the q-table
        state: current state
        epsilon: epsilon to use for the calculation
    Return: the next action index
    """
    # determine if your algorithm should explore or exploit
    p = np.random.uniform(0, 1)
    if p > epsilon:
        # exploit
        action = np.argmax(Q[state, :])
    else:
        # explore
        action = np.random.randint(0, Q.shape[1])

    return action
