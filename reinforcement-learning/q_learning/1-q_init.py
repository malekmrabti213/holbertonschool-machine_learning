#!/usr/bin/env python3
"""
module containing function q_init
"""
import numpy as np


def q_init(env):
    """
    function that initializes the Q-table
    Args:
        env: FrozenLakeEnv instance
    Return: Q-table as a numpy.ndarray of zeros
    """
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    return q_table
