#!/usr/bin/env python3
"""
    Policy Gradient
"""

import numpy as np


def policy(state, weight):
    """

    """
    z = state.dot(weight)
    # Apply numerical stability fix by subtracting max of z
    z -= np.max(z)
    exp = np.exp(z)

    return exp / np.sum(exp)
