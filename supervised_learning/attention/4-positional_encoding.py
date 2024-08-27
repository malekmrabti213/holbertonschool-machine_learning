#!/usr/bin/env python3
"""
Task 4
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]
    angles = positions / (10000 ** (2 * (dimensions // 2) / np.float32(dm)))
    angles[:, ::2] = np.sin(angles[:, ::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles
