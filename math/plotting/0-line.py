#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    """
    y = np.arange(0, 11) ** 3
    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
