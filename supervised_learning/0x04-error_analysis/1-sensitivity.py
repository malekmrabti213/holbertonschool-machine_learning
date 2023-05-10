#!/usr/bin/env python3

import numpy as np

def sensitivity(confusion):
    actual = np.sum(confusion, axis=1)
    diagonal = np.diagonal(confusion)
    return diagonal / actual