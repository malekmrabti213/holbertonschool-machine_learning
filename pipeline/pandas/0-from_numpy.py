#!/usr/bin/env python3
"""
Task 0
"""
import pandas as pd


def from_numpy(array):
    """
    """
    columns = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    c = array.shape[1]
    df = pd.DataFrame(array, columns=list(columns[:c]))
    return df
