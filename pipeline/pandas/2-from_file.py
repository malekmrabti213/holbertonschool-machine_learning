#!/usr/bin/env python3
"""
Task 3
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    """
    return pd.read_csv(filename, delimiter=delimiter)
