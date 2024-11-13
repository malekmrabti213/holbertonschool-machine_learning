#!/usr/bin/env python3
"""
Task 8
"""


def prune(df):
    """
    """
    df = df.dropna(subset=['Close'])
    return df
