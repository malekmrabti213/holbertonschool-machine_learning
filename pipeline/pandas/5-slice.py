#!/usr/bin/env python3
"""
Task 5
"""


def slice(df):
    """
    """
    df = df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df
