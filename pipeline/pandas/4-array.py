#!/usr/bin/env python3
"""
Task 4
"""


def array(df):
    """
    """
    df = df[['High', 'Close']].tail(10).to_numpy()
    return df
