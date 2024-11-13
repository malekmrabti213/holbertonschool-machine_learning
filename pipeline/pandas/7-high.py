#!/usr/bin/env python3
"""
Task 7
"""


def high(df):
    """
    """
    df = df.sort_values(by='High', ascending=False)
    return df
