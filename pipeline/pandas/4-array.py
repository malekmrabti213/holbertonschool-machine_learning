#!/usr/bin/env python3
"""
Task 4
"""
import pandas as pd


def array(df):
    """
    """
    df = df[['High', 'Close']].tail(10).to_numpy()
    return df
