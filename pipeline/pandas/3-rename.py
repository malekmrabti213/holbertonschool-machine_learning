#!/usr/bin/env python3
"""
Task 3
"""
import pandas as pd


def rename(df):
    """
    """
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df = df[['Datetime', 'Close']]
    return df
