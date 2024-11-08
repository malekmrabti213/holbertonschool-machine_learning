#!/usr/bin/env python3
"""
Task 1
"""
import pandas as pd


d = {'First': [0, 0.5, 1, 1.5], 'Second': ['one', 'two', 'three', 'four']}
df = pd.DataFrame(d, index=list('ABCD'))
