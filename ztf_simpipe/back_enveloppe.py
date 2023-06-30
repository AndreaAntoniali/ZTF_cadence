#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:22:11 2023

@author: cosmostage
"""

import pandas as pd

# Sample DataFrame
data = {'Column': ['apple', 'banana', 'apple', 'orange', 'kiwi', 'apple']}
df = pd.DataFrame(data)

def count_string_occurrences(dataframe, column_name, target_string):
    count = dataframe[column_name].str.count(target_string).sum()
    return count

# Usage example
target = 'apple'
occurrences = count_string_occurrences(df, 'Column', target)
print(f"The string '{target}' appears {occurrences} times in the column.")