#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:19:16 2023

@author: cosmostage
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../../dataLC/fake_data_obs.csv')
print(df.columns)
print(df)
n = 3
df['skynoise'] = df['skynoise'] + 1.25 *np.log10(n)
print(df)