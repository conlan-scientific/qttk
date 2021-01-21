# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# movgAvg.py - Moving Average Study
# performance evaluation of different moving average algorithms

import pandas as pd
import numpy as np
import os
from profiler import time_this

@time_this
def moving_average(df_slice, window):
    # Complexity O(n * m) for n = df_slice.shape[0] and m = window.
    # Get it down to O(n)
    # 2021-01-17v3 moving_average algorithm - correctly validates vs. Excel
    '''
    Completed moving_average in 1543.335 milliseconds
    '''
    i = window
    mvgAvg = pd.DataFrame(0.0, index=np.arange(len(df_slice)), columns=['mvgAvg'])
    while i < len(df_slice) + 1:
    # for i in range(window,df.shape[0] + 1):
        window_start = i - window
        window_end = i         # numpy doesn't select the ending index in a slice
        j = window_end - 1     # row index reference
        mvgAvg.iloc[j] = np.sum(df_slice[window_start:window_end])/window
        # Just a suggestion ... untested
        # if i > window and window_end < df.shape[0]:
        #     mvgAvg.iloc[j] = np.sum(df_slice[window_start:window_end])/window
        # else:
        #     mvgAvg.iloc[j] = 0.0
        try:
            mvgAvg.iloc[j] = np.sum(df_slice[window_start:window_end])/window
        except:
            mvgAvg.iloc[j] = 0.0
        i = i + 1
    return mvgAvg

@time_this
def mvgAvg2(df_slice, window):
    # Complexity O(n * m) for n = df_slice.shape[0] and m = window.
    # Get it down to O(n)
    '''
    Completed mvgAvg2 in 816.912 milliseconds
    '''
    mvgAvg = pd.DataFrame(0.0, index=np.arange(len(df_slice)), columns=['mvgAvg'])
    i = window
    for i in range(window,df_slice.shape[0] + 1):
        window_start = i - window
        window_end = i
        j = window_end - 1
        mvgAvg.iloc[j] = np.sum(df_slice[window_start:window_end])/window
        i = i + 1
    return mvgAvg
