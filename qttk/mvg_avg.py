'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# movgAvg.py - Moving Average Study
# performance evaluation of different moving average algorithms

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/mvgAvg.py
'''
__all__ = ['mvgAvg2']

import pandas as pd
import numpy as np
import os
# from qttk.profiler import time_this
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange


#@time_this(lambda *args, **kwargs: args[0].shape[0])
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

#@time_this(lambda *args, **kwargs: args[0].shape[0])
def mvgAvg2(df_slice:pd.DataFrame, window:int)->pd.DataFrame:
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


if __name__ == '__main__':
    exp_range = ExponentialRange(1, 5, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    with timed_report():
        for i in exp_range.iterator():
            moving_average(series.iloc[:i], window=20)

        for i in exp_range.iterator():
           mvgAvg2(series.iloc[:i], window=20)

    # exp_range = ExponentialRange(1, 4, 1/4)
    # with timed_report():
    #     for i in exp_range.iterator():
    #         for j in [5, 10, 20, 50, 100]:
    #             mvgAvg2(series.iloc[:i], j)
    exit
