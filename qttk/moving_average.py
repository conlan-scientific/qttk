'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# moving_average.py - Moving Average Study
# performance evaluation of different moving average algorithms

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/mvgAvg.py
'''
__all__ = ['moving_average_v3']

import pandas as pd
import numpy as np
import os
# from qttk.profiler import time_this
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange
from qttk.utils.data_validation import load_moving_average


def moving_average_v1(df_slice, window):
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

        if i > window and window_end < df_slice.shape[0]:
            mvgAvg.iloc[j] = np.sum(df_slice[window_start:window_end])/window
        else:
            mvgAvg.iloc[j] = 0.0
        i = i + 1
    return mvgAvg


def moving_average_v2(df_slice:pd.DataFrame, window:int)->pd.DataFrame:
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


def moving_average_v3(values: pd.Series, window: int = 20) -> pd.Series:
    '''
    This is an O(n) time implementation of a simple moving average.
    '''
    cumsum = values.cumsum()
    return (cumsum - cumsum.shift(window))/window


def moving_average_v4(values: pd.Series, window: int = 20) -> pd.Series:
    '''
    Pandas moving average with .rolling
    '''
    return values.rolling(window).mean()


if __name__ == '__main__':

    #validation testing
    data, target = load_moving_average()

    exp_range = ExponentialRange(1, 4, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])

        for i in exp_range.iterator():            
            tt(moving_average_v1)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(moving_average_v2)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(moving_average_v3)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(moving_average_v4)(series.iloc[:i], window=20)

    # test performance of window size
    '''
    exp_range = ExponentialRange(1, 4, 1/4)
    with timed_report():
        for i in exp_range.iterator():
            for j in [5, 10, 20, 50, 100]:
                mvgAvg2(series.iloc[:i], j)'''

