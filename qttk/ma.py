'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# moving_average.py - Moving Average Study
# performance evaluation of different moving average algorithms

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/mvgAvg.py

# 1/31/2021 changes:
  * renamed functions 
  * Removed while loop from moving_avg_v1
  * Removed try/except from moving_avg_v1
  * Simplified moving_avg_v1 by removing duplicate position var j
  * Added functions from testma.py
  * Updated to use validation testing utility
  * Added date index to validation data
  * Corrected offset in moving_avg_v3, sadly v3 is slower now
  * Created new validation files:
         ma_input_data, ma_target_data with fewer samples
  * Removed decorators and call profilerv2.time_this as a function
  * Conformed to code quality guideline: 
        Don't return unnecessarily complex objects.
  * Adjusted moving_avg_v1 and v2 to respect date indexes
  * Changed min_periods parameter name to window to be consistent
'''
__all__ = ['moving_avg_v4']

import pandas as pd
import numpy as np
import os
# from qttk.profiler import time_this
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange
from qttk.utils.data_validation import load_sma_validation_data
from pandas._testing import assert_almost_equal, assert_series_equal


def moving_avg_v1(values:pd.Series, window) -> pd.Series:
    # Complexity O(n * m) for n = values.shape[0] and m = window.
    # Get it down to O(n)
    # 2021-01-17v3 moving_average algorithm - correctly validates vs. Excel
    '''
    Completed moving_average in 1543.335 milliseconds
    '''
    mvgAvg = pd.Series(0.0, index=values.index, name='mvgAvg')
    
    for i in range(window-1, values.shape[0]):
        window_start = i - (window-1)
        window_end = i+1         # numpy doesn't select the ending index in a slice

        mvgAvg.iloc[i] = (np.sum(values[window_start:window_end])/window)

        if i >= window-1 and window_end <= values.shape[0]:
            mvgAvg.iloc[i] = (values[window_start:window_end].sum()/window)
        else:
            mvgAvg.iloc[i] = 0.0
            
    return mvgAvg


def moving_avg_v2(values:pd.Series, window:int) -> pd.Series:
    # Complexity O(n * m) for n = values.shape[0] and m = window.
    # Get it down to O(n)
    '''
    Completed mvgAvg2 in 816.912 milliseconds
    '''
    mvgAvg = pd.Series(0.0, index=values.index, name='mvgAvg')
    i = window
    for i in range(window, values.shape[0] + 1):
        window_start = i - window
        window_end = i
        j = window_end - 1
        mvgAvg.iloc[j] = np.sum(values[window_start:window_end])/window
        i = i + 1
    return mvgAvg


def moving_avg_v3(values: pd.Series, window: int = 20) -> pd.Series:
    '''
    This is an O(n) time implementation of a simple moving average.
    It appears shift(window) starts at window + 0
      for example: with a window of 20 and zero based index
        was expecting shift to start at index 19.
        Shift started at index 20, the 21st position
    The workaround may be costly.  
    '''
    original_index = values.index.copy()
    cumsum = values.cumsum()
    cumsum = pd.concat([pd.Series(0, name=values.name), cumsum])
    mvg_avg = ((cumsum - cumsum.shift(window))/window)
    mvg_avg = mvg_avg.iloc[1:]
    mvg_avg.index = original_index
    return mvg_avg


def moving_avg_v4(values: pd.Series, window: int = 20) -> pd.Series:
    '''
    Pandas moving average with .rolling
    '''
    return values.rolling(window).mean()


if __name__ == '__main__':

    #validation testing
    data, target = load_sma_validation_data()

    v1 = moving_avg_v1(data, window=20)
    assert_series_equal(v1, target.fillna(0), check_names=False)

    #v2 = moving_avg_v2(data, window=20)
    v2 = moving_avg_v2(data, window=20)
    assert_series_equal(v2, target.fillna(0), check_names=False)

    v3 = moving_avg_v3(data, window=20)
    assert_series_equal(v3, target, check_names=False)
    
    v4 = moving_avg_v4(data, window=20)
    assert_series_equal(v4, target, check_names=False)

    exp_range = ExponentialRange(1, 4, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])

        for i in exp_range.iterator():            
            tt(moving_avg_v1)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(moving_avg_v2)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(moving_avg_v3)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(moving_avg_v4)(series.iloc[:i], window=20)

    # test performance of window size
    '''
    exp_range = ExponentialRange(1, 4, 1/4)
    with timed_report():
        for i in exp_range.iterator():
            for j in [5, 10, 20, 50, 100]:
                mvgAvg2(series.iloc[:i], j)'''

