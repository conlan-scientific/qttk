'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

cumulative_moving_average.py - Three Cumulative Moving Average Functions

The Cumulative moving average (MA) is a technical indicator
that represents the unweighted mean of the previous values up to current time.

from: Investopedia.com
https://www.investopedia.com/terms/m/movingaverage.asp

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/cumulative_moving_average.py

Profiled and tested moving average functions

Updates 1/31/2021:
  * Using validation test utility
  * Validation data updates discussed in detail in data_validation.py
  * Profiler_v2 is being called as a function
  * Adjusted cumulative_moving_avg_v1 to respect date index
  * Fixed misshaped series bug in cumulative_moving_avg_v2
  * Skipped bug with cumulative_moving_avg_v1 that happens
      when window is larger than values
'''
import os
import pandas as pd
import numpy as np
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange
from typing import Any, List, Dict
from pandas.testing import assert_series_equal
from pandas.testing import assert_frame_equal
from qttk.utils.data_validation import load_cma_validation_data


# @time_this(lambda *args, **kwargs: args[0].shape[0])
def cumulative_moving_avg_v1(values: pd.Series, window: int = 5) -> pd.Series:
    '''
    Slow cumulative moving average.
    '''
    counter = 0
    cma = pd.Series([np.nan]*(window-1))
    temp = list()

    for i in range(window, len(values)+1):

        temp.append(sum(values.iloc[:i])/(window+counter))
        counter += 1
    cma = cma.append(pd.Series(temp))
    cma.index = values.index
    return cma


# @time_this(lambda *args, **kwargs: args[0].shape[0])
def cumulative_moving_avg_v2(values: pd.Series, window: int = 5) -> pd.Series:
    '''
    Pandas simple moving average pd.Series.expand.mean()
    '''

    return values.expanding(window).mean()


# @time_this(lambda *args, **kwargs: args[0].shape[0])
def cumulative_moving_avg_v3(values: pd.Series, window: int = 5) -> pd.Series:
    '''
    Cumulative Moving Average 
    '''

    # denominator is one-indexed location of the element in the cumsum
    denominator = pd.Series(np.arange(1, values.shape[0]+1), index=values.index)
    result = values.cumsum() / denominator
    # Set the first window elements to nan
    result.iloc[:(window-1)] = np.nan

    return result


if __name__ == '__main__':

    '''
    # Test assert_frame_equal
    data, result = load_cma_validation_data()
    print(f'difference: {(data - result).mean()}')
    assert_frame_equal(data, result)'''
    window = 20
    # validation tests on known data
    data, target = load_cma_validation_data()
    
    v1 = cumulative_moving_avg_v1(data, window=window)
    assert_series_equal(v1, target, check_names=False)

    v2 = cumulative_moving_avg_v2(data, window=window)
    assert_series_equal(v2, target, check_names=False)
    
    v3 = cumulative_moving_avg_v3(data, window=window)
    assert_series_equal(v3, target, check_names=False)

    # performance tests
    exp_range = ExponentialRange(1, 4, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    tt = time_this(lambda *args, **kwargs: args[0].shape[0])
    window=10
    with timed_report():
        
        for i in exp_range.iterator():
            #cumulative_moving_avg_v1(series.iloc[:i])
            tt(cumulative_moving_avg_v1)(series.iloc[:i], window=window)
        
        for i in exp_range.iterator():      
            tt(cumulative_moving_avg_v2)(series.iloc[:i], window=window)

        for i in exp_range.iterator():
            tt(cumulative_moving_avg_v3)(series.iloc[:i], window=window)
            #cumulative_moving_avg_v3(series.iloc[:i])

