'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

testma.py - Moving Average Functions

The moving average (MA) is a technical indicator
that helps smooth out the price of a stock over a specified
time-frame.

from: Investopedia.com
https://www.investopedia.com/terms/m/movingaverage.asp

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/testma.py

Profiled and tested moving average functions
'''
import pandas as pd
import numpy as np
from profiler import time_this


#@time_this
def pd_simple_moving_avg(values: pd.Series, min_periods: int = 20) -> pd.Series:
    '''
    This is an O(n) time implementation of a simple moving average.
    Simple moving average

    >>> pd_simple_moving_avg(pd.Series(range(0,14,2)),2)
    0     NaN
    1     NaN
    2     3.0
    3     5.0
    4     7.0
    5     9.0
    6    11.0
    dtype: float64
    '''
    cumsum = values.cumsum()
    return (cumsum - cumsum.shift(min_periods))/min_periods

#@time_this
def cumulative_moving_avg(values: pd.Series, min_periods: int = 5) -> pd.Series:
    '''
    The Cumulative Moving Average is the unweighted mean of the previous values up to the current time t.
    pandas has an expand O(nm) because n number of periods in a loop and sum(m)

    '''
    counter=0

    # Initialize series
    cma = pd.Series([np.nan]*(min_periods-1), dtype='float64')

    temp: List[float] = list()

    for i in range(min_periods, len(values)+1):

        temp.append(sum(values[:i])/(min_periods+counter))
        counter += 1

    return cma.append(pd.Series(temp)).reset_index(drop=True)


#@time_this
def cumulative_moving_avg_v2(values: pd.Series, min_periods: int = 5) -> pd.Series:
    '''
    pd.Series.expand.mean() exists, but I tried to work out something vectorized
    '''

    # Initialize series
    cma = pd.Series([np.nan]*(min_periods-1), dtype='float64')

    cma = cma.append((values.cumsum() / pd.Series(range(1, len(values)+1)))[min_periods-1:])

    return cma.reset_index(drop=True)


#@time_this
def cumulative_moving_avg_v3(values: pd.Series, min_periods: int = 5) -> pd.Series:
    '''
    Cumulative moving average with reduced memory complexity
    '''

    # denominator is one-indexed location of the element in the cumsum
    denominator = pd.Series(np.arange(1, series.shape[0]+1))
    result = values.cumsum() / denominator

    # Set the first min_periods elements to nan
    result.iloc[:(min_periods-1)] = np.nan

    return result


if __name__ == '__main__':
    # test datsets
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    x = pd_simple_moving_avg(series, min_periods=1)

    y = cumulative_moving_avg(series, min_periods=1)
    z = cumulative_moving_avg_v2(series, min_periods=1)
    w = cumulative_moving_avg_v3(series, min_periods=1)

    truth_series_ma = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    truth_series_cma = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])

    assert x.equals(truth_series_ma)

    assert y.equals(truth_series_cma)
    assert z.equals(truth_series_cma)
    assert w.equals(truth_series_cma)

    exit
