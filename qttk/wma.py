'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

testwma.py - Weighted Moving Average Functions

Weighted moving averages assign a heavier weighting to more current
data points since they are more relevant than dat points in the distant
past.  The sum of the weighting should add up to 1 (or 100 percent).

from: Investopedia.com
http://bit.ly/364HCPk

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/weighted_moving_average.py

Profiled and tested weighted moving average functions

1/31/2021 updates:
* replaced min_periods parameter name with window for consistency
* Converted to profiler_v2
* Implemented function call of profiler_v2
* Added validation testing

'''
import pandas as pd
import numpy as np
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange
from qttk.utils.data_validation import load_wma_validation_data
from pandas._testing import assert_almost_equal, assert_series_equal


def weighted_moving_avg_v1(values: pd.Series,
                           window: int = 5) -> pd.Series:
    '''
    Description:
      Backwards looking weighted moving average.

    Complexity:
      O(nm) because n values * window sized window

    Memory Usage:

    Inputs:
      values: pd.Series of closing prices
      m     : period to be calculated

    '''
    # Constant
    weights = pd.Series(range(1, window+1))
    weights = weights / weights.sum()

    # when period is greater than values, return
    if values.shape[0] <= window:
        return pd.Series([np.nan]*len(values))

    # initialize and copy index from input series for timeseries index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)

    for i in range(window, values.shape[0]+1):
        wma.iloc[i-1] = (values.iloc[i-window:i].values * weights.values).sum()

    return wma


def weighted_moving_avg_v2(values: pd.Series,
                           window: int = 5) -> pd.Series:
    '''
    Description:
      Backwards looking weighted moving average.
    Complexity:
      O(n)?  This is the slowest method, hidden complexity?
    Memory Usage:
    '''
    # Constant
    weights = pd.Series(range(1, window+1))
    weights = weights / weights.sum()

    # when period is greater than values, return
    if values.shape[0] <= window:
        return pd.Series([np.nan]*len(values))

    # initialize series and copy index from input series to return a matching index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)
    group = values.iloc[0:window-1]

    for idx in values.iloc[window-1:].index:
        group[idx] = values[idx]
        wma[idx] = (group.values * weights.values).sum()
        del group[group.index[0]]

    return wma

def _np_weighted_moving_avg(values: np.array, window: int = 5) -> np.array:
    '''
    np convolution method

    Description:
      Backwards looking weighted moving average with numpy.

    Complexity:
      O(n) because n values

    Memory Usage:

    References:
      https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
      https://en.wikipedia.org/wiki/Convolution

    '''
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    return np.convolve(values, weights[::-1], 'valid')


def weighted_moving_avg_v3(values: pd.Series, window: int = 5) -> pd.Series:
    '''
    Wrapper to use np_weighted_moving_avg function
    .00257 ÷ .000706 = 3.64x slower than talib.WMA with arguments:
        series = pd.Series(np.random.random((1000 * 100)))
        window = 12
    '''

    # when period is greater than values, return
    if values.shape[0] <= window:
        return pd.Series([np.nan]*len(values))

    # initialize series and copy index from input series to return a matching index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)
    wma.iloc[window-1:] = _np_weighted_moving_avg(values.values, window)
    return wma



if __name__ == '__main__':
    #validation testing
    data, target = load_wma_validation_data()
    window = 20

    v1 = weighted_moving_avg_v1(data, window=window)
    assert_series_equal(v1, target, check_names=False)

    v2 = weighted_moving_avg_v2(data, window=window)
    assert_series_equal(v2, target, check_names=False)

    v3 = weighted_moving_avg_v3(data, window=window)
    assert_series_equal(v3, target, check_names=False)
    

    exp_range = ExponentialRange(1, 4, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])

        for i in exp_range.iterator():            
            tt(weighted_moving_avg_v1)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(weighted_moving_avg_v2)(series.iloc[:i], window=20)

        for i in exp_range.iterator():
            tt(weighted_moving_avg_v3)(series.iloc[:i], window=20)

