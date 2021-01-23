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
    C:/Users/user/qttk>ipython -i ./qttk/testwma.py

Profiled and tested weighted moving average functions
'''
import pandas as pd
import numpy as np
from profiler import time_this


@time_this
def weighted_moving_avg_v1(values: pd.Series,
                           min_periods: int = 5) -> pd.Series:
    '''
    Description:
      Backwards looking weighted moving average.

    Complexity:
      O(nm) because n values * min_periods sized window

    Memory Usage:

    Inputs:
      values: pd.Series of closing prices
      m     : period to be calculated

    '''
    # Constant
    weights = pd.Series(range(1, min_periods+1))
    weights = weights / weights.sum()

    # when period is greater than values, return
    if values.shape[0] <= min_periods:
        return pd.Series([np.nan]*len(values))

    # initialize and copy index from input series for timeseries index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)

    for i in range(min_periods, values.shape[0]+1):
        wma.iloc[i-1] = (values.iloc[i-min_periods:i].values * weights.values).sum()

    return wma


@time_this
def weighted_moving_avg_v2(values: pd.Series,
                           min_periods: int = 5) -> pd.Series:
    '''
    Description:
      Backwards looking weighted moving average.

    Complexity:
      O(n)?  This is the slowest method, hidden complexity?

    Memory Usage:

    '''
    # Constant
    weights = pd.Series(range(1, min_periods+1))
    weights = weights / weights.sum()

    # when period is greater than values, return
    if values.shape[0] <= min_periods:
        return pd.Series([np.nan]*len(values))

    # initialize series and copy index from input series to return a matching index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)
    window = values.iloc[0:min_periods-1]

    for idx in values.iloc[min_periods-1:].index:
        window[idx] = values[idx]
        wma[idx] = (window.values * weights.values).sum()
        del window[window.index[0]]

    return wma


def _np_weighted_moving_avg(values: np.array, min_periods: int = 5) -> np.array:
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
    weights = np.arange(1, min_periods + 1)
    weights = weights / weights.sum()
    return np.convolve(values, weights[::-1], 'valid')


@time_this
def weighted_moving_avg_v3(values: pd.Series, min_periods: int = 5) -> pd.Series:
    '''
    Wrapper to use np_weighted_moving_avg function
    .00257 ÷ .000706 = 3.64x slower than talib.WMA with arguments:
        series = pd.Series(np.random.random((1000 * 100)))
        min_periods = 12
    '''

    # when period is greater than values, return
    if values.shape[0] <= min_periods:
        return pd.Series([np.nan]*len(values))

    # initialize series and copy index from input series to return a matching index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)
    wma.iloc[min_periods-1:] = _np_weighted_moving_avg(values.values, min_periods)
    return wma



if __name__ == '__main__':
    series = pd.Series(np.random.random((100 * 75)))

    x = weighted_moving_avg_v1(series, 12)
    y = weighted_moving_avg_v2(series, 12)
    z = weighted_moving_avg_v3(series, 12)
    assert x.equals(y)
    if not x.equals(z):
        print((x[12:] == z[12:]).value_counts())
        print(f'mean of difference: {np.mean((x[12:] - z[12:])) :.7f}\n')
    # assert x.equals(z) # inconsistent

    print('test with a timeseries and assert datatype\n')
    series.index = pd.date_range(start='12-1-2010', periods=series.shape[0])

    x = weighted_moving_avg_v1(series, 10)
    y = weighted_moving_avg_v2(series, 10)
    z = weighted_moving_avg_v3(series, 10)
    assert x.index.dtype == np.dtype('<M8[ns]')
    assert y.index.dtype == np.dtype('<M8[ns]')
    assert z.index.dtype == np.dtype('<M8[ns]')

    # truth series
    test_series = pd.Series(list(range(21, 31)))
    truth_series = pd.Series([
        np.nan,
        np.nan,
        np.nan,
        (21 * 1 + 22 * 2 + 23 * 3 + 24 * 4) / 10,
        (22 * 1 + 23 * 2 + 24 * 3 + 25 * 4) / 10,
        (23 * 1 + 24 * 2 + 25 * 3 + 26 * 4) / 10,
        (24 * 1 + 25 * 2 + 26 * 3 + 27 * 4) / 10,
        (25 * 1 + 26 * 2 + 27 * 3 + 28 * 4) / 10,
        (26 * 1 + 27 * 2 + 28 * 3 + 29 * 4) / 10,
        (27 * 1 + 28 * 2 + 29 * 3 + 30 * 4) / 10,
    ])
    test_results = weighted_moving_avg_v3(test_series, 4)
    # assert truth_series.equals(test_results)
    pd._testing.assert_series_equal(truth_series, test_results)
