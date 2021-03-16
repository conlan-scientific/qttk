'''
Profiled and tested weighted moving average functions.

Weighted moving averages assign a heavier weighting to more current
data points since they are more relevant than data points in the distant
past.  The sum of the weighting should add up to 1 (or 100 percent).

Weighted moving average is a convolution of data points with a fixed
weighting function.

References:
    `Investopedia Weighted Moving Average <http://bit.ly/364HCPk>`_

'''
import pandas as pd
import numpy as np
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange
from qttk.utils.data_validation import load_wma_validation_data
from pandas._testing import assert_series_equal


def weighted_moving_avg_v1(values: pd.Series,
                           window: int = 5) -> pd.Series:
    """
    Weighted Moving Average used for benchmarking.  This implementation
    uses a pandas series with a for loop to assign the weighted moving average.

    Args:
        values (pd.Series): price series
        window (int, optional): Defaults to 5.

    Returns:
        pd.Series: weighted moving average named wma
    """

    # Constant
    weights = pd.Series(range(1, window+1))
    weights = weights / weights.sum()

    # when period is greater than values, return
    if values.shape[0] <= window:
        return pd.Series([np.nan]*len(values))

    # initialize and copy index from input series for timeseries index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index, name='wma')

    for i in range(window, values.shape[0]+1):
        wma.iloc[i-1] = (values.iloc[i-window:i].values * weights.values).sum()

    return wma


def weighted_moving_avg_v2(values: pd.Series,
                           window: int = 5) -> pd.Series:
    """
    Weighted Moving Average used for benchmarking.  This implementation is
    using two pd.series like v1 and has additional feature: it deletes the
    window of values from the original series in an attempt to reduce memory.
    Index operations add drag and the function performs worse than v1.

    Args:
        values (pd.Series): price series
        window (int, optional): Defaults to 5.

    Returns:
        pd.Series: weighted moving average series
    """

    # Constant
    weights = pd.Series(range(1, window+1))
    weights = weights / weights.sum()

    # when period is greater than values, return
    if values.shape[0] <= window:
        return pd.Series([np.nan]*len(values))

    # initialize series and copy index from input series to return a matching index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index, name='wma')
    group = values.iloc[0:window-1]

    for idx in values.iloc[window-1:].index:
        group[idx] = values[idx]
        wma[idx] = (group.values * weights.values).sum()
        del group[group.index[0]]

    return wma


def _np_weighted_moving_avg(values: np.array, window: int = 5) -> np.array:
    '''
    Implementation of np.convolve weighted moving average.  'valid' is in
    context of signal processing and simply means to only compute values
    where the series overlap.  Full and Same options return an array with
    a shape greater than the input array.

    References:
        https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        https://en.wikipedia.org/wiki/Convolution

    '''
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    return np.convolve(values, weights[::-1], 'valid')


def weighted_moving_avg_v3(values: pd.Series, window: int = 5) -> pd.Series:
    """
    Backwards looking weighted moving average implemented with Numpy's
    convolution method. As the window changes, the weighted series is applied
    to different parts of the input series.

    References:
        https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        https://en.wikipedia.org/wiki/Convolution

    Args:
        values (pd.Series): price series
        window (int, optional): Defaults to 5.

    Returns:
        pd.Series: weighted moving average series

    """

    # when period is greater than values, return
    if values.shape[0] <= window:
        return pd.Series([np.nan]*len(values), index=values.index)

    # initialize series and copy index from input series to return a matching index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index, name='wma')
    wma.iloc[window-1:] = _np_weighted_moving_avg(values.values, window)

    return wma


if __name__ == '__main__':
    # validation testing
    data, target = load_wma_validation_data()
    window = 20

    v1 = weighted_moving_avg_v1(data, window=window)
    assert_series_equal(v1, target, check_names=False)

    v2 = weighted_moving_avg_v2(data, window=window)
    assert_series_equal(v2, target, check_names=False)

    v3 = weighted_moving_avg_v3(data, window=window)
    assert_series_equal(v3, target, check_names=False)
    
    # performance profiling
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
