"""
Exponential Moving Average Functions

The exponential moving average (EMA) is a technical indicator
that tracks the price of an investment (like a stock or commodity)
over time. The EMA is a type of weighted moving average (WMA) that gives
more weighting or importance to more recent price data.

Reference:
    `Investopedia.com <http://bit.ly/3sRRxBx>`_

Profiled and tested exponential moving average functions
"""

from typing import List, Dict, Any, Optional, Union
from datetime import timedelta
from pandas._typing import FrameOrSeries
from scipy.signal import lfiltic, lfilter  
from qttk.profiler import time_this
import pandas as pd
import numpy as np
import os
import gc
from qttk.profiler_v2 import time_this, ExponentialRange, timed_report


def simple_ema(values: pd.Series, alpha: float = 0.0, window: int = 5) -> pd.Series:
    """
    Simple Exponential Moving Average using pandas series.  Use this function
    for a benchmark comparison.

    Args:
        values (pd.Series): closing price
        alpha (float): smoothing factor, usually between .1 - .3
        window (int): periods for window function

    Returns:
        pd.Series: exponential moving average
    """
    
    if not alpha:
        alpha = 2 / (min_periods + 1)
    
    ema = pd.Series([np.nan]*values.shape[0], name='ema')
    sma = pd.Series(values[:window].sum()/window)
    ema.iloc[window - 1] = (alpha * values.iloc[window-1]) + ((1-alpha) * sma)

    for i in range(window, values.shape[0]):
        ema.iloc[i] = alpha * values.iloc[i-1] + (1-alpha) * ema.iloc[i-1]

    return ema


def python_simple_ema(values: List, alpha: float = 0.0, window: int = 5) -> List:
    """
    Simple Exponential Moving Average written in python.  It is used here to 
    study the performance attributes of different approaches.  This function
    can be useful for SBCs or other tiny devices.  

    Args:
        values (List): closing prices
        alpha (float): smoothing factor
        window (int): periods to consider in EMA calculation

    Returns:
        List: exponential moving average
    """

    ema = [None] * len(values)

    if not alpha:
        alpha = 2 / (min_periods + 1)

    #initialize first ema with simple moving average
    sma = sum(values[:window])/window
    ema[window-1] = (alpha * values[window-1]) + (1-alpha) * sma

    for i in range(window, len(values)):
        ema[i] = (alpha*values[i-1]) + (1 - alpha) * ema[i-1]

    return ema


def exponential_moving_average_v1(values:      pd.Series,
                                  com:         Optional[float] = None,
                                  span:        Optional[float] = None,
                                  halflife:    Union[None, float, str, \
                                  timedelta] = None,
                                  alpha:       Optional[float] = None,
                                  min_periods: int = 5,
                                  adjust:      bool = True,
                                  ignore_na:   bool = False,
                                  axis:        int = 0,
                                  times:       Union[None, str, np.ndarray, \
                                  FrameOrSeries] = None) -> pd.Series:
    """
    Pandas implementation fo exponential moving average.  

    Args:
        values (pd.Series): [description]
        com (Optional[float], optional): [description]. Defaults to None.
        span (Optional[float], optional): [description]. Defaults to None.
        halflife (Union[None, float, str,, optional): [description]. Defaults to None.
        alpha (Optional[float], optional): [description]. Defaults to None.
        min_periods (int, optional): [description]. Defaults to 5.
        adjust (bool, optional): [description]. Defaults to True.
        ignore_na (bool, optional): [description]. Defaults to False.
        axis (int, optional): [description]. Defaults to 0.
        times (Union[None, str, np.ndarray,, optional): [description]. Defaults to None.

    Returns:
        pd.Series: exponential moving average
    """
    return values.ewm(com=com,
                      span=span,
                      halflife=halflife,
                      alpha=alpha,
                      min_periods=min_periods,
                      adjust=adjust,
                      ignore_na=ignore_na,
                      axis=axis,
                      times=times).mean()


def _numpy_ewm_alpha_v2(values: np.array,
                        alpha: float = 0,
                        min_periods: int = 0) -> np.array:
    '''
    numpy's convolve method to get a performance boost
    modified Divakar's answer
    http://bit.ly/36487o9
    '''
    # numpy's error: TypeError: No loop matching the specified signature and
    # casting was found for ufunc true_divide
    if not isinstance(alpha, float):
        raise TypeError("Please set alpha value to a float")

    weights = (1-alpha)**np.arange(min_periods)
    weights /= weights.sum()
    out = np.convolve(values, weights)
    out[:min_periods-1] = np.nan
    out = pd.Series(out)
    return out[:values.size]


def exponential_moving_average_v2(values: pd.Series,
                                  alpha: float = 0.0,
                                  min_periods: int = 5) -> pd.Series:
    """
    Uses Numpy's convolve method to calculate an exponential moving average.
    This function is vectorized and fast, but it has accuracy issues to resolve.

    Args:
        values (pd.Series): closing prices
        alpha (float, optional): smoothing factor, usually between .1 - .3 Defaults to 2/(N+1).
        min_periods (int, optional): periods for window function. Defaults to 5.

    Returns:
        pd.Series: exponential moving average series
    """

    if not alpha: # argument is blank
        alpha = 2 / (min_periods + 1)

    a = _numpy_ewm_alpha_v2(values.values, alpha=alpha, min_periods=min_periods)
    values = a
    return values


def exponential_moving_average_v3(series: pd.Series,
                                  alpha: float = 0.0,
                                  min_periods: int = 5):
    """
    The exponential moving average (EMA) is a technical indicator
    that tracks the price of an investment (like a stock or commodity)
    over time. The EMA is a type of weighted moving average (WMA) that gives
    more weighting or importance to more recent price data.

    Args:
        values (pd.Series): closing price 
        alpha (float, optional): smoothing factor, usually between .1 - .3 Defaults to 2/(N+1).
        min_periods (int, optional): periods for window function. Defaults to 5.

    Returns:
        EMA (pd.Series): series named EMA 
    """
    if not alpha:
        alpha = 2 / (min_periods + 1)

    values = series.to_numpy()    
    b = [alpha]
    a = [1, alpha-1]
    zi = lfiltic(b, a, values[0:1], [0])
    output = pd.Series(lfilter(b, a, values, zi=zi)[0])
    output.index = series.index
    output.rename('EMA', inplace=True)
    return output


if __name__ == '__main__':

    i = 1 # exponent to generate 10**i range

    series = pd.Series(np.arange(0, 10**i))
    mp = 5
    alpha = 2/(mp + 1)

    x = exponential_moving_average_v1(series, min_periods=mp, alpha=alpha)
    y = exponential_moving_average_v2(series, min_periods=mp, alpha=alpha)
    z = exponential_moving_average_v3(series, min_periods=mp)
    z = pd.Series(z) # convert numpy.ndarray to Pandas series
    a = simple_ema(series, alpha=alpha, window=mp)
    b = python_simple_ema(series.to_list(), alpha=alpha, window=mp)
    b = pd.Series(b)

    if not x.equals(y):
        print('\nX not equal to Y: ', (x[mp:] != y[mp:]).value_counts())
        print(f'mean of difference: {np.mean((x[mp:] - y[mp:])) :.7f}\n')
    if not y.equals(z):
        print('Y not equal to Z: ', (y[mp:] != z[mp:]).value_counts())
        print(f'mean of difference: {np.mean((y[mp:] - z[mp:])) :.7f}\n')
    if not z.equals(x):
        print('Z not equal to X: ', (z[mp:] != x[mp:]).value_counts())
        print(f'mean of difference: {np.mean((z[mp:] - x[mp:])) :.7f}\n')

    if not x.equals(a):
        print('a not equal to X: ', (z[mp:] != x[mp:]).value_counts())
        print(f'mean of difference: {np.mean((a[mp:] - x[mp:])) :.7f}\n')

    if not x.equals(b):
        print('b not equal to X: ', (b[mp:] != x[mp:]).value_counts())
        print(f'mean of difference: {np.mean((b[mp:] - x[mp:])) :.7f}\n')


    truth_series1 = pd.Series([np.nan, np.nan, np.nan, np.nan, 2.758294,\
    3.577444, 4.435163, 5.324822, 6.240363, 7.176476])

    truth_series2 = pd.Series([np.nan, np.nan, np.nan, np.nan, 2.758294,\
    3.758294, 4.758294, 5.758294, 6.758294, 7.758294])

    truth_series3 = pd.Series([0.0, 0.333333, 0.888889, 1.592593, 2.395062,\
    3.263374, 4.175583, 5.117055, 6.078037, 7.052025])

    assert x.round(4).equals(truth_series1.round(4)), "truth series .4f failed"
    assert y.round(4).equals(truth_series2.round(4)), "truth series .4f failed"
    assert z.round(4).equals(truth_series3.round(4)), "truth series .4f failed"
    #assert a.round(4).equals(truth_series3.round(4)), "truth series .4f failed"
    #assert b.round(4).equals(truth_series3.round(4)), "truth series .4f failed"

    exp_range = ExponentialRange(1, 8, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])

        for i in exp_range.iterator():
            tt(exponential_moving_average_v1)(series.iloc[:i],
                                              min_periods=mp,
                                              alpha=alpha)

        gc.collect()
        for i in exp_range.iterator():
            tt(exponential_moving_average_v2)(series.iloc[:i],
                                              min_periods=mp,
                                              alpha=alpha)

        gc.collect()
        for i in exp_range.iterator():
            tt(exponential_moving_average_v3)(series.iloc[:i],
                                              min_periods=mp)

        gc.collect()
        for i in exp_range.iterator(4):
            tt(simple_ema)(series.iloc[:i],
                           alpha=alpha,
                           window=mp)

        gc.collect()
        tt = time_this(lambda *args, **kwargs: len(args[0]))
        for i in exp_range.iterator(4):
            tt(python_simple_ema)(series.iloc[:i].tolist(),
                                  alpha=alpha,
                                  window=mp)
