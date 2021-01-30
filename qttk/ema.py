'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

testema.py - Exponential Moving Average Functions

The exponential moving average (EMA) is a technical indicator
that tracks the price of an investment (like a stock or commodity)
over time. The EMA is a type of weighted moving average (WMA) that gives
more weighting or importance to more recent price data.

from: Investopedia.com
http://bit.ly/3sRRxBx

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/testema.py

Profiled and tested exponential moving average functions

for a list 10**4 long:
exponential_moving_average_v1: 0.650 milliseconds
exponential_moving_average_v2: 0.623 milliseconds
exponential_moving_average_v3: 0.339 milliseconds
'''
from typing import List, Dict, Any, Optional, Union
from datetime import timedelta
from pandas._typing import FrameOrSeries
from scipy.signal import lfiltic, lfilter  # testing purposes only
from qttk.profiler import time_this
import pandas as pd
import numpy as np
import os

#@time_this
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
    '''
    Accepts a series of values and returns an exponentially
       weighted moving average series
    pandas.DataFrame.ewm() function
    '''
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

#@time_this
def exponential_moving_average_v2(values: pd.Series,
                                  alpha: float = 0,
                                  min_periods: int = 5) -> pd.Series:
    '''
    Uses numpy's convolve method to slightly outperform native pandas.DataFrame.ewm
      at the cost of some features.
    '''
    a = _numpy_ewm_alpha_v2(values.values, alpha=alpha, min_periods=min_periods)
    values = a
    return values

#@time_this
def exponential_moving_average_v3(values: pd.Series, min_periods: int = 5):
    '''
    Scipy alternative
    '''
    values = values.to_numpy()
    alpha = 2 / (min_periods + 1)
    b = [alpha]
    a = [1, alpha-1]
    zi = lfiltic(b, a, values[0:1], [0])
    return lfilter(b, a, values, zi=zi)[0]


if __name__ == '__main__':

    i = 1 # exponent to generate 10**i range

    #from numpy.random import default_rng
    #rng = default_rng()
    #vals = rng.standard_normal(10**i)
    #series = pd.Series(vals)
    series = pd.Series(np.arange(0, 10**i))
    mp = 5
    a = 2/(mp + 1)

    x = exponential_moving_average_v1(series, min_periods=mp, alpha=a)
    y = exponential_moving_average_v2(series, min_periods=mp, alpha=a)
    z = exponential_moving_average_v3(series, min_periods=mp)
    z = pd.Series(z) # convert numpy.ndarray to Pandas series

    if not x.equals(y):
        print('\nX not equal to Y: ', (x[mp:] != y[mp:]).value_counts())
        print(f'mean of difference: {np.mean((x[mp:] - y[mp:])) :.7f}\n')
    if not y.equals(z):
        print('Y not equal to Z: ', (y[mp:] != z[mp:]).value_counts())
        print(f'mean of difference: {np.mean((y[mp:] - z[mp:])) :.7f}\n')
    if not z.equals(x):
        print('Z not equal to X: ', (z[mp:] != x[mp:]).value_counts())
        print(f'mean of difference: {np.mean((z[mp:] - x[mp:])) :.7f}\n')

    truth_series1 = pd.Series([np.nan, np.nan, np.nan, np.nan, 2.758294,\
    3.577444, 4.435163, 5.324822, 6.240363, 7.176476])

    truth_series2 = pd.Series([np.nan, np.nan, np.nan, np.nan, 2.758294,\
    3.758294, 4.758294, 5.758294, 6.758294, 7.758294])

    truth_series3 = pd.Series([0.0, 0.333333, 0.888889, 1.592593, 2.395062,\
    3.263374, 4.175583, 5.117055, 6.078037, 7.052025])

    assert x.round(4).equals(truth_series1.round(4))
    assert y.round(4).equals(truth_series2.round(4))
    assert z.round(4).equals(truth_series3.round(4))

    exit
