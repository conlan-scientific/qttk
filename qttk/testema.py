'''
Profiled and tested exponential moving average functions
'''
from typing import List, Dict, Any, Optional, Union
from datetime import timedelta
from pandas._typing import FrameOrSeries
import pandas as pd
import numpy as np
from scipy.signal import lfiltic, lfilter  # testing purposes only
from profiler import time_this


def exponential_moving_average_v1(values:      pd.Series,
                                  com:         Optional[float] = None,
                                  span:        Optional[float] = None,
                                  halflife:    Union[None, float, str, timedelta] = None,
                                  alpha:       Optional[float] = None,
                                  min_periods: int = 5,
                                  adjust:      bool = True,
                                  ignore_na:   bool = False,
                                  axis:        int = 0,
                                  times:       Union[None, str, np.ndarray, FrameOrSeries] = None) -> pd.Series:
    '''
    Accepts a series of values and returns an exponentially
       weighted moving average series

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
    modified Divakar's answer
    https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    '''
    # numpy's error: TypeError: No loop matching the specified signature and casting was found for ufunc true_divide
    if not isinstance(alpha, float):
        raise TypeError("Please set alpha value to a float")

    weights = (1-alpha)**np.arange(min_periods)
    weights /= weights.sum()
    out = np.convolve(values, weights)
    out[:min_periods-1] = np.nan
    return out[:values.size]


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


def exponential_moving_average_v3(values: np.array, min_periods: int = 5):
    '''
    Scipy alternative 
    '''
    alpha = 2 / (min_periods + 1)
    b = [alpha]
    a = [1, alpha-1]
    zi = lfiltic(b, a, values[0:1], [0])
    return lfilter(b, a, values, zi=zi)[0]

if __name__ == '__main__':
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mp = 5
    a = 2/(mp + 1)

    x = exponential_moving_average_v1(series, min_periods=mp, alpha=a)
    y = exponential_moving_average_v2(series, min_periods=mp, alpha=a)
    z = exponential_moving_average_v3(series, min_periods=mp)
    #print(x)
    #print(y)
    if not x.equals(y):
        print((x[mp:] == y[mp:]).value_counts())
        print(f'mean of difference: {np.mean((x[mp:] - y[mp:])) :.7f}\n')