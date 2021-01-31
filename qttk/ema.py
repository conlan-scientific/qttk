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
#from qttk.profiler import time_this
import pandas as pd
import numpy as np
import os
from profiler_v2 import time_this, timed_report
from profiler_v2 import ExponentialRange
from qttk.utils.data_validation import load_ema_validation_data
from pandas._testing import assert_series_equal


def exponential_moving_average_v1(values:      pd.Series,
                                  com:         Optional[float] = None,
                                  span:        Optional[float] = None,
                                  halflife:    Union[None,
                                                     float,
                                                     str,
                                                     timedelta] = None,
                                  alpha:       Optional[float] = None,
                                  min_periods: int = 5,
                                  adjust:      bool = True,
                                  ignore_na:   bool = False,
                                  axis:        int = 0,
                                  times:       Union[None,
                                                     str,
                                                     np.ndarray,
                                                     FrameOrSeries] = None) -> pd.Series:
    '''
    Accepts a series of values and returns an exponentially
       weighted moving average series
    vanilla pandas.DataFrame.ewm() function
    '''
    # default to alpha
    check_args = (com, span, halflife, alpha)
    if not [x for x in check_args  if x is not None]:
        pass


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
                        window: int = 0) -> np.array:
    '''
    numpy's convolve method to get a performance boost
    modified Divakar's answer
    http://bit.ly/36487o9
    '''
    # numpy's error: TypeError: No loop matching the specified signature
    if not isinstance(alpha, float):
        raise TypeError("Please set alpha value to a float")

    weights = (1-alpha)**np.arange(window)
    weights /= weights.sum()
    out = np.convolve(values, weights)
    out[:window-1] = np.nan
    out = pd.Series(out)
    return out[:values.size]


def exponential_moving_average_v2(values: pd.Series,
                                  alpha: float = 0,
                                  window: int = 5) -> pd.Series:
    '''
    Uses numpy's convolve method to slightly outperform native pandas.DataFrame.ewm
      at the cost of some features.
    '''
    values = _numpy_ewm_alpha_v2(values.values, alpha=alpha, window=window)
    return values


if __name__ == '__main__':
    #validation testing
    window = 20
    '''data, target = load_ema_validation_data()    

    v1 = exponential_moving_average_v1(data,
                                       adjust=False,
                                       span=window,
                                       min_periods=window)
    assert_series_equal(v1, target, check_names=False)

    v2 = exponential_moving_average_v2(data, window=window, alpha=.1)
    assert_series_equal(v2, target, check_names=False)'''

    exp_range = ExponentialRange(1, 4, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])

        for i in exp_range.iterator():            
            tt(exponential_moving_average_v1)(series.iloc[:i],
                                              min_periods=window,
                                              span=window)

        for i in exp_range.iterator():
            tt(exponential_moving_average_v2)(series.iloc[:i], window=window, alpha=.1)


