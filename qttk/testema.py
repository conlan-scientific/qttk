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
#from profiler_v2 import time_this, timed_report
#from profiler_v2 import ExponentialRange

@time_this
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
    vanilla pandas.DataFrame.ewm() function
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

@time_this
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

@time_this
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

def save_validation_data(function_name: str, data: pd.Series):
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-{}.csv'.format(function_name))
    #data = fillinValues(data)
    data.to_csv(filename)

def test(function_name: str, i):
    from pandas.testing import assert_series_equal
    from pandas.testing import assert_frame_equal

    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_{}.csv'.format(function_name))
    validation_data = pd.read_csv(filename, index_col=0)

    # to cross validate the functions against one another
    if function_name == 'v1':
        filename2 = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_v2.csv')
    elif function_name == 'v2':
        filename2 = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_v1.csv')
    else:
        filename2 = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_v2.csv')

    # defines input data and parameters for the EMA functions
    series = pd.read_csv(os.path.join(path, '..', 'data', 'validation_data', \
    'Validation-data_input.csv'))
    mp = 5
    a = 2/(mp + 1)
    '''
    # unit test cases:
    x == x, _x == y, y == y, y == x, z == z, z == y

    todo resolve DataFrame shape mismatch
    line 153
    [left]:  (1000000, 2)
    [right]: (1000000, 1)
    '''
    if function_name == 'v1':
        x = exponential_moving_average_v1(series, min_periods=mp, alpha=a)
        x = pd.DataFrame(x)
        assert_frame_equal(x, validation_data, check_dtype=False)
        y = pd.read_csv(filename2)
        assert_frame_equal(x, y, check_dtype=False)
    elif function_name == 'v2':
        y = exponential_moving_average_v2(series, min_periods=mp, alpha=a)
        y = pd.DataFrame(y)
        assert_frame_equal(y, validation_data, check_dtype=False)
        x = pd.read_csv(filename2)
        assert_frame_equal(y, x, check_dtype=False)
    else:
        z = exponential_moving_average_v2(series, min_periods=mp)
        z = pd.DataFrame(z)
        assert_frame_equal(z, validation_data, check_dtype=False)
        y = pd.read_csv(filename2)
        assert_frame_equal(z, y, check_dtype=False)


if __name__ == '__main__':

    i = 4 # exponent to generate 10**i range

    from numpy.random import default_rng
    rng = default_rng()
    vals = rng.standard_normal(10**i)
    series = pd.Series(vals)
    #series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mp = 5
    a = 2/(mp + 1)

    x = exponential_moving_average_v1(series, min_periods=mp, alpha=a)
    y = exponential_moving_average_v2(series, min_periods=mp, alpha=a)
    z = exponential_moving_average_v3(series, min_periods=mp)

    # these lines generate validation data for unit test cases:
    #save_validation_data('data_input', series)
    #save_validation_data('output_v1', x)
    #save_validation_data('output_v2', x)
    #save_validation_data('output_v3', x)

    # implement unit test cases
    test('v1', i)
    test('v2', i)
    test('v3', i)

    if not x.equals(y):
        print((x[mp:] == y[mp:]).value_counts())
        print(f'mean of difference: {np.mean((x[mp:] - y[mp:])) :.7f}\n')

    '''
    todo implement timed_report() - raises n_values key value error
    with timed_report():
        for i in exp_range.iterator():
            x = exponential_moving_average_v1(series.iloc[:i], min_periods=mp, alpha=a)

        for i in exp_range.iterator():
            y = exponential_moving_average_v2(series.iloc[:i], min_periods=mp, alpha=a)

        for i in exp_range.iterator():
            z = exponential_moving_average_v3(series.iloc[:i], min_periods=mp)
    '''
