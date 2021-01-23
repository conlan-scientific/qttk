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
import os
import pandas as pd
import numpy as np
from profiler import time_this


@time_this
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

@time_this
def cumulative_moving_avg(values: pd.Series, min_periods: int = 5) -> pd.Series:
    '''
    The Cumulative Moving Average is the unweighted mean of the previous values up to the current time t.
    pandas has an expand O(nm) because n number of periods in a loop and sum(m)

    '''
    counter=0

    # Initialize series
    cma = pd.Series([np.nan]*(min_periods-1))

    temp: List[float] = list()

    for i in range(min_periods, len(values)+1):

        temp.append(sum(values[:i])/(min_periods+counter))
        counter += 1

    return cma.append(pd.Series(temp)).reset_index(drop=True)


@time_this
def cumulative_moving_avg_v2(values: pd.Series, min_periods: int = 5) -> pd.Series:
    '''
    pd.Series.expand.mean() exists, but I tried to work out something vectorized
    '''

    # Initialize series
    cma = pd.Series([np.nan]*(min_periods-1))

    cma = cma.append((values.cumsum() / pd.Series(range(1, len(values)+1)))[min_periods-1:])

    return cma.reset_index(drop=True)


@time_this
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

def save_validation_data(function_name: str, data: pd.Series):
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'validation_data', \
    'ValidationData-{}.csv'.format(function_name))
    #data = fillinValues(data)
    data.to_csv(filename)

def test(function_name: str):
    from pandas.testing import assert_series_equal
    from pandas.testing import assert_frame_equal

    path = os.path.dirname(__file__)

    if function_name == 'simpleMA':
        # load validated output data
        filename = os.path.join(path, '..', 'data', 'validation_data', \
        'MA-ValidationData-{}.csv'.format(function_name))
        ma_validated = pd.read_csv(filename, index_col=0)
        validation_data = ma_validated[['0']]
        # load known input data matching validated output data
        filename_input = os.path.join(path, '..', 'data', 'validation_data', \
        'Validation-data_input.csv')
        df = pd.read_csv(filename_input, index_col=0)
        series = df[['0']]
    else:
        # load validated output data
        filename = os.path.join(path, '..', 'data', 'validation_data', \
        'ValidationData-CMA_output.csv')
        cma_validated = pd.read_csv(filename, index_col=0)
        validation_data = cma_validated[['0']]
        # load known input data matching validated output data
        filename_input = os.path.join(path, '..', 'data', 'validation_data', \
        'ValidationData-CMA_input.csv')
        df = pd.read_csv(filename_input, index_col=0)
        series = df[['0']]

    if function_name == 'simpleMA':
        x = pd_simple_moving_avg(series, min_periods=5)
        x = pd.DataFrame(x)
        assert_frame_equal(x, validation_data, check_dtype=False)
    elif function_name == 'CMA':
        y = cumulative_moving_avg(series, min_periods=5)
        y = pd.DataFrame(y)
        assert_frame_equal(y, validation_data, check_dtype=False)
    elif function_name == 'CMAv2':
        z = cumulative_moving_avg_v2(series, min_periods=5)
        z = pd.DataFrame(z)
        assert_frame_equal(z, validation_data, check_dtype=False)
    elif function_name == 'CMAv3':
        w = cumulative_moving_avg_v3(series, min_periods=5)
        w = pd.DataFrame(w)
        assert_frame_equal(w, validation_data, check_dtype=False)


if __name__ == '__main__':

    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'validation_data', \
    'ValidationData-CMA_input.csv')

    # load known input dataset
    df = pd.read_csv(filename, index_col=0)
    series = df[['0']]
    # test datsets
    #series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #series = pd.Series(np.random.random((1000 * 1000)))

    x = pd_simple_moving_avg(series, min_periods=5)
    '''
    # todo test y, z, w with ValidationData-CMA_input.csv
    '''
    #y = cumulative_moving_avg(series, min_periods=5)
    #z = cumulative_moving_avg_v2(series, min_periods=5)
    #w = cumulative_moving_avg_v3(series, min_periods=5)

    # save validation data
    #save_validation_data('simpleMA', x)
    #save_validation_data('CMA_output', z)

    # unit test cases - verify CMA output with known input
    test('simpleMA')
    '''
    # todo resolve MemoryError: Unable to allocate 7.28 TiB for
    # an array with shape (1000000, 1000000) and data type float64
    #test('CMAv2')
    '''
    #series = pd.Series(list(range(21, 41)))
    #x = pd_simple_moving_avg(series, min_periods=5)
    #y = cumulative_moving_avg(series, min_periods=5)
    #z = cumulative_moving_avg_v2(series, min_periods=5)
    #w = cumulative_moving_avg_v3(series, min_periods=5)

    '''
    truth_series = pd.Series([
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        23.0,
        23.5,
        24.0,
        24.5,
        25.0,
        25.5,
        26.0,
        26.5,
        27.0,
        27.5,
        28.0,
        28.5,
        29.0,
        29.5,
        30.0,
        30.5,
    ])
    assert y.equals(truth_series)
    assert z.equals(truth_series)
    assert w.equals(truth_series)
    '''
