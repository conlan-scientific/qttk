'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# rsi.py - Relative Strength Index

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/rsi.py

# production version: 2021-01-27
'''
__all__ = ['compute_net_returns', 'compute_rsi']

from datetime import datetime
import pandas as pd
import numpy as np
import os
#from qttk.profiler import time_this
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange

def load_sample_ticker():
    '''
    Loads example EOD data for YPS
    '''
    path = os.path.dirname(__file__)
    filename = os.path.join(path, 'data', 'eod', 'YPS.csv')
    dataframe = pd.read_csv(filename, index_col=0, parse_dates=True)
    return dataframe

def _fillinValues(dataframe:pd.DataFrame)->pd.DataFrame:
    '''
    Fill in NaN values
    '''
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)
    return dataframe

def compute_net_returns(series:pd.Series)->pd.Series:
    '''
    Net return(t) = Price(t)/Price(t-1) - 1
    from: page 13, Machine Trading by Chan, E.P.

    returns an instrument's net return

    dataframe is a dataframe that needs to be in the following format:
    index        0    1     2   3    4
    YYYY-MM-DD   open close low high volume
    '''
    price = series
    rets = price/price.shift(1)-1
    # fill in NaN values
    rets = _fillinValues(rets)
    return rets

def _save_data(filename, dataframe: pd.DataFrame):
    # _save_data is a convenience method to save data to .csv file
    # dataframe needs to be a Pandas dataframe
    path = os.path.dirname(__file__)
    filename = os.path.join(path, 'data', 'validation_data', '{}.csv'.format(filename))
    dataframe.to_csv(filename)


def compute_rsi(series:pd.Series, window=14) -> pd.Series:
    '''
    rsi: Relative Strength Index
        rsi = 100 - (100/(1+RS))
        RS = (avg of x days' up closes)/(avg of x days' down closes)

        avg of x days' up closes = total points gained on up days/weeks divide by x days/weeks
        avg of x days' down closes = total points lost on down days/weeks divide by x days/weeks

        from: page 239 Technical Analysis of the Financial Markets, 1st ed. by Murphy, John J.

        rets are the net returns. use function compute_net_returns() to calculate.

        window is x days for the moving average calculation

    returns a series of rsi values

    Completed rsi in 4.283 milliseconds
    '''
    # rsi algorithm validated against Excel: 2021-01-17v3

    # calculate daily net returns
    rets = compute_net_returns(series)
    # date_range is used to reindex after separating days up from days down
    date_range = rets.index
    up = rets.loc[rets.iloc[:] >= 0.0]
    up = up.reindex(date_range, fill_value = 0.0)
    #_save_data('up', up)

    up_avg = up.rolling(window=window).mean()

    up_avg = up_avg.fillna(value = 0.0)
    #_save_data('up_avg', up_avg)
    down = rets.loc[rets.iloc[:] < 0.0]
    down = down.reindex(date_range, fill_value = 0.0)
    #_save_data('down', down)

    down_avg = down.rolling(window=window).mean()*-1

    down_avg = down_avg.fillna(value = 0.0)
    # replace 0s with 1s
    down_avg.replace(to_replace = 0.0, value = 1.0)
    #_save_data('down_avg', down_avg)
    # calculate rsi
    rs = up_avg/down_avg
    rsi = 100 - (100/(1+rs))
    rsi = rsi.rename('rsi')
    rsi.fillna(value=1.0, inplace=True)
    return rsi

def _test(window):
    '''
    This function defines unit tests for the two main functions:
    compute_net_returns() and rsi()
    '''
    # load a known dataset to execute tests against
    dataframe = load_sample_ticker()

    # test compute_net_returns() to validate the results
    # if a test fails, an assertion error will be shown
    # no result is shown when a test passes
    from pandas._testing import assert_frame_equal
    from pandas._testing import assert_series_equal

    path = os.path.dirname(__file__)
    filename_rets = os.path.join(path, 'data', 'validation_data', 'rets_YPS.csv')
    test_rets_validated = pd.read_csv(filename_rets, index_col=0, parse_dates=True)
    test_rets = pd.DataFrame(compute_net_returns(dataframe['close']))
    assert_frame_equal(test_rets_validated, test_rets)

    # window must be equal to 14 for test to pass
    if window != 14:
        window = 14

    filename_rsi = os.path.join(path, 'data', 'validation_data', 'rsi_YPS.csv')
    test_rsi_validated = pd.read_csv(filename_rsi, index_col=0, parse_dates=True)
    test_rsi_series = test_rsi_validated.iloc[:, 0]
    test_rsi = compute_rsi(dataframe['close'], window)
    assert_series_equal(test_rsi_series, test_rsi)


if __name__ == '__main__':
    # load sample data
    dataframe = load_sample_ticker()
    # window defines the period used for rsi
    # a shorter window makes rsi more sensitive to daily price changes
    window = 14

    rsi = compute_rsi(dataframe['close'], window)

    # Execute unit tests
    _test(window)

    # Performance Characterization
    exp_range = ExponentialRange(1, 5, 1/4)

    test_columns = ['date', 'open', 'close', 'low', 'high', 'volume']
    test_df = pd.DataFrame(np.random.rand(exp_range.max,6), columns=test_columns)

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])
        for i in exp_range.iterator():
            tt(compute_rsi)(test_df.iloc[:i, 2], window)
