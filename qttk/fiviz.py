'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# plot.py - Financial Visualization
# candles chart with rsi

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/plot.py

# production version: 2021-01-25
'''
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
import os
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange
from Bollinger_1 import bollinger


def plot(dataframe: pd.DataFrame) -> None:
    '''
    Plots 2 subplots, an equity curve with Bollinger bands and an rsi plot
    '''
    fig, axs = plt.subplots(2, 1, figsize=(10,6))
    plt.subplots_adjust(hspace=0.3)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    '''
    characters {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}, which are short-hand
    notations for shades of blue, green, red, cyan, magenta, yellow, black, and white
    '''
    axs[0].set_title('Ticker Price')
    axs[0].xaxis.set_major_locator(locator)
    axs[0].xaxis.set_major_formatter(formatter)
    axs[0].plot(dataframe[['BOLU']], c='g', lw=1.0, ls='-.', label='BB-Upper')
    axs[0].plot(dataframe[['MA_Close']], c='y', lw=1.0, ls='-.', label='MA-Close')
    axs[0].plot(dataframe[['BOLD']], c='r', lw=1.0, ls='-.', label='BB-Lower')
    axs[0].plot(dataframe[['close']], c='k', lw=1.0, ls='dotted', label='Close')
    axs[0].scatter(dataframe.index, dataframe[['open']], s=4.0, c='b',\
     marker=".")
    axs[0].scatter(dataframe.index, dataframe[['high']], s=4.0, c='g',\
    marker=".")
    axs[0].scatter(dataframe.index, dataframe[['low']], s=4.0, c='r',\
    marker=".")
    axs[0].scatter(dataframe.index, dataframe[['close']], s=4.0, c='k',\
    marker=".")
    axs[0].set_ylabel('Price')
    axs[0].legend(loc=0)
    axs[0].grid(True)

    axs[1].set_title('Ticker RSI')
    axs[1].xaxis.set_major_locator(locator)
    axs[1].xaxis.set_major_formatter(formatter)
    axs[1].set_ylim(0, 100)
    axs[1].plot(dataframe[['rsi']], c='k', lw=1.0, ls='dotted',\
     marker=".", label="RSI")
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('rsi')
    axs[1].legend(loc='lower right')
    axs[1].grid(True)

    plt.show()

def load_sample_ticker():
    '''
    Loads example EOD data for SPY
    '''
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'SPY.csv')
    dataframe = pd.read_csv(filename, index_col=0, parse_dates=True)
    return dataframe

def _fillinValues(dataframe:pd.DataFrame)->pd.DataFrame:
    '''
    Fill in NaN values
    '''
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)
    return dataframe

def compute_net_returns(dataframe:pd.DataFrame)->pd.DataFrame:
    '''
    Net return(t) = Price(t)/Price(t-1) - 1
    from: page 13, Machine Trading by Chan, E.P.

    returns an instrument's net return

    dataframe is a dataframe that needs to be in the following format:
    index        0    1     2   3    4
    YYYY-MM-DD   open close low high volume
    '''
    price = dataframe['close']
    rets = price/price.shift(1)-1
    # fill in NaN values
    rets = _fillinValues(rets)
    return rets

def _save_data(filename, dataframe: pd.DataFrame):
    # _save_data is a convenience method to save data to .csv file
    # dataframe needs to be a Pandas dataframe
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'validation_data', '{}.csv'.format(filename))
    dataframe.to_csv(filename)

def compute_rsi(dataframe:pd.DataFrame, window=14) -> pd.DataFrame:
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
    rets = compute_net_returns(dataframe)
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
    rsi = rsi.to_frame()
    rsi.rename(columns={0:'rsi'}, inplace=True)
    rsi.set_index(date_range, inplace=True)
    rsi.fillna(value=1.0, inplace=True)
    return rsi

def test(window):
    '''
    This function defines unit tests for the two main functions:
    compute_net_returns() and rsi()
    '''
    # load a known dataset to execute tests against
    dataframe = load_sample_ticker()

    # test compute_net_returns() to validate the results
    # if a test fails, an assertion error will be shown
    # no result is showon when a test passes
    from pandas._testing import assert_frame_equal

    path = os.path.dirname(__file__)
    filename_rets = os.path.join(path, '..', 'data', 'validation_data', 'rets_SPY.csv')
    test_rets_validated = pd.read_csv(filename_rets, index_col=0, parse_dates=True)
    test_rets = pd.DataFrame(compute_net_returns(dataframe))
    #_save_data('test_rets', test_rets)
    assert_frame_equal(test_rets_validated, test_rets)

    # test rsi() to validate the results
    # window must be equal to 14 for test to pass
    if window != 14:
        window = 14
    filename_rsi = os.path.join(path, '..', 'data', 'validation_data', 'rsi_SPY.csv')
    test_rsi_validated = pd.read_csv(filename_rsi, index_col=0, parse_dates=True)
    test_rsi = compute_rsi(dataframe, window)
    assert_frame_equal(test_rsi_validated, test_rsi)


if __name__ == '__main__':
    # load sample data
    dataframe = load_sample_ticker()
    # window defines the period used for rsi
    # a shorter window makes rsi more sensitive to daily price changes
    window = 14

    rsi = compute_rsi(dataframe, window)
    to_plot = bollinger(dataframe)

    x = -window                  # define the date range for plot to plot
    to_plot = to_plot.iloc[x:]
    to_plot['rsi'] = rsi.iloc[x:, [0]]
    plot(to_plot)

    # Execute unit tests
    test(window)
    ''' todo: timed_report() raises 'n_values' error
    # Performance Characterization
    # timed_report()
    exp_range = ExponentialRange(1, 5, 1/4)

    with timed_report():
        for i in exp_range.iterator():
            rsi_SPY = compute_rsi(dataframe, window)
    '''
    exit
