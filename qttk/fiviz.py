# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# fiviz.py - Financial Visualization
# candles chart with rsi

# run from prompt: /qttk>ipython -i ./qttk/fiviz.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
import os
from profiler_v2 import time_this, timed_report
from profiler_v2 import ExponentialRange


def fiviz(x, y1, y2):
    fig, axs = plt.subplots(2, 1, figsize=(10,6))
    plt.subplots_adjust(hspace=0.3)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=30)
    formatter = mdates.ConciseDateFormatter(locator)

    axs[0].set_title('SPY OHLC Price')
    axs[0].xaxis.set_major_locator(locator)
    axs[0].xaxis.set_major_formatter(formatter)
    # transpose data for boxplots (i.e. candle plots)
    y1_transposed = y1.T
    axs[0].boxplot(y1_transposed, whis=[0,100])
    axs[0].set_ylabel('Price')
    axs[0].grid(True)

    axs[1].set_title('SPY RSI')
    axs[1].xaxis.set_major_locator(locator)
    axs[1].xaxis.set_major_formatter(formatter)
    axs[1].plot(x, y2)
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('RSI')
    axs[1].grid(True)

    plt.show()

def fillinValues(dataframe):
    # fill in NaN values
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)
    return dataframe

def net_returns(df):
    '''
    Net return(t) = Price(t)/Price(t-1) - 1
    from: page 13, Machine Trading by Chan, E.P.

    returns an instrument's net return

    df is a dataframe that needs to be in the following format:
    index        0    1     2   3    4
    YYYY-MM-DD   open close low high volume
    '''
    price = df['close']
    rets = price/price.shift(1)-1
    # fill in NaN values
    rets = fillinValues(rets)
    return rets

def save_data(filename, df):
    # save_data is a convenience method to save data to .csv file
    # df needs to be a Pandas dataframe
    filename = os.path.join(path, 'data', 'validation_data', '{}.csv'.format(filename))
    df.to_csv(filename)

def rsi(rets, window):
    '''
    RSI: Relative Strength Index
        RSI = 100 - (100/(1+RS))
        RS = (avg of x days' up closes)/(avg of x days' down closes)

        avg of x days' up closes = total points gained on up days/weeks divide by x days/weeks
        avg of x days' down closes = total points lost on down days/weeks divide by x days/weeks

        from: page 239 Technical Analysis of the Financial Markets, 1st ed. by Murphy, John J.

        rets are the net returns. use function net_returns() to calculate.

        window is x days for the moving average calculation

    returns a series of RSI values

    Completed rsi in 4.283 milliseconds
    '''
    # RSI algorithm validated against Excel: 2021-01-17v3

    # date_range is used to reindex after separating days up from days down
    date_range = rets.index
    up = rets.loc[rets.iloc[:] >= 0.0]
    up = up.reindex(date_range, fill_value = 0.0)
    #save_data('up', up)

    up_avg = up.rolling(window=window).mean()

    up_avg = up_avg.fillna(value = 0.0)
    #save_data('up_avg', up_avg)
    down = rets.loc[rets.iloc[:] < 0.0]
    down = down.reindex(date_range, fill_value = 0.0)
    #save_data('down', down)

    down_avg = down.rolling(window=window).mean()*-1

    down_avg = down_avg.fillna(value = 0.0)
    # replace 0s with 1s
    down_avg.replace(to_replace = 0.0, value = 1.0)
    #save_data('down_avg', down_avg)
    # calculate rsi
    rs = up_avg/down_avg
    rsi = 100 - (100/(1+rs))
    rsi = rsi.to_frame()
    rsi.rename(columns={0:'RSI'}, inplace=True)
    rsi.set_index(date_range, inplace=True)
    rsi.fillna(value=1.0, inplace=True)
    #save_data('rsi_SPY', rsi)
    return rsi

def test(window):
    '''
    This function defines unit tests for the two main functions:
    net_returns() and rsi()
    '''
    # load a known dataset to execute tests against
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'SPY.csv')
    data = pd.read_csv(filename, index_col=0, parse_dates=True)

    # test net_returns() to validate the results
    # if a test fails, an assertion error will be shown
    # no result is showon when a test passes
    from pandas._testing import assert_frame_equal

    filename_rets = os.path.join(path, '..', 'data', 'validation_data', 'rets_SPY.csv')
    test_rets_validated = pd.read_csv(filename_rets, index_col=0, parse_dates=True)
    test_rets = pd.DataFrame(net_returns(data))
    #save_data('test_rets', test_rets)
    assert_frame_equal(test_rets_validated, test_rets)

    # test rsi() to validate the results
    # window must be equal to 14 for test to pass
    if window != 14:
        window = 14
    filename_rsi = os.path.join(path, '..', 'data', 'validation_data', 'rsi_SPY.csv')
    test_rsi_validated = pd.read_csv(filename_rsi, index_col=0, parse_dates=True)
    test_rsi = rsi(test_rets['close'], window)
    assert_frame_equal(test_rsi_validated, test_rsi)


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'SPY.csv')
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    # window needs to be optimized for trading frequency/price autocorrelation
    # lag period
    window = 14

    date = datetime.now()
    now = date.strftime("%Y-%m-%d-%H%M%S")

    filename_rets_output = os.path.join(path, 'data', 'performance', \
    'retsSPYPerformanceData{}.txt'.format(now))
    rets_SPY = net_returns(data)

    filename_rsi_output = os.path.join(path, 'data', 'performance', \
    'rsiSPYPerformanceData{}.txt'.format(now))
    rsi_SPY = rsi(rets_SPY, window)

    # Execute unit tests
    test(window)
    ''' TODO: time_report() raises 'n_values' error
    # Performance Characterization
    # timed_report()
    exp_range = ExponentialRange(1, 5, 1/4)

    with timed_report():
        for i in exp_range.iterator():
            rsi_SPY = rsi(rets_SPY, window)
    '''
    x = -window*3                  # define the date range for fiviz to plot
    price = data[['open', 'close', 'low', 'high']]
    fiviz(data.index[x:], price.iloc[x:], rsi_SPY.iloc[x:])
