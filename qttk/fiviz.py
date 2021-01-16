# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk
#
# fiviz.py - Financial Visualization
# candles chart with rsi

import pandas as pd
import numpy as np
import matplotlib
import os
from profiler import time_this

def fiviz(data):
    data.iloc[-10:, 2].plot()

@time_this
def moving_average(series, window):
    i = window
    mvgAvg = pd.DataFrame(np.nan, index=range(0, len(series)), columns=['mvgAvg'])
    mvgAvg.iloc[:,0] = mvgAvg.iloc[:,0] * np.nan # fill with NaNs
    while i < len(series):
        mvgAvg.iloc[i] = np.mean(series[:i])
        i = i + 1
    return mvgAvg

@time_this
def dropna(series):
    return series[~np.isnan(series)]

@time_this
def rsi(data, window):
    '''
    RSI: Relative Strength index
            RSI = 100 - (100/(1+RS))
            RS = (avg of x days' up closes)/(avg of x days' down closes)

            avg of x days' up closes = total points gained on up days/weeks divide by x days/weeks
            avg of x days' down closes = total points lost on down days/weeks divide by x days/weeks

            from: page 239 Technical Analysis of the Financial Markets, 1st ed. by Murphy, John J.

            data is a dataframe in the following format:
            index        0    1     2   3    4
            YYYY-MM-DD   open close low high volume

            window is x days/weeks for the moving average calculation

            returns a series of RSI values
    '''
    up = data.loc[data['close'] > data['open'], 'close']
    up_avg = moving_average(up, window)
    up_avg = dropna(up_avg)  # drop nan's

    down = data.loc[data['close'] < data['open'], 'close']
    down_avg = moving_average(down, window)
    down_avg = dropna(down_avg) # drop nan's

    rs = up_avg/down_avg
    rsi = 100 - (100/(1+rs))

    return rsi


if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    filename = path+'\data\SPY.csv'
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    window = 14
    rsi_SPY = rsi(data, window)
    rsi_SPY.plot()
