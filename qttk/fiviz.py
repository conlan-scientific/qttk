# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk
#
# fiviz.py - Financial Visualization
# candles chart with rsi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from profiler import time_this

def fiviz(x, y1, y2):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x, y1)
    axs[0].set_ylabel('Close')
    axs[0].grid(True)

    axs[1].plot(x, y2)
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('RSI')
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()

@time_this
def fillinValues(dataframe):
    # fill in NaN values
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)
    return dataframe

@time_this
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

@time_this
def moving_average(df_slice, window):
    i = window
    new_index = df_slice.index.tolist()
    mvgAvg = pd.DataFrame(np.nan, index=new_index, columns=[0])
    while i < len(df_slice)+1:
        window_start = i - window
        j = i - 1  # row index reference
        window_end = j
        mvgAvg.iloc[j] = np.sum(df_slice[window_start:window_end])/window
        i = i + 1
    return mvgAvg

@time_this
def rsi(rets, window):
    '''
    RSI: Relative Strength Index
            RSI = 100 - (100/(1+RS))
            RS = (avg of x days' up closes)/(avg of x days' down closes)

            avg of x days' up closes = total points gained on up days/weeks divide by x days/weeks
            avg of x days' down closes = total points lost on down days/weeks divide by x days/weeks

            from: page 239 Technical Analysis of the Financial Markets, 1st ed. by Murphy, John J.

            rets are the net returns. use function net_returns() to calculate.

            window is x days/weeks for the moving average calculation

            returns a series of RSI values
    '''
    # date_range is used to reindex after separating days up from days down
    date_range = rets.index

    up = rets.loc[rets.iloc[:]>=0]
    up = up.reindex(date_range, fill_value=0)
    up_avg = moving_average(up, window)
    up_avg = up_avg.fillna(value=0)

    down = rets.loc[rets.iloc[:]<0]
    down = down.reindex(date_range, fill_value=0)
    down_avg = moving_average(down, window)*-1
    down_avg = down_avg.fillna(value=0)
    # replace 0s with last non-zero value
    down_avg.replace(to_replace=0, method='ffill')

    # calculate rsi
    rs = up_avg/down_avg
    rsi = 100 - (100/(1+rs))
    return rsi

if __name__ == '__main__':
    path = os.path.dirname(os.getcwd())
    filename = path+'\data\SPY.csv'
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    window = 14
    rets_SPY = net_returns(data)
    rsi_SPY = rsi(rets_SPY, window)
    x = -window*3
    fiviz(data.index[x:], data['close'].iloc[x:], rsi_SPY.iloc[x:])
