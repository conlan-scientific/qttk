'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# ma_crossover.py - Moving Average Crossover Strategy Example
# http://bit.ly/3iQ9k7y

# run from project directory:
    C:/Users/user/qttk>ipython -i ./examples/ma_crossover.py

# production version: 2021-01-28
'''
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
import os
from qttk.indicators import ma
'''
Moving Average function:
ma(dataframe_slice:pd.DataFrame, window:int)
returns a moving average as a dataframe
'''
def _plot(ticker:str, ticker_data: pd.DataFrame) -> None:
    '''
    Plots an equity curve with short and long moving average lines
    '''
    fig, axs = plt.subplots(1, 1, figsize=(10,6))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    '''
    characters {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}, which are short-hand
    notations for shades of blue, green, red, cyan, magenta, yellow, black, and white
    '''
    axs.set_title(ticker+' Price')
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(formatter)
    axs.plot(ticker_data[['ma_short']], c='c', lw=1.0, ls='-.', label='MA-Short')
    axs.plot(ticker_data[['ma_long']], c='b', lw=1.0, ls='-.', label='MA-Long')
    axs.plot(ticker_data[['close']], c='k', lw=1.0, ls='dotted', label='Close')
    axs.scatter(ticker_data.index, ticker_data[['open']], s=4.0, c='b',\
    marker=".")
    axs.scatter(ticker_data.index, ticker_data[['high']], s=4.0, c='g',\
    marker=".")
    axs.scatter(ticker_data.index, ticker_data[['low']], s=4.0, c='r',\
    marker=".")
    axs.scatter(ticker_data.index, ticker_data[['close']], s=4.0, c='k',\
    marker=".")
    axs.set_ylabel('Price')
    axs.legend(loc=0)
    axs.grid(True)

    plt.show()


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    ticker = 'DLVY'
    filename = os.path.join(path, '..', 'data', 'eod', ticker+'.csv')
    dataframe = pd.read_csv(filename, index_col=0, parse_dates=True)

    ma_short = ma(dataframe['close'], 10)
    ma_long = ma(dataframe['close'], 25)

    ma_short = ma_short.set_index(dataframe.index)
    ma_long = ma_long.set_index(dataframe.index)

    dataframe['ma_short'] = ma_short.iloc[:, [0]]
    dataframe['ma_long'] = ma_long.iloc[:, [0]]

    to_plot = dataframe.copy()

    window = 30                  # define the date range for plot to plot
    x = -window
    to_plot = to_plot.iloc[x:]
    _plot(ticker, to_plot)
    exit
