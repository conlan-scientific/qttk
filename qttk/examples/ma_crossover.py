'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# ma_crossover.py - Moving Average Crossover Strategy Example
# http://bit.ly/3iQ9k7y

# run from project directory:
    C:/Users/user/qttk>ipython -i ./examples/ma_crossover.py

# production version: 2021-02-01
'''
from qttk.indicators import compute_ma
from qttk.utils.sample_data import load_sample_data

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
import os

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
    axs.plot(ticker_data[['ma_short']], c='c', lw=1.5, ls='-.', label='MA-Short')
    axs.plot(ticker_data[['ma_long']], c='b', lw=1.5, ls='-.', label='MA-Long')
    axs.plot(ticker_data[['close']], c='k', lw=1.5, ls='dotted', label='Close')
    axs.set_ylabel('Price')
    axs.legend(loc=0)
    axs.grid(True)

    plt.show()


if __name__ == '__main__':
    ticker = 'AWU'
    dataframe = load_sample_data(ticker)

    ma_short = compute_ma(dataframe['close'], 5)
    ma_long = compute_ma(dataframe['close'], 25)

    dataframe['ma_short'] = ma_short
    dataframe['ma_long'] = ma_long

    to_plot = dataframe.copy()

    window = 30                  # define the date range for plot to plot
    x = -window
    to_plot = to_plot.iloc[x:]
    _plot(ticker, to_plot)
    exit
