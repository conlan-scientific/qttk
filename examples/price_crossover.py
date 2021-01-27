'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# price_crossover.py - Price Crossover Example

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/price_crossover.py

# production version: 2021-01-27
'''
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
import os

from qttk.indicators import compute_rsi, bollinger

def _plot(ticker: str, dataframe: pd.DataFrame) -> None:
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
    axs[0].set_title(ticker+' Price')
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


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    ticker = 'DLVY'
    filename = os.path.join(path, '..', 'data', 'eod', ticker+'.csv')
    dataframe = pd.read_csv(filename, index_col=0, parse_dates=True)

    window = 14

    rsi = compute_rsi(dataframe, window)
    to_plot = dataframe.copy()
    bollinger(to_plot)

    x = -window                  # define the date range for plot to plot
    to_plot = to_plot.iloc[x:]
    to_plot['rsi'] = rsi.iloc[x:, [0]]
    _plot(ticker, to_plot)
    exit