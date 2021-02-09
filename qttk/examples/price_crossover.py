'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# price_crossover.py - Price Crossover Example

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/price_crossover.py

# production version: 2021-02-08
'''
from qttk.indicators import compute_rsi, compute_bb
from qttk.utils.sample_data import load_sample_data

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import os


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
    axs[0].plot(dataframe[['BOLU']], c='g', lw=1.5, ls='-.', label='BB-Upper')
    axs[0].plot(dataframe[['MA_Close']], c='y', lw=1.5, ls='-.', label='MA-Close')
    axs[0].plot(dataframe[['BOLD']], c='r', lw=1.5, ls='-.', label='BB-Lower')
    axs[0].plot(dataframe[['close']], c='k', lw=1.5, ls='dotted', label='Close')
    axs[0].set_ylabel('Price')
    axs[0].legend(loc=0)
    axs[0].grid(True)

    axs[1].set_title(ticker+' RSI')
    axs[1].xaxis.set_major_locator(locator)
    axs[1].xaxis.set_major_formatter(formatter)
    axs[1].set_ylim(0, 100)
    axs[1].plot(dataframe[['rsi']], c='k', lw=1.5, ls='dotted', label="RSI")
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('rsi')
    axs[1].legend(loc='lower right')
    axs[1].grid(True)

    plt.show()


if __name__ == '__main__':
    ticker = 'DLVY'
    dataframe = load_sample_data(ticker)

    window = 30

    rsi = compute_rsi(dataframe, window)
    to_plot = dataframe.copy()
    compute_bb(to_plot)

    x = -window                  # define the date range for plot to plot
    to_plot = to_plot.iloc[x:]
    to_plot['rsi'] = rsi[x:]
    _plot(ticker, to_plot)
    exit
