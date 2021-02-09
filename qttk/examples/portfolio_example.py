'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# portfolio_example.py - Portfolio Example

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/portfolio_example.py

# production version: 2021-02-05
'''
from qttk.indicators import calculate_sharpe_ratio, portfolio_price_series
from qttk.indicators import compute_rsi, compute_bb
from qttk.utils.sample_data import load_portfolio

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    # define portfolio- stocks and weights
    stocks = ['AWU', 'AXC', 'BGN', 'BMG', 'DVRL', 'EHH', 'EUZ', 'EXY', 'FJKV', 'KUAQ']
    # an equally weighted portfolio is assumed
    # weights must add up to 1.0 (100%)
    weights = np.full((1,len(stocks)), 1/len(stocks))
    portfolio = pd.DataFrame(weights, columns=stocks)
    dataframe = load_portfolio(portfolio.columns.values)
    series = portfolio_price_series(weights, dataframe.iloc[-252:])
    sharpe = np.around(calculate_sharpe_ratio(series, 0.04), 2)
    print(dataframe.describe().round(2))
    print('Sharpe Ratio: ', sharpe)

    ticker = 'Portfolio'
    window = 30  # interval needed for compute_rsi

    to_plot=pd.DataFrame(series, columns=['close'])
    rsi = compute_rsi(to_plot, window)
    compute_bb(to_plot)

    x = -window                  # define the date range for plot to plot
    to_plot = to_plot.iloc[x:]
    to_plot['rsi'] = rsi[x:]
    _plot(ticker, to_plot)
    exit
