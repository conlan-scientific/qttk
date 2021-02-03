'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# portfolio.py - Portfolio Summary

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/portfolio.py

# production version: 2021-02-02
'''
from datetime import datetime
import pandas as pd
import numpy as np
import os
from qttk.indicators import calculate_sharpe_ratio
from qttk.profiler import time_this


def load_portfolio(stocks: list) -> pd.DataFrame:
    '''
    Loads example EOD data for portfolio of 10 instruments
    '''
    path = os.path.dirname(__file__)
    dataframe = pd.DataFrame()
    for stock in stocks:
        filename = os.path.join(path, '..', 'data', 'eod', stock+'.csv')
        dataload = pd.read_csv(filename, index_col=0, parse_dates=True)
        dataframe[stock] = dataload['close']
    return dataframe

def portfolio_price_series(wt: list, df: pd.DataFrame) -> pd.DataFrame:
    port_price = np.sum(wt * df, axis=1)
    return port_price

def _fillinValues(dataframe:pd.DataFrame)->pd.DataFrame:
    '''
    Fill in NaN values
    '''
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)
    return dataframe


if __name__ == '__main__':
    # load sample data
    stocks = ['AWU', 'AXC', 'BGN', 'BMG', 'DVRL', 'EHH', 'EUZ', 'EXY',\
     'FJKV', 'KUAQ']
    # weights must add up to 1.0 (100%)
    weights = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    dataframe = load_portfolio(stocks)
    series = portfolio_price_series(weights, dataframe)
    sharpe = calculate_sharpe_ratio(series)
    print('Portfolio: \n', dataframe.columns.values, '\nWeights: ', weights)
    print('Sharpe Ratio: ', sharpe)
