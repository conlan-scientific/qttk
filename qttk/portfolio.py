'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# portfolio.py - Portfolio Summary

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/portfolio.py

# production version: 2021-02-04
'''
from datetime import datetime
import pandas as pd
import numpy as np
import os
from qttk.indicators import calculate_sharpe_ratio
from qttk.utils.sample_data import load_portfolio
from qttk.profiler import time_this


def portfolio_price_series(wt: list, df: pd.DataFrame) -> pd.DataFrame:
    port_price = np.sum(wt * df, axis=1)
    return port_price

def _fillinValues(dataframe:pd.DataFrame)->pd.DataFrame:
    '''
    Fill in NaN values
    '''
    dataframe.fillna(method='ffill', inplace=True)
    return dataframe
    dataframe.fillna(method='bfill', inplace=True)


if __name__ == '__main__':
    # define portfolio- stocks and weights
    # weights must add up to 1.0 (100%)
    stocks = ['AWU', 'HECP', 'HRVC', 'HXX', 'NSLG', 'PQCE', 'RZW', 'TRE', 'WFS', 'ZGL']
    weights = np.full((1,len(stocks)), 1/len(stocks)) # an equally weighted portfolio is assumed
    portfolio = pd.DataFrame(weights, columns=stocks)
    dataframe = load_portfolio(portfolio.columns.values)
    series = portfolio_price_series(weights, dataframe.iloc[:252])
    sharpe = np.around(calculate_sharpe_ratio(series, 0.04), 2)
    print(dataframe.describe().round(2))
    print('Sharpe Ratio: ', sharpe)
    assert sharpe
