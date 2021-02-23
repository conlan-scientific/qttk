'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# portfolio_optimization.py - Portfolio Optimization

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/portfolio_optimization.py

# production version: 2021-02-23
'''
from qttk.indicators import calculate_sharpe_ratio, portfolio_price_series
from qttk.utils.sample_data import load_portfolio
from qttk.utils.qttk_plot import plot

import pandas as pd
import numpy as np


if __name__ == '__main__':
    '''
    an equally weighted portfolio is initially assumed
    weights must add up to approximately 1.0 (100%)
    a portfolio with a dictionary of stocks with weights of 0.09
    is defined below to establish a baseline for comparison:
    '''
    stocks = {'AWU': 0.09, 'HECP': 0.09, 'HRVC': 0.09, 'HXX': 0.09, 'NSLG': 0.09, \
    'PQCE': 0.09, 'RZW': 0.09, 'TRE': 0.09, 'WFS': 0.09, 'YPS': 0.09, 'ZGL': 0.09}

    '''
    Results-
    Sharpe Ratio: 0.68
    Returns:      0.18
    '''

    '''
    1 stock portfolio: HECP
    '''
    #stocks = {'AWU': 0.0, 'HECP': 1.0, 'HRVC': 0.0, 'HXX': 0.0, 'NSLG': 0.0, \
    #'PQCE': 0.0, 'RZW': 0.0, 'TRE': 0.0, 'WFS': 0.0, 'YPS': 0.0, 'ZGL': 0.0}
    '''
    Results-
    Sharpe Ratio: 0.61
    Returns:      0.29
    '''

    '''
    Stocks selected based upon returns and volatility:
    2 stock portfolio: NSLG, PQCE
    '''
    #stocks = {'AWU': 0.0, 'HECP': 0.0, 'HRVC': 0.0, 'HXX': 0.0, 'NSLG': 0.5, \
    #'PQCE': 0.5, 'RZW': 0.0, 'TRE': 0.0, 'WFS': 0.0, 'YPS': 0.0, 'ZGL': 0.0}
    '''
    Results-
    Sharpe Ratio: 0.70
    Returns:      0.25
    '''

    dataframe = load_portfolio(stocks.keys())
    series = portfolio_price_series(stocks.values(), dataframe)
    sharpe = np.around(calculate_sharpe_ratio(series, 0.04), 2)

    returns = np.log(series/series.shift(1))
    volatility = returns.rolling(window=2).std()*np.sqrt(252)
    volatility.rename({'portfolio':'port_volatility'})

    # Plot portfolio normalized price series vs. time
    plot(series[-30:]/series[0], test=False, title='Portfolio Normalized Price Series',\
     kind='line', legend=False)

    # Plot individual stock returns and volitility vs. time
    plot(dataframe.iloc[-30:, [0, 1, 2, 3, 4, 5]]/dataframe.iloc[0, [0, 1, 2, 3, 4, 5]], test=False, \
    title='Stocks 1-6 Normalized Price Series', kind='line')

    plot(dataframe.iloc[-30:, [6, 7 , 8, 9, 10]]/dataframe.iloc[0, [6, 7 , 8, 9, 10]], test=False, \
    title='Stocks 7-11  Normalized Price Series', kind='line')

    print('Sharpe Ratio: ', sharpe.round(2))
    print('Portfolio Returns: ', \
    np.around((series[-1]/series[0])**(1/(len(series)/252))-1, 2))

    exit
