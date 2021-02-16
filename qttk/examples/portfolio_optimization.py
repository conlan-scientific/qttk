'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# portfolio_optimization.py - Portfolio Optimization

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/portfolio_optimization.py

# production version: 2021-02-15
'''
from qttk.indicators import calculate_sharpe_ratio, portfolio_price_series
from qttk.utils.sample_data import load_portfolio
from qttk.utils.qttk_plot import plot

from datetime import datetime
import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    # define portfolio- stocks and weights
    stocks = ['AWU', 'HECP', 'HRVC', 'HXX', 'NSLG', 'PQCE', 'RZW', 'TRE', 'WFS',\
     'YPS', 'ZGL']
    # an equally weighted portfolio is assumed
    # weights must add up to 1.0 (100%)
    weights = np.full((1,len(stocks)), 1/len(stocks))
    '''
    Adjusting portfolio weights to demonstrate affects on Sharpe Ratio:
    '''
    #weights = np.array([[0.125, 0.125, 0.0, 0.125, 0.125, 0.125, 0.0, 0.125,\
    #0.125, 0.125, 0.0]])
    #weights = np.array([[0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    #weights = np.array([[0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 0.34]])
    portfolio = pd.DataFrame(weights, columns=stocks)
    dataframe = load_portfolio(portfolio.columns.values)
    series = portfolio_price_series(weights, dataframe.iloc[-252:])
    sharpe = np.around(calculate_sharpe_ratio(series, 0.04), 2)

    dataframe['portfolio'] = series
    returns = np.log(dataframe/dataframe.shift(1))
    volatility = returns.rolling(window=2).std()*np.sqrt(252)
    volatility.rename(columns={'portfolio':'port_volatility'})

    # Plot portfolio returns and volatility vs. time
    start = -200 # row to start plotting at, a negative number
    end = -180 # row to end plotting at, a negative number
    plot(returns.iloc[start:end, [11]], test=False, \
    title='Portfolio Returns', kind='line', legend=False)
    plot(volatility.iloc[start:end, [11]], test=False, \
    title='Portfolio Volatility', kind='line', legend=False)

    # Plot individual stock returns and volitility vs. time
    plot(returns.iloc[start:end, [0, 1, 2, 3, 4, 5]], test=False, \
    title='Stocks 1-5 Returns', kind='line')
    plot(volatility.iloc[start:end, [0, 1, 2, 3, 4, 5]], test=False, \
    title='Stocks 1-5 Volatility', kind='line')

    plot(returns.iloc[start:end, [6, 7 , 8, 9, 10]], test=False, \
    title='Stocks 6-10  Returns', kind='line')
    plot(volatility.iloc[start:end, [6, 7 , 8, 9, 10]], test=False,\
    title='Stocks 6-10 Volatility', kind='line')

    print('Sharpe Ratio: ', sharpe)
    print('Portfolio Returns: ', \
    np.around(returns.iloc[:, [11]].sum(), 5))

    exit
