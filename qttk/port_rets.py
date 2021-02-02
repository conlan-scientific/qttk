'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# port_rets.py - Portfolio Returns

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/port_rets.py

# production version: 2021-02-01
'''
from datetime import datetime
import pandas as pd
import numpy as np
import os
from qttk.profiler_v2 import time_this, timed_report
from qttk.profiler_v2 import ExponentialRange

def load_portfolio():
    '''
    Loads example EOD data for portfolio of 10 instruments
    '''
    path = os.path.dirname(__file__)
    stocks = ['AWU', 'AXC', 'BGN', 'BMG', 'DVRL',\
     'EHH', 'EUZ', 'EXY', 'FJKV', 'KUAQ']
     dataframe = pd.DataFrame()
    for stock in stocks:
        filename = os.path.join(path, '..', 'data', stock+'.csv')
        dataload = pd.read_csv(filename, index_col=0, parse_dates=True)
        dataframe[stock] = dataload['close']
    return dataframe

def _fillinValues(dataframe:pd.DataFrame)->pd.DataFrame:
    '''
    Fill in NaN values
    '''
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)
    return dataframe

def compute_cagr(series: pd.Series) -> float:
    '''
    CAGR = Price(T)/Price(0)**(1/k) - 1
    k = T/252, T = last date in series
    from: page 26, Algorithmic Trading with Python by C. Conlan

    returns the Compounded Annual Growth Rate (CAGR) of the series
    '''

    return cagr


if __name__ == '__main__':
    # load sample data
    dataframe = load_sample_ticker()
    # window defines the period used for rsi
    # a shorter window makes rsi more sensitive to daily price changes
    window = 14

    rsi = compute_rsi(dataframe, window)

    # Execute unit tests
    test(window)

    # Performance Characterization
    exp_range = ExponentialRange(1, 5, 1/4)

    test_columns = ['date', 'open', 'close', 'low', 'high', 'volume']
    test_df = pd.DataFrame(np.random.rand(exp_range.max,6), columns=test_columns)

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])
        for i in exp_range.iterator():
            # rsi_SPY = compute_rsi(dataframe, window)
            tt(compute_rsi)(test_df.iloc[:i], window)
