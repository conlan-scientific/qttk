'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# bollinger.py - Bollinger(R) Bands Example

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/bollinger.py

# production version: 2021-01-28
'''
import pandas as pd
import os
from qttk.indicators import bollinger, demo

if __name__ == '__main__':
    required_ohlcv_columns = pd.Series(['open', 'high', 'low', 'close', 'volume'])
    path = os.path.dirname(__file__)
    ticker = 'AWU'
    filename = os.path.join(path, '..', 'data', 'eod', ticker+'.csv')
    dataframe = pd.read_csv(filename, index_col=0, parse_dates=True)

    data = 'AWU.csv' # name of data file to use
    demo(data, required_columns=required_ohlcv_columns)

    exit
