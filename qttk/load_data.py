'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# load_data.py - a convenience utlity that loads sample data
# input: accepts a single ticker symbol or a list of sample stock symbols
# from /data/eod/
# output: returns a dataframe with the closing prices of each symbol

# production version: 2021-02-06
'''
import pandas as pd
import numpy as np
import os
from typing import Optional

def load_data(stocks) -> pd.DataFrame:
    '''
    Loads example EOD data
    '''
    path = os.path.dirname(__file__)
    dataframe = pd.DataFrame()

    if type(stocks) == str:
        data_file_path = os.path.join(path, 'data', 'eod', stocks+'.csv')
        if os.path.exists(data_file_path):
            dataframe = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        else:
            print('Could not locate data. Verify {} is correct.'.format(data_file_path))

    elif type(stocks) == np.ndarray:
        for stock in stocks:
            data_file_path = os.path.join(path, 'data', 'eod', stock+'.csv')
            if os.path.exists(data_file_path):
                dataload = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
                dataframe[stock] = dataload['close']
            else:
                print('Could not locate data. Verify {} is correct.'.format(data_file_path))
    else:
        print('Could not locate data. Verify {} is correct.'.format(ticker))
    return dataframe
