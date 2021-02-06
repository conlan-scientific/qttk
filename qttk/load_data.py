'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# load_data.py - a convenience utlity that loads sample data
# input: a list of sample stock symbols from /data/eod/
# returns a dataframe with the closing prices of each symbol

# production version: 2021-02-05
'''
import pandas as pd
import os

def load_data(stocks: list) -> pd.DataFrame:
    '''
    Loads example EOD data for a list of sample stock symbols
    '''
    path = os.path.dirname(__file__)
    dataframe = pd.DataFrame()
    for stock in stocks:
        data_file_path = os.path.join(path, 'data', 'eod', stock+'.csv')
        if data_file_path:
            assert os.path.exists(data_file_path), f"{data_file_path} not found"
            dataload = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
            dataframe[stock] = dataload['close']
        else:
            print('Could not locate data. Verify {} is correct.'.format(data_file_path))
    return dataframe
