'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# bollinger.py - Bollinger(R) Bands Example

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/bollinger.py

# production version: 2021-02-01
'''
from qttk.indicators import compute_bb, demo_bollinger
import pandas as pd
import os


if __name__ == '__main__':
    required_ohlcv_columns = pd.Series(['open', 'high', 'low', 'close', 'volume'])
    path = os.path.dirname(__file__)

    data = 'AWU.csv' # name of data file to use
    demo_bollinger(data, required_columns=required_ohlcv_columns)

    exit
