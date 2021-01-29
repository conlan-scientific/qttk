'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# macd_example.py - Moving Average Convergence Divergence MACD Example

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/macd_example.py

# production version: 2021-01-28
'''
import pandas as pd
import os
from qttk.indicators import macd

if __name__ == '__main__':
    path = os.path.dirname(__file__)
    ticker = 'AWU'
    dataset = os.path.join(path, '..', 'data', 'eod', ticker+'.csv')

    n1 = 12   # for moving average short
    n2 = 21  # for moving average long

    macd(dataset, n1, n2)

    exit
