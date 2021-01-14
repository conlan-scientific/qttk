# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk
#
# fiviz.py - Financial Visualization
# candles chart with rsi

import pandas as pd
import numpy as np
import matplotlib
import os

def fiviz(filename):
    raw = pd.read_csv(filename, index_col=0, parse_dates=True)
    raw['close'].plot()

if __name__ == '__main__':
    filename = './data/SPY.csv'
    fiviz(filename)
