'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# r-squared.py - R-Squared, Coefficient of determination
# https://en.wikipedia.org/wiki/Coefficient_of_determination

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/r_squared.py

# production version: 2021-02-21
'''
from qttk.utils.sample_data import load_sample_data
from qttk.profiler_v2 import time_this, timed_report, ExponentialRange

from datetime import datetime
import pandas as pd
import numpy as np

'''
ss_tot total sum of squares
       SStot = sum(y_i-y_bar)**2 where
       y_i is an element of series Y
       y_bar is the mean of series Y
'''
def ss_tot(series: pd.Series)->np.float64:
    mean = series.mean()
    ss_tot = np.sum((series-mean)**2)
    return ss_tot
'''
ss_res residual sum of squares
       SSres = sum(y_i - f_i)**2 where
       y_i is an element of series Y
       f_i is the predicted or modeled value of y_i
       y_i - f_i is the residual error
'''
def ss_res(series: pd.Series, predicted_values: pd.Series)->np.float64:
    ss_res = np.sum((series-predicted_values)**2)
    return ss_res

'''
R-Squared coefficient of determination
    R**2 = 1 - SSres/SStot
'''
def r_squared(series: pd.Series, predicted_values: pd.Series)->np.float64:
    ss_res_series = ss_res(series, predicted_values)
    ss_tot_series = ss_tot(series)
    r_squared = 1 - (ss_res_series/ss_tot_series)
    return r_squared


if __name__ == '__main__':
    # load data
    ticker = 'HECP'
    series = load_sample_data(ticker)
    close_price = series.iloc[-30:, 1]
    print('Symbol: ', ticker)

    print('naive prediction--')
    predicted_values = series.iloc[-30:, 1].shift(1)
    r2 = r_squared(close_price, predicted_values).round(3)
    print('R squared: ', r2)

    exp_range = ExponentialRange(4, 8, 1/4)
    test_columns = ['date', 'open', 'close', 'low', 'high', 'volume']
    test_df = pd.DataFrame(
        np.random.rand(exp_range.max, 6),
        columns=test_columns,
        index=pd.date_range('01-01-1900', periods=exp_range.max, freq=pd.Timedelta(seconds=10))
    )

    test_df['predict_close'] = test_df['close'].shift(1)

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])
        for i in exp_range.iterator():
            tt(r_squared)(test_df['close'].iloc[:i], test_df['predict_close'].iloc[:i])
    exit
