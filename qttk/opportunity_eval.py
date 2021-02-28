'''
****************************************************************
# Quantitative Trading Toolkit (qttk)
# https://github.com/conlan-scientific/qttk

# opportunity_eval.py - Opportunity Evaluation

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/opportunity_eval.py

# production version: 2021-02-27
****************************************************************
Background:

There may be a situation where it is desireable to evaluate an investment
opportunity.  This may present itself as evaluating a predictive model
or an investment instrument.

This module calculates the mean return based upon the following inputs:

series: an equity series that varies with respect to time

    E_t* = C_t + P_t where:

    E_t is the equity series over time
    C_t is the cash over time
    P_t are the stock holdings' value over time

    *any time-varying price series may be used as input data

C_0: the portion of assets under management being invested.

r_squared: the accuracy of the model
    r_squared is an input of this function, but it can be calculated if given
    an equity series and predicted values for that series. Ref: r_squared.py
'''
from qttk.utils.sample_data import load_sample_data
from qttk.sharpe import calculate_return_series, calculate_annualized_volatility
from qttk.r_squared import r_squared
from qttk.profiler_v2 import time_this, timed_report, ExponentialRange

from datetime import datetime
import pandas as pd
import numpy as np

def mean_return(price_series: pd.Series, C_0: float, r_squared: float)->float:
    return_series = calculate_return_series(price_series)
    volatility = calculate_annualized_volatility(return_series)
    return C_0*np.sqrt(2/np.pi)*volatility*(2*r_squared-1)

'''
An alternative form of the function is provided for the situation where mean
return is known, or there is a desired value for mean return, and R-squared is
unknown (i.e. the model hasn't been developed yet).  This version of the function
returns the R-Squared needed to provide the desired mean return.
'''
def r_squared_min(price_series: pd.Series, C_0: float, mean_return: float)->float:
    return_series = calculate_return_series(price_series)
    volatility = calculate_annualized_volatility(return_series)
    return mean_return/(2*C_0*np.sqrt(2/np.pi)*volatility) + 0.5


if __name__ == '__main__':
    # load data
    ticker = 'HECP'
    series = load_sample_data(ticker)
    close_price = series.iloc[-30:, 1]
    print('\nSymbol: ', ticker)

    print('naive prediction--')
    predicted_values = series.iloc[-30:, 1].shift(1)
    r2 = r_squared(close_price, predicted_values).round(3)
    print('R squared: ', r2)

    C_0 = 1000.00  # invested capital, a portion of AUM
    mean = mean_return(close_price, C_0, r2)
    r2_min = r_squared_min(close_price, C_0, mean)

    print('\nSymbol: {0}, C_0: ${1}'.format(ticker, C_0))
    print('mean return: ', np.around(mean, decimals=2))
    print('percent mean return: ', np.around(mean/C_0*100, decimals=2))
    print('minimum R squared needed: ', np.around(r2_min, decimals=3))
    print('')

    exp_range = ExponentialRange(4, 6, 1/4)
    test_columns = ['date', 'open', 'close', 'low', 'high', 'volume']
    test_df = pd.DataFrame(
        np.random.rand(exp_range.max, 6),
        columns=test_columns,
        index=pd.date_range('01-01-1900', periods=exp_range.max, freq=pd.Timedelta(seconds=10))
    )

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])
        for i in exp_range.iterator():
            tt(mean_return)(test_df['close'].iloc[:i], C_0, r2)
    exit
