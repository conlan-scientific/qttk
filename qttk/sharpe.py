'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# Listings from:
# https://github.com/chrisconlan/algorithmic-trading-with-python/tree/master/listings/chapter_2

# sharpe.py - Sharpe Ratio

# run from project root directory:
    C:/Users/user/qttk>ipython -i ./qttk/sharpe.py

# production version: 2021-02-04
'''
from datetime import datetime
import pandas as pd
import numpy as np
import os
from qttk.profiler_v2 import time_this, timed_report, ExponentialRange


def load_ticker(ticker: str) -> pd.Series:
    """
    Loads EOD data
    """
    path = os.path.dirname(__file__)
    filename = os.path.join(path, 'data', 'eod', ticker+'.csv')
    dataframe = pd.read_csv(filename, index_col=0, parse_dates=True)
    series = dataframe['close']
    return series

def get_years_past(series: pd.Series) -> float:
    """
    Calculate the years past according to the index of the series for use with
    functions that require annualization
    """
    start_date = series.index[0]
    end_date = series.index[-1]
    return (end_date - start_date).days / 365.25

def calculate_return_series(series: pd.Series) -> pd.Series:
    """
    Calculates the return series of a time series.
    The first value will always be NaN.
    Output series retains the index of the input series.
    ticker: /data/eod/IRX.csv
    baseline time: 3.506 milliseconds
    """
    shifted_series = series.shift(1, axis=0)
    return series / shifted_series - 1

def calculate_annualized_volatility(return_series: pd.Series) -> float:
    """
    Calculates annualized volatility for a date-indexed return series.
    Works for any interval of date-indexed prices and returns.
    """
    years_past = get_years_past(return_series)
    entries_per_year = return_series.shape[0] / years_past
    return return_series.std() * np.sqrt(entries_per_year)

def calculate_cagr(series: pd.Series) -> float:
    '''
    CAGR = Price(T)/Price(0)**(1/k) - 1
    k = 1/T, T = years spanned in series
    from: page 26, Algorithmic Trading with Python by C. Conlan

    returns the Compounded Annual Growth Rate (CAGR) of the series
    '''
    value_factor = series.iloc[-1] / series.iloc[0]
    year_past = get_years_past(series)
    return (value_factor ** (1/year_past)) - 1

def calculate_sharpe_ratio(price_series: pd.Series,
    benchmark_rate: float=0) -> float:
    """
    Calculates the sharpe ratio given a price series. Defaults to benchmark_rate
    of zero.
    ticker: /data/eod/IRX.csv
    baseline time: 5.5 milliseconds
    """
    cagr = calculate_cagr(price_series)
    return_series = calculate_return_series(price_series)
    volatility = calculate_annualized_volatility(return_series)
    return (cagr - benchmark_rate) / volatility


if __name__ == '__main__':
    # load data
    ticker = 'HECP'
    series = load_ticker(ticker)
    sharpe = calculate_sharpe_ratio(series)
    print('Symbol: ', ticker)
    print('Sharpe Ratio: ', sharpe)

    exp_range = ExponentialRange(4, 8, 1/4)
    test_columns = ['date', 'open', 'close', 'low', 'high', 'volume']
    test_df = pd.DataFrame(
        np.random.rand(exp_range.max, 6),
        columns=test_columns,
        index=pd.date_range('01-01-1900', periods=exp_range.max, freq=pd.Timedelta(seconds=10))
    )

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])
        for i in exp_range.iterator():
            # rsi_SPY = compute_rsi(dataframe, window)
            tt(calculate_sharpe_ratio)(test_df['close'].iloc[:i])
