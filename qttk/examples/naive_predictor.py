'''
# Quantitative Trading ToolKit (qttk)
# https://github.com/conlan-scientific/qttk

# naive_predictor.py - Naive Predictor

The naive predictor assumes that the future value will be equal to the current value.
This provides a comparative baseline for sophisticated modeling techniques.
A sophisticated modeling technique needs to outperform the naive predictor for
the same time series data being predicted.

# run from project directory:
    C:/Users/user/qttk>ipython -i ./qttk/examples/naive_predictor.py

# production version: 2021-02-21
'''
from qttk.utils.sample_data import load_sample_data
from qttk.utils.qttk_plot import plot
from qttk.indicators import r_squared

if __name__ == '__main__':
    # load data
    ticker = 'HECP'
    series = load_sample_data(ticker)
    close_price = series.iloc[-30:, 1]
    # make naive prediction
    predicted_values = series.iloc[-30:, 1].shift(1)
    predicted_values = predicted_values.rename('predicted_close')
    r_squared = r_squared(close_price, predicted_values).round(3)

    print('Symbol: ', ticker)
    print('naive prediction--')
    print('R squared: ', r_squared)

    plot(close_price, title=ticker)
    plot(predicted_values, title=ticker)

exit
