"""
Chaikin Money Flow (CMF) is an oscillator that depends on both volume
and candlestick data. CMF produces a divergence signal that reveals
upside/downside pressures by volume activity. The output is bound 
between 1 and -1.  CMF is used to help gauge and confirm strength of a trend.

Interpretation:
    Trends above zero indicate a bullish sentiment. When there is selling 
    pressure (CMF below zero), the trend is bearish. Higher readings indicate
    a stronger trend.  A false signal may occur when the trend is weak,
    barely crossing zero over zero.


Reference:
    `Algorithmic Trading with Python <https://github.com/chrisconlan/algorithmic-trading-with-python>`_

    `Chaikin Money Flow (CMF) <https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/chaikin-money-flow-cmf/>`_

"""
import sys
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qttk.utils.data_utils import check_dataframe_columns
from qttk.utils.sample_data import load_sample_data


def calculate_money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Money Flow Series

    Args:
        df (pd.DataFrame): Open High Low Close Volume dataframe

    Returns:
        pd.Series: money flow volume series
    """

    check_dataframe_columns(df,pd.Series(['volume', 'close', 'high', 'low']))

    mfv = df['volume'] * (2*df['close'] - df['high'] - df['low']) / \
        (df['high'] - df['low'])

    return mfv


def calculate_money_flow_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculate money flow volume, or q_t in the module's formula
    """

    return calculate_money_flow_volume_series(df).rolling(n).sum()


def calculate_chaikin_money_flow(df: pd.DataFrame, n: int=20) -> pd.Series:
    """
    Calculates the Chaikin Money Flow which is the n period average
    over n period average volume.  

    Args:
        df (pd.DataFrame):  OHLCV data
        n (int, optional):  number of periods.  Defaults to 20.

    Returns:
        pd.Series: Chaikin Money Flow series named cmf
    """
    
    cmf = calculate_money_flow_volume(df, n) / df['volume'].rolling(n).sum()
    cmf.rename('cmf', inplace=True)
    return cmf


def calculate_chaikin_money_flow_signal(cmf: pd.Series) -> pd.Series:
    """    
    Generates a trend reversal signal from Chaikin Money Flow.  A false signal
    may occur when the trend is weak, barely crossing over zero. 

    Args:
        :cmf (pd.Series): Chaikin Money Flow series

    Returns:
        pd.Series: signal -1, 0 ,1 
    """

    cmf_sign = np.sign(cmf)
    cmf_shifted_sign = cmf_sign.shift(1)
    cmf = cmf_sign * (cmf_sign != cmf_shifted_sign)
    cmf.Name = 'chakin_signal'
    return cmf


def _format_chaikin_plot(close_price: pd.Series,
                         chaikin_money_flow: pd.Series,
                         signal: pd.Series,
                         ticker: str) -> (plt.figure, plt.axes):
    """
    Helper function to format Chaikin Money FLow Plot with Signal

    Returns:
        plt.figure: matplotlib.pyplot figure
        plt.axes: matplotlib.pyplot axes
    """
    cutoff = str(close_price.index.max().year - 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 2, 2]})
    plt.subplots_adjust(top=0.947,
                        bottom=0.087,
                        left=0.071,
                        right=0.989,
                        hspace=0.918,
                        wspace=0.2)

    axs[0].set_title = 'Close Prices: {ticker}'.format(ticker=ticker)
    axs[0].plot(close_price.loc[close_price.index > cutoff])

    axs[1].set_title = f'Chaikin Money Flow: {ticker}'
    axs[1].plot(cmf.loc[chaikin_money_flow.index > cutoff])

    axs[2].set_title = f'Signal: {ticker}'
    axs[2].plot(cmf_signal.loc[signal.index > cutoff])
    return fig, axs


if __name__ == '__main__':
    ticker = 'AWU'
    demo_plot = True

    if sys.argv[1:]:
        demo_plot = False

    demo_df = load_sample_data(ticker)

    cmf = calculate_chaikin_money_flow(demo_df)
    cmf_signal = calculate_chaikin_money_flow_signal(cmf)

    if demo_plot:
        fig, axs = _format_chaikin_plot(demo_df.close,
                                        cmf,
                                        cmf_signal,
                                        ticker)

        plt.show(block=False)