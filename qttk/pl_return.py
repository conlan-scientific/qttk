"""The simple return of a portfolio is the weighted sum
    of the simple returns of the constituents of the portfolio.

    The log return for a time period is the sum of the log returns
    of partitions of the time period.

    This module outputs a series along with a graph:
        Series can be either percent return or log return
        Graph will output either percent or log return
"""

import os
import numpy as np
import pandas as pd
# from profiler import time_this
from typing import Optional
from qttk.utils.data_utils import check_dataframe_columns


# log return function
# @time_this
def compute_logr(df: pd.DataFrame) -> pd.Series:
    """
    Completed compute_log in 2.028 milliseconds

    Args:
        df: Close from DataFrame
    Returns:
         log return pd.Series
    """

    shifted_series = df['close'].shift(1, axis=0)  # shifting data to begin calc
    log_ret = (np.log(df['close'] / shifted_series))  # calculating data

    return log_ret.fillna(0)


"""Log returns are useful due to their symmetrical characteristic.

    Log returns are not easily interpretable for large price swings
    but good for small price swings.
"""


# percent return function
# @time_this
def compute_perr(df: pd.DataFrame) -> pd.Series:
    """
    Completed compute_perr in 0.448 milliseconds

    Args:
        df: Close from DataFrame
    Returns:
         percent return pd.Series
    """

    shifted_series = df['close'].shift(1, axis=0)
    percent_ret = (df['close'] / shifted_series) - 1  # calculating percent return

    return percent_ret.fillna(0)


# graphing of either series
def graph_returns(series: pd.Series) -> None:
    series.plot()


def demo_logr(data: str = None, data_file_path: Optional[str] = None,
              required_columns: Optional[pd.Series] = None) -> None:
    """Main entry ponit for graph generating tool
    Args:
        data_file_path (Optional[str], optional): [description]. Defaults to None.
        required_columns (Optional[pd.Series], optional): [description]. Defaults to None.
        :param data_file_path:
        :param required_columns:
        :param data:
    """

    if data_file_path:
        assert os.path.exists(data_file_path), f"{data_file_path} not found"
        csv_file = data_file_path

    else:  # use relative path and example file
        script_dir = os.path.dirname(__file__)
        csv_files = os.path.join(script_dir, 'data', 'eod')
        csv_file = os.path.join(csv_files, data)

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.iloc[-180:]  # trim dataframe or autofit axis ?

    if required_columns is not None:
        check_dataframe_columns(df, required_columns)

    # df.set_index('date', inplace=True) # set index when csv read
    df = compute_logr(df)
    graph_returns(df)


if __name__ == '__main__':
    required_ohlcv_columns = pd.Series(['close'])
    # removed date as a required column because it is set as the dataframe index
    # when the csv is read
    # required_ohlcv_columns = pd.Series(['close'])
    data = 'AWU.csv'  # name of data file to use
    demo_logr(data, required_columns=required_ohlcv_columns)

    # optional loop
    # script_dir = os.path.dirname(__file__)
    # csv_files = os.path.join(script_dir,  'data', 'eod', '*.csv')
    # for csv_file in glob.glob(csv_files):
    #    main(csv_file)
