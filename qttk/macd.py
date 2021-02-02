
"""
Moving average convergence divergence (MACD) is a trend-following momentum
    indicator that shows the relationship between two moving averages of a security’s
    price. The MACD is calculated by subtracting the 26-period exponential moving
    average (EMA) from the 12-period EMA.
"""

import os
import sys
import glob
import datetime as dt
from typing import Any, Optional, Iterable
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from qttk.utils.data_utils import check_dataframe_columns


def compute_macd(eod_data: pd.DataFrame,
                 sm_window: int = 12,
                 lg_window: int = 26) -> pd.DataFrame:
    """
       Assumes close feature is adjusted close.
       Prepares dataframe by adding features:
         Small Exponential Moving Average(sm_exp)
         Large Exponential Moving Average(lg_exp)
         MACD
       Args:
           eod_data (pd.DataFrame): Open, High, Low, Close Volume dataframe


       Returns:
           pd.DataFrame: Columns added, MA_Close, std, BOLU, BOLD, MA_Volume, Bandwidth
       """

    sm_exp = eod_data['close'].ewm(span=sm_window,
                                   adjust=False).mean()  # getting the exp. moving average of the first period
    lg_exp = eod_data['close'].ewm(span=lg_window,
                                   adjust=False).mean()  # getting the exp. moving average of the second period

    macd_calc = sm_exp - lg_exp  # obtaining the MACD from subtracting the EMA's
    eod_data['MACD'] = macd_calc  # putting MACD into the dataframe


    eod_data['MACD_MA'] = macd_calc.rolling(window=9).mean()  # obtaining the moving average of the MACD

    columns_to_fill = ['MACD', 'MACD_MA']
    eod_data.loc[:, columns_to_fill].fillna(0, inplace=True)

    return eod_data


def macd_graph_formatter(data_frame: pd.DataFrame) -> None:
    '''
    Relies on global import matplotlib.pyplot as plt
    '''
    # macd_data = data_frame[['open', 'close', 'low', 'high']]
    fig, axs = plt.subplots(3, 1, figsize=(40, 20), gridspec_kw={'height_ratios': [3, 1, 2]})
    plt.subplots_adjust(hspace=0.4, bottom=0.3)

    # added formatting for axis labels
    locator = mdates.AutoDateLocator(minticks=5, maxticks=30)
    formatter = mdates.ConciseDateFormatter(locator)

    axs[0].set_title('MACD')
    axs[0].plot(data_frame[['MACD', 'MACD_MA']])
    axs[0].autoscale(enable=True)
    axs[0].set_xticklabels(data_frame.index, rotation=45)
    axs[0].set_ylabel('MACD')
    axs[0].grid(True)

    # histogram with moving average Subplot
    axs[1].set_title('MACD')
    axs[1].bar(data_frame.index, data_frame['MACD'])
    axs[1].plot(data_frame.index, data_frame['MACD_MA'], color='black')
    axs[1].set_xticklabels(data_frame.index, rotation=45)
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('MACD')
    axs[1].grid(True)

    # price Subplot
    axs[2].set_title('Price')
    axs[2].plot(data_frame.index, data_frame['close'], color='black')
    axs[2].set_xticklabels(data_frame.index, rotation=45)
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Price')
    axs[2].grid(True)


def macd_demo(data: str = None, data_file_path: Optional[str] = None,
              save_figure: bool = False,
              required_columns: Optional[pd.Series] = None) -> None:
    """Main entry ponit for graph generating tool
    Args:
        data_file_path (Optional[str], optional): [description]. Defaults to None.
        save_figure (bool, optional): [description]. Defaults to True.
        required_columns (Optional[pd.Series], optional): [description]. Defaults to None.
    """

    if data_file_path:
        assert os.path.exists(data_file_path), f"{data_file_path} not found"
        csv_file = data_file_path

    else:  # use relative path and example file
        script_dir = os.path.dirname(__file__)
        csv_files = os.path.join(script_dir, '..', 'data', 'eod')
        csv_file = os.path.join(csv_files, data)

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.iloc[-180:]  # trim dataframe or autofit axis ?

    # if required_columns is not None:
    #     check_dataframe_columns(df, required_columns)

    # df.set_index('date', inplace=True) # set index when csv read
    df = compute_macd(df)
    macd_graph_formatter(df)
    plt.autoscale()

    if save_figure:
        plt.savefig(f"{csv_file.split('/')[-1]}_{dt.datetime.now().strftime('%y_%m_%d_%H_%M_%S')}.png")
    else:
        plt.show()


if __name__ == '__main__':
    required_ohlcv_columns = pd.Series(['open', 'high', 'low', 'close', 'volume'])
    # removed date as a required column because it is set as the dataframe index
    # when the csv is read
    # required_ohlcv_columns = pd.Series(['date', 'open', 'high', 'low', 'close', 'volume'])
    data = 'AWU.csv'  # name of data file to use
    macd_demo(data, required_columns=required_ohlcv_columns)

    # optional loop
    # script_dir = os.path.dirname(__file__)
    # csv_files = os.path.join(script_dir, '..', 'data', 'eod', '*.csv')
    # for csv_file in glob.glob(csv_files):
    #    main(csv_file)