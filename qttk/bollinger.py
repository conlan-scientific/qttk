"""
A Bollinger Band® is a technical analysis tool defined by a set of trendlines plotted two standard deviations
    positively and negatively) away from a simple moving average (SMA) of a security's price, but which
    can be adjusted to user preferences.

This Module outputs a stacked graph featuring:
  Bollinger Bands with candlestick close prices
  Volume
  %b
  Bandwidth

"""
import os
#import sys
#import glob
#import datetime as dt
from typing import Any, Optional, Iterable
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
#from progiler import time_this
from qttk.utils.data_utils import check_dataframe_columns


#@time_this
def compute_bb(eod_data: pd.DataFrame,
               moving_avg_window: int = 21,
               std_window: int = 21,
               volume_window: Optional[int] = 50,
               multiplier: int = 2) -> pd.DataFrame:
    """
    Assumes close feature is adjusted close.
    Prepares dataframe by adding features:
      Moving Average(min_periods)
      Standard deviation(min_periods)
      Upper Bollinger Band
      Lower Bollinger Band

      Completed compute_bb in 6.263 milliseconds

    Args:
        eod_data (pd.DataFrame): Open, High, Low, Close Volume dataframe
        moving_avg_window (int, optional): [description]. Defaults to 21.
        std_window (int, optional): [description]. Defaults to 21.
        volume_window (int, optional): [description]. Defaults to 50.
        multiplier (int, optional): [description]. Defaults to 2.

    Returns:
        pd.DataFrame: Columns added, MA_Close, std, BOLU, BOLD, MA_Volume, Bandwidth
    """

    # Calculating a 21 day moving average
    eod_data['MA_Close'] = eod_data['close'].rolling(window=moving_avg_window).mean()

    # Calculating the standard deviation of the adjusted close
    # Standard deviation emphasizes the outliers
    # It is sensitive to breakouts, adaptive to changes in regime
    eod_data['std'] = eod_data['close'].rolling(window=std_window).std()

    # This is calculating the upper band
    eod_data['BOLU'] = eod_data['MA_Close'] + (multiplier * eod_data['std'])

    # This is calculating the lower band
    eod_data['BOLD'] = eod_data['MA_Close'] - (multiplier * eod_data['std'])

    # Calculating the 50 day average volume
    # Both volume and the 50 day average will be plotted as line graphs
    try:
        eod_data['MA_Volume'] = eod_data['volume'] \
        .rolling(window=volume_window) \
        .mean()
    except KeyError: None

    # Tells us where we are in relation to the BB
    # chart will have 2 additional lines that are
    # 21 day high and low to see how it fluctuates
    eod_data['pct_b'] = ((eod_data['close'] - eod_data['BOLD']) / (eod_data['BOLD'] - eod_data['BOLU']))

    # Tells us how wide the BB are
    # Lines are the highest and lowest values of bandwidth in the last 125 days
    # High is bulge low is squeeze
    eod_data['Bandwidth'] = (eod_data['BOLU'] - eod_data['BOLD']) / eod_data['MA_Close']
    eod_data.loc[:, :].fillna(0, inplace=True)

    return eod_data


def graph_bb(data_frame: pd.DataFrame) -> None:
    """
    Relies on global import matplotlib.pyplot as plt
    """
    # bb_data = data_frame[['open', 'close', 'low', 'high']]
    fig, axs = plt.subplots(4, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    plt.subplots_adjust(top=0.947, bottom=0.087, left=0.071, right=0.989, hspace=0.918, wspace=0.2)
    '''
    Figure parameters:
    top=0.947, bottom=0.087, left=0.071, right=0.989, hspace=0.918,wspace=0.2
    '''
    # added formatting for axis labels
    locator = mdates.AutoDateLocator(minticks=5, maxticks=30)

    axs[0].set_title('Bollinger Bands')
    # axs[0].boxplot(bb_data.T, whis=[0,100])
    axs[0].plot(data_frame[['MA_Close', 'BOLU', 'BOLD']])
    '''
    characters {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}, which are short-hand
    notations for shades of blue, green, red, cyan, magenta, yellow, black, and white
    '''
    axs[0].scatter(data_frame.index, data_frame[['close']], s=1.0, c='k', marker=',')
    axs[0].xaxis.set_major_locator(locator)
    axs[0].set_ylabel('Price')
    axs[0].grid(True)

    # x day Moving Average Volume Subplot
    axs[1].set_title('Moving Average Volume')
    axs[1].bar(data_frame.index, data_frame['volume'])
    # axs[1].plot(data_frame.index, data_frame['MA_Volume'], color='black')
    axs[1].xaxis.set_major_locator(locator)
    axs[1].set_ylabel('Volume')
    axs[1].grid(True)

    # %b Subplot
    axs[2].set_title('%B')
    axs[2].plot(data_frame.index, data_frame['pct_b'], color='black')
    axs[2].xaxis.set_major_locator(locator)
    axs[2].set_ylabel('%B')
    axs[2].grid(True)

    # bandwidth Subplot
    axs[3].set_title('Bandwidth')
    axs[3].plot(data_frame.index, data_frame['Bandwidth'], color='black')
    axs[3].xaxis.set_major_locator(locator)
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('Bandwidth')
    axs[3].grid(True)
    plt.show()


def demo_bollinger(data: str = None, data_file_path: Optional[str] = None,
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
        csv_files = os.path.join(script_dir,  'data', 'eod')
        csv_file = os.path.join(csv_files, data)

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.iloc[-180:]  # trim dataframe or autofit axis ?

    if required_columns is not None:
        check_dataframe_columns(df, required_columns)

    # df.set_index('date', inplace=True) # set index when csv read
    df = compute_bb(df)
    graph_bb(df)
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
    demo_bollinger(data, required_columns=required_ohlcv_columns)

    # optional loop
    # script_dir = os.path.dirname(__file__)
    # csv_files = os.path.join(script_dir,  'data', 'eod', '*.csv')
    # for csv_file in glob.glob(csv_files):
    #    main(csv_file)
