'''
A Bollinger Band® is a technical analysis tool defined by a set of trendlines plotted two standard deviations 
    positively and negatively) away from a simple moving average (SMA) of a security's price, but which
    can be adjusted to user preferences.

This Module outputs a stacked graph featuring:
  Bollinger Bands with candlestick close prices
  Volume
  %b
  Bandwidth

'''

import os
import sys
import glob
import datetime as dt
from typing import Any, Optional, Iterable
import pandas as pd
import matplotlib.pyplot as plt
from qttk.utils.data_utils import check_dataframe_columns

def bollinger(eod_data: pd.DataFrame,
              moving_avg_window: int = 21,
              std_window: int = 21,
              volume_window: int = 50,
              multiplier: int = 2) -> pd.DataFrame:
    """
    Assumes close feature is adjusted close.
    Prepares dataframe by adding features:
      Moving Average(min_periods)
      Standard deviation(min_periods)
      Upper Bollinger Band
      Lower Bollinger Band

    Args:
        eod_data (pd.DataFrame): Open, High, Low, Close Volume dataframe
        moving_avg_window (int, optional): [description]. Defaults to 21.
        std_window (int, optional): [description]. Defaults to 21.
        volume_window (int, optional): [description]. Defaults to 50.
        multiplier (int, optional): [description]. Defaults to 2.

    Returns:
        pd.DataFrame: Columns added, MA_Close, std, BOLU, BOLD, MA_Volume, Bandwidth
    """

    # Calcutlating a 21 day moving average
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
    eod_data['MA_Volume'] = eod_data['volume']\
        .rolling(window=volume_window)\
        .mean()

    # Tells us where we are in relation to the BB
    # chart will have 2 additonal lines that are 
    #   21 day high and low to see how it fluctates
    eod_data['pct_b'] = ((eod_data['close']-eod_data['BOLD'])/(eod_data['BOLD']-eod_data['BOLU']))

    # Tells us how wide the BB are
    # Lines are the highest and lowest values of bandwidth in the last 125 days 
    # High is bulge low is squeeze
    eod_data['Bandwidth'] = (eod_data['BOLU']-eod_data['BOLD'])/eod_data['MA_Close']
    columns_to_fill = ['MA_Close', 'std', 'BOLU', 'BOLD', 'MA_Volume', 'Bandwidth']
    eod_data.loc[:, columns_to_fill].fillna(0, inplace=True)

    return eod_data


def bb_graph_formatter(data_frame: pd.DataFrame) -> None:
    '''
    Relies on global import matplotlib.pyplot as plt

    todo: 
      [x] create line chart into a candlestick
      [ ] Change inf to zero or a better variable
      [x] Create the 4 graphs on top of each other
      [ ] Have a visual indicator on the graph to show it is buy/sell
          I tried to added a 'signal' column and annotations - it was messy
      [ ] Have an out response that says 3/4 of charts say this is a buy signal
      [x] Can this run for all stocks in market/sector... Make a list of all companies and for loop for data
          May want to add a CLI using argparse to produce a bunch of graphs
      [n/a] turn parameter dataset to *args
           Function expects a dataframe and can be called in a loop
      [ ] Keep axis labels neat

    '''
    bb_data = data_frame[['open', 'close', 'low', 'high']]
    fig, axs = plt.subplots(4, 1, figsize=(40, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    plt.subplots_adjust(hspace=0.4, bottom=0.3)

    axs[0].set_title('Bollinger Bands')
    axs[0].boxplot(bb_data.T, whis=[0,100])
    axs[0].plot(data_frame[['MA_Volume', 'BOLU', 'BOLD']])
    axs[0].autoscale(enable=True)
    axs[0].set_xticklabels(data_frame.index, rotation=45)
    axs[0].set_ylabel('Price')
    axs[0].grid(True)

    # x day Moving Average VOlumn Subplot
    axs[1].set_title('Moving Average Volume')
    axs[1].bar(data_frame.index, data_frame['volume'])
    axs[1].plot(data_frame.index, data_frame['MA_Volume'], color='black')
    axs[1].set_xticklabels(data_frame.index, rotation=45)
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Volume')
    axs[1].grid(True)

    # %b Subplot
    axs[2].set_title('%B')
    axs[2].plot(data_frame.index, data_frame['pct_b'], color='black')
    axs[2].set_xticklabels(data_frame.index, rotation=45)
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('%B')
    axs[2].grid(True)

    # bandwidth Subplot
    axs[3].set_title('Bandwidth')
    axs[3].plot(data_frame.index, data_frame['Bandwidth'], color='black')
    axs[3].set_xticklabels(data_frame.index, rotation=45)
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('Bandwidth')
    axs[3].grid(True)


def main(data_file_path: Optional[str] = None,
         save_figure: bool = True,         
         required_columns: Optional[pd.Series] = None ) -> None:

    """Create Bollinger Band Plot

    Args:
        single_file (bool, optional): [description]. Defaults to True.
        save_figure (bool, optional): [description]. Defaults to True.
        required_columns (pd.Series, optional): [description]. Defaults to None.
    """
    if data_file_path:
        assert os.path.exists(data_file_path), f"{data_file_path} not found"
        csv_file = data_file_path

    else:  # use relative path and example file
        script_dir = os.path.dirname(__file__)
        csv_files = os.path.join(script_dir, '..', 'data', 'eod')
        csv_file = os.path.join(csv_files, 'AWU.csv')



    df = pd.read_csv(csv_file)
    df = df.iloc[-180:] # trim dataframe or autofit axis ? 

    if required_columns is not None:
        check_dataframe_columns(df, required_columns)

    df.set_index('date', inplace=True)
    df = bollinger(df)
    bb_graph_formatter(df)
    plt.autoscale()

    if save_figure:
        plt.savefig(f"{csv_file.split('/')[-1]}_{dt.datetime.now().strftime('%y_%m_%d_%H_%M_%S')}.png")
    else:
        plt.show()


if __name__ == '__main__':
    required_ohlcv_columns = pd.Series(['date', 'open', 'high', 'low', 'close', 'volume'])
    main(required_columns=required_ohlcv_columns)

    # optional loop
    #script_dir = os.path.dirname(__file__)
    #csv_files = os.path.join(script_dir, '..', 'data', 'eod', '*.csv')
    #for csv_file in glob.glob(csv_files):
    #    main(csv_file)

