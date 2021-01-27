'''
Load sample data from qttk/data/eod directory

Todo:
    Load alternative data
'''
import os
from typing import List
import pandas as pd


def load_sample_data(ticker: str) -> pd.DataFrame:
    """
    Load sample data from qttk/data/eod directory

    Args:
        ticker (str): case insensitive, extension not required

    Returns:
        pd.DataFrame: 
            date       object
            open      float64
            close     float64
            low       float64
            high      float64
            volume      int64
            dtype: object
    """
    pwd = os.path.dirname(__file__)
    data_dir = os.path.join(pwd, '..', '..', 'data', 'eod')
    ticker_csv = os.path.join(data_dir, f'{ticker.upper()}.csv')
    assert os.path.exists(ticker_csv), f"{ticker.upper()}.csv not found"
    return pd.read_csv(ticker_csv)


def list_tickers() -> List[str]:
    """
    Directory listing of qttk/data/eod directory

    Returns:
        List: unfiltered list of all files in directory
    """
    pwd = os.path.dirname(__file__)
    data_dir = os.path.join(pwd, '..', '..', 'data', 'eod')
    return os.listdir(data_dir)


if __name__ == '__main__':
    present_working_directory = os.path.dirname(__file__)
    data_directory = os.path.join(present_working_directory, '..', '..', 'data', 'eod')
    assert os.path.exists(data_directory)
