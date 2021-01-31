"""
Module for data utilities

1/25/2021: check_column_names created

"""

from typing import Any, Dict, List
import pandas as pd

def check_dataframe_columns(dataframe: pd.DataFrame,
                            required_columns: pd.Series,
                            strict: bool = True) -> None:

    """Utility to check columns are present in a dataframe

    Args:
        dataframe (pd.DataFrame): DataFrame columns to be tested
        required_columns (pd.Series): required columns
        strict (bool, optional): match all required columns. Defaults to True.
    """

    column_test = required_columns.isin(dataframe.columns)

    if strict:
        columns_valid = column_test.all()
    else:
        columns_valid = column_test.any()

    if not columns_valid:
        column_difference = required_columns[~column_test]
        missing_column = ', '.join(column_difference.values)
        required_columnsf = ', '.join(required_columns)
        error_message = (f" Missing: {missing_column} "
                        f"Required: {required_columnsf}")
        raise IndexError(error_message)


if __name__ == '__main__':
    import numpy as np
    test_columns = ['date', 'open', 'close', 'low', 'high', 'volume']
    test_df = pd.DataFrame(np.random.rand(4,6), columns=test_columns)
    
    # Test core use
    test_columns_series = pd.Series(test_columns)
    assert check_dataframe_columns(test_df, test_columns_series) is None

    # test partial match in strict mode
    test_columns_series = pd.Series(['date', 'open', 'close',
                                     'low', 'high', 'volume', 'rsi'])
    error_string = None
    try:
        check_dataframe_columns(test_df, test_columns_series)
    except IndexError as e:
        error_string = str(e)

    assert error_string is not None # tests error

    # test custom error string contains missing column
    assert 'rsi' in error_string
