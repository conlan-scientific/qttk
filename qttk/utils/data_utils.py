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
        raise IndexError(f"Required: {required_columnsf} Missing: {missing_column}")
