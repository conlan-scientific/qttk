"""
Given a list of numbers, compute their cumulative sum
Output the cumulative sum to a new sequence

Source:
  Fast Python - Master the Basics to Write Faster Code

As seen on the log-log plot, the functions have the same complexity
  but there are differences in performance 
"""
import gc
import pandas as pd
import numpy as np
from typing import List
from profiler_v2 import time_this, ExponentialRange, timed_report


def series_accumulator_cumsum(values: pd.Series) -> pd.Series:
    '''
    Cumsum by means of accumulator and pd.Series datatype
    '''
    cumsum = pd.Series([], name='cumsum', dtype=float)
    accumulator = 0.0

    for value in values:
        accumulator += value
        cumsum.append(pd.Series([accumulator]))

    return cumsum


def series_accumulator_cumsum_idx(values: pd.Series) -> pd.Series:
    '''
    Cumsum by means of accumulator, index and pd.Series datatype
    '''
    cumsum = pd.Series([np.nan]*values.shape[0], name='cumsum', dtype=float)
    accumulator = 0.0

    for i, value in enumerate(values):
        accumulator += value
        cumsum[i] = accumulator

    return cumsum


def pandas_fast_cumsum(values: pd.Series) -> pd.Series:
    '''
    this s O(n) and optimized with C code
    Uses memory optimized array under the hood and pandas
      has additional functionality that impacts performance
    '''
    return values.cumsum()


def numpy_fast_cumsum(values: np.ndarray) -> np.ndarray:
    '''
    This is O(n) and optimized with C code
    uses memory optimized array
    '''
    return values.cumsum()


def pure_python_cumsum(values: List[float]) -> List[float]:
    '''
    This is O(n) time because it does addition for n values
    Python lists use a heap instead of a memory optimized array.
      Heaps have less overhead
    '''
    cumsum = []
    accumulator = 0

    for value in values:
        accumulator += value
        cumsum.append(accumulator)
    
    return cumsum


if __name__ == '__main__':

    exp_range = ExponentialRange(1, 7, 1/4)
    series = pd.Series(np.random.random(exp_range.max))

    with timed_report():
        tt = time_this(lambda *args, **kwargs: args[0].shape[0])

        for i in exp_range.iterator(4):
            tt(series_accumulator_cumsum)(series.iloc[:i])
        
        gc.collect()
        for i in exp_range.iterator(4):            
            tt(series_accumulator_cumsum_idx)(series.iloc[:i])

        gc.collect()
        for i in exp_range.iterator():
            tt(pandas_fast_cumsum)(series.iloc[:i])

        gc.collect()
        for i in exp_range.iterator():
            tt(numpy_fast_cumsum)(series.iloc[:i].values)

        gc.collect()
        tt = time_this(lambda *args, **kwargs: len(args[0]))
        for i in exp_range.iterator():
            tt(pure_python_cumsum)(series.iloc[:i].tolist())