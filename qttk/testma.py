import pandas as pd
import numpy as np
from qttk.profiler import time_this

@time_this
def pd_simple_moving_avg(values: pd.Series, m: int=20) -> pd.Series:
    '''
    This is an O(n) time implementation of a simple moving average.
    Simple moving average 
    
    >>> pd_simple_moving_avg(pd.Series(range(0,14,2)),2)
    0     NaN
    1     NaN
    2     3.0
    3     5.0
    4     7.0
    5     9.0
    6    11.0
    dtype: float64
    '''
    
    cumsum = values.cumsum()
    return (cumsum - cumsum.shift(m))/m

@time_this
def cumulative_moving_avg(values: pd.Series, min_periods: int=5) -> pd.Series:
    '''
    The Cumulative Moving Average is the unweighted mean of the previous values up to the current time t.
    pandas has an expand
    
    O(nm) because n number of periods in a loop and sum(m), 
    
    '''
    counter = 0
    
    # Initialize series
    cumulative_moving_avg = pd.Series([np.nan]*(min_periods-1))
    
    temp: List[float] = list()
        
    for i in range(min_periods,len(values)+1):
    
        temp.append(sum(values[:i])/(min_periods+counter))    
        counter +=1

    return cumulative_moving_avg.append(pd.Series(temp)).reset_index(drop=True)
    
@time_this
def cumulative_moving_avg_v2(values: pd.Series, min_periods: int=5) -> pd.Series:
    '''
    pd.Series.expand.mean() exists, but I tried to work out something vectorized
    
    
    '''
    
    # Initialize series
    cumulative_moving_avg = pd.Series([np.nan]*(min_periods-1))
    
    cumulative_moving_avg = cumulative_moving_avg.append((values.cumsum() / pd.Series(range(1,len(values)+1)))[min_periods-1:])
    
    return cumulative_moving_avg.reset_index(drop=True)
        
        
@time_this
def cumulative_moving_avg_v3(values: pd.Series, min_periods: int=5) -> pd.Series:

    # denominator is one-indexed location of the element in the cumsum
    denominator = pd.Series(np.arange(1, series.shape[0]+1))
    result = series.cumsum() / denominator

    # Set the first min_periods elements to nan
    result.iloc[:(min_periods-1)] = np.nan

    return result

    
if __name__ == '__main__':

    
    series = pd.Series([1,2,3,4,5,6,7,8,9,10])

    import numpy as np
    series = pd.Series(np.random.random((1000 * 1000)))
    x = pd_simple_moving_avg(series, m=5)
    # y = cumulative_moving_avg(series, min_periods=5)
    z = cumulative_moving_avg_v2(series, min_periods=5)
    w = cumulative_moving_avg_v3(series, min_periods=5)

    series = pd.Series(list(range(21, 41)))
    x = pd_simple_moving_avg(series, m=5)
    y = cumulative_moving_avg(series, min_periods=5)
    z = cumulative_moving_avg_v2(series, min_periods=5)
    w = cumulative_moving_avg_v3(series, min_periods=5)

    truth_series = pd.Series([
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        23.0,
        23.5,
        24.0,
        24.5,
        25.0,
        25.5,
        26.0,
        26.5,
        27.0,
        27.5,
        28.0,
        28.5,
        29.0,
        29.5,
        30.0,
        30.5,
    ])
    assert y.equals(truth_series)
    assert z.equals(truth_series)
    assert w.equals(truth_series)





