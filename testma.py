import pandas as pd
import numpy as np

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
    

def cumulative_moving_avg_2(values: pd.Series, min_periods: int=5) -> pd.Series:
    '''
    pd.Series.expand.mean() exists, but I tried to work out something vectorized
    
    
    '''
    
    # Initialize series
    cumulative_moving_avg = pd.Series([np.nan]*(min_periods-1))
    
    cumulative_moving_avg = cumulative_moving_avg.append((values.cumsum() / pd.Series(range(1,len(values)+1)))[min_periods-1:])
    
    return cumulative_moving_avg.reset_index(drop=True)
        
        
    