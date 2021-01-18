import pandas as pd
import numpy as np
from profiler import time_this

'''

'''

@time_this
def weighted_moving_avg_v1(values: pd.Series, m: int=5) -> pd.Series:
    '''
    Description:
      Backwards looking weighted moving average.
    
    Complexity:
      O(nm) because n values * m sized window
      
    Memory Usage:

    Inputs:
      values: pd.Series of closing prices
      m     : period to be calculated 
    
    '''
    # Constant
    w = pd.Series(range(1,m+1))
    w = w /w.sum()
    
    # when period is greater than values, return
    if values.shape[0] <= m:
        return pd.Series([np.nan]*len(values))
    
    #initialize and copy index from input series for timeseries index compatibility
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)

    for i in range(m,values.shape[0]+1):
        wma.iloc[i-1] = (values.iloc[i-m:i].values * w.values).sum()
        
    return wma

@time_this
def weighted_moving_avg_v2(values: pd.Series, m: int=5) -> pd.Series:
    '''
    Description:
      Backwards looking weighted moving average.
    
    Complexity:
      O(n)?  This is the slowest method, hidden complexity?
      
    Memory Usage:
    
    '''
    # Constant
    weights = pd.Series(range(1,m+1))
    weights = weights /weights.sum()
    
    # when period is greater than values, return
    if values.shape[0] <= m:
        return pd.Series([np.nan]*len(values))
    
    #initialize series and copy index from input series to return a matching index
    wma = pd.Series([np.nan]*values.shape[0], index=values.index)
    window = values.iloc[0:m-1]
    

    for idx in values.iloc[m-1:].index:
        window[idx] = values[idx]
        wma[idx] = (window.values * weights.values).sum()
        del window[window.index[0]]
     
    return wma


def _np_weighted_moving_avg(values: np.array, m:int=5) -> np.array:
    '''
    np convolution method
    
    Description:
      Backwards looking weighted moving average with numpy.
    
    Complexity:
      O(n) because n values.  
      
    Memory Usage:

    References:
      https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
      https://en.wikipedia.org/wiki/Convolution      
    
    '''
    weights = np.arange(1,m+1)
    weights = weights/weights.sum()
    
    # when period is greater than values, return - for standalone
    #if values.shape[0] <= m:
    #    return pd.Series([np.nan]*len(values))
    
        
    weights = np.arange(1,m+1)
    weights = weights/weights.sum()
    return np.convolve(values,weights[::-1],'valid')
    
    
@time_this
def weighted_moving_avg_v3(values: pd.Series, m: int=5) -> pd.Series:
    '''
    Wrapper to use np_weighted_moving_avg function
    .00257 ÷ .000706 = 3.64x slower than talib.WMA with arguments:
        series = pd.Series(np.random.random((1000 * 100)))
        m = 12
    '''
    
    # when period is greater than values, return
    if values.shape[0] <= m:
        return pd.Series([np.nan]*len(values))
    
    #initialize series and copy index from input series to return a matching index
    wma:pd.Series = pd.Series([np.nan]*values.shape[0], index=values.index)
    wma.iloc[m-1:] = _np_weighted_moving_avg(values.values,m)
    return wma
    
    
    
if __name__ == '__main__':
    import numpy as np
    series = pd.Series(np.random.random((100 * 75)))

    x = weighted_moving_avg_v1(series,12)
    y = weighted_moving_avg_v2(series,12)
    z = weighted_moving_avg_v3(series,12)
    assert x.equals(y)
    if not x.equals(z):
      print((x[12:] == z[12:]).value_counts())
      print(f'mean of difference: {np.mean((x[12:] - z[12:])) :.7f}\n')
    #assert x.equals(z) # inconsistent

    print('test with a timeseries and assert datatype\n')
    series.index = pd.date_range(start='12-1-2010', periods=series.shape[0])

    x = weighted_moving_avg_v1(series,10)
    y = weighted_moving_avg_v2(series,10)
    z = weighted_moving_avg_v3(series,10)
    assert x.index.dtype == np.dtype('<M8[ns]')
    assert y.index.dtype == np.dtype('<M8[ns]')
    assert z.index.dtype == np.dtype('<M8[ns]')
