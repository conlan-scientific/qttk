'''
General use plot function

Parameters
----------
test : bool, default=True 
    suppresses plot figures
    
plot : bool, default=True
    shows figures using Pandas plot

ticker_data : pd.DataFrame
    a dataframe required for generating a plot

'''
import pandas as pd
import matplotlib as mpl

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

def plot(ticker_data: pd.DataFrame, test: bool=False)->None:
    if test:
        print('testing-- plots not shown')
        return # no plot
    else:
        ticker_data.plot() # show plot
    return
