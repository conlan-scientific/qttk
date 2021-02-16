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

kind : str
 the kind of plot to produce:
    line
    bar = vertical bar plot
    barh = horizontal bar plot
    hist = histogram
    box = boxplot
    kde = Kernel Density Estimation plot
    area
    pie
    scatter
    hexbin
from: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
'''
import pandas as pd
import matplotlib as mpl

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

def plot(ticker_data: pd.DataFrame, test: bool=False, title: str=None,\
 kind: str=None, legend: bool=True)->None:
    if test:
        print('testing-- plots not shown')
        return # no plot
    else:
        ticker_data.plot(title=title, kind=kind, legend=legend) # show plot
    return
