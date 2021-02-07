'''
Exploration of natural log return series
Plot 1: How compariable are natural log and percent returns
Plot 2: Histogram like the plot in Algorithmic Trading with Python

Logarithms explained Bob Ross style:
  https://www.youtube.com/watch?v=up21mvokyQ4


'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from qttk.load_data import load_data

if __name__ == '__main__':
    
    # Todo: create figure with subplots to display all plots at once.

    # create simulated returns dataframe
    r1 = pd.Series(range(0,200,2))
    p1 = pd.Series([100]*100)

    returns_df = pd.concat([p1,r1],axis=1)
    returns_df.columns = ['pv','fv']
    returns_df['pct'] = (returns_df['fv'] - returns_df['pv'])/returns_df['pv']
    returns_df['log'] = np.log(returns_df['fv']/returns_df['pv'])
    
    # Create plot for log return vs percent return
    line_plot = returns_df[['pct', 'log']].plot(figsize=(10,10))
    line_plot.set_ylabel('return')
    line_plot.set_xlabel('closing price')
    line_plot.grid(color='b', linestyle='-', linewidth=.5)
    line_plot.axis([0, 100, -1.5, 1.5])
    line_plot.title = plt.title('Natural Log and Percent Returns ~ .20 +/-')
    plt.yticks(np.arange(-1.5, 1.5, step=0.05))    
    plt.show()

    # create histogram for log and percent return
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    close_df = load_data('AWU')[['close']]
    close_df['fv'] = close_df.shift(50)
    close_df = close_df[50:]
    close_df['pct'] = (close_df['fv']-close_df['close'])/close_df['close']
    close_df['log'] = np.log(close_df['fv']/close_df['close'])
    
    close_pct_hist = close_df['pct'].hist(ax=axes[0])
    close_pct_hist.title = plt.title('Percent Return')
    #plt.show()
    
    close_log_hist = close_df['log'].hist(ax=axes[1])
    close_log_hist.title = plt.title('Log Return')
    plt.show()
    


