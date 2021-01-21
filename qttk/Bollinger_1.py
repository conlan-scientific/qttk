#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt


# In[6]:


style.use('ggplot')  #Setting the style of the plot


# In[7]:

def bollinger(close: pd.Series):
    df = pd.read_csv(dataset)                 #reading the csv file that was made
    df.set_index('Date', inplace = True)      #setting the index to the date


    '''A Bollinger Band® is a technical analysis tool defined by a set of trendlines    plotted two standard deviations (positively and negatively) away from a simple
    moving average (SMA) of a security's price, but which can be adjusted to user
    preferences.'''
    
    #Calcutlating a 21 day moving average
    df['21 MA'] = df['Adj Close'].rolling(window =21 , min_periods = 5).mean()
    df['21 MA'] = df['21 MA'].fillna(0) 
    
    #Calculating the standard deviation of the adjusted close
    #Standard deviation emphasizes the outliers
    #It is sensitive to breakouts, adaptive to changes in regime
    df['21 std'] = df['Adj Close'].rolling(window = 21).std()
    df['21 std'] = df['21 std'].fillna(0)
    
    
    #This is calculating the upper band
    df['BOLU'] = df['21 MA'] + (2 * df['21 std'])
    df['BOLU'] = df['BOLU'].fillna(0)
    
    
    #This is calculating the lower band
    df['BOLD'] = df['21 MA'] - (2 * df['21 std'])
    df['BOLD'] = df['BOLD'].fillna(0)
    
    
    #Calculating the 50 day average volume
    #Both volume and the 50 day average will be plotted as line graphs
    df['50d Vol'] = df['Volume'].rolling(window = 50).mean()
    df['50d Vol'] = df['50d Vol'].fillna(0)
    
    
    #Tells us where we are in relation to the BB
    #chart will have 2 additonal lines that are 21 day high and low to see how it fluctates
    df['%b'] = ((df['Close']-df['BOLD'])/(df['BOLD']-df['BOLU']))
    
    
    #Tells us how wide the BB are
    #Lines are the highest and lowest values of bandwidth in the last 125 days 
    #High is bulge low is squeeze
    df['Bandwidth'] = (df['BOLU']-df['BOLD'])/df['21 MA']
    df['Bandwidth'] = df['Bandwidth'].fillna(0)
    
    
    #Plotting
    df[['Close','21 MA','BOLU','BOLD']].plot(figsize=(40,20))
    plt.grid(True)
    plt.title(' Bollinger Bands')
    plt.axis('tight')
    plt.ylabel('Price')
    
    return df



# Setup
# BB
# %b
# Bandwidth
# Volume

#     1) create line chart into a candlestick
#     2) Change inf to zero or a better variable
#     3) Create the 4 graphs on top of each other
#     4) Have a visual indicator on the graph to show it is buy/sell
#     5) Have an out response that says 3/4 of charts say this is a buy signal
#     6) Can this run for all stocks in market/sector... Make a list of all companies and for loop for data
#     7) turn parameter dataset to *args




# In[8]:
# import os
# script_dir = os.path.dirname(__file__)
# filepath = os.path.join(script_dir, '..', 'data', 'eod', 'AWU.csv')

x = bollinger('TSLA.csv')
x


# In[9]:


def volume(dataset):
    df = pd.read_csv(dataset) 
    df.set_index('Date', inplace = True)
    
    df['50d Vol'] = df['Volume'].rolling(window = 50).mean()
    
    df[['50d Vol']].plot(figsize=(40,20))

    ax1 = plt.subplot2grid((6,1),(0,0),rowspan = 5, colspan = 1)
    ax1.bar(df.index,df['Volume'])
    ax1.plot(df.index,df['50d Vol'], color = 'black')
    
    return df


# In[10]:


x = volume('TSLA.csv')
x


# In[11]:


def percent_b(dataset):
    df = pd.read_csv(dataset) 
    df.set_index('Date', inplace = True)
    
    df['%b'] = ((df['Close']-df['BOLD'])/(df['BOLD']-df['BOLU']))
    
    #Last - lower Band =  The distance from the closing price to the lower band
    return df


# In[12]:


def bandwidth(dataset):
    df = pd.read_csv(dataset) 
    df.set_index('Date', inplace = True)
    
    df['Bandwidth'] = (df['BOLU']-df['BOLD'])/df['21 MA']
    


# In[ ]:




