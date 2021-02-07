#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
import datetime as dt
from datetime import datetime
#importing module so that I can train for linear regression
from sklearn.model_selection import train_test_split


# In[16]:


style.use('ggplot')  #declaring style

def inv_rgsn(ticker,start_year,start_month):
    start = dt.datetime(start_year,start_month,1)   #Creating a start date of beg. of 2020 for stock
    end = dt.datetime.now()

    df = web.DataReader(ticker,'yahoo',start,end)   #reading data from yf to then store as a df

    ticker_str = ticker + '.csv'            #creating the ticker into a str that can create csv files

    df.to_csv(ticker_str)
    dataset = pd.read_csv(ticker_str)       #reading the csv file that was made

    x = dataset.iloc[:,0].values            #Getting the correct datasets
    x= pd.to_datetime(x)                    #beginning the process of changing the date strings to ints
    x = x.map(dt.datetime.toordinal)        #completing the str to int process

    dep_v = dataset.iloc[:,6].values        #obtaining the y dataset
    y = dep_v

    X = np.reshape(x,(-1,1))                #reshaping the array so that it can be used


    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .6, random_state = 0)    #training the given data
    # what is a good test_size/random state???

    from sklearn.linear_model import LinearRegression        #importing linear regression
    regressor = LinearRegression()                           #calling the regression
    regressor.fit(x_train,y_train)                           #fitting the regression


    y_pred = regressor.predict(x_test)


    plt.scatter(x_train,y_train,color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title('Stock Price vs Time (Training Set)' )
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt_show = plt.show()

    plt.scatter(x_test,y_test,color = 'red')
    plt.plot(x_train, regressor.predict(x_train), color = 'blue')
    plt.title('Stock Price vs Time (Test Set)' )
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt_show_test = plt.show()

    return plt_show, plt_show_test


# In[18]:


#b = inv_rgsn(, ,)


# In[ ]:
