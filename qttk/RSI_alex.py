import pandas as pd


def rsi(dataframe, window):  # defining rsi with needed csv. and desired window
    df = pd.read_csv(dataframe)  # reading the csv file that was made
    df.set_index('date', inplace=True)  # setting the index to the date

    adj_close = df['close']  # assigning the adj. close

    # close_diff = adj_close.diff()                     #subtracts the lower value with the upper with respect to time
    close_diff = adj_close / adj_close.shift(1)-1  # divides the lower value with the upper with respect to time
    close_diff = close_diff[:]  # putting the difference into a list
    #df['close diff'] = close_diff

    up, down = close_diff.copy(), close_diff.copy()  # assigning list to 2 variables
    up[up < 0] = 0  # puts num > 0 into up
    down[down > 0] = 0  # puts num < 0 into down

    up_sma = up.rolling(window=window).mean()  # creating a sma of the up values w/ window
    #df['up_sma'] = up_sma                                                                                       
    #df['up'] = up

    down_sma = down.abs().rolling(window=window).mean()  # creating a sma of the absolute down values w/ window

    rs = up_sma / down_sma  # getting relative strength
    rsi = 100.0 - (100.0 / (1.0 + rs))  # getting RSI

    df['RSI'] = rsi  # adding RSI to the dataframe
    df['RSI'] = df['RSI'].fillna(0)  # filling nan with 0

    df[['RSI']].plot(figsize=(40, 20))  # plotting RSI vs time

    return df



