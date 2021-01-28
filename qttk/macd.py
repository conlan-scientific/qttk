import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def macd(dataset, n1, n2):
    df = pd.read_csv(dataset)  # reading the csv file that was made
    df.set_index('date', inplace=True)  # setting the index to the date

    assert n1 < n2

    exp1 = df['close'].ewm(span=n1, adjust=False).mean()  # getting the exp. moving average of the first period
    exp2 = df['close'].ewm(span=n2, adjust=False).mean()  # getting the exp. moving average of the second period

    macd_calc = exp1 - exp2  # obtaining the MACD from subtracting the EMA's
    df['MACD'] = macd_calc  # putting MACD into the dataframe
    df['MACD'] = df['MACD'].fillna(0)  # filling the nan with 0

    exp3 = macd_calc.rolling(window=9).mean()  # obtaining the moving average of the MACD

    # Plotting
    macd_calc.plot(label='MACD', color='g')
    ax = exp3.plot(label='Signal Line', color='b')
    df['close'].plot(ax=ax, secondary_y=True, label='Price')

    ax.set_ylabel('MACD')
    ax.right_ax.set_ylabel('Price $')
    ax.set_xlabel('Date')
    ax.legend(loc=0)
    plt.show()

    return df
