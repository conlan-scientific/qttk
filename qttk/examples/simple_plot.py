'''
# Example demonstrating the use of qttk_plot.py and sample_data.py

  qttk_plot.py: plotting can be suppressed by passing an argument
  test = True. the default is test = False, i.e. show plots.

  example with pd.DataFrame datatype:
  plot(ticker_data: pd.DataFrame, test: bool=False)

  example with pd.Series datatype:
  plot(ticker_data: pd.Series, test: bool=False)
'''
from qttk.utils.qttk_plot import plot
from qttk.utils.sample_data import load_sample_data

'''
plot pd.DataFrame
'''
ticker = 'TRE'
dataframe = load_sample_data(ticker)

print(dataframe.info()) # show dataframe structure

print('Simple Plot demo- test=True, don\'t show plot')
plot(dataframe.iloc[-30:, 1], test=True, title=ticker)

print('Simple Plot demo- test=False, show plot:')
plot(dataframe.iloc[-30:, 1], test=False, title=ticker)

'''
# alternative form:
# plot(dataframe['close'])
'''

'''
plot pd.Series
'''
print('Plot series demo- test=False, show plot:')
series = dataframe['open']
print(type(series))
plot(series[-30:], test=False, title=ticker)
