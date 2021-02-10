'''
# Example demonstrating the use of qttk_plot.py and sample_data.py
# qttk_plot.py: plotting can be suppressed by passing an argument
# test = True. the default is test = False, i.e. show plots.
# plot(ticker_data: pd.DataFrame, test: bool=False)
'''
from qttk.utils.qttk_plot import plot
from qttk.utils.sample_data import load_sample_data

ticker = 'EUZ'
dataframe = load_sample_data(ticker)

print('Simple Plot demo- test=True, don\'t show plot')
plot(dataframe['close'], test=True)

print('Simple Plot demo- test=False, show plot:')
plot(dataframe['close'], test=False)
'''
# alternative form:
# plot(dataframe['close'])
'''
