# Indicators module contains core qttk functionality
from qttk.indicators import compute_rsi

# qttk is 'batteries included' with simulated data
from qttk.utils.sample_data import load_sample_data

# Load a ticker from csv into dataframe
df = load_sample_data('AWU')

print(df.head())

rsi = compute_rsi(df)

print(rsi[-10:])
