# qttk
[qttk] Quantitative Trading ToolKit - Quant trading library developed by Conlan Scientific Open-Source Research Cohort

## Requirements:
* python >= 3.7
* numpy, pandas, matplotlib
* checkout setup.md for conda environment setup

## Philosophy
* Transparent and open source
* Well documented - function signatures use type hints
* Performance tested

## Getting Started with QTTK

```python
# Indicators module contains core qttk functionality
from qttk.indicators import compute_rsi

# qttk is 'batteries included' with simulated data
from qttk.utils.sample_data import load_sample_data

# Load a ticker from csv into dataframe
df = load_sample_data('AWU')

print(df.head())

rsi = compute_rsi(df)

print(rsi[-10:])
```

## Next Steps:
* Checkout examples/ for cool qttk demos
* Feel free to create a bug or feature request ticket: [qttk issue tracker](https://github.com/conlan-scientific/qttk/issues) 
