# qttk
[qttk] Quantitative Trading ToolKit - Quant trading library developed by Conlan Scientific Open-Source Research Cohort

## Design Philosophy

+ **Consistent** Date-indexed `pandas` objects are the core data structure.
+ **Transparent** Test cases are clear and serve as an additional layer of documentation. Type hints are used liberally.
+ **Performant** Execution speed tests are built into test cases. Only the fastest functions get published.


## Getting Started with `qttk`

Install from PyPI using `pip install qttk`. Try the following sample.

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

## Next Steps

+ Check out [some demos on GitHub](https://github.com/conlan-scientific/qttk/tree/master/qttk/examples).
+ Feel free to create a bug or feature request ticket: [qttk issue tracker](https://github.com/conlan-scientific/qttk/issues).
