:: Not tested on windows
:: This should wait for each file to run
start /wait python qttk\utils\data_utils.py
start /wait python qttk\utils\data_validation.py
start /wait python qttk\utils\sample_data.py
start /wait python qttk\bollinger.py
start /wait python qttk\cma.py
start /wait python qttk\cumulative_sum.py
start /wait python qttk\ema.py
start /wait python qttk\indicators.py
start /wait python qttk\ma.py
start /wait python qttk\macd.py
start /wait python qttk\portfolio.py
start /wait python qttk\rsi.py
start /wait python qttk\sharpe.py
start /wait python qttk\wma.py