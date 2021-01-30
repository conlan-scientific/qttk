'''
Indicators Interface
  rsi
  bollinger
  macd
  ma
  ema
  wma
'''
from qttk.rsi import compute_net_returns, compute_rsi
from qttk.bollinger import bollinger, bollinger_demo
from qttk.macd import macd
from qttk.mvg_avg import mvgAvg2 as ma
from qttk.ema import exponential_moving_average_v2 as ema
from qttk.wma import weighted_moving_avg_v3 as wma
