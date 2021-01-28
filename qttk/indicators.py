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
from qttk.Bollinger_1 import bollinger, demo
from qttk.macd import macd
from qttk.mvgAvg import mvgAvg2 as ma
from qttk.testema import exponential_moving_average_v2 as ema
from qttk.testwma import weighted_moving_avg_v3 as wma
