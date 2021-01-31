'''
Production Ready Indicators Interface
  rsi
  bollinger
  macd
  ma
  ema
  wma
'''
from qttk.rsi import compute_net_returns, compute_rsi
from qttk.Bollinger_1 import bollinger, bollinger_demo
from qttk.macd import macd
from qttk.moving_average import moving_average_v3 as compute_ma
from qttk.cumulative_moving_average import cumulative_moving_avg_v2 as compute_cma
from qttk.exponential_moving_average import exponential_moving_average_v2 as compute_ema
from qttk.weighted_moving_average import weighted_moving_avg_v3 as compute_wma
