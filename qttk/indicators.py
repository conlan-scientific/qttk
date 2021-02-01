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
from qttk.bollinger import bollinger, bollinger_demo
from qttk.macd import macd
from qttk.ma import moving_avg_v4 as compute_ma
from qttk.cma import cumulative_moving_avg_v2 as compute_cma
from qttk.ema import exponential_moving_average_v2 as compute_ema
from qttk.wma import weighted_moving_avg_v3 as compute_wma


