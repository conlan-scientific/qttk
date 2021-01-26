'''
Indicators Interface
  rsi
  bollinger
  ma
  ema
  wma
'''
from qttk.fiviz import compute_rsi as rsi
from qttk.Bollinger_1 import bollinger as bollinger
from qttk.mvgAvg import mvgAvg2 as ma
from qttk.testema import exponential_moving_average_v2 as ema
from qttk.testwma import weighted_moving_avg_v3 as wma
