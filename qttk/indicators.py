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
from qttk.bollinger import compute_bb, graph_bb, demo_bollinger
from qttk.macd import compute_macd, graph_macd
from qttk.ma import moving_avg_v4 as compute_ma
from qttk.cma import cumulative_moving_avg_v2 as compute_cma
from qttk.ema import exponential_moving_average_v2 as compute_ema
from qttk.wma import weighted_moving_avg_v3 as compute_wma
from qttk.sharpe import calculate_return_series, calculate_sharpe_ratio
from qttk.portfolio import portfolio_price_series
from qttk.pl_return import compute_logr, compute_perr, graph_returns