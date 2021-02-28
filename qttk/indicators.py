'''
The Indicators module gathers **Production** Ready Indicators.
This is the primary entry point for using qttk

.. currentmodule:: indicators

.. autosummary::

   qttk.rsi.compute_net_returns
   qttk.rsi.compute_rsi
   qttk.bollinger.compute_bb
   qttk.bollinger.graph_bb
   qttk.bollinger.demo_bollinger
   qttk.macd.compute_macd
   qttk.macd.graph_macd
   qttk.ma.moving_avg_v4
   qttk.cma.cumulative_moving_avg_v2
   qttk.ema.exponential_moving_average_v2
   qttk.wma.weighted_moving_avg_v3
   qttk.sharpe.calculate_return_series
   qttk.sharpe.calculate_sharpe_ratio
   qttk.portfolio.portfolio_price_series
   qttk.pl_return.compute_logr
   qttk.pl_return.compute_perr
   qttk.pl_return.graph_returns
   qttk.r_squared.r_squared

'''
from qttk.rsi import compute_net_returns, compute_rsi
from qttk.bollinger import compute_bb, graph_bb, demo_bollinger
from qttk.macd import compute_macd, graph_macd
from qttk.ma import moving_avg_v4 as compute_ma
from qttk.cma import cumulative_moving_avg_v2 as compute_cma
from qttk.ema import exponential_moving_average_v3 as compute_ema
from qttk.wma import weighted_moving_avg_v3 as compute_wma
from qttk.sharpe import calculate_return_series, calculate_sharpe_ratio
from qttk.portfolio import portfolio_price_series
from qttk.pl_return import compute_logr, compute_perr, graph_returns
from qttk.r_squared import r_squared
from qttk.opportunity_eval import mean_return, r_squared_min
