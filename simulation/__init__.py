"""Couche de simulation : backtest des stratégies sans clé API ni capital."""

from .backtest import BacktestResult, load_strategy, run_backtest, run_sweep
from .execution import ExecutionConfig, ExecutionEngine, Fill, Portfolio
from .market_sim import MarketConfig, MarketState, SimulatedMarket, logit, sigmoid

__all__ = [
    "BacktestResult",
    "ExecutionConfig",
    "ExecutionEngine",
    "Fill",
    "MarketConfig",
    "MarketState",
    "Portfolio",
    "SimulatedMarket",
    "load_strategy",
    "logit",
    "run_backtest",
    "run_sweep",
    "sigmoid",
]
