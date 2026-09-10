"""Offline simulation layer: backtest strategies without API keys or capital."""

from .backtest import BacktestResult, load_strategy, run_backtest, run_sweep
from .execution import ExecutionConfig, ExecutionEngine, Fill, Portfolio
from .market_sim import (
    MarketConfig,
    MarketState,
    OrderFlow,
    SimulatedMarket,
    logit,
    sigmoid,
)
from .signal import (
    FEATURE_NAMES,
    FeatureState,
    RidgeRegression,
    SignalConfig,
    SignalModel,
    extract_features,
)
from .training import Dataset, collect_dataset, train_signal

__all__ = [
    "BacktestResult",
    "Dataset",
    "ExecutionConfig",
    "ExecutionEngine",
    "FEATURE_NAMES",
    "FeatureState",
    "Fill",
    "MarketConfig",
    "MarketState",
    "OrderFlow",
    "Portfolio",
    "RidgeRegression",
    "SignalConfig",
    "SignalModel",
    "SimulatedMarket",
    "collect_dataset",
    "extract_features",
    "load_strategy",
    "logit",
    "run_backtest",
    "run_sweep",
    "sigmoid",
    "train_signal",
]
