"""Collect training data and fit the signal model.

Training and evaluation run on **disjoint seeds**: the model is fitted on one
set of simulated paths and judged on paths it has never seen. Reporting an
in-sample Sharpe would measure the fitting procedure, not the strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from poly_market_maker.orderbook import OrderBook
from poly_market_maker.token import Token

from .backtest import load_strategy
from .execution import ExecutionConfig, ExecutionEngine, Portfolio
from .market_sim import MarketConfig, SimulatedMarket
from .signal import FEATURE_NAMES, FeatureState, RidgeRegression, extract_features

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Feature matrix and target vector."""

    X: List[List[float]]
    y: List[float]

    def __len__(self) -> int:
        return len(self.y)

    def split(self, train_fraction: float = 0.7) -> Tuple["Dataset", "Dataset"]:
        """Chronological split — never shuffled.

        Shuffling a time series before splitting lets the model see the
        future of its own training set.
        """
        cut = int(len(self) * train_fraction)
        return (
            Dataset(self.X[:cut], self.y[:cut]),
            Dataset(self.X[cut:], self.y[cut:]),
        )


def collect_dataset(
    strategy_name: str = "bands",
    config_path: str = "config/bands.json",
    seeds: Sequence[int] = range(20),
    steps: int = 400,
    horizon: int = 3,
    market_config: Optional[MarketConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
    initial_collateral: float = 1000.0,
    initial_shares: float = 500.0,
) -> Dataset:
    """Run backtests and record (features, forward return) pairs.

    Args:
        horizon: Number of steps over which the target return is measured.
            One step is mostly tick noise; too many and the signal is no
            longer actionable by a maker requoting every step.

    Returns:
        The dataset. Features are observed at step ``t``, the target is the
        mid return from ``t`` to ``t + horizon``.
    """
    X: List[List[float]] = []
    y: List[float] = []

    for seed in seeds:
        market = SimulatedMarket(market_config or MarketConfig(), seed=seed)
        portfolio = Portfolio(initial_collateral)
        portfolio.positions[Token.A] = initial_shares
        portfolio.positions[Token.B] = initial_shares

        engine = ExecutionEngine(
            market, portfolio, execution_config or ExecutionConfig(), seed=seed + 10_000
        )
        strategy = load_strategy(strategy_name, config_path)
        state = FeatureState()

        pending: List[Tuple[List[float], float]] = []  # (features, mid at t)

        for _ in range(steps):
            price_a = market.mid_price
            flow = market.flow

            state.update(price_a, flow.imbalance)

            features = extract_features(
                state,
                mid_price=price_a,
                imbalance=flow.imbalance,
                order_count=flow.buy_orders + flow.sell_orders,
                inventory_skew=portfolio.inventory_skew,
            )
            pending.append((features, price_a))

            # Label the observation made `horizon` steps ago, now that its
            # forward return is known.
            if len(pending) > horizon:
                past_features, past_price = pending.pop(0)
                if past_price > 0:
                    X.append(past_features)
                    y.append((price_a - past_price) / past_price)

            token_prices = {Token.A: price_a, Token.B: round(1.0 - price_a, 2)}
            orderbook = OrderBook(
                orders=list(engine.open_orders),
                balances=portfolio.balances(),
                orders_being_placed=False,
                orders_being_cancelled=False,
            )
            try:
                to_cancel, to_place = strategy.get_orders(orderbook, token_prices)
            except Exception as exc:  # a strategy failure must not stop collection
                logger.debug("Strategy error at step %d: %s", market.step, exc)
                to_cancel, to_place = [], []

            engine.cancel_orders(to_cancel)
            engine.place_orders(to_place)
            engine.match(flow)
            market.advance()

            if market.resolved:
                break

    return Dataset(X, y)


def train_signal(
    dataset: Dataset,
    alpha: float = 1.0,
    train_fraction: float = 0.7,
) -> Tuple[RidgeRegression, dict]:
    """Fit the ridge model and report in- and out-of-sample R².

    Returns:
        The fitted model and a report holding both R² values and the
        standardised coefficients, which are directly comparable across
        features.
    """
    train, test = dataset.split(train_fraction)

    if len(train) < 10:
        raise ValueError(f"Training set too small ({len(train)} samples)")

    model = RidgeRegression(alpha=alpha).fit(train.X, train.y)

    report = {
        "n_train": len(train),
        "n_test": len(test),
        "r2_in_sample": round(model.score(train.X, train.y), 5),
        "r2_out_of_sample": round(model.score(test.X, test.y), 5) if len(test) > 1 else None,
        "coefficients": {
            name: round(coefficient, 5)
            for name, coefficient in zip(FEATURE_NAMES, model.coefficients)
        },
    }
    return model, report
