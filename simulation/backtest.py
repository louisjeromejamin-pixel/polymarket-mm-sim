"""Backtest loop: runs an upstream strategy against the simulated market."""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from poly_market_maker.order import Order, Side
from poly_market_maker.orderbook import OrderBook
from poly_market_maker.strategies.amm_strategy import AMMStrategy
from poly_market_maker.strategies.bands_strategy import BandsStrategy
from poly_market_maker.token import Token

from .execution import ExecutionConfig, ExecutionEngine, Fill, Portfolio
from .market_sim import MarketConfig, SimulatedMarket
from .signal import FeatureState, SignalModel, extract_features

logger = logging.getLogger(__name__)

#: Steps per year, for annualising the Sharpe ratio. One step is taken to be
#: one minute of a continuously trading market.
STEPS_PER_YEAR = 525_600


@dataclass
class BacktestResult:
    """Backtest outcome, with the series needed for plots."""

    strategy: str
    steps: int
    initial_value: float
    final_value: float
    equity_curve: List[float] = field(default_factory=list)
    price_series: List[float] = field(default_factory=list)
    inventory_series: List[float] = field(default_factory=list)
    skew_series: List[int] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)
    fees_paid: float = 0.0
    resolved: bool = False
    outcome: Optional[int] = None

    @property
    def pnl(self) -> float:
        return self.final_value - self.initial_value

    @property
    def return_pct(self) -> float:
        return 100.0 * self.pnl / self.initial_value if self.initial_value else 0.0

    @property
    def num_fills(self) -> int:
        return len(self.fills)

    @property
    def informed_fill_ratio(self) -> float:
        """Share of fills taken by informed counterparties.

        The adverse-selection rate. A skew that works lowers it: the maker
        steps away from flow that is about to move against it.
        """
        if not self.fills:
            return 0.0
        return sum(f.informed_counterparty for f in self.fills) / len(self.fills)

    @property
    def volume(self) -> float:
        """Total notional traded."""
        return sum(f.notional for f in self.fills)

    @property
    def spread_captured(self) -> float:
        """Average sale price minus average purchase price."""
        buys = [f for f in self.fills if f.side == Side.BUY]
        sells = [f for f in self.fills if f.side == Side.SELL]

        buy_size = sum(f.size for f in buys)
        sell_size = sum(f.size for f in sells)
        if buy_size == 0 or sell_size == 0:
            return 0.0

        return (
            sum(f.notional for f in sells) / sell_size
            - sum(f.notional for f in buys) / buy_size
        )

    @property
    def max_drawdown_pct(self) -> float:
        """Largest peak-to-trough loss on the equity curve."""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        worst = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            if peak > 0:
                worst = max(worst, (peak - value) / peak)
        return 100.0 * worst

    @property
    def volatility_pct(self) -> float:
        """Standard deviation of per-step returns, in percent."""
        rets = self._step_returns()
        if len(rets) < 2:
            return 0.0
        return 100.0 * statistics.stdev(rets)

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio, no risk-free rate.

        Annualised at one step per minute. The figure serves to compare runs
        of this simulator, not for publication.
        """
        rets = self._step_returns()
        if len(rets) < 2:
            return 0.0
        sigma = statistics.stdev(rets)
        if sigma == 0:
            return 0.0
        return (statistics.mean(rets) / sigma) * math.sqrt(STEPS_PER_YEAR)

    @property
    def max_inventory_skew(self) -> float:
        """Largest inventory imbalance reached."""
        return max((abs(x) for x in self.inventory_series), default=0.0)

    def _step_returns(self) -> List[float]:
        return [
            (b - a) / a
            for a, b in zip(self.equity_curve, self.equity_curve[1:])
            if a != 0
        ]

    def summary(self) -> Dict[str, float]:
        return {
            "strategy": self.strategy,
            "steps": self.steps,
            "initial_value": round(self.initial_value, 2),
            "final_value": round(self.final_value, 2),
            "pnl": round(self.pnl, 2),
            "return_pct": round(self.return_pct, 3),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 3),
            "fills": self.num_fills,
            "informed_fill_pct": round(100 * self.informed_fill_ratio, 1),
            "spread_captured": round(self.spread_captured, 4),
            "volume": round(self.volume, 2),
            "fees": round(self.fees_paid, 4),
            "max_inventory": round(self.max_inventory_skew, 1),
        }


def load_strategy(name: str, config_path: str):
    """Instantiate an upstream strategy from its config file."""
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    key = name.lower()
    if key == "amm":
        return AMMStrategy(config)
    if key == "bands":
        return BandsStrategy(config)
    raise ValueError(f"Unknown strategy: {name!r} (expected amm or bands)")


def _apply_skew(orders: List[Order], skew_ticks: int, tick: float = 0.01) -> None:
    """Shift every quote by ``skew_ticks``, in place.

    Both sides move together: a predicted rise makes the maker bid higher
    *and* ask higher, so it is less likely to sell into the move and more
    likely to accumulate ahead of it. Prices are clamped one tick inside
    (0, 1), where a binary contract cannot trade.
    """
    if skew_ticks == 0:
        return

    shift = skew_ticks * tick
    for order in orders:
        order.price = round(min(0.99, max(0.01, order.price + shift)), 2)


def run_backtest(
    strategy_name: str,
    config_path: str,
    steps: int = 500,
    market_config: Optional[MarketConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
    signal: Optional[SignalModel] = None,
    initial_collateral: float = 1000.0,
    initial_shares: float = 500.0,
    seed: Optional[int] = None,
) -> BacktestResult:
    """Run a strategy against the simulated market.

    Args:
        strategy_name: ``"amm"`` or ``"bands"``.
        config_path: JSON config for the strategy.
        steps: Simulation steps.
        market_config: Market parameters; defaults otherwise.
        execution_config: Execution parameters; defaults otherwise.
        signal: Fitted signal model used to skew quotes. Without one the
            strategy quotes symmetrically, which is the baseline.
        initial_collateral: Starting USDC.
        initial_shares: Shares held on each leg at the start. A maker must
            hold inventory to be able to sell.
        seed: Random seed, for reproducibility.

    Returns:
        The backtest result, including time series.
    """
    market = SimulatedMarket(market_config or MarketConfig(), seed=seed)
    portfolio = Portfolio(initial_collateral)
    portfolio.positions[Token.A] = initial_shares
    portfolio.positions[Token.B] = initial_shares

    engine = ExecutionEngine(
        market,
        portfolio,
        execution_config or ExecutionConfig(),
        seed=None if seed is None else seed + 1,
    )
    strategy = load_strategy(strategy_name, config_path)
    state = FeatureState()

    initial_value = portfolio.mark_to_market(market.mid_price)
    result = BacktestResult(
        strategy=strategy_name,
        steps=steps,
        initial_value=initial_value,
        final_value=initial_value,
    )

    for _ in range(steps):
        price_a = market.mid_price
        flow = market.flow
        token_prices = {Token.A: price_a, Token.B: round(1.0 - price_a, 2)}

        state.update(price_a, flow.imbalance)

        orderbook = OrderBook(
            orders=list(engine.open_orders),
            balances=portfolio.balances(),
            orders_being_placed=False,
            orders_being_cancelled=False,
        )

        try:
            to_cancel, to_place = strategy.get_orders(orderbook, token_prices)
        except Exception as exc:  # a strategy failure must not kill the run
            logger.warning("Strategy error at step %d: %s", market.step, exc)
            to_cancel, to_place = [], []

        skew = 0
        if signal is not None and signal.is_fitted:
            features = extract_features(
                state,
                mid_price=price_a,
                imbalance=flow.imbalance,
                order_count=flow.buy_orders + flow.sell_orders,
                inventory_skew=portfolio.inventory_skew,
            )
            skew = signal.skew_ticks(features)
            _apply_skew(to_place, skew)

        engine.cancel_orders(to_cancel)
        engine.place_orders(to_place)

        # Orders are matched against the flow arriving on this step, then the
        # market moves. Informed flow therefore fills just before the move it
        # anticipates, which is what adverse selection means.
        result.fills.extend(engine.match(flow))

        result.equity_curve.append(portfolio.mark_to_market(market.mid_price))
        result.price_series.append(market.mid_price)
        result.inventory_series.append(portfolio.inventory_skew)
        result.skew_series.append(skew)

        market.advance()
        if market.resolved:
            break

    if market.resolved and market.outcome is not None:
        result.final_value = portfolio.value_at_resolution(market.outcome)
        result.resolved = True
        result.outcome = market.outcome
    else:
        result.final_value = portfolio.mark_to_market(market.mid_price)

    result.fees_paid = portfolio.fees_paid
    return result


def run_sweep(
    strategy_name: str,
    config_path: str,
    seeds: Sequence[int],
    steps: int = 500,
    market_config: Optional[MarketConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
    signal: Optional[SignalModel] = None,
) -> List[BacktestResult]:
    """Replay the same backtest across several seeds.

    A single backtest on a random market proves nothing: the outcome depends
    as much on the draw as on the strategy.
    """
    return [
        run_backtest(
            strategy_name,
            config_path,
            steps=steps,
            market_config=market_config,
            execution_config=execution_config,
            signal=signal,
            seed=seed,
        )
        for seed in seeds
    ]
