"""Synthetic market for a Polymarket-style binary contract.

A binary prediction market resolves to 0 or 1, so its price is bounded in
``(0, 1)`` and reads directly as a probability. An arithmetic random walk
would leave that interval and would assign the same absolute volatility at
0.50 as at 0.02, where a near-resolved contract barely moves.

The price therefore evolves in **log-odds**:

    logit(p) = log(p / (1 - p))

The walk is unconstrained on the real line and the sigmoid maps it back into
``(0, 1)`` by construction, compressing moves near the boundaries.

Order flow is split in two, because the distinction determines whether market
making is viable at all:

**Informed flow** trades ahead of the price move. A market maker who fills it
is on the wrong side — this is adverse selection.

**Uninformed flow** trades for reasons unrelated to the next price move
(hedging, liquidity, noise). Filling it earns the spread. It is the actual
source of a market maker's revenue, and a simulator without it makes every
strategy lose by construction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


def logit(p: float) -> float:
    """Convert a probability to log-odds."""
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Convert log-odds to a probability, overflow-safe."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class MarketConfig:
    """Parameters of the simulated market.

    Attributes:
        initial_price: Starting probability, in ``(0, 1)``.
        volatility: Standard deviation of the log-odds step.
        drift: Log-odds drift per step. Zero is a martingale, the honest
            setting for evaluating a market maker.
        informed_ratio: Share of order flow that trades ahead of the price
            move. At 1.0 every counterparty is informed and no strategy can
            profit; empirically prediction markets sit well below that.
        flow_intensity: Expected number of counterparty orders per step.
        spread_ticks: Half-width of the external book, in one-cent ticks.
        tick: Price granularity (Polymarket quotes in cents).
        resolution_step: Step at which the market resolves, or ``None``.
    """

    initial_price: float = 0.50
    volatility: float = 0.15
    drift: float = 0.0
    informed_ratio: float = 0.35
    flow_intensity: float = 2.0
    spread_ticks: int = 2
    tick: float = 0.01
    resolution_step: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.initial_price < 1.0:
            raise ValueError("initial_price must lie strictly between 0 and 1")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if not 0.0 <= self.informed_ratio <= 1.0:
            raise ValueError("informed_ratio must lie in [0, 1]")
        if self.flow_intensity < 0:
            raise ValueError("flow_intensity must be non-negative")
        if self.spread_ticks < 1:
            raise ValueError("spread_ticks must be at least 1")


@dataclass
class OrderFlow:
    """Counterparty orders arriving on one step.

    Attributes:
        buy_orders: Number of counterparties wanting to buy (they lift asks).
        sell_orders: Number wanting to sell (they hit bids).
        informed: Whether this batch trades ahead of the price move.
    """

    buy_orders: int = 0
    sell_orders: int = 0
    informed: bool = False

    @property
    def imbalance(self) -> float:
        """Signed flow imbalance in ``[-1, 1]``.

        The most informative single feature available to a market maker:
        persistent one-sided flow precedes price moves.
        """
        total = self.buy_orders + self.sell_orders
        if total == 0:
            return 0.0
        return (self.buy_orders - self.sell_orders) / total


@dataclass
class MarketState:
    """Snapshot of the market at one step."""

    step: int
    mid_price: float
    best_bid: float
    best_ask: float
    flow: OrderFlow = field(default_factory=OrderFlow)
    resolved: bool = False
    outcome: Optional[int] = None


class SimulatedMarket:
    """Binary market whose price follows a log-odds random walk.

    The market does not know the maker's orders or positions; the execution
    engine matches the two.
    """

    def __init__(self, config: MarketConfig, seed: Optional[int] = None):
        self.config = config
        self._rng = random.Random(seed)
        self._logit_price = logit(config.initial_price)
        self.step = 0
        self.resolved = False
        self.outcome: Optional[int] = None
        self.history: List[MarketState] = []

        # The next step's shock is drawn one step early so informed flow can
        # be aligned with it: informed counterparties trade in the direction
        # the price is about to move.
        self._next_shock = self._draw_shock()
        self.flow = self._draw_flow(self._next_shock)
        self._record()

    def _draw_shock(self) -> float:
        return self._rng.gauss(self.config.drift, self.config.volatility)

    def _draw_flow(self, next_shock: float) -> OrderFlow:
        """Generate the counterparty orders for this step.

        Informed batches lean in the direction of the upcoming move;
        uninformed batches are symmetric around zero.
        """
        cfg = self.config
        n_orders = _poisson(self._rng, cfg.flow_intensity)

        if n_orders == 0:
            return OrderFlow()

        informed = self._rng.random() < cfg.informed_ratio

        if informed:
            # Directional: most orders lean the way the price will move.
            p_buy = 0.85 if next_shock > 0 else 0.15
        else:
            p_buy = 0.5

        buys = sum(1 for _ in range(n_orders) if self._rng.random() < p_buy)
        return OrderFlow(buy_orders=buys, sell_orders=n_orders - buys, informed=informed)

    @property
    def mid_price(self) -> float:
        """Mid price, rounded to the tick."""
        raw = sigmoid(self._logit_price)
        ticks = round(raw / self.config.tick)
        # Keep at least one tick away from 0 and 1: an unresolved binary
        # contract never trades exactly at its bounds.
        ticks = max(1, min(ticks, int(round(1 / self.config.tick)) - 1))
        return round(ticks * self.config.tick, 2)

    @property
    def best_bid(self) -> float:
        return round(self.mid_price - self.config.spread_ticks * self.config.tick, 2)

    @property
    def best_ask(self) -> float:
        return round(self.mid_price + self.config.spread_ticks * self.config.tick, 2)

    def advance(self) -> MarketState:
        """Advance one step and return the new state."""
        if self.resolved:
            return self.history[-1]

        self._logit_price += self._next_shock
        self.step += 1

        self._next_shock = self._draw_shock()
        self.flow = self._draw_flow(self._next_shock)

        if (
            self.config.resolution_step is not None
            and self.step >= self.config.resolution_step
        ):
            self._resolve()

        return self._record()

    def _resolve(self) -> None:
        """Resolve the market, drawing the outcome from the current price.

        Drawing the outcome with probability equal to the price keeps the
        market coherent: a contract at 0.80 resolves yes eight times in ten.
        """
        self.outcome = 1 if self._rng.random() < self.mid_price else 0
        self.resolved = True

    def _record(self) -> MarketState:
        state = MarketState(
            step=self.step,
            mid_price=self.mid_price,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            flow=self.flow,
            resolved=self.resolved,
            outcome=self.outcome,
        )
        self.history.append(state)
        return state


def _poisson(rng: random.Random, lam: float) -> int:
    """Poisson draw by Knuth's method.

    ``random.Random`` has no Poisson sampler and numpy is not a dependency
    here. Adequate for the small intensities used (lambda of a few units).
    """
    if lam <= 0:
        return 0

    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= target:
            return k
        k += 1
        if k > 1000:  # guard against pathological intensities
            return k
