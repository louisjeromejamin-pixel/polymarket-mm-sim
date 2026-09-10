"""Execution engine and portfolio accounting for the simulated market.

Replaces the two external dependencies of the upstream keeper — the CLOB API
and the blockchain — with a local simulation. Strategies are untouched: they
receive the same ``OrderBook`` and return the same lists of orders to cancel
and place.

Fills are driven by counterparty flow rather than a bare probability. A maker
order fills when a counterparty wants the other side and the price is
marketable, which is what makes the maker's revenue depend on *who* it trades
against.
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from poly_market_maker.order import Order, Side
from poly_market_maker.token import Collateral, Token

from .market_sim import OrderFlow, SimulatedMarket

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    """A full or partial execution."""

    step: int
    price: float
    size: float
    side: Side
    token: Token
    order_id: str
    informed_counterparty: bool = False

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class ExecutionConfig:
    """Execution model parameters.

    Attributes:
        maker_fee: Proportional fee on notional. Polymarket charges no maker
            fee today; the parameter exists to measure sensitivity if one
            were introduced.
        queue_priority: Probability of being served when a counterparty order
            arrives at the maker's price. Below 1 it models queue position:
            being at the right price does not guarantee being filled.
        size_per_order: Shares demanded by one counterparty order.
        uninformed_tolerance: How far from the mid an uninformed
            counterparty will still trade. This concession is what the maker
            earns; informed counterparties concede nothing.
    """

    maker_fee: float = 0.0
    queue_priority: float = 0.7
    size_per_order: float = 25.0
    uninformed_tolerance: float = 0.04


class Portfolio:
    """Tracks collateral, positions and mark-to-market value.

    Polymarket convention: holding one share of token A and one of token B in
    the same market guarantees exactly 1 USDC at resolution, the outcomes
    being complementary.
    """

    def __init__(self, initial_collateral: float = 1000.0):
        self.initial_collateral = initial_collateral
        self.collateral = initial_collateral
        self.positions: Dict[Token, float] = {Token.A: 0.0, Token.B: 0.0}
        self.fills: List[Fill] = []
        self.fees_paid = 0.0

    def apply_fill(self, fill: Fill, fee_rate: float) -> None:
        """Update the portfolio after an execution."""
        fee = fill.notional * fee_rate
        self.fees_paid += fee

        if fill.side == Side.BUY:
            self.collateral -= fill.notional + fee
            self.positions[fill.token] += fill.size
        else:
            self.collateral += fill.notional - fee
            self.positions[fill.token] -= fill.size

        self.fills.append(fill)

    def balances(self) -> dict:
        """Balances in the shape upstream strategies expect."""
        return {
            Collateral: self.collateral,
            Token.A: self.positions[Token.A],
            Token.B: self.positions[Token.B],
        }

    def mark_to_market(self, price_a: float) -> float:
        """Portfolio value at current prices."""
        price_b = 1.0 - price_a
        return (
            self.collateral
            + self.positions[Token.A] * price_a
            + self.positions[Token.B] * price_b
        )

    def value_at_resolution(self, outcome: int) -> float:
        """Final value once the market resolves: winner pays 1, loser 0."""
        winning = Token.A if outcome == 1 else Token.B
        return self.collateral + self.positions[winning]

    @property
    def inventory_skew(self) -> float:
        """Signed imbalance between the two legs, in shares.

        A market maker targets a balanced inventory; this gap is its exposure
        to price moves.
        """
        return self.positions[Token.A] - self.positions[Token.B]


class ExecutionEngine:
    """Matches the maker's orders against counterparty flow.

    A counterparty buy order lifts the maker's sell quotes; a counterparty
    sell order hits the maker's buy quotes. Filling informed flow leaves the
    maker on the wrong side of the next move; filling uninformed flow earns
    the spread.
    """

    def __init__(
        self,
        market: SimulatedMarket,
        portfolio: Portfolio,
        config: Optional[ExecutionConfig] = None,
        seed: Optional[int] = None,
    ):
        self.market = market
        self.portfolio = portfolio
        self.config = config or ExecutionConfig()
        self.open_orders: List[Order] = []
        self._rng = random.Random(seed)
        self._id_counter = itertools.count(1)

    def place_orders(self, orders: List[Order]) -> None:
        """Register new orders, assigning an id to each."""
        for order in orders:
            if order.id is None:
                order.id = f"sim-{next(self._id_counter)}"
            self.open_orders.append(order)

    def cancel_orders(self, orders: List[Order]) -> None:
        """Remove orders from the book."""
        to_cancel = {order.id for order in orders}
        self.open_orders = [o for o in self.open_orders if o.id not in to_cancel]

    def cancel_all(self) -> None:
        self.open_orders = []

    def match(self, flow: OrderFlow) -> List[Fill]:
        """Match open orders against this step's counterparty flow."""
        fills: List[Fill] = []

        # Counterparty buys consume the maker's sell quotes, and vice versa.
        demand = {
            Side.SELL: flow.buy_orders * self.config.size_per_order,
            Side.BUY: flow.sell_orders * self.config.size_per_order,
        }
        if not any(demand.values()):
            return fills

        still_open: List[Order] = []

        # Best prices first: a counterparty takes the most favourable quote.
        for order in self._by_price_priority():
            remaining_demand = demand.get(order.side, 0.0)

            if remaining_demand <= 0 or not self._is_marketable(order, flow):
                still_open.append(order)
                continue

            if self._rng.random() > self.config.queue_priority:
                still_open.append(order)  # someone else was ahead in the queue
                continue

            filled = min(order.size, remaining_demand)
            fill = Fill(
                step=self.market.step,
                price=order.price,
                size=round(filled, 2),
                side=order.side,
                token=order.token,
                order_id=order.id,
                informed_counterparty=flow.informed,
            )

            if fill.size <= 0 or not self._can_afford(fill):
                still_open.append(order)
                continue

            self.portfolio.apply_fill(fill, self.config.maker_fee)
            fills.append(fill)
            demand[order.side] = remaining_demand - fill.size

            remainder = round(order.size - fill.size, 4)
            if remainder > 0:
                order.size = remainder
                still_open.append(order)

        self.open_orders = still_open
        return fills

    def _by_price_priority(self) -> List[Order]:
        """Orders in the sequence a counterparty would take them.

        Counterparties hit the highest bid and lift the lowest ask.
        """
        buys = sorted(
            (o for o in self.open_orders if o.side == Side.BUY),
            key=lambda o: -o.price,
        )
        sells = sorted(
            (o for o in self.open_orders if o.side == Side.SELL),
            key=lambda o: o.price,
        )
        return buys + sells

    def _is_marketable(self, order: Order, flow: OrderFlow) -> bool:
        """Whether a counterparty would accept this quote.

        Uninformed counterparties trade for reasons unrelated to price and
        accept quotes within ``uninformed_tolerance`` of the mid — that
        concession is the maker's revenue.

        Informed counterparties know where the price is going and will not
        pay a worse price than the mid. Requiring them to cross the mid is
        what makes adverse selection bite: the maker only trades with them
        when its own quote is already unfavourable.
        """
        market_price = (
            self.market.mid_price
            if order.token == Token.A
            else round(1.0 - self.market.mid_price, 2)
        )
        tolerance = 0.0 if flow.informed else self.config.uninformed_tolerance

        if order.side == Side.BUY:
            return order.price >= market_price - tolerance
        return order.price <= market_price + tolerance

    def _can_afford(self, fill: Fill) -> bool:
        """Whether the portfolio can absorb the execution.

        Buying without collateral or selling shares not held is refused: a
        simulator that allows either produces meaningless backtests.
        """
        if fill.side == Side.BUY:
            cost = fill.notional * (1 + self.config.maker_fee)
            return self.portfolio.collateral >= cost
        return self.portfolio.positions[fill.token] >= fill.size
