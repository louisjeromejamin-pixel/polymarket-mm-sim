"""Tests for the market model and execution engine."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from poly_market_maker.order import Order, Side
from poly_market_maker.token import Collateral, Token
from simulation.execution import ExecutionConfig, ExecutionEngine, Fill, Portfolio
from simulation.market_sim import (
    MarketConfig,
    OrderFlow,
    SimulatedMarket,
    _poisson,
    logit,
    sigmoid,
)


# --- Market ---------------------------------------------------------------


def test_logit_and_sigmoid_are_inverse():
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)


def test_sigmoid_does_not_overflow():
    assert sigmoid(-1000) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(1000) == pytest.approx(1.0, abs=1e-12)


def test_price_stays_within_bounds():
    """Even under extreme volatility a probability must stay in (0, 1)."""
    market = SimulatedMarket(MarketConfig(volatility=2.0), seed=7)
    for _ in range(500):
        market.advance()
        assert 0.0 < market.mid_price < 1.0


def test_same_seed_same_path():
    a = SimulatedMarket(MarketConfig(), seed=123)
    b = SimulatedMarket(MarketConfig(), seed=123)
    for _ in range(50):
        a.advance()
        b.advance()
    assert [s.mid_price for s in a.history] == [s.mid_price for s in b.history]


def test_bid_below_ask():
    market = SimulatedMarket(MarketConfig(), seed=1)
    for _ in range(30):
        market.advance()
        assert market.best_bid < market.mid_price < market.best_ask


def test_market_resolves_at_the_requested_step():
    market = SimulatedMarket(MarketConfig(resolution_step=10), seed=3)
    for _ in range(15):
        market.advance()
    assert market.resolved
    assert market.outcome in (0, 1)


def test_config_rejects_invalid_values():
    for kwargs in (
        {"initial_price": 0.0},
        {"initial_price": 1.5},
        {"volatility": -1.0},
        {"informed_ratio": 1.5},
        {"flow_intensity": -1.0},
        {"spread_ticks": 0},
    ):
        with pytest.raises(ValueError):
            MarketConfig(**kwargs)


# --- Order flow -----------------------------------------------------------


def test_poisson_mean_and_variance():
    """A Poisson variable has equal mean and variance."""
    rng = random.Random(0)
    for lam in (0.5, 2.0, 5.0):
        sample = [_poisson(rng, lam) for _ in range(20000)]
        assert statistics.mean(sample) == pytest.approx(lam, rel=0.05)
        assert statistics.variance(sample) == pytest.approx(lam, rel=0.10)


def test_poisson_of_zero():
    assert _poisson(random.Random(0), 0.0) == 0


def test_informed_share_matches_configuration():
    market = SimulatedMarket(MarketConfig(informed_ratio=0.35), seed=42)
    flows = []
    for _ in range(2000):
        flows.append(market.flow)
        market.advance()

    active = [f for f in flows if f.buy_orders + f.sell_orders > 0]
    informed = sum(f.informed for f in active)
    assert informed / len(active) == pytest.approx(0.35, abs=0.05)


def test_informed_flow_is_more_directional():
    """Informed counterparties lean one way; uninformed ones do not."""
    market = SimulatedMarket(MarketConfig(informed_ratio=0.5), seed=0)
    informed, uninformed = [], []

    for _ in range(3000):
        flow = market.flow
        if flow.buy_orders + flow.sell_orders > 0:
            (informed if flow.informed else uninformed).append(abs(flow.imbalance))
        market.advance()

    assert statistics.mean(informed) > statistics.mean(uninformed)


def test_imbalance_bounds():
    assert OrderFlow(buy_orders=5, sell_orders=0).imbalance == 1.0
    assert OrderFlow(buy_orders=0, sell_orders=5).imbalance == -1.0
    assert OrderFlow(buy_orders=3, sell_orders=3).imbalance == 0.0
    assert OrderFlow().imbalance == 0.0


def test_no_flow_without_intensity():
    market = SimulatedMarket(MarketConfig(flow_intensity=0.0), seed=1)
    for _ in range(50):
        assert market.flow.buy_orders + market.flow.sell_orders == 0
        market.advance()


# --- Portfolio ------------------------------------------------------------


def test_buy_debits_collateral_and_credits_position():
    portfolio = Portfolio(100.0)
    portfolio.apply_fill(
        Fill(0, price=0.40, size=10.0, side=Side.BUY, token=Token.A, order_id="x"),
        fee_rate=0.0,
    )
    assert portfolio.collateral == pytest.approx(96.0)
    assert portfolio.positions[Token.A] == 10.0


def test_sell_credits_collateral():
    portfolio = Portfolio(100.0)
    portfolio.positions[Token.A] = 20.0
    portfolio.apply_fill(
        Fill(0, price=0.60, size=10.0, side=Side.SELL, token=Token.A, order_id="x"),
        fee_rate=0.0,
    )
    assert portfolio.collateral == pytest.approx(106.0)
    assert portfolio.positions[Token.A] == 10.0


def test_fees_charged_on_both_sides():
    buy = Portfolio(100.0)
    buy.apply_fill(Fill(0, 0.50, 10.0, Side.BUY, Token.A, "x"), fee_rate=0.01)
    assert buy.collateral == pytest.approx(100.0 - 5.0 - 0.05)

    sell = Portfolio(100.0)
    sell.positions[Token.A] = 10.0
    sell.apply_fill(Fill(0, 0.50, 10.0, Side.SELL, Token.A, "x"), fee_rate=0.01)
    assert sell.collateral == pytest.approx(100.0 + 5.0 - 0.05)


def test_value_at_resolution():
    portfolio = Portfolio(50.0)
    portfolio.positions[Token.A] = 30.0
    portfolio.positions[Token.B] = 20.0
    assert portfolio.value_at_resolution(outcome=1) == pytest.approx(80.0)
    assert portfolio.value_at_resolution(outcome=0) == pytest.approx(70.0)


def test_balanced_position_is_outcome_independent():
    """One share of A and one of B guarantee 1 USDC whatever the outcome."""
    portfolio = Portfolio(0.0)
    portfolio.positions[Token.A] = 40.0
    portfolio.positions[Token.B] = 40.0
    assert portfolio.value_at_resolution(1) == pytest.approx(40.0)
    assert portfolio.value_at_resolution(0) == pytest.approx(40.0)


def test_balances_use_the_upstream_shape():
    assert set(Portfolio(10.0).balances()) == {Collateral, Token.A, Token.B}


def test_inventory_skew():
    portfolio = Portfolio(0.0)
    portfolio.positions[Token.A] = 30.0
    portfolio.positions[Token.B] = 12.0
    assert portfolio.inventory_skew == pytest.approx(18.0)


# --- Execution ------------------------------------------------------------


def _engine(price=0.50, **kwargs):
    market = SimulatedMarket(MarketConfig(initial_price=price, flow_intensity=0.0), seed=0)
    portfolio = Portfolio(1000.0)
    portfolio.positions[Token.A] = 100.0
    portfolio.positions[Token.B] = 100.0
    config = ExecutionConfig(queue_priority=1.0, **kwargs)
    return market, portfolio, ExecutionEngine(market, portfolio, config, seed=0)


def test_no_fill_without_counterparty():
    """A resting order needs someone on the other side."""
    _, _, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.48, side=Side.BUY, token=Token.A)])
    assert engine.match(OrderFlow()) == []
    assert len(engine.open_orders) == 1


def test_counterparty_sell_hits_the_maker_bid():
    _, _, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.48, side=Side.BUY, token=Token.A)])

    fills = engine.match(OrderFlow(sell_orders=1))
    assert len(fills) == 1
    assert fills[0].side == Side.BUY


def test_counterparty_buy_lifts_the_maker_ask():
    _, portfolio, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.52, side=Side.SELL, token=Token.A)])

    fills = engine.match(OrderFlow(buy_orders=1))
    assert len(fills) == 1
    assert fills[0].side == Side.SELL


def test_informed_counterparty_will_not_pay_a_worse_price():
    """Informed flow crosses the mid only; uninformed flow concedes a little.

    At a mid of 0.50 a bid of 0.47 is three cents away, inside the four-cent
    uninformed tolerance but outside the zero tolerance of informed flow.
    That gap is the maker's revenue.
    """
    _, _, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.47, side=Side.BUY, token=Token.A)])

    assert engine.match(OrderFlow(sell_orders=2, informed=True)) == []
    assert len(engine.match(OrderFlow(sell_orders=2, informed=False))) == 1


def test_uninformed_tolerance_is_bounded():
    """Beyond the tolerance, even uninformed flow walks away."""
    _, _, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.40, side=Side.BUY, token=Token.A)])
    assert engine.match(OrderFlow(sell_orders=2, informed=False)) == []


def test_fill_records_the_counterparty_type():
    _, _, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.50, side=Side.BUY, token=Token.A)])

    fills = engine.match(OrderFlow(sell_orders=1, informed=True))
    assert fills and fills[0].informed_counterparty is True


def test_demand_is_shared_across_orders():
    """One counterparty order cannot fill more than its own size."""
    _, _, engine = _engine(price=0.50, size_per_order=10.0)
    engine.place_orders(
        [
            Order(size=100.0, price=0.49, side=Side.BUY, token=Token.A),
            Order(size=100.0, price=0.48, side=Side.BUY, token=Token.A),
        ]
    )
    fills = engine.match(OrderFlow(sell_orders=1))
    assert sum(f.size for f in fills) == pytest.approx(10.0)


def test_best_price_fills_first():
    """A counterparty selling hits the highest bid."""
    _, _, engine = _engine(price=0.50, size_per_order=5.0)
    engine.place_orders(
        [
            Order(size=5.0, price=0.47, side=Side.BUY, token=Token.A),
            Order(size=5.0, price=0.49, side=Side.BUY, token=Token.A),
        ]
    )
    fills = engine.match(OrderFlow(sell_orders=1))
    assert len(fills) == 1
    assert fills[0].price == 0.49


def test_token_b_price_is_complementary():
    """With A at 0.70, B is 0.30: a bid on B at 0.28 is reachable."""
    _, _, engine = _engine(price=0.70)
    engine.place_orders([Order(size=10.0, price=0.28, side=Side.BUY, token=Token.B)])
    assert len(engine.match(OrderFlow(sell_orders=1))) == 1


def test_buy_refused_without_collateral():
    market = SimulatedMarket(MarketConfig(initial_price=0.50, flow_intensity=0.0), seed=0)
    portfolio = Portfolio(1.0)  # too little for 10 shares at 0.49
    engine = ExecutionEngine(market, portfolio, ExecutionConfig(queue_priority=1.0), seed=0)

    engine.place_orders([Order(size=10.0, price=0.49, side=Side.BUY, token=Token.A)])
    assert engine.match(OrderFlow(sell_orders=1)) == []
    assert portfolio.collateral == 1.0


def test_short_selling_refused():
    market = SimulatedMarket(MarketConfig(initial_price=0.50, flow_intensity=0.0), seed=0)
    portfolio = Portfolio(100.0)  # holds no shares
    engine = ExecutionEngine(market, portfolio, ExecutionConfig(queue_priority=1.0), seed=0)

    engine.place_orders([Order(size=10.0, price=0.51, side=Side.SELL, token=Token.A)])
    assert engine.match(OrderFlow(buy_orders=1)) == []
    assert portfolio.positions[Token.A] == 0.0


def test_cancel_removes_from_the_book():
    _, _, engine = _engine()
    order = Order(size=10.0, price=0.48, side=Side.BUY, token=Token.A)
    engine.place_orders([order])
    engine.cancel_orders([order])
    assert engine.open_orders == []


def test_partial_fill_leaves_the_remainder():
    _, _, engine = _engine(price=0.50, size_per_order=10.0)
    engine.place_orders([Order(size=100.0, price=0.50, side=Side.BUY, token=Token.A)])

    fills = engine.match(OrderFlow(sell_orders=1))
    assert fills and fills[0].size == pytest.approx(10.0)
    assert engine.open_orders  # the rest is still resting


def test_queue_priority_can_skip_a_fill():
    """Below full priority, being at the right price is not enough."""
    market = SimulatedMarket(MarketConfig(initial_price=0.50, flow_intensity=0.0), seed=0)
    portfolio = Portfolio(1000.0)
    engine = ExecutionEngine(
        market, portfolio, ExecutionConfig(queue_priority=0.0), seed=0
    )
    engine.place_orders([Order(size=10.0, price=0.50, side=Side.BUY, token=Token.A)])
    assert engine.match(OrderFlow(sell_orders=5)) == []


def test_fill_notional():
    fill = Fill(0, price=0.25, size=8.0, side=Side.BUY, token=Token.A, order_id="x")
    assert fill.notional == pytest.approx(2.0)
