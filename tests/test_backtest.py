"""Tests for the backtest loop and the upstream strategies."""

from __future__ import annotations

import pytest

from poly_market_maker.token import Token
from simulation.backtest import load_strategy, run_backtest, run_sweep
from simulation.execution import ExecutionConfig
from simulation.market_sim import MarketConfig
from simulation.signal import FEATURE_NAMES, RidgeRegression, SignalConfig, SignalModel

AMM = "config/amm.json"
BANDS = "config/bands.json"


@pytest.mark.parametrize("name,path", [("amm", AMM), ("bands", BANDS)])
def test_strategy_loads(name, path):
    assert load_strategy(name, path) is not None


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        load_strategy("nonexistent", AMM)


@pytest.mark.parametrize("strategy,path", [("amm", AMM), ("bands", BANDS)])
def test_backtest_produces_fills(strategy, path):
    result = run_backtest(strategy, path, steps=200, seed=42)
    assert result.num_fills > 0, "no fills: the engine matched nothing"
    assert len(result.equity_curve) == 200


def test_backtest_is_reproducible():
    a = run_backtest("bands", BANDS, steps=100, seed=7)
    b = run_backtest("bands", BANDS, steps=100, seed=7)
    assert a.equity_curve == b.equity_curve
    assert a.num_fills == b.num_fills


def test_different_seeds_give_different_results():
    a = run_backtest("bands", BANDS, steps=100, seed=1)
    b = run_backtest("bands", BANDS, steps=100, seed=2)
    assert a.equity_curve != b.equity_curve


def test_price_stays_bounded_throughout():
    result = run_backtest("amm", AMM, steps=300, seed=5)
    assert all(0.0 < p < 1.0 for p in result.price_series)


def test_no_strategy_error_at_the_price_bounds(caplog):
    """The `phi()` fix must hold when the price reaches p_min/p_max.

    Without it the AMM strategy raises ZeroDivisionError as soon as a market
    drifts towards resolution.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        run_backtest(
            "amm",
            AMM,
            steps=400,
            market_config=MarketConfig(volatility=0.5),
            seed=42,
        )
    errors = [r for r in caplog.records if "Strategy error" in r.getMessage()]
    assert not errors, f"the strategy failed {len(errors)} times"


def test_backtest_on_a_resolving_market():
    result = run_backtest(
        "bands",
        BANDS,
        steps=200,
        market_config=MarketConfig(resolution_step=50),
        seed=11,
    )
    assert result.resolved
    assert result.outcome in (0, 1)


def test_fees_reduce_the_result():
    """At an identical market, higher fees cannot improve PnL."""
    without = run_backtest(
        "bands", BANDS, steps=200, seed=3,
        execution_config=ExecutionConfig(maker_fee=0.0),
    )
    with_fees = run_backtest(
        "bands", BANDS, steps=200, seed=3,
        execution_config=ExecutionConfig(maker_fee=0.02),
    )
    assert with_fees.fees_paid > 0
    assert with_fees.pnl <= without.pnl


def test_metrics_are_coherent():
    result = run_backtest("bands", BANDS, steps=200, seed=9)
    assert result.max_drawdown_pct >= 0
    assert result.volatility_pct >= 0
    assert result.volume >= 0
    assert result.pnl == pytest.approx(result.final_value - result.initial_value)
    assert 0.0 <= result.informed_fill_ratio <= 1.0


def test_maker_captures_a_positive_spread():
    """Selling above the average purchase price is the source of revenue."""
    result = run_backtest("bands", BANDS, steps=400, seed=0)
    assert result.spread_captured > 0


def test_profitability_falls_as_informed_flow_rises():
    """More adverse selection, less profit — and none at all when every
    counterparty is informed."""
    import statistics

    means = []
    for ratio in (0.0, 0.5, 1.0):
        results = run_sweep(
            "bands",
            BANDS,
            range(8),
            steps=300,
            market_config=MarketConfig(informed_ratio=ratio),
        )
        means.append(statistics.mean(r.pnl for r in results))

    assert means[0] > means[-1]
    assert means[-1] == pytest.approx(0.0, abs=1e-9)


def test_sweep_returns_one_result_per_seed():
    results = run_sweep("bands", BANDS, seeds=[1, 2, 3], steps=100)
    assert len(results) == 3
    assert len({r.equity_curve[-1] for r in results}) > 1


def test_summary_contains_the_expected_keys():
    summary = run_backtest("amm", AMM, steps=100, seed=4).summary()
    for key in ("pnl", "sharpe", "fills", "informed_fill_pct", "spread_captured"):
        assert key in summary


# --- Signal integration ---------------------------------------------------


def _constant_signal(skew_prediction: float) -> SignalModel:
    """A model that always predicts the same value."""
    model = RidgeRegression()
    model.coefficients = [0.0] * len(FEATURE_NAMES)
    model.intercept = skew_prediction
    model._means = [0.0] * len(FEATURE_NAMES)
    model._scales = [1.0] * len(FEATURE_NAMES)
    return SignalModel(model, SignalConfig(skew_strength=1000.0, max_skew_ticks=2))


def test_backtest_runs_without_a_signal():
    result = run_backtest("bands", BANDS, steps=150, seed=1, signal=None)
    assert all(s == 0 for s in result.skew_series)


def test_signal_shifts_the_quotes():
    """A constant positive prediction must move every quote up."""
    result = run_backtest(
        "bands", BANDS, steps=150, seed=1, signal=_constant_signal(0.01)
    )
    assert all(s > 0 for s in result.skew_series)


def test_negative_prediction_shifts_down():
    result = run_backtest(
        "bands", BANDS, steps=150, seed=1, signal=_constant_signal(-0.01)
    )
    assert all(s < 0 for s in result.skew_series)


def test_skew_is_capped_in_the_backtest():
    result = run_backtest(
        "bands", BANDS, steps=150, seed=1, signal=_constant_signal(10.0)
    )
    assert max(result.skew_series) == 2


def test_skewed_quotes_stay_within_bounds():
    """Shifted prices must remain tradeable, i.e. strictly inside (0, 1)."""
    result = run_backtest(
        "bands",
        BANDS,
        steps=300,
        seed=1,
        market_config=MarketConfig(initial_price=0.95, volatility=0.4),
        signal=_constant_signal(10.0),
    )
    assert all(0.0 < f.price < 1.0 for f in result.fills)
