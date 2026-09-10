"""Tests de la boucle de backtest et des stratégies amont."""

import pytest

from poly_market_maker.token import Token
from simulation.backtest import load_strategy, run_backtest, run_sweep
from simulation.execution import ExecutionConfig
from simulation.market_sim import MarketConfig

AMM = "config/amm.json"
BANDS = "config/bands.json"


@pytest.mark.parametrize("name,path", [("amm", AMM), ("bands", BANDS)])
def test_strategie_se_charge(name, path):
    assert load_strategy(name, path) is not None


def test_strategie_inconnue_rejetee():
    with pytest.raises(ValueError):
        load_strategy("inexistante", AMM)


@pytest.mark.parametrize("strategy,path", [("amm", AMM), ("bands", BANDS)])
def test_backtest_produit_des_executions(strategy, path):
    result = run_backtest(strategy, path, steps=200, seed=42)
    assert result.num_fills > 0, "aucune exécution : le moteur ne confronte rien"
    assert len(result.equity_curve) == 200


def test_backtest_reproductible():
    a = run_backtest("bands", BANDS, steps=100, seed=7)
    b = run_backtest("bands", BANDS, steps=100, seed=7)
    assert a.equity_curve == b.equity_curve
    assert a.num_fills == b.num_fills


def test_graines_differentes_donnent_resultats_differents():
    a = run_backtest("bands", BANDS, steps=100, seed=1)
    b = run_backtest("bands", BANDS, steps=100, seed=2)
    assert a.equity_curve != b.equity_curve


def test_prix_borne_pendant_tout_le_backtest():
    result = run_backtest("amm", AMM, steps=300, seed=5)
    assert all(0.0 < p < 1.0 for p in result.price_series)


def test_aucune_erreur_de_strategie_aux_bornes(caplog):
    """Le correctif de `phi()` doit tenir même quand le prix atteint p_min/p_max.

    Sans lui, la stratégie AMM lève ZeroDivisionError dès que le marché
    dérive vers une résolution.
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
    erreurs = [r for r in caplog.records if "Stratégie en erreur" in r.getMessage()]
    assert not erreurs, f"la stratégie a échoué {len(erreurs)} fois"


def test_backtest_sur_marche_resolu():
    result = run_backtest(
        "bands",
        BANDS,
        steps=200,
        market_config=MarketConfig(resolution_step=50),
        seed=11,
    )
    assert result.resolved
    assert result.outcome in (0, 1)


def test_frais_degradent_le_resultat():
    """À marché identique, des frais plus élevés ne peuvent pas améliorer le PnL."""
    sans = run_backtest(
        "bands", BANDS, steps=200, seed=3,
        execution_config=ExecutionConfig(maker_fee=0.0),
    )
    avec = run_backtest(
        "bands", BANDS, steps=200, seed=3,
        execution_config=ExecutionConfig(maker_fee=0.02),
    )
    assert avec.fees_paid > 0
    assert avec.pnl <= sans.pnl


def test_metriques_coherentes():
    result = run_backtest("bands", BANDS, steps=200, seed=9)
    assert result.max_drawdown_pct >= 0
    assert result.volatility_pct >= 0
    assert result.volume >= 0
    assert result.pnl == pytest.approx(result.final_value - result.initial_value)


def test_sweep_renvoie_un_resultat_par_graine():
    results = run_sweep("bands", BANDS, seeds=[1, 2, 3], steps=100)
    assert len(results) == 3
    assert len({r.equity_curve[-1] for r in results}) > 1


def test_summary_contient_les_cles_attendues():
    summary = run_backtest("amm", AMM, steps=100, seed=4).summary()
    for key in ("pnl", "executions", "drawdown_max_pct", "sharpe"):
        assert key in summary
