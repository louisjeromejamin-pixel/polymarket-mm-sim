"""Tests de la couche de simulation."""

import math

import pytest

from poly_market_maker.order import Order, Side
from poly_market_maker.token import Collateral, Token
from simulation.execution import ExecutionConfig, ExecutionEngine, Fill, Portfolio
from simulation.market_sim import MarketConfig, SimulatedMarket, logit, sigmoid


# --- Marché ---------------------------------------------------------------


def test_logit_sigmoid_sont_inverses():
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)


def test_sigmoid_ne_deborde_pas():
    assert sigmoid(-1000) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(1000) == pytest.approx(1.0, abs=1e-12)


def test_prix_reste_dans_les_bornes():
    """Même sous forte volatilité, un prix de probabilité reste dans (0, 1)."""
    market = SimulatedMarket(MarketConfig(volatility=2.0), seed=7)
    for _ in range(500):
        market.advance()
        assert 0.0 < market.mid_price < 1.0


def test_meme_graine_meme_trajectoire():
    a = SimulatedMarket(MarketConfig(), seed=123)
    b = SimulatedMarket(MarketConfig(), seed=123)
    for _ in range(50):
        a.advance()
        b.advance()
    assert [s.mid_price for s in a.history] == [s.mid_price for s in b.history]


def test_bid_inferieur_a_ask():
    market = SimulatedMarket(MarketConfig(), seed=1)
    for _ in range(30):
        market.advance()
        assert market.best_bid < market.mid_price < market.best_ask


def test_marche_se_resout_au_pas_demande():
    market = SimulatedMarket(MarketConfig(resolution_step=10), seed=3)
    for _ in range(15):
        market.advance()
    assert market.resolved
    assert market.outcome in (0, 1)


def test_config_rejette_prix_invalide():
    with pytest.raises(ValueError):
        MarketConfig(initial_price=0.0)
    with pytest.raises(ValueError):
        MarketConfig(initial_price=1.5)


# --- Portefeuille ---------------------------------------------------------


def test_achat_debite_le_collateral_et_credite_la_position():
    portfolio = Portfolio(100.0)
    fill = Fill(0, price=0.40, size=10.0, side=Side.BUY, token=Token.A, order_id="x")
    portfolio.apply_fill(fill, fee_rate=0.0)
    assert portfolio.collateral == pytest.approx(96.0)
    assert portfolio.positions[Token.A] == 10.0


def test_vente_credite_le_collateral():
    portfolio = Portfolio(100.0)
    portfolio.positions[Token.A] = 20.0
    fill = Fill(0, price=0.60, size=10.0, side=Side.SELL, token=Token.A, order_id="x")
    portfolio.apply_fill(fill, fee_rate=0.0)
    assert portfolio.collateral == pytest.approx(106.0)
    assert portfolio.positions[Token.A] == 10.0


def test_frais_preleves_dans_les_deux_sens():
    """Les frais réduisent le gain d'une vente comme ils alourdissent un achat."""
    buy = Portfolio(100.0)
    buy.apply_fill(
        Fill(0, 0.50, 10.0, Side.BUY, Token.A, "x"), fee_rate=0.01
    )
    assert buy.collateral == pytest.approx(100.0 - 5.0 - 0.05)

    sell = Portfolio(100.0)
    sell.positions[Token.A] = 10.0
    sell.apply_fill(
        Fill(0, 0.50, 10.0, Side.SELL, Token.A, "x"), fee_rate=0.01
    )
    assert sell.collateral == pytest.approx(100.0 + 5.0 - 0.05)


def test_valeur_a_la_resolution():
    portfolio = Portfolio(50.0)
    portfolio.positions[Token.A] = 30.0
    portfolio.positions[Token.B] = 20.0
    assert portfolio.value_at_resolution(outcome=1) == pytest.approx(80.0)
    assert portfolio.value_at_resolution(outcome=0) == pytest.approx(70.0)


def test_positions_equilibrees_valent_leur_nombre_de_paires():
    """Une part de A et une de B garantissent 1 USDC, quelle que soit l'issue."""
    portfolio = Portfolio(0.0)
    portfolio.positions[Token.A] = 40.0
    portfolio.positions[Token.B] = 40.0
    assert portfolio.value_at_resolution(1) == pytest.approx(40.0)
    assert portfolio.value_at_resolution(0) == pytest.approx(40.0)


def test_balances_au_format_amont():
    portfolio = Portfolio(10.0)
    balances = portfolio.balances()
    assert set(balances) == {Collateral, Token.A, Token.B}


def test_inventory_skew():
    portfolio = Portfolio(0.0)
    portfolio.positions[Token.A] = 30.0
    portfolio.positions[Token.B] = 12.0
    assert portfolio.inventory_skew == pytest.approx(18.0)


# --- Exécution ------------------------------------------------------------


def _engine(price=0.50, **kwargs):
    market = SimulatedMarket(MarketConfig(initial_price=price), seed=0)
    portfolio = Portfolio(1000.0)
    portfolio.positions[Token.A] = 100.0
    portfolio.positions[Token.B] = 100.0
    config = ExecutionConfig(fill_probability=1.0, max_fill_ratio=1.0, **kwargs)
    return market, portfolio, ExecutionEngine(market, portfolio, config, seed=0)


def test_achat_sous_le_marche_reste_ouvert():
    """Un achat à 0,40 quand le marché cote 0,50 ne doit pas s'exécuter."""
    _, _, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.40, side=Side.BUY, token=Token.A)])
    assert engine.match() == []
    assert len(engine.open_orders) == 1


def test_achat_atteint_par_le_marche_sexecute():
    """Le marché descend jusqu'à l'ordre : il passe."""
    _, _, engine = _engine(price=0.40)
    engine.place_orders([Order(size=10.0, price=0.45, side=Side.BUY, token=Token.A)])
    fills = engine.match()
    assert len(fills) == 1
    assert fills[0].side == Side.BUY


def test_vente_au_dessus_du_marche_reste_ouverte():
    _, _, engine = _engine(price=0.50)
    engine.place_orders([Order(size=10.0, price=0.60, side=Side.SELL, token=Token.A)])
    assert engine.match() == []


def test_vente_atteinte_sexecute():
    _, _, engine = _engine(price=0.65)
    engine.place_orders([Order(size=10.0, price=0.60, side=Side.SELL, token=Token.A)])
    assert len(engine.match()) == 1


def test_prix_du_token_b_est_complementaire():
    """À 0,70 sur A, le token B vaut 0,30 : un achat de B à 0,35 passe."""
    _, _, engine = _engine(price=0.70)
    engine.place_orders([Order(size=10.0, price=0.35, side=Side.BUY, token=Token.B)])
    assert len(engine.match()) == 1


def test_achat_refuse_sans_collateral():
    market = SimulatedMarket(MarketConfig(initial_price=0.40), seed=0)
    portfolio = Portfolio(1.0)  # trop peu pour 10 parts à 0,45
    engine = ExecutionEngine(
        market, portfolio, ExecutionConfig(fill_probability=1.0, max_fill_ratio=1.0), seed=0
    )
    engine.place_orders([Order(size=10.0, price=0.45, side=Side.BUY, token=Token.A)])
    assert engine.match() == []
    assert portfolio.collateral == 1.0


def test_vente_a_decouvert_refusee():
    market = SimulatedMarket(MarketConfig(initial_price=0.65), seed=0)
    portfolio = Portfolio(100.0)  # aucune part détenue
    engine = ExecutionEngine(
        market, portfolio, ExecutionConfig(fill_probability=1.0, max_fill_ratio=1.0), seed=0
    )
    engine.place_orders([Order(size=10.0, price=0.60, side=Side.SELL, token=Token.A)])
    assert engine.match() == []
    assert portfolio.positions[Token.A] == 0.0


def test_annulation_retire_du_carnet():
    _, _, engine = _engine()
    order = Order(size=10.0, price=0.40, side=Side.BUY, token=Token.A)
    engine.place_orders([order])
    engine.cancel_orders([order])
    assert engine.open_orders == []


def test_execution_partielle_laisse_le_reste_ouvert():
    market = SimulatedMarket(MarketConfig(initial_price=0.40), seed=0)
    portfolio = Portfolio(1000.0)
    engine = ExecutionEngine(
        market,
        portfolio,
        ExecutionConfig(fill_probability=1.0, max_fill_ratio=0.5),
        seed=0,
    )
    engine.place_orders([Order(size=100.0, price=0.45, side=Side.BUY, token=Token.A)])
    fills = engine.match()
    assert fills and fills[0].size < 100.0
    assert engine.open_orders  # le solde reste en carnet


def test_fill_notional():
    fill = Fill(0, price=0.25, size=8.0, side=Side.BUY, token=Token.A, order_id="x")
    assert fill.notional == pytest.approx(2.0)
