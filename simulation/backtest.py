"""Boucle de backtest : fait tourner une stratégie amont sur le marché simulé."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from poly_market_maker.order import Order
from poly_market_maker.orderbook import OrderBook
from poly_market_maker.strategies.amm_strategy import AMMStrategy
from poly_market_maker.strategies.bands_strategy import BandsStrategy
from poly_market_maker.token import Token

from .execution import ExecutionConfig, ExecutionEngine, Fill, Portfolio
from .market_sim import MarketConfig, SimulatedMarket

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Résultat d'un backtest, avec les séries nécessaires aux graphiques."""

    strategy: str
    steps: int
    initial_value: float
    final_value: float
    equity_curve: List[float] = field(default_factory=list)
    price_series: List[float] = field(default_factory=list)
    inventory_series: List[float] = field(default_factory=list)
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
    def volume(self) -> float:
        """Notionnel total échangé."""
        return sum(f.notional for f in self.fills)

    @property
    def max_drawdown_pct(self) -> float:
        """Perte maximale depuis un sommet de la courbe de capital."""
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
        """Écart-type des rendements par pas, en pourcentage."""
        returns = self._step_returns()
        if len(returns) < 2:
            return 0.0
        return 100.0 * statistics.stdev(returns)

    @property
    def sharpe(self) -> float:
        """Ratio rendement/risque par pas, sans taux sans risque.

        Non annualisé : un pas de simulation ne correspond à aucune durée
        calendaire précise. L'indicateur sert à comparer des stratégies
        entre elles sur un même jeu de paramètres, pas à être publié.
        """
        returns = self._step_returns()
        if len(returns) < 2:
            return 0.0
        sigma = statistics.stdev(returns)
        if sigma == 0:
            return 0.0
        return statistics.mean(returns) / sigma

    @property
    def max_inventory_skew(self) -> float:
        """Plus grand déséquilibre d'inventaire atteint."""
        return max((abs(x) for x in self.inventory_series), default=0.0)

    def _step_returns(self) -> List[float]:
        return [
            (b - a) / a
            for a, b in zip(self.equity_curve, self.equity_curve[1:])
            if a != 0
        ]

    def summary(self) -> Dict[str, float]:
        return {
            "strategie": self.strategy,
            "pas": self.steps,
            "valeur_initiale": round(self.initial_value, 2),
            "valeur_finale": round(self.final_value, 2),
            "pnl": round(self.pnl, 2),
            "rendement_pct": round(self.return_pct, 3),
            "executions": self.num_fills,
            "volume": round(self.volume, 2),
            "frais": round(self.fees_paid, 4),
            "drawdown_max_pct": round(self.max_drawdown_pct, 3),
            "volatilite_pct": round(self.volatility_pct, 4),
            "sharpe": round(self.sharpe, 4),
            "inventaire_max": round(self.max_inventory_skew, 2),
        }


def load_strategy(name: str, config_path: str):
    """Instancie une stratégie amont depuis son fichier de configuration."""
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    key = name.lower()
    if key == "amm":
        return AMMStrategy(config)
    if key == "bands":
        return BandsStrategy(config)
    raise ValueError(f"Stratégie inconnue : {name!r} (attendu : amm ou bands)")


def run_backtest(
    strategy_name: str,
    config_path: str,
    steps: int = 500,
    market_config: Optional[MarketConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
    initial_collateral: float = 1000.0,
    initial_shares: float = 500.0,
    seed: Optional[int] = None,
) -> BacktestResult:
    """Fait tourner une stratégie sur un marché simulé.

    Args:
        strategy_name: ``"amm"`` ou ``"bands"``.
        config_path: Fichier JSON de configuration de la stratégie.
        steps: Nombre de pas de simulation.
        market_config: Paramètres du marché ; valeurs par défaut sinon.
        execution_config: Paramètres d'exécution ; valeurs par défaut sinon.
        initial_collateral: Collatéral de départ en USDC.
        initial_shares: Parts détenues au départ sur chaque jambe. Un market
            maker doit posséder des parts pour pouvoir en vendre.
        seed: Graine aléatoire, pour des résultats reproductibles.

    Returns:
        Le résultat du backtest, séries temporelles comprises.
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

    initial_value = portfolio.mark_to_market(market.mid_price)
    result = BacktestResult(
        strategy=strategy_name,
        steps=steps,
        initial_value=initial_value,
        final_value=initial_value,
    )

    for _ in range(steps):
        price_a = market.mid_price
        token_prices = {Token.A: price_a, Token.B: round(1.0 - price_a, 2)}

        orderbook = OrderBook(
            orders=list(engine.open_orders),
            balances=portfolio.balances(),
            orders_being_placed=False,
            orders_being_cancelled=False,
        )

        try:
            to_cancel, to_place = strategy.get_orders(orderbook, token_prices)
        except Exception as exc:  # une stratégie ne doit pas tuer le backtest
            logger.warning("Stratégie en erreur au pas %d : %s", market.step, exc)
            to_cancel, to_place = [], []

        engine.cancel_orders(to_cancel)
        engine.place_orders(to_place)

        # Le marche bouge APRES le placement : un ordre passe parce que le
        # prix vient le chercher. Confronter avant le mouvement ne remplirait
        # jamais rien, la strategie recentrant ses ordres a chaque pas.
        market.advance()
        result.fills.extend(engine.match())

        result.equity_curve.append(portfolio.mark_to_market(market.mid_price))
        result.price_series.append(market.mid_price)
        result.inventory_series.append(portfolio.inventory_skew)

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
    seeds: List[int],
    steps: int = 500,
    market_config: Optional[MarketConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
) -> List[BacktestResult]:
    """Rejoue le même backtest sur plusieurs graines.

    Un backtest unique sur un marché aléatoire ne prouve rien : le résultat
    dépend autant du tirage que de la stratégie. Comparer des distributions
    sur plusieurs graines est le minimum pour conclure.
    """
    return [
        run_backtest(
            strategy_name,
            config_path,
            steps=steps,
            market_config=market_config,
            execution_config=execution_config,
            seed=seed,
        )
        for seed in seeds
    ]
