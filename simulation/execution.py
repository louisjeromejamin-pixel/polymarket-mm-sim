"""Moteur d'exécution et comptabilité du portefeuille simulé.

Ce module remplace les deux dépendances externes du keeper amont — l'API CLOB
et la blockchain — par une simulation locale. Les stratégies restent
inchangées : elles reçoivent le même ``OrderBook`` et renvoient les mêmes
listes d'ordres à annuler et à placer.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from poly_market_maker.order import Order, Side
from poly_market_maker.token import Collateral, Token

from .market_sim import SimulatedMarket

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    """Exécution d'un ordre, totale ou partielle."""

    step: int
    price: float
    size: float
    side: Side
    token: Token
    order_id: str

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class ExecutionConfig:
    """Paramètres du modèle d'exécution.

    Attributes:
        maker_fee: Frais proportionnels au notionnel. Polymarket ne
            prélève pas de frais maker à ce jour ; le paramètre existe pour
            mesurer la sensibilité de la stratégie s'ils étaient introduits.
        fill_probability: Probabilité qu'un ordre au prix touché soit
            exécuté sur un pas. En dessous de 1, il modélise la file
            d'attente : être au bon prix ne garantit pas d'être servi.
        max_fill_ratio: Fraction maximale d'un ordre exécutée en un pas.
    """

    maker_fee: float = 0.0
    fill_probability: float = 0.6
    max_fill_ratio: float = 0.5


class Portfolio:
    """Suit le collatéral, les positions et la valeur liquidative.

    Convention Polymarket : détenir une part du token A et une part du
    token B du même marché garantit exactement 1 USDC à la résolution,
    puisque les deux issues sont complémentaires.
    """

    def __init__(self, initial_collateral: float = 1000.0):
        self.initial_collateral = initial_collateral
        self.collateral = initial_collateral
        self.positions: Dict[Token, float] = {Token.A: 0.0, Token.B: 0.0}
        self.fills: List[Fill] = []
        self.fees_paid = 0.0

    def apply_fill(self, fill: Fill, fee_rate: float) -> None:
        """Met à jour le portefeuille après une exécution."""
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
        """Renvoie les soldes au format attendu par les stratégies amont."""
        return {
            Collateral: self.collateral,
            Token.A: self.positions[Token.A],
            Token.B: self.positions[Token.B],
        }

    def mark_to_market(self, price_a: float) -> float:
        """Valeur liquidative aux prix courants."""
        price_b = 1.0 - price_a
        return (
            self.collateral
            + self.positions[Token.A] * price_a
            + self.positions[Token.B] * price_b
        )

    def value_at_resolution(self, outcome: int) -> float:
        """Valeur finale une fois le marché résolu.

        Le token gagnant vaut 1, le perdant 0.
        """
        winning = Token.A if outcome == 1 else Token.B
        return self.collateral + self.positions[winning]

    @property
    def inventory_skew(self) -> float:
        """Déséquilibre entre les deux jambes, en parts.

        Un market maker vise un inventaire équilibré : c'est cet écart qui
        l'expose au mouvement du prix.
        """
        return self.positions[Token.A] - self.positions[Token.B]


class ExecutionEngine:
    """Confronte les ordres du market maker au marché simulé.

    Un ordre d'achat s'exécute quand le marché descend à son prix ou en
    dessous ; un ordre de vente quand le marché y monte. L'exécution est
    partielle et probabiliste, pour ne pas surestimer la performance : sur
    un vrai carnet, un ordre au meilleur prix n'est pas systématiquement
    servi.
    """

    def __init__(
        self,
        market: SimulatedMarket,
        portfolio: Portfolio,
        config: Optional[ExecutionConfig] = None,
        seed: Optional[int] = None,
    ):
        import random

        self.market = market
        self.portfolio = portfolio
        self.config = config or ExecutionConfig()
        self.open_orders: List[Order] = []
        self._rng = random.Random(seed)
        self._id_counter = itertools.count(1)

    def place_orders(self, orders: List[Order]) -> None:
        """Enregistre de nouveaux ordres, en leur attribuant un identifiant."""
        for order in orders:
            if order.id is None:
                order.id = f"sim-{next(self._id_counter)}"
            self.open_orders.append(order)

    def cancel_orders(self, orders: List[Order]) -> None:
        """Retire des ordres du carnet."""
        to_cancel = {order.id for order in orders}
        self.open_orders = [o for o in self.open_orders if o.id not in to_cancel]

    def cancel_all(self) -> None:
        self.open_orders = []

    def match(self) -> List[Fill]:
        """Confronte les ordres ouverts au marché et applique les exécutions."""
        fills: List[Fill] = []
        still_open: List[Order] = []

        for order in self.open_orders:
            filled_size = self._fill_size(order)

            if filled_size <= 0:
                still_open.append(order)
                continue

            fill = Fill(
                step=self.market.step,
                price=order.price,
                size=filled_size,
                side=order.side,
                token=order.token,
                order_id=order.id,
            )

            if not self._can_afford(fill):
                still_open.append(order)
                continue

            self.portfolio.apply_fill(fill, self.config.maker_fee)
            fills.append(fill)

            remaining = round(order.size - filled_size, 4)
            if remaining > 0:
                order.size = remaining
                still_open.append(order)

        self.open_orders = still_open
        return fills

    def _fill_size(self, order: Order) -> float:
        """Détermine la quantité exécutée d'un ordre sur ce pas."""
        # Le prix du token B est le complément de celui du token A.
        market_price = (
            self.market.mid_price
            if order.token == Token.A
            else round(1.0 - self.market.mid_price, 2)
        )

        # Un achat passe quand le marche descend jusqu'au prix de l'ordre ;
        # une vente quand il y monte.
        touched = (
            market_price <= order.price
            if order.side == Side.BUY
            else market_price >= order.price
        )
        if not touched:
            return 0.0

        if self._rng.random() > self.config.fill_probability:
            return 0.0

        ratio = self._rng.uniform(0.1, self.config.max_fill_ratio)
        return round(order.size * ratio, 2)

    def _can_afford(self, fill: Fill) -> bool:
        """Vérifie que le portefeuille peut absorber l'exécution.

        On refuse un achat à découvert de collatéral et une vente de parts
        qu'on ne détient pas : le simulateur doit rester réaliste, sinon les
        résultats de backtest n'ont aucune valeur.
        """
        if fill.side == Side.BUY:
            cost = fill.notional * (1 + self.config.maker_fee)
            return self.portfolio.collateral >= cost
        return self.portfolio.positions[fill.token] >= fill.size
