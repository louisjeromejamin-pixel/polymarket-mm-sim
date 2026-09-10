"""Marché synthétique pour un contrat binaire de type Polymarket.

Un marché de prédiction binaire se résout à 0 ou 1. Son prix, borné dans
``(0, 1)``, s'interprète directement comme une probabilité. Une marche
aléatoire arithmétique classique conviendrait mal : elle sortirait de
l'intervalle et attribuerait la même volatilité absolue à un prix de 0,50 et
à un prix de 0,02, alors qu'un contrat quasi résolu bouge beaucoup moins.

On simule donc le prix en **log-odds** (le logit de la probabilité) :

    logit(p) = log(p / (1 - p))

où la marche aléatoire est libre sur tout l'axe réel. La transformation
inverse (la sigmoïde) ramène le prix dans ``(0, 1)`` par construction, et
compresse naturellement les mouvements près des bornes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


def logit(p: float) -> float:
    """Convertit une probabilité en log-odds."""
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Convertit des log-odds en probabilité, sans risque d'overflow."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class MarketConfig:
    """Paramètres du marché simulé.

    Attributes:
        initial_price: Probabilité initiale du contrat, dans ``(0, 1)``.
        volatility: Écart-type du pas de la marche aléatoire en log-odds.
        drift: Dérive par pas en log-odds. Positif = le marché tend vers
            "oui". Zéro correspond à une martingale, cas le plus honnête
            pour évaluer un market maker.
        spread_ticks: Demi-écart du carnet externe, en ticks d'un cent.
        liquidity_size: Taille disponible à chaque niveau du carnet externe.
        tick: Granularité des prix (Polymarket cote au cent).
        resolution_step: Pas auquel le marché se résout, ou ``None`` pour
            qu'il ne se résolve jamais.
    """

    initial_price: float = 0.50
    volatility: float = 0.15
    drift: float = 0.0
    spread_ticks: int = 2
    liquidity_size: float = 500.0
    tick: float = 0.01
    resolution_step: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.initial_price < 1.0:
            raise ValueError("initial_price doit être strictement entre 0 et 1")
        if self.volatility < 0:
            raise ValueError("volatility doit être positive")
        if self.spread_ticks < 1:
            raise ValueError("spread_ticks doit valoir au moins 1")


@dataclass
class MarketState:
    """Instantané du marché à un pas donné."""

    step: int
    mid_price: float
    best_bid: float
    best_ask: float
    resolved: bool = False
    outcome: Optional[int] = None


class SimulatedMarket:
    """Marché binaire dont le prix suit une marche aléatoire en log-odds.

    Le marché expose un prix milieu et un carnet à deux faces. Il ne connaît
    ni les ordres du market maker ni ses positions : c'est le simulateur
    d'exécution qui confronte les deux.
    """

    def __init__(self, config: MarketConfig, seed: Optional[int] = None):
        self.config = config
        self._rng = random.Random(seed)
        self._logit_price = logit(config.initial_price)
        self.step = 0
        self.resolved = False
        self.outcome: Optional[int] = None
        self.history: List[MarketState] = []
        self._record()

    @property
    def mid_price(self) -> float:
        """Prix milieu, arrondi au tick."""
        raw = sigmoid(self._logit_price)
        ticks = round(raw / self.config.tick)
        # On garde au moins un tick de marge avec 0 et 1 : un contrat
        # binaire non résolu ne cote jamais exactement à ses bornes.
        ticks = max(1, min(ticks, int(round(1 / self.config.tick)) - 1))
        return round(ticks * self.config.tick, 2)

    @property
    def best_bid(self) -> float:
        return round(self.mid_price - self.config.spread_ticks * self.config.tick, 2)

    @property
    def best_ask(self) -> float:
        return round(self.mid_price + self.config.spread_ticks * self.config.tick, 2)

    def advance(self) -> MarketState:
        """Fait avancer le marché d'un pas et renvoie le nouvel état."""
        if self.resolved:
            return self.history[-1]

        shock = self._rng.gauss(self.config.drift, self.config.volatility)
        self._logit_price += shock
        self.step += 1

        if (
            self.config.resolution_step is not None
            and self.step >= self.config.resolution_step
        ):
            self._resolve()

        return self._record()

    def _resolve(self) -> None:
        """Résout le marché en tirant l'issue selon le prix courant.

        Tirer l'issue avec probabilité égale au prix garde le marché
        cohérent : un contrat à 0,80 se résout « oui » huit fois sur dix.
        """
        self.outcome = 1 if self._rng.random() < self.mid_price else 0
        self.resolved = True

    def _record(self) -> MarketState:
        state = MarketState(
            step=self.step,
            mid_price=self.mid_price,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            resolved=self.resolved,
            outcome=self.outcome,
        )
        self.history.append(state)
        return state
