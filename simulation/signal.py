"""Short-horizon price signal and quote skewing.

A market maker quoting symmetrically around the mid holds inventory it did
not choose. If the next move is partly predictable, shifting both quotes in
that direction reduces the adverse fills and biases the inventory the right
way.

The predictor is a ridge regression, solved in closed form with the standard
library alone. The model-selection work behind that choice — linear
regression, ridge, random forest, gradient boosting — lives in
``research/signal_models.py``; ridge won on out-of-sample R², and nothing
heavier justified its dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

#: Names of the features produced by :func:`extract_features`, in order.
FEATURE_NAMES = [
    "flow_imbalance",
    "flow_imbalance_ema",
    "order_count",
    "realised_volatility",
    "mid_return",
    "distance_from_half",
    "inventory_skew_norm",
]


@dataclass
class FeatureState:
    """Rolling state needed to build features without look-ahead.

    Every quantity is computed from information available *before* the step
    being predicted. The state is updated after the prediction, never before.
    """

    mid_history: List[float] = field(default_factory=list)
    imbalance_ema: float = 0.0
    ema_alpha: float = 0.2
    window: int = 20

    def update(self, mid_price: float, imbalance: float) -> None:
        """Record one step of observed market data."""
        self.mid_history.append(mid_price)
        if len(self.mid_history) > self.window * 3:
            del self.mid_history[0]
        self.imbalance_ema = (
            self.ema_alpha * imbalance + (1 - self.ema_alpha) * self.imbalance_ema
        )

    def realised_volatility(self) -> float:
        """Standard deviation of recent mid returns."""
        history = self.mid_history[-self.window :]
        if len(history) < 3:
            return 0.0

        rets = [
            (b - a) / a for a, b in zip(history, history[1:]) if a > 0
        ]
        if len(rets) < 2:
            return 0.0

        mean = sum(rets) / len(rets)
        variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(variance)

    def mid_return(self) -> float:
        """Return of the mid over the last step."""
        if len(self.mid_history) < 2 or self.mid_history[-2] <= 0:
            return 0.0
        return (self.mid_history[-1] - self.mid_history[-2]) / self.mid_history[-2]


def extract_features(
    state: FeatureState,
    mid_price: float,
    imbalance: float,
    order_count: int,
    inventory_skew: float,
    inventory_scale: float = 500.0,
) -> List[float]:
    """Build the feature vector for the current step.

    All inputs are observable at decision time. ``distance_from_half``
    captures the compression of moves near the bounds of a binary contract;
    ``inventory_skew_norm`` lets the model account for the maker's own
    position, since a maker long the market has a different optimal skew from
    one that is flat.
    """
    return [
        imbalance,
        state.imbalance_ema,
        min(order_count / 10.0, 1.0),
        state.realised_volatility(),
        state.mid_return(),
        abs(mid_price - 0.5) * 2.0,
        max(-1.0, min(1.0, inventory_skew / inventory_scale)),
    ]


class RidgeRegression:
    """Ridge regression solved in closed form.

    Minimises

        ||y - Xw||^2 + lambda ||w||^2

    whose solution is w = (X'X + lambda I)^-1 X'y. The penalty also
    guarantees the matrix is invertible, which matters here: the two
    imbalance features are strongly correlated, and ordinary least squares
    would be numerically unstable on them.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coefficients: List[float] = []
        self.intercept: float = 0.0
        self._means: List[float] = []
        self._scales: List[float] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "RidgeRegression":
        """Fit the model, standardising features first.

        Standardisation matters for ridge: the penalty is scale-dependent, so
        without it a feature measured in large units would be penalised less
        than one in small units.
        """
        if not X or not y:
            raise ValueError("Cannot fit on empty data")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        n_features = len(X[0])
        n_samples = len(X)

        self._means = [sum(row[j] for row in X) / n_samples for j in range(n_features)]
        self._scales = []
        for j in range(n_features):
            variance = sum((row[j] - self._means[j]) ** 2 for row in X) / n_samples
            self._scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

        Z = [
            [(row[j] - self._means[j]) / self._scales[j] for j in range(n_features)]
            for row in X
        ]
        y_mean = sum(y) / n_samples
        y_centred = [value - y_mean for value in y]

        # Normal equations: (Z'Z + alpha I) w = Z'y
        gram = [
            [
                sum(Z[i][a] * Z[i][b] for i in range(n_samples))
                + (self.alpha if a == b else 0.0)
                for b in range(n_features)
            ]
            for a in range(n_features)
        ]
        rhs = [sum(Z[i][a] * y_centred[i] for i in range(n_samples)) for a in range(n_features)]

        self.coefficients = _solve(gram, rhs)
        self.intercept = y_mean
        return self

    def predict_one(self, features: Sequence[float]) -> float:
        """Predict for a single feature vector."""
        if not self.coefficients:
            return 0.0
        return self.intercept + sum(
            coefficient * (value - mean) / scale
            for coefficient, value, mean, scale in zip(
                self.coefficients, features, self._means, self._scales
            )
        )

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_one(row) for row in X]

    def score(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> float:
        """Coefficient of determination R^2.

        Negative values are meaningful and common on financial data: the
        model does worse than predicting the mean.
        """
        if not y:
            return 0.0
        predictions = self.predict(X)
        mean = sum(y) / len(y)

        ss_res = sum((a - b) ** 2 for a, b in zip(y, predictions))
        ss_tot = sum((value - mean) ** 2 for value in y)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _solve(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    """Solve a linear system by Gaussian elimination with partial pivoting."""
    n = len(matrix)
    augmented = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]

    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            continue  # singular column; the ridge penalty normally prevents this
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]

    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        if abs(augmented[row][row]) < 1e-12:
            continue
        total = augmented[row][n] - sum(
            augmented[row][k] * solution[k] for k in range(row + 1, n)
        )
        solution[row] = total / augmented[row][row]
    return solution


@dataclass
class SignalConfig:
    """Configuration of the predictive quote skew.

    Attributes:
        skew_strength: Conversion from predicted return to a price shift, in
            ticks per unit of predicted return.
        max_skew_ticks: Cap on the shift. Bounding it matters: an unbounded
            skew turns a market-making strategy into a directional bet on the
            model, which is a different business with a different risk.
        min_confidence: Predictions below this magnitude are ignored, so
            noise around zero does not move the quotes.
    """

    skew_strength: float = 200.0
    max_skew_ticks: int = 3
    min_confidence: float = 0.0005


class SignalModel:
    """Predicts the next mid move and turns it into a quote skew."""

    def __init__(
        self,
        model: Optional[RidgeRegression] = None,
        config: Optional[SignalConfig] = None,
    ):
        self.model = model or RidgeRegression()
        self.config = config or SignalConfig()

    @property
    def is_fitted(self) -> bool:
        return bool(self.model.coefficients)

    def predict(self, features: Sequence[float]) -> float:
        """Predicted return of the mid over the next step."""
        return self.model.predict_one(features) if self.is_fitted else 0.0

    def skew_ticks(self, features: Sequence[float]) -> int:
        """Signed quote shift, in ticks.

        Positive means the price is expected to rise, so both quotes move up:
        the maker becomes less willing to sell cheaply and more willing to
        buy.
        """
        prediction = self.predict(features)

        if abs(prediction) < self.config.min_confidence:
            return 0

        raw = prediction * self.config.skew_strength
        capped = max(-self.config.max_skew_ticks, min(self.config.max_skew_ticks, raw))
        return int(round(capped))
