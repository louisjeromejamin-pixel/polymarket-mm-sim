"""Tests for the signal model and its training pipeline."""

from __future__ import annotations

import random

import pytest

from simulation.signal import (
    FEATURE_NAMES,
    FeatureState,
    RidgeRegression,
    SignalConfig,
    SignalModel,
    _solve,
    extract_features,
)
from simulation.training import Dataset, collect_dataset, train_signal


# --- Linear solver --------------------------------------------------------


def test_solver_on_a_known_system():
    solution = _solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert solution == pytest.approx([1.0, 3.0], abs=1e-9)


def test_solver_on_the_identity():
    assert _solve([[1.0, 0.0], [0.0, 1.0]], [3.0, 7.0]) == pytest.approx([3.0, 7.0])


def test_solver_needs_pivoting():
    """A zero leading coefficient requires row exchange."""
    solution = _solve([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0])
    assert solution == pytest.approx([3.0, 2.0], abs=1e-9)


# --- Ridge regression -----------------------------------------------------


def test_ridge_recovers_a_linear_relation():
    """On y = 3*x0 - 2*x1 + noise the coefficients must come back."""
    rng = random.Random(0)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(2000)]
    y = [3 * a - 2 * b + 1 + rng.gauss(0, 0.1) for a, b in X]

    model = RidgeRegression(alpha=0.01).fit(X, y)
    unscaled = [model.coefficients[j] / model._scales[j] for j in range(2)]

    assert unscaled == pytest.approx([3.0, -2.0], abs=0.05)
    assert model.score(X, y) > 0.99


def test_ridge_intercept_is_the_target_mean():
    X = [[1.0], [2.0], [3.0]]
    y = [10.0, 20.0, 30.0]
    model = RidgeRegression(alpha=1.0).fit(X, y)
    assert model.intercept == pytest.approx(20.0)


def test_ridge_r2_is_negative_out_of_sample_on_noise():
    """Fitting noise must not generalise; R^2 below zero is the tell."""
    rng = random.Random(1)
    X_train = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(200)]
    y_train = [rng.gauss(0, 1) for _ in range(200)]
    X_test = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(200)]
    y_test = [rng.gauss(0, 1) for _ in range(200)]

    model = RidgeRegression(alpha=1.0).fit(X_train, y_train)
    assert model.score(X_test, y_test) < 0.05


def test_stronger_penalty_shrinks_coefficients():
    rng = random.Random(2)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
    y = [2 * a + b + rng.gauss(0, 1) for a, b in X]

    weak = RidgeRegression(alpha=0.01).fit(X, y)
    strong = RidgeRegression(alpha=1000.0).fit(X, y)

    assert sum(abs(c) for c in strong.coefficients) < sum(
        abs(c) for c in weak.coefficients
    )


def test_ridge_handles_a_constant_feature():
    """A zero-variance column must not divide by zero."""
    X = [[1.0, float(i)] for i in range(50)]
    y = [float(i) for i in range(50)]
    model = RidgeRegression(alpha=1.0).fit(X, y)
    assert all(c == c for c in model.coefficients)  # no NaN


def test_ridge_rejects_empty_or_mismatched_input():
    with pytest.raises(ValueError):
        RidgeRegression().fit([], [])
    with pytest.raises(ValueError):
        RidgeRegression().fit([[1.0]], [1.0, 2.0])


def test_unfitted_model_predicts_zero():
    assert RidgeRegression().predict_one([1.0, 2.0]) == 0.0


# --- Features -------------------------------------------------------------


def test_feature_vector_matches_declared_names():
    state = FeatureState()
    features = extract_features(state, 0.5, 0.0, 0, 0.0)
    assert len(features) == len(FEATURE_NAMES)


def test_features_are_finite_on_a_fresh_state():
    """No history must still produce usable numbers, not NaN."""
    features = extract_features(FeatureState(), 0.5, 0.3, 2, 100.0)
    assert all(f == f and abs(f) < 1e6 for f in features)


def test_distance_from_half():
    state = FeatureState()
    at_half = extract_features(state, 0.5, 0.0, 0, 0.0)[5]
    near_edge = extract_features(state, 0.95, 0.0, 0, 0.0)[5]
    assert at_half == pytest.approx(0.0)
    assert near_edge == pytest.approx(0.9)


def test_inventory_feature_is_bounded():
    state = FeatureState()
    huge = extract_features(state, 0.5, 0.0, 0, 1e9)[6]
    assert huge == 1.0


def test_realised_volatility_grows_with_dispersion():
    calm = FeatureState()
    volatile = FeatureState()
    rng = random.Random(3)

    for _ in range(30):
        calm.update(0.50, 0.0)
        volatile.update(0.50 + rng.uniform(-0.1, 0.1), 0.0)

    assert volatile.realised_volatility() > calm.realised_volatility()


def test_ema_tracks_persistent_imbalance():
    state = FeatureState()
    for _ in range(50):
        state.update(0.5, 1.0)
    assert state.imbalance_ema > 0.9


def test_history_is_bounded():
    """The rolling state must not grow without limit over a long run."""
    state = FeatureState(window=20)
    for i in range(10_000):
        state.update(0.5, 0.0)
    assert len(state.mid_history) <= 20 * 3


# --- Quote skew -----------------------------------------------------------


def _fitted_signal(coefficient: float) -> SignalModel:
    """Build a model whose prediction is proportional to the first feature."""
    model = RidgeRegression()
    model.coefficients = [coefficient] + [0.0] * (len(FEATURE_NAMES) - 1)
    model.intercept = 0.0
    model._means = [0.0] * len(FEATURE_NAMES)
    model._scales = [1.0] * len(FEATURE_NAMES)
    return SignalModel(model, SignalConfig(skew_strength=100.0, max_skew_ticks=3))


def test_unfitted_signal_gives_no_skew():
    signal = SignalModel()
    assert not signal.is_fitted
    assert signal.skew_ticks([0.5] * len(FEATURE_NAMES)) == 0


def test_skew_follows_the_prediction_sign():
    signal = _fitted_signal(1.0)
    features = [0.0] * len(FEATURE_NAMES)

    features[0] = 0.02
    assert signal.skew_ticks(features) > 0

    features[0] = -0.02
    assert signal.skew_ticks(features) < 0


def test_skew_is_capped():
    signal = _fitted_signal(1.0)
    features = [0.0] * len(FEATURE_NAMES)
    features[0] = 1000.0
    assert signal.skew_ticks(features) == 3


def test_small_predictions_are_ignored():
    """Noise around zero must not move the quotes."""
    signal = _fitted_signal(1.0)
    features = [0.0] * len(FEATURE_NAMES)
    features[0] = 1e-6
    assert signal.skew_ticks(features) == 0


# --- Dataset and training -------------------------------------------------


def test_dataset_split_is_chronological():
    dataset = Dataset(X=[[float(i)] for i in range(100)], y=[float(i) for i in range(100)])
    train, test = dataset.split(0.7)

    assert len(train) == 70 and len(test) == 30
    assert train.y[-1] < test.y[0], "the split must not shuffle"


def test_collect_dataset_produces_labelled_observations():
    dataset = collect_dataset(seeds=[0, 1], steps=120, horizon=3)
    assert len(dataset) > 50
    assert len(dataset.X[0]) == len(FEATURE_NAMES)
    assert len(dataset.X) == len(dataset.y)


def test_collected_targets_are_forward_returns():
    """Labels must be finite and of a plausible magnitude."""
    dataset = collect_dataset(seeds=[0], steps=120, horizon=3)
    assert all(abs(value) < 5.0 for value in dataset.y)


def test_collection_is_reproducible():
    a = collect_dataset(seeds=[7], steps=100, horizon=3)
    b = collect_dataset(seeds=[7], steps=100, horizon=3)
    assert a.y == b.y


def test_train_signal_reports_both_scores():
    dataset = collect_dataset(seeds=range(4), steps=150, horizon=3)
    model, report = train_signal(dataset, alpha=1.0)

    assert report["n_train"] > 0 and report["n_test"] > 0
    assert set(report["coefficients"]) == set(FEATURE_NAMES)
    assert model.coefficients


def test_train_signal_rejects_a_tiny_dataset():
    with pytest.raises(ValueError, match="too small"):
        train_signal(Dataset(X=[[1.0]] * 5, y=[1.0] * 5))
