"""Model selection for the short-horizon price signal.

Compares candidate predictors on the same dataset with the same
chronological split, and reports the out-of-sample R² that decided which one
went into production.

This file is research, not production: it may import scikit-learn and
LightGBM if they are installed. The shipped simulator depends on neither —
`simulation/signal.py` implements ridge regression against the standard
library alone.

    pip install scikit-learn lightgbm    # optional
    python research/signal_models.py
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, ".")

from simulation.signal import FEATURE_NAMES, RidgeRegression
from simulation.training import Dataset, collect_dataset


def r_squared(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination.

    Negative values are meaningful here: the model does worse than
    predicting the mean, which is the common outcome on financial data.
    """
    if not y_true:
        return 0.0
    mean = sum(y_true) / len(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((value - mean) ** 2 for value in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Share of predictions with the correct sign.

    More relevant than R² for a quote skew: only the direction of the shift
    matters, not the magnitude of the predicted move.
    """
    considered = [(a, b) for a, b in zip(y_true, y_pred) if a != 0 and b != 0]
    if not considered:
        return 0.0
    correct = sum(1 for a, b in considered if (a > 0) == (b > 0))
    return correct / len(considered)


# --- Candidate models -----------------------------------------------------


def fit_mean_baseline(train: Dataset, test: Dataset) -> Tuple[List[float], str]:
    """Predict the training mean for everything.

    The reference any model must beat. Its R² on the test set is not zero
    unless the two sets share a mean, which is exactly the point.
    """
    mean = sum(train.y) / len(train.y)
    return [mean] * len(test.y), "mean"


def fit_ridge(train: Dataset, test: Dataset, alpha: float = 1.0) -> Tuple[List[float], str]:
    """Ridge regression — the implementation used in production."""
    model = RidgeRegression(alpha=alpha).fit(train.X, train.y)
    return model.predict(test.X), f"ridge(alpha={alpha})"


def fit_ols(train: Dataset, test: Dataset) -> Tuple[List[float], str]:
    """Ordinary least squares, i.e. ridge with a negligible penalty.

    Included to show what the penalty buys: the two imbalance features are
    strongly correlated, so OLS is the unstable case.
    """
    model = RidgeRegression(alpha=1e-8).fit(train.X, train.y)
    return model.predict(test.X), "ols"


def fit_sklearn(
    train: Dataset, test: Dataset, estimator, label: str
) -> Optional[Tuple[List[float], str]]:
    """Fit any scikit-learn estimator, if the library is available."""
    try:
        estimator.fit(train.X, train.y)
    except Exception as exc:  # noqa: BLE001 - research script
        print(f"  {label}: failed ({exc})")
        return None
    return list(estimator.predict(test.X)), label


def _lightgbm_available() -> bool:
    """Whether LightGBM can be imported."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def fit_lightgbm(
    train: Dataset, test: Dataset, max_depth: int, n_estimators: int
) -> Optional[Tuple[List[float], str]]:
    """Fit LightGBM in a subprocess.

    LightGBM ships its own OpenMP runtime. On Windows, loading it into a
    process that has already initialised scikit-learn's causes a hard crash,
    so the fit runs in a clean interpreter and the predictions come back as
    JSON.
    """
    import json
    import subprocess
    import tempfile

    label = f"lightgbm(d={max_depth})"

    script = """
import json, sys
from lightgbm import LGBMRegressor
payload = json.load(open(sys.argv[1]))
model = LGBMRegressor(
    n_estimators=payload["n_estimators"],
    max_depth=payload["max_depth"],
    learning_rate=0.05,
    random_state=0,
    verbose=-1,
)
model.fit(payload["train_X"], payload["train_y"])
json.dump([float(v) for v in model.predict(payload["test_X"])], open(sys.argv[2], "w"))
"""

    with tempfile.TemporaryDirectory() as directory:
        script_path = f"{directory}/fit.py"
        input_path = f"{directory}/in.json"
        output_path = f"{directory}/out.json"

        with open(script_path, "w") as handle:
            handle.write(script)
        with open(input_path, "w") as handle:
            json.dump(
                {
                    "train_X": train.X,
                    "train_y": train.y,
                    "test_X": test.X,
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                },
                handle,
            )

        completed = subprocess.run(
            [sys.executable, script_path, input_path, output_path],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = (completed.stderr or "unknown error").strip().splitlines()[-1]
            print(f"  {label}: failed ({message})")
            return None

        with open(output_path) as handle:
            return json.load(handle), label


def build_candidates() -> List[Callable[[Dataset, Dataset], Optional[Tuple]]]:
    """Assemble the candidate list, skipping unavailable libraries."""
    candidates: List[Callable] = [
        fit_mean_baseline,
        fit_ols,
        lambda tr, te: fit_ridge(tr, te, alpha=0.1),
        lambda tr, te: fit_ridge(tr, te, alpha=1.0),
        lambda tr, te: fit_ridge(tr, te, alpha=10.0),
    ]

    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Lasso

        candidates += [
            lambda tr, te: fit_sklearn(tr, te, Lasso(alpha=1e-5, max_iter=5000), "lasso"),
            lambda tr, te: fit_sklearn(
                tr,
                te,
                RandomForestRegressor(
                    n_estimators=200, max_depth=6, random_state=0, n_jobs=1
                ),
                "random_forest(d=6)",
            ),
            lambda tr, te: fit_sklearn(
                tr,
                te,
                GradientBoostingRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
                ),
                "gradient_boosting",
            ),
        ]
    except ImportError:
        print("scikit-learn not installed: skipping lasso, forest and boosting")

    if _lightgbm_available():
        candidates += [
            lambda tr, te: fit_lightgbm(tr, te, max_depth=4, n_estimators=300),
            lambda tr, te: fit_lightgbm(tr, te, max_depth=8, n_estimators=600),
        ]
    else:
        print("lightgbm not installed: skipping gradient-boosted trees")

    return candidates


def evaluate(dataset: Dataset, train_fraction: float = 0.7) -> None:
    """Fit every candidate and report in- and out-of-sample scores."""
    train, test = dataset.split(train_fraction)
    print(f"\n{len(train)} training samples, {len(test)} test samples")
    print(f"{len(FEATURE_NAMES)} features: {', '.join(FEATURE_NAMES)}\n")

    header = f"  {'model':<24}{'R2 test':>10}{'dir. acc.':>11}{'fit (s)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    for candidate in build_candidates():
        started = time.perf_counter()
        outcome = candidate(train, test)
        elapsed = time.perf_counter() - started

        if outcome is None:
            continue

        predictions, label = outcome
        r2 = r_squared(test.y, predictions)
        accuracy = directional_accuracy(test.y, predictions)
        results.append((label, r2, accuracy, elapsed))

        print(f"  {label:<24}{r2:>+10.5f}{accuracy:>10.1%}{elapsed:>10.2f}")

    if not results:
        return

    best = max(results, key=lambda row: row[1])
    print(f"\n  Best out-of-sample R2: {best[0]} ({best[1]:+.5f})")

    linear = [r for r in results if r[0].startswith(("ridge", "ols", "lasso"))]
    trees = [r for r in results if r[0].startswith(("random_forest", "gradient", "lightgbm"))]

    if linear and trees:
        best_linear = max(linear, key=lambda row: row[1])
        best_tree = max(trees, key=lambda row: row[1])
        gap = best_tree[1] - best_linear[1]
        print(
            f"  Best tree model beats the best linear one by {gap:+.5f} R2"
            f" at {best_tree[3] / max(best_linear[3], 1e-9):.0f}x the fit time"
        )


def coefficient_report(dataset: Dataset, alpha: float = 1.0) -> None:
    """Print the standardised ridge coefficients.

    Standardised, so magnitudes are directly comparable across features.
    """
    train, _ = dataset.split()
    model = RidgeRegression(alpha=alpha).fit(train.X, train.y)

    print(f"\n  Standardised ridge coefficients (alpha={alpha}):")
    pairs = sorted(
        zip(FEATURE_NAMES, model.coefficients), key=lambda kv: -abs(kv[1])
    )
    for name, coefficient in pairs:
        print(f"    {name:<24}{coefficient:>+10.5f}")


def horizon_sweep(seeds: Sequence[int], steps: int) -> None:
    """Out-of-sample R² as a function of the prediction horizon.

    One step is mostly tick noise; too many steps and the signal is no longer
    actionable by a maker that requotes every step. The sweep uses the same
    seeds as the model comparison, since R² at this magnitude is sensitive to
    sample size.
    """
    print("\n  R2 by prediction horizon (ridge, alpha=1.0):")
    print(f"    {'horizon':>8}{'R2 test':>12}{'samples':>10}")

    for horizon in (1, 2, 3, 5, 10, 20):
        dataset = collect_dataset(seeds=seeds, steps=steps, horizon=horizon)
        train, test = dataset.split()
        model = RidgeRegression(alpha=1.0).fit(train.X, train.y)
        print(
            f"    {horizon:>8}{model.score(test.X, test.y):>+12.5f}{len(dataset):>10}"
        )


def main() -> None:
    seeds = range(30)
    steps = 400

    print("Collecting simulated data...")
    dataset = collect_dataset(seeds=seeds, steps=steps, horizon=3)

    evaluate(dataset)
    coefficient_report(dataset)
    horizon_sweep(seeds, steps)


if __name__ == "__main__":
    main()
