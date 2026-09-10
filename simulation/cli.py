"""Command-line interface for the simulator.

    python -m simulation.cli backtest --strategy amm --steps 500 --seed 42
    python -m simulation.cli compare --seeds 25 --steps 400
    python -m simulation.cli signal --train-seeds 25 --test-seeds 50
    python -m simulation.cli sensitivity --param informed_ratio --values 0,0.3,0.6
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import sys
from typing import List, Optional

from .backtest import BacktestResult, run_backtest, run_sweep
from .execution import ExecutionConfig
from .market_sim import MarketConfig
from .signal import SignalConfig, SignalModel
from .training import collect_dataset, train_signal

CONFIGS = {"amm": "config/amm.json", "bands": "config/bands.json"}


def _enable_utf8_output() -> None:
    """Force UTF-8 output (the Windows console defaults to cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _market_from_args(args: argparse.Namespace) -> MarketConfig:
    return MarketConfig(
        initial_price=args.initial_price,
        volatility=args.volatility,
        drift=args.drift,
        informed_ratio=args.informed_ratio,
        flow_intensity=args.flow_intensity,
        resolution_step=args.resolution_step,
    )


def _execution_from_args(args: argparse.Namespace) -> ExecutionConfig:
    return ExecutionConfig(
        maker_fee=args.maker_fee,
        queue_priority=args.queue_priority,
    )


def _print_result(result: BacktestResult) -> None:
    summary = result.summary()
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"  {key.replace('_', ' '):<{width}}  {value}")
    if result.resolved:
        outcome = "YES" if result.outcome == 1 else "NO"
        print(f"  {'resolution':<{width}}  market resolved {outcome}")


def cmd_backtest(args: argparse.Namespace) -> int:
    result = run_backtest(
        args.strategy,
        args.config or CONFIGS[args.strategy],
        steps=args.steps,
        market_config=_market_from_args(args),
        execution_config=_execution_from_args(args),
        seed=args.seed,
    )
    print(f"\nBacktest - {args.strategy.upper()} strategy\n")
    _print_result(result)
    return 0


def _aggregate(results: List[BacktestResult]) -> dict:
    pnls = [r.pnl for r in results]
    sharpes = [r.sharpe for r in results]
    return {
        "runs": len(results),
        "mean_pnl": round(statistics.mean(pnls), 2),
        "median_pnl": round(statistics.median(pnls), 2),
        "stdev_pnl": round(statistics.stdev(pnls), 2) if len(pnls) > 1 else 0.0,
        "min_pnl": round(min(pnls), 2),
        "max_pnl": round(max(pnls), 2),
        "median_sharpe": round(statistics.median(sharpes), 2),
        "profitable_pct": round(100 * sum(p > 0 for p in pnls) / len(pnls), 1),
        "mean_fills": round(statistics.mean([r.num_fills for r in results]), 1),
        "informed_fill_pct": round(
            100 * statistics.mean([r.informed_fill_ratio for r in results]), 1
        ),
        "mean_drawdown_pct": round(
            statistics.mean([r.max_drawdown_pct for r in results]), 2
        ),
        "mean_max_inventory": round(
            statistics.mean([r.max_inventory_skew for r in results]), 1
        ),
    }


def _print_table(rows: dict, columns: List[str]) -> None:
    metrics = list(next(iter(rows.values())).keys())
    width = max(len(m) for m in metrics)

    header = f"  {'metric':<{width}}" + "".join(f"{c:>14}" for c in columns)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for metric in metrics:
        line = f"  {metric.replace('_', ' '):<{width}}"
        line += "".join(f"{rows[c][metric]:>14}" for c in columns)
        print(line)


def cmd_compare(args: argparse.Namespace) -> int:
    seeds = list(range(args.seeds))
    market = _market_from_args(args)
    execution = _execution_from_args(args)

    print(f"\nComparison over {len(seeds)} seeds, {args.steps} steps per run\n")

    rows = {}
    for strategy in ("amm", "bands"):
        rows[strategy] = _aggregate(
            run_sweep(
                strategy,
                args.config or CONFIGS[strategy],
                seeds,
                steps=args.steps,
                market_config=market,
                execution_config=execution,
            )
        )

    _print_table(rows, ["amm", "bands"])
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    """Fit the signal model and measure its effect out of sample."""
    market = _market_from_args(args)
    execution = _execution_from_args(args)

    train_seeds = range(args.train_seeds)
    # Test seeds are disjoint from training seeds by construction.
    test_seeds = range(1000, 1000 + args.test_seeds)

    print(
        f"\nTraining on {len(train_seeds)} seeds, "
        f"testing on {len(test_seeds)} unseen seeds\n"
    )

    dataset = collect_dataset(
        strategy_name=args.strategy,
        config_path=args.config or CONFIGS[args.strategy],
        seeds=train_seeds,
        steps=args.steps,
        horizon=args.horizon,
        market_config=market,
        execution_config=execution,
    )
    model, report = train_signal(dataset, alpha=args.alpha)

    print(f"  observations        {len(dataset)}")
    print(f"  R2 in sample        {report['r2_in_sample']:+.5f}")
    print(f"  R2 out of sample    {report['r2_out_of_sample']:+.5f}")
    print("\n  standardised coefficients:")
    for name, coefficient in sorted(
        report["coefficients"].items(), key=lambda kv: -abs(kv[1])
    ):
        print(f"    {name:<24}{coefficient:>+10.5f}")

    signal = SignalModel(
        model,
        SignalConfig(skew_strength=args.skew_strength, max_skew_ticks=args.max_skew),
    )

    baseline = run_sweep(
        args.strategy,
        args.config or CONFIGS[args.strategy],
        test_seeds,
        steps=args.steps,
        market_config=market,
        execution_config=execution,
    )
    with_signal = run_sweep(
        args.strategy,
        args.config or CONFIGS[args.strategy],
        test_seeds,
        steps=args.steps,
        market_config=market,
        execution_config=execution,
        signal=signal,
    )

    print()
    _print_table(
        {"baseline": _aggregate(baseline), "with signal": _aggregate(with_signal)},
        ["baseline", "with signal"],
    )

    # Paired difference: both arms share seeds, so the market draw cancels.
    differences = [a.pnl - b.pnl for a, b in zip(with_signal, baseline)]
    mean_difference = statistics.mean(differences)

    if len(differences) > 1 and statistics.stdev(differences) > 0:
        t_statistic = mean_difference / (
            statistics.stdev(differences) / math.sqrt(len(differences))
        )
        print(f"\n  paired PnL difference   {mean_difference:+.2f}")
        print(f"  t statistic             {t_statistic:+.2f}  (n = {len(differences)})")
        if abs(t_statistic) < 2:
            print(
                "  |t| < 2: not significant at this sample size. Roughly "
                "370 paired runs are\n  needed to resolve an effect of "
                "this magnitude, given the PnL dispersion across seeds."
            )

    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    values = [float(v) for v in args.values.split(",")]
    seeds = list(range(args.seeds))

    print(f"\nSensitivity to '{args.param}' - {args.strategy.upper()} strategy")
    print(f"{len(seeds)} seeds per value, {args.steps} steps per run\n")

    width = max(len(args.param), 12)
    print(
        f"  {args.param:<{width}}{'mean pnl':>12}{'stdev':>12}"
        f"{'profitable':>12}{'fills':>10}"
    )
    print("  " + "-" * (width + 46))

    for value in values:
        market = _market_from_args(args)
        execution = _execution_from_args(args)

        if hasattr(market, args.param):
            setattr(market, args.param, value)
        elif hasattr(execution, args.param):
            setattr(execution, args.param, value)
        else:
            raise SystemExit(f"Unknown parameter: {args.param}")

        stats = _aggregate(
            run_sweep(
                args.strategy,
                args.config or CONFIGS[args.strategy],
                seeds,
                steps=args.steps,
                market_config=market,
                execution_config=execution,
            )
        )
        print(
            f"  {value:<{width}}{stats['mean_pnl']:>12}{stats['stdev_pnl']:>12}"
            f"{stats['profitable_pct']:>11}%{stats['mean_fills']:>10}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulation",
        description="Offline backtesting of Polymarket market-making strategies.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--steps", type=int, default=400, help="Simulation steps")
        p.add_argument("--config", default=None, help="Strategy JSON config")
        p.add_argument("--initial-price", type=float, default=0.50)
        p.add_argument("--volatility", type=float, default=0.15)
        p.add_argument("--drift", type=float, default=0.0)
        p.add_argument("--informed-ratio", type=float, default=0.35)
        p.add_argument("--flow-intensity", type=float, default=2.0)
        p.add_argument("--resolution-step", type=int, default=None)
        p.add_argument("--maker-fee", type=float, default=0.0)
        p.add_argument("--queue-priority", type=float, default=0.7)

    bt = sub.add_parser("backtest", help="Single backtest on one seed")
    bt.add_argument("--strategy", choices=["amm", "bands"], default="bands")
    bt.add_argument("--seed", type=int, default=42)
    add_common(bt)
    bt.set_defaults(func=cmd_backtest)

    cmp_ = sub.add_parser("compare", help="Compare AMM and Bands across seeds")
    cmp_.add_argument("--seeds", type=int, default=25, help="Number of seeds")
    add_common(cmp_)
    cmp_.set_defaults(func=cmd_compare)

    sig = sub.add_parser("signal", help="Fit the signal and measure it out of sample")
    sig.add_argument("--strategy", choices=["amm", "bands"], default="bands")
    sig.add_argument("--train-seeds", type=int, default=30)
    # PnL varies enough between seeds that ~370 paired runs are needed to
    # detect an effect of the size this signal produces.
    sig.add_argument("--test-seeds", type=int, default=400)
    sig.add_argument("--horizon", type=int, default=3, help="Prediction horizon")
    sig.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty")
    sig.add_argument("--skew-strength", type=float, default=50.0)
    sig.add_argument("--max-skew", type=int, default=3, help="Skew cap in ticks")
    add_common(sig)
    sig.set_defaults(func=cmd_signal)

    sens = sub.add_parser("sensitivity", help="Sweep one parameter")
    sens.add_argument("--strategy", choices=["amm", "bands"], default="bands")
    sens.add_argument("--param", default="informed_ratio")
    sens.add_argument("--values", default="0.0,0.2,0.35,0.5,0.7,1.0")
    sens.add_argument("--seeds", type=int, default=20)
    add_common(sens)
    sens.set_defaults(func=cmd_sensitivity)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _enable_utf8_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
