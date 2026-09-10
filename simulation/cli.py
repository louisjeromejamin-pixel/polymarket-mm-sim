"""Interface en ligne de commande du simulateur.

Exemples
--------
    python -m simulation.cli backtest --strategy amm --steps 500 --seed 42
    python -m simulation.cli compare --seeds 20 --steps 500
    python -m simulation.cli sensitivity --param volatility --values 0.01,0.05,0.1
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from typing import List

from .backtest import BacktestResult, run_backtest, run_sweep
from .execution import ExecutionConfig
from .market_sim import MarketConfig

CONFIGS = {"amm": "config/amm.json", "bands": "config/bands.json"}


def _enable_utf8_output() -> None:
    """Force la sortie en UTF-8 (la console Windows utilise cp1252)."""
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
        resolution_step=args.resolution_step,
    )


def _execution_from_args(args: argparse.Namespace) -> ExecutionConfig:
    return ExecutionConfig(
        maker_fee=args.maker_fee,
        fill_probability=args.fill_probability,
    )


def _print_result(result: BacktestResult) -> None:
    summary = result.summary()
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"  {key.replace('_', ' '):<{width}}  {value}")
    if result.resolved:
        issue = "OUI" if result.outcome == 1 else "NON"
        print(f"  {'resolution':<{width}}  marché résolu sur « {issue} »")


def cmd_backtest(args: argparse.Namespace) -> int:
    result = run_backtest(
        args.strategy,
        args.config or CONFIGS[args.strategy],
        steps=args.steps,
        market_config=_market_from_args(args),
        execution_config=_execution_from_args(args),
        seed=args.seed,
    )
    print(f"\nBacktest — stratégie {args.strategy.upper()}\n")
    _print_result(result)
    return 0


def _aggregate(results: List[BacktestResult]) -> dict:
    pnls = [r.pnl for r in results]
    return {
        "runs": len(results),
        "pnl_moyen": round(statistics.mean(pnls), 2),
        "pnl_median": round(statistics.median(pnls), 2),
        "pnl_ecart_type": round(statistics.stdev(pnls), 2) if len(pnls) > 1 else 0.0,
        "pnl_min": round(min(pnls), 2),
        "pnl_max": round(max(pnls), 2),
        "part_gagnante_pct": round(100 * sum(p > 0 for p in pnls) / len(pnls), 1),
        "executions_moy": round(statistics.mean([r.num_fills for r in results]), 1),
        "drawdown_moy_pct": round(
            statistics.mean([r.max_drawdown_pct for r in results]), 3
        ),
        "inventaire_max_moy": round(
            statistics.mean([r.max_inventory_skew for r in results]), 1
        ),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    seeds = list(range(args.seeds))
    market = _market_from_args(args)
    execution = _execution_from_args(args)

    print(f"\nComparaison sur {len(seeds)} graines, {args.steps} pas par run\n")

    rows = {}
    for strategy in ("amm", "bands"):
        results = run_sweep(
            strategy,
            args.config or CONFIGS[strategy],
            seeds,
            steps=args.steps,
            market_config=market,
            execution_config=execution,
        )
        rows[strategy] = _aggregate(results)

    metrics = list(next(iter(rows.values())).keys())
    width = max(len(m) for m in metrics)
    header = f"  {'métrique':<{width}}  {'AMM':>12}  {'BANDS':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric in metrics:
        print(
            f"  {metric.replace('_', ' '):<{width}}"
            f"  {rows['amm'][metric]:>12}  {rows['bands'][metric]:>12}"
        )
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    values = [float(v) for v in args.values.split(",")]
    seeds = list(range(args.seeds))

    print(f"\nSensibilité à « {args.param} » — stratégie {args.strategy.upper()}")
    print(f"{len(seeds)} graines par valeur, {args.steps} pas par run\n")

    width = max(len(args.param), 10)
    print(f"  {args.param:<{width}}  {'pnl moyen':>12}  {'ecart-type':>12}  {'% gagnant':>10}")
    print("  " + "-" * (width + 40))

    for value in values:
        market = _market_from_args(args)
        execution = _execution_from_args(args)
        if hasattr(market, args.param):
            setattr(market, args.param, value)
        elif hasattr(execution, args.param):
            setattr(execution, args.param, value)
        else:
            raise SystemExit(f"Paramètre inconnu : {args.param}")

        results = run_sweep(
            args.strategy,
            args.config or CONFIGS[args.strategy],
            seeds,
            steps=args.steps,
            market_config=market,
            execution_config=execution,
        )
        stats = _aggregate(results)
        print(
            f"  {value:<{width}}  {stats['pnl_moyen']:>12}"
            f"  {stats['pnl_ecart_type']:>12}  {stats['part_gagnante_pct']:>10}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulation",
        description="Backtest des stratégies de market making Polymarket, hors ligne.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--steps", type=int, default=500, help="Pas de simulation")
        p.add_argument("--config", default=None, help="Config JSON de la stratégie")
        p.add_argument("--initial-price", type=float, default=0.50)
        p.add_argument("--volatility", type=float, default=0.15)
        p.add_argument("--drift", type=float, default=0.0)
        p.add_argument("--resolution-step", type=int, default=None)
        p.add_argument("--maker-fee", type=float, default=0.0)
        p.add_argument("--fill-probability", type=float, default=0.6)

    bt = sub.add_parser("backtest", help="Un backtest sur une graine")
    bt.add_argument("--strategy", choices=["amm", "bands"], default="amm")
    bt.add_argument("--seed", type=int, default=42)
    add_common(bt)
    bt.set_defaults(func=cmd_backtest)

    cmp_ = sub.add_parser("compare", help="Compare AMM et Bands sur plusieurs graines")
    cmp_.add_argument("--seeds", type=int, default=20, help="Nombre de graines")
    add_common(cmp_)
    cmp_.set_defaults(func=cmd_compare)

    sens = sub.add_parser("sensitivity", help="Balaye un paramètre")
    sens.add_argument("--strategy", choices=["amm", "bands"], default="amm")
    sens.add_argument("--param", default="volatility")
    sens.add_argument("--values", default="0.01,0.03,0.05,0.10")
    sens.add_argument("--seeds", type=int, default=20)
    add_common(sens)
    sens.set_defaults(func=cmd_sensitivity)

    return parser


def main(argv: List[str] | None = None) -> int:
    _enable_utf8_output()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
