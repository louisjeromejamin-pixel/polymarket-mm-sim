# polymarket-mm-sim

Offline backtesting of market-making strategies on Polymarket-style binary markets, with a learned quote skew. No API key, wallet or capital required.

Built on [Polymarket's official keeper](https://github.com/Polymarket/poly-market-maker) (MIT). The upstream strategies are unchanged; the two external dependencies — the CLOB API and the blockchain — are replaced by a simulated market and a local execution engine.

## Market model

A binary contract resolves to 0 or 1, so its price is a probability. The price evolves in log-odds

$$\ell_{t+1} = \ell_t + \mu + \sigma \varepsilon_t, \qquad p_t = \frac{1}{1 + e^{-\ell_t}}$$

which keeps $p_t \in (0,1)$ by construction and compresses moves near the bounds, where a near-resolved contract barely trades.

Order flow arrives as $N_t \sim \mathrm{Poisson}(\lambda)$ counterparty orders per step, split into two populations:

- **Informed** (probability $\phi$): trades in the direction of the coming move, $\mathbb{P}(\text{buy}) = 0.85$ if $\varepsilon_t > 0$. Filling it leaves the maker on the wrong side — adverse selection.
- **Uninformed**: symmetric, $\mathbb{P}(\text{buy}) = 0.5$, and willing to concede up to 4 cents from the mid. That concession is the maker's revenue.

The split is what makes the simulator informative. With $\phi = 1$ every counterparty is informed, no quote is ever hit at a favourable price, and no strategy can profit:

| Informed share $\phi$ | Mean PnL | Profitable runs | Fills |
|---:|---:|---:|---:|
| 0.00 | 169.1 | 87.5% | 852 |
| 0.20 | 126.0 | 82.5% | 686 |
| 0.35 | 102.2 | 75.0% | 555 |
| 0.50 | 65.4 | 70.0% | 417 |
| 0.70 | 29.3 | 60.0% | 246 |
| 1.00 | **0.0** | 0.0% | **0** |

40 seeds per value, 400 steps per run, Bands strategy.

## Strategy comparison

50 seeds, 400 steps, $\phi = 0.35$, starting capital 1500 USDC:

| Metric | AMM | Bands |
|---|---:|---:|
| Mean PnL | 82.4 | **101.1** |
| Median PnL | 96.3 | 111.0 |
| PnL std dev | 240.7 | **132.7** |
| Median Sharpe | 14.3 | **29.5** |
| Profitable runs | 66% | **78%** |
| Mean drawdown | 19.9% | **10.7%** |
| Max inventory | 1035 | **849** |

Bands earns slightly more with half the dispersion and half the drawdown. The AMM quotes across a wider price range, so it accumulates inventory it did not choose.

## Learned quote skew

A maker quoting symmetrically around the mid holds whatever inventory the flow gives it. If the next move is partly predictable, shifting both quotes in that direction reduces adverse fills.

Seven features, all observable at decision time: flow imbalance and its EMA, order count, realised volatility, last mid return, distance from 0.50, and normalised inventory. Target: the mid return over the next 3 steps.

Production uses **ridge regression**, solved in closed form against the standard library:

$$\hat{w} = (Z^\top Z + \alpha I)^{-1} Z^\top y$$

on standardised features — the penalty is scale-dependent, and it also keeps the matrix invertible when the two imbalance features are collinear.

The skew is capped at 3 ticks. Without a cap, the strategy stops making markets and becomes a directional bet on the model.

### Model selection

`research/signal_models.py` compares candidates on the same chronological split. 8337 training rows, 3573 test rows:

| Model | Test R² | Directional acc. | Fit time |
|---|---:|---:|---:|
| mean baseline | −0.00031 | 50.4% | 0.00 s |
| OLS | +0.01750 | 54.8% | 0.04 s |
| **ridge (α = 1)** | **+0.01750** | **54.8%** | 0.11 s |
| lasso | +0.01742 | 54.9% | 0.02 s |
| random forest (d = 6) | **+0.02247** | 54.4% | 5.11 s |
| gradient boosting | +0.00955 | 53.0% | 3.92 s |
| LightGBM (d = 4) | −0.01103 | 52.9% | 1.75 s |
| LightGBM (d = 8) | −0.12842 | 52.9% | 2.01 s |

The random forest has the best R² but a *lower* directional accuracy, and only the sign matters for a quote skew. On 60 held-out seeds it also produced less PnL than ridge (+148 vs +154). LightGBM overfits outright: R² of −0.128 at depth 8.

Ridge therefore ships, and the repo depends on neither scikit-learn nor LightGBM.

Signal quality falls with the prediction horizon, as expected of microstructure:

| Horizon (steps) | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| Test R² | +0.0355 | +0.0179 | +0.0175 | +0.0111 | +0.0111 | +0.0080 |

### Out-of-sample result

Trained on 30 seeds, evaluated on 400 seeds never used in training:

| | Baseline | With signal |
|---|---:|---:|
| Mean PnL | 107.0 | **120.7** |
| Median PnL | 132.6 | **150.1** |
| Median Sharpe | 33.96 | **35.05** |
| Profitable runs | 78.0% | **79.0%** |

Paired difference **+13.71**, $t = +2.25$, $n = 400$.

The sample size is not incidental. PnL has a standard deviation of ~144 across seeds, so detecting an effect of this magnitude needs roughly 370 paired runs: at $n = 100$ the same test swung between $t = +2.9$ and $t = +0.3$ depending on which seeds were drawn. The effect is real but small, and any claim from a smaller sample would have been noise.

## Usage

```bash
pip install -r requirements-sim.txt   # simulation only: standard library
```

```bash
python -m simulation.cli backtest --strategy bands --steps 400 --seed 42
python -m simulation.cli compare --seeds 50
python -m simulation.cli signal --train-seeds 30 --test-seeds 400
python -m simulation.cli sensitivity --param informed_ratio --values 0,0.35,0.7
```

```python
from simulation import collect_dataset, train_signal, run_sweep, SignalModel

dataset = collect_dataset(seeds=range(30), steps=400, horizon=3)
model, report = train_signal(dataset, alpha=1.0)
print(report["r2_out_of_sample"])

results = run_sweep("bands", "config/bands.json", range(1000, 1400),
                    steps=400, signal=SignalModel(model))
```

## Upstream bug

Running the AMM strategy under high volatility surfaced an exception in the upstream code:

```
File "poly_market_maker/strategies/amm.py", line 107, in phi
    return (1 / (sqrt(self.p_i) - sqrt(self.p_l))) * (...)
ZeroDivisionError: float division by zero
```

When the price reaches a bound of the configured range (`p_min` = 0.05, `p_max` = 0.95), `set_price()` yields `p_l == p_i` and an empty `buy_prices` list — division by zero on one path, `IndexError` on the other. Every prediction market drifts towards 0 or 1 as it approaches resolution, so a keeper left running on a settling market reaches this state.

Fixed in [`amm.py`](poly_market_maker/strategies/amm.py) by returning a zero allocation in the degenerate cases; a test asserts no strategy error over 400 steps at volatility 0.5.

## Layout

```
poly_market_maker/       upstream code (Polymarket, MIT), strategies unchanged
  strategies/amm.py      + fix for the degenerate cases in phi()
  order.py, utils.py     + py_clob_client / web3 imports made optional
simulation/
  market_sim.py          log-odds price, informed and uninformed flow
  execution.py           flow-driven matching, portfolio, adverse selection
  signal.py              features, ridge regression, quote skew
  training.py            data collection, train/test split, fitting
  backtest.py            backtest loop and risk metrics
  cli.py                 backtest / compare / signal / sensitivity
research/
  signal_models.py       model comparison (needs scikit-learn, LightGBM)
tests/                   83 tests
```

```bash
pytest
```

Tests cover the logit/sigmoid round trip, price bounds under extreme volatility, the Poisson generator's mean and variance, ridge recovering a known linear relation, negative out-of-sample R² on pure noise, refusal of short sales and uncollateralised buys, and the fact that informed flow never trades at a price favourable to the maker.

## Limitations

- **The market ignores the maker.** Its orders do not move the price, which overstates performance at size.
- **Informed flow is a two-state model.** Real flow has a continuum of information content.
- **No liquidity rewards**, a significant part of real Polymarket revenue.
- **A step has no calendar duration**, so the annualised Sharpe is comparable only across runs of this simulator.
- **Synthetic prices.** Replaying historical Polymarket order books would be the next step.

## Credits

`poly_market_maker/` comes from [Polymarket/poly-market-maker](https://github.com/Polymarket/poly-market-maker), MIT licensed, copyright (c) 2023 Polymarket. The original licence is kept in [LICENSE](LICENSE).

> Study tool. Not investment advice; simulated results say nothing about live performance.
