"""Parameter sweeps and walk-forward validation."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics
from backtester.strategies import generate


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def sweep(
    bars: pl.DataFrame,
    strategy: str,
    grid: dict[str, list],
    config: BacktestConfig | None = None,
    base_params: dict | None = None,
    sort_by: str = "sharpe",
) -> pl.DataFrame:
    """Backtest every parameter combination; return one row of metrics per combo."""
    rows = []
    for params in param_grid(grid):
        p = {**(base_params or {}), **params}
        res = run_backtest(bars, generate(strategy, bars, **p), config, params=p)
        rows.append({**params, **compute_metrics(res)})
    return pl.DataFrame(rows).sort(sort_by, descending=True, nulls_last=True)


def walk_forward(
    bars: pl.DataFrame,
    strategy: str,
    grid: dict[str, list],
    n_splits: int = 5,
    train_frac: float = 0.6,
    config: BacktestConfig | None = None,
    objective: str = "sharpe",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Anchored walk-forward.

    The first `train_frac` of the data is the initial in-sample window; the remainder is split
    into `n_splits` out-of-sample folds. For each fold, the best params on all data before it
    are chosen by `objective` and then evaluated on the fold only.

    Signals are generated once on the full history (strategies are causal, so this leaks
    nothing) so indicator warm-up doesn't eat the start of every fold.

    Returns (per-fold table, stitched out-of-sample equity/returns frame).
    """
    config = config or BacktestConfig()
    n = bars.height
    oos_start = int(n * train_frac)
    edges = np.linspace(oos_start, n, n_splits + 1, dtype=int)
    combos = param_grid(grid)
    signals = [generate(strategy, bars, **p) for p in combos]

    folds, oos_parts = [], []
    capital = config.initial_capital
    for k in range(n_splits):
        lo, hi = edges[k], edges[k + 1]
        if hi - lo < 2:
            continue
        train = bars.slice(0, lo)
        scores = [
            compute_metrics(run_backtest(train, sig[:lo], config))[objective] for sig in signals
        ]
        best = int(np.nanargmax(scores))

        fold_cfg = BacktestConfig(
            initial_capital=capital,
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
            stop_loss=config.stop_loss,
        )
        # Start one bar early so the decision made at the close of bar lo-1 is executed at the
        # open of bar lo (the engine always trades the previous bar's target). That lead-in bar
        # is flat and is dropped from the stitched equity curve.
        lead = 1 if lo > 0 else 0
        test = bars.slice(lo - lead, hi - lo + lead)
        res = run_backtest(test, signals[best][lo - lead : hi], fold_cfg)
        m = compute_metrics(res)
        capital = float(res.equity[-1])
        folds.append(
            {
                "fold": k + 1,
                "start": test["date"][lead],
                "end": test["date"][-1],
                **{f"best_{key}": v for key, v in combos[best].items()},
                f"is_{objective}": float(scores[best]),
                **{f"oos_{key}": m[key] for key in ("total_return", "sharpe", "max_drawdown")},
                "oos_trades": m["trades"],
            }
        )
        oos_parts.append(
            res.bars.slice(lead).select("date", "close", "position", "equity", "returns")
        )

    return pl.DataFrame(folds), pl.concat(oos_parts) if oos_parts else pl.DataFrame()
