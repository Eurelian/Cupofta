"""Jobs shared by the CLI and the UI: load data, run, compute metrics, save to memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from backtester import store
from backtester.analysis import sweep, walk_forward
from backtester.data import DEFAULT_DATA_DIR, load_bars, synthetic_bars
from backtester.engine import BacktestConfig, BacktestResult, run_backtest
from backtester.metrics import compute_metrics
from backtester.strategies import generate, get_strategy

SYNTH = "SYNTH"


def get_bars(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pl.DataFrame:
    if symbol.upper() == SYNTH:
        bars = synthetic_bars()
        if start:
            bars = bars.filter(pl.col("date") >= start)
        if end:
            bars = bars.filter(pl.col("date") <= end)
        return bars
    return load_bars(symbol, data_dir, start, end)


def _bars_range(bars: pl.DataFrame) -> tuple[date, date]:
    return bars["date"][0], bars["date"][-1]


@dataclass
class BacktestJob:
    result: BacktestResult
    metrics: dict
    benchmark: dict
    bench_equity: pl.Series
    run_id: str | None = None


@dataclass
class SweepJob:
    table: pl.DataFrame
    bars: pl.DataFrame
    run_id: str | None = None


@dataclass
class WalkForwardJob:
    folds: pl.DataFrame
    oos: pl.DataFrame
    metrics: dict
    benchmark: dict
    bench_equity: pl.Series
    run_id: str | None = None
    settings: dict = field(default_factory=dict)


def backtest(
    symbol: str,
    strategy: str,
    params: dict | None = None,
    config: BacktestConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    save: bool = True,
    source: str = "cli",
    notes: str = "",
    tags: list[str] | None = None,
    parent_id: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> BacktestJob:
    config = config or BacktestConfig()
    bars = get_bars(symbol, start, end, data_dir)
    full_params = {**get_strategy(strategy).defaults, **(params or {})}
    res = run_backtest(bars, generate(strategy, bars, **full_params), config, full_params)
    bench = run_backtest(bars, np.ones(bars.height), config)
    job = BacktestJob(res, compute_metrics(res), compute_metrics(bench), bench.equity)
    if save:
        job.run_id = store.save_backtest(
            res,
            symbol,
            strategy,
            job.metrics,
            job.benchmark,
            bench_equity=bench.equity,
            source=source,
            notes=notes,
            tags=tags,
            parent_id=parent_id,
        )
    return job


def run_sweep(
    symbol: str,
    strategy: str,
    grid: dict[str, list],
    config: BacktestConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    sort_by: str = "sharpe",
    *,
    save: bool = True,
    source: str = "cli",
    notes: str = "",
    tags: list[str] | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> SweepJob:
    config = config or BacktestConfig()
    bars = get_bars(symbol, start, end, data_dir)
    table = sweep(bars, strategy, grid, config, sort_by=sort_by)
    job = SweepJob(table, bars)
    if save:
        job.run_id = store.save_sweep(
            table, bars, symbol, strategy, grid, config, source=source, notes=notes, tags=tags
        )
    return job


def run_walkforward(
    symbol: str,
    strategy: str,
    grid: dict[str, list],
    n_splits: int = 5,
    train_frac: float = 0.6,
    objective: str = "sharpe",
    config: BacktestConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    save: bool = True,
    source: str = "cli",
    notes: str = "",
    tags: list[str] | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> WalkForwardJob:
    config = config or BacktestConfig()
    bars = get_bars(symbol, start, end, data_dir)
    folds, oos = walk_forward(bars, strategy, grid, n_splits, train_frac, config, objective)
    if oos.is_empty():
        raise ValueError(
            "walk-forward produced no out-of-sample data; use more bars or fewer splits"
        )

    # Metrics of the stitched out-of-sample curve, and buy & hold over the same dates.
    oos_res = BacktestResult(bars=oos, trades=pl.DataFrame(), config=config)
    metrics = compute_metrics(oos_res)
    metrics["trades"] = int(folds["oos_trades"].sum())
    # Per-trade stats aren't tracked across folds; blank them rather than report zeros.
    for k in ("win_rate", "profit_factor", "avg_trade", "avg_bars_held"):
        metrics[k] = None
    lo = bars["date"].search_sorted(oos["date"][0])
    lead = 1 if lo > 0 else 0
    window = bars.slice(lo - lead, oos.height + lead)
    bench = run_backtest(window, np.ones(window.height), config)
    bench_bars = bench.bars.slice(lead)
    bench_res = BacktestResult(bars=bench_bars, trades=pl.DataFrame(), config=config)
    settings = {"n_splits": n_splits, "train_frac": train_frac, "objective": objective}
    job = WalkForwardJob(
        folds, oos, metrics, compute_metrics(bench_res), bench_bars["equity"], settings=settings
    )
    if save:
        job.run_id = store.save_walkforward(
            folds,
            oos,
            bars,
            symbol,
            strategy,
            grid,
            settings,
            config,
            metrics,
            benchmark=job.benchmark,
            bench_equity=job.bench_equity,
            source=source,
            notes=notes,
            tags=tags,
        )
    return job


def rerun(run_id: str, source: str = "cli") -> str:
    """Re-run a saved run with the same settings on the current data. Returns the new run id."""
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"no run {run_id}")
    cfg = BacktestConfig(**run["config"])
    common = dict(
        symbol=run["symbol"],
        strategy=run["strategy"],
        config=cfg,
        start=run["start_date"],
        end=run["end_date"],
        source=source,
        notes=f"re-run of {run_id}",
        tags=list(run["tags"] or []),
    )
    params = dict(run["params"] or {})
    if run["kind"] == "backtest":
        new_id = backtest(params=params, parent_id=run_id, **common).run_id
    elif run["kind"] == "sweep":
        new_id = run_sweep(grid=params.pop("grid"), **common).run_id
    else:
        new_id = run_walkforward(grid=params.pop("grid"), **params, **common).run_id
    store.log_action("rerun", {"new_run": new_id}, run_id=run_id, source=source)
    return new_id
