"""Persistent memory: every backtest, sweep and walk-forward run plus an activity log.

Backed by a single DuckDB file (default data/backtests.duckdb, override with the
BACKTESTER_DB environment variable). Connections are opened per call and closed
immediately so the CLI and the UI can share the file.

Tables
  runs          one row per saved run (kind = backtest | sweep | walkforward)
  run_series    bar-level date/close/position/equity for a run (stitched OOS for walk-forward)
  run_trades    closed trades for a backtest run
  run_table     tabular output (sweep grid rows, walk-forward folds) as JSON rows
  actions       activity log: what was done, when, from where, on which run
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

from backtester.engine import BacktestConfig, BacktestResult

DEFAULT_DB = Path("data") / "backtests.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          VARCHAR PRIMARY KEY,
    created_at  TIMESTAMP NOT NULL,
    kind        VARCHAR NOT NULL,
    source      VARCHAR,
    symbol      VARCHAR NOT NULL,
    strategy    VARCHAR NOT NULL,
    params      VARCHAR,          -- JSON
    config      VARCHAR,          -- JSON
    start_date  DATE,
    end_date    DATE,
    n_bars      INTEGER,
    metrics     VARCHAR,          -- JSON (headline metrics of the run)
    benchmark   VARCHAR,          -- JSON (buy & hold over the same bars)
    notes       VARCHAR DEFAULT '',
    tags        VARCHAR[] DEFAULT [],
    starred     BOOLEAN DEFAULT FALSE,
    parent_id   VARCHAR           -- set when a run is a re-run of another
);
CREATE TABLE IF NOT EXISTS run_series (
    run_id VARCHAR, date DATE, close DOUBLE, position DOUBLE, equity DOUBLE,
    bench_equity DOUBLE           -- buy & hold over the same bars
);
CREATE TABLE IF NOT EXISTS run_trades (
    run_id VARCHAR, entry_date DATE, exit_date DATE, direction DOUBLE,
    entry_price DOUBLE, exit_price DOUBLE, "return" DOUBLE, bars_held BIGINT
);
CREATE TABLE IF NOT EXISTS run_table (
    run_id VARCHAR, row_idx INTEGER, data VARCHAR
);
CREATE TABLE IF NOT EXISTS actions (
    ts      TIMESTAMP NOT NULL,
    action  VARCHAR NOT NULL,
    source  VARCHAR,
    run_id  VARCHAR,
    details VARCHAR                -- JSON
);
"""


def db_path() -> Path:
    return Path(os.environ.get("BACKTESTER_DB", DEFAULT_DB))


@contextmanager
def connect(path: Path | None = None, retries: int = 20):
    """Open the store, retrying briefly if another process holds the write lock."""
    path = Path(path or db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            con = duckdb.connect(str(path))
            break
        except duckdb.IOException:
            if attempt == retries - 1:
                raise
            time.sleep(0.1)
    try:
        con.execute(SCHEMA)
        yield con
    finally:
        con.close()


# ---------------------------------------------------------------------------
# JSON helpers (metrics may contain inf/nan; keep them round-trippable)
# ---------------------------------------------------------------------------


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    return v


def _dumps(obj) -> str:
    return json.dumps(_clean(obj))


def _loads(s):
    if s is None:
        return None
    obj = json.loads(s)
    if isinstance(obj, dict):
        return {k: (float(v) if v in ("inf", "-inf") else v) for k, v in obj.items()}
    return obj


def _new_id() -> str:
    return uuid.uuid4().hex[:10]


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def log_action(
    action: str,
    details: dict | None = None,
    run_id: str | None = None,
    source: str = "cli",
    path: Path | None = None,
) -> None:
    with connect(path) as con:
        con.execute(
            "INSERT INTO actions VALUES (?, ?, ?, ?, ?)",
            [datetime.now(), action, source, run_id, _dumps(details or {})],
        )


def _insert_run(
    con,
    *,
    kind: str,
    symbol: str,
    strategy: str,
    params: dict,
    config: BacktestConfig,
    bars: pl.DataFrame,
    metrics: dict,
    benchmark: dict | None,
    source: str,
    notes: str,
    tags: list[str],
    parent_id: str | None,
) -> str:
    run_id = _new_id()
    con.execute(
        """INSERT INTO runs (id, created_at, kind, source, symbol, strategy, params, config,
                             start_date, end_date, n_bars, metrics, benchmark, notes, tags,
                             starred, parent_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, ?)""",
        [
            run_id,
            datetime.now(),
            kind,
            source,
            symbol.upper(),
            strategy,
            _dumps(params),
            _dumps(asdict(config)),
            bars["date"][0],
            bars["date"][-1],
            bars.height,
            _dumps(metrics),
            _dumps(benchmark) if benchmark else None,
            notes,
            tags,
            parent_id,
        ],
    )
    return run_id


def _insert_series(
    con, run_id: str, frame: pl.DataFrame, bench_equity: pl.Series | None = None
) -> None:
    s = frame.select("date", "close", "position", "equity")
    s = s.with_columns(  # noqa: F841 - read by DuckDB
        bench_equity=bench_equity if bench_equity is not None else pl.lit(None, pl.Float64)
    )
    con.execute(
        "INSERT INTO run_series SELECT ?, date, close, position, equity, bench_equity FROM s",
        [run_id],
    )


def _insert_table(con, run_id: str, table: pl.DataFrame) -> None:
    rows = [(run_id, i, _dumps(r)) for i, r in enumerate(table.to_dicts())]
    if rows:
        con.executemany("INSERT INTO run_table VALUES (?, ?, ?)", rows)


def save_backtest(
    result: BacktestResult,
    symbol: str,
    strategy: str,
    metrics: dict,
    benchmark: dict | None = None,
    bench_equity: pl.Series | None = None,
    source: str = "cli",
    notes: str = "",
    tags: list[str] | None = None,
    parent_id: str | None = None,
    path: Path | None = None,
) -> str:
    with connect(path) as con:
        run_id = _insert_run(
            con,
            kind="backtest",
            symbol=symbol,
            strategy=strategy,
            params=result.params,
            config=result.config,
            bars=result.bars,
            metrics=metrics,
            benchmark=benchmark,
            source=source,
            notes=notes,
            tags=tags or [],
            parent_id=parent_id,
        )
        _insert_series(con, run_id, result.bars, bench_equity)
        if result.trades.height:
            t = result.trades.select(  # noqa: F841 - read by DuckDB
                "entry_date",
                "exit_date",
                "direction",
                "entry_price",
                "exit_price",
                "return",
                "bars_held",
            )
            con.execute("INSERT INTO run_trades SELECT ?, * FROM t", [run_id])
        con.execute(
            "INSERT INTO actions VALUES (?, 'backtest', ?, ?, ?)",
            [
                datetime.now(),
                source,
                run_id,
                _dumps({"symbol": symbol, "strategy": strategy, "params": result.params}),
            ],
        )
    return run_id


def save_sweep(
    table: pl.DataFrame,
    bars: pl.DataFrame,
    symbol: str,
    strategy: str,
    grid: dict,
    config: BacktestConfig,
    source: str = "cli",
    notes: str = "",
    tags: list[str] | None = None,
    path: Path | None = None,
) -> str:
    best = table.row(0, named=True) if table.height else {}
    with connect(path) as con:
        run_id = _insert_run(
            con,
            kind="sweep",
            symbol=symbol,
            strategy=strategy,
            params={"grid": grid},
            config=config,
            bars=bars,
            metrics=best,
            benchmark=None,
            source=source,
            notes=notes,
            tags=tags or [],
            parent_id=None,
        )
        _insert_table(con, run_id, table)
        con.execute(
            "INSERT INTO actions VALUES (?, 'sweep', ?, ?, ?)",
            [
                datetime.now(),
                source,
                run_id,
                _dumps({"symbol": symbol, "strategy": strategy, "combos": table.height}),
            ],
        )
    return run_id


def save_walkforward(
    folds: pl.DataFrame,
    oos: pl.DataFrame,
    bars: pl.DataFrame,
    symbol: str,
    strategy: str,
    grid: dict,
    settings: dict,
    config: BacktestConfig,
    metrics: dict,
    benchmark: dict | None = None,
    bench_equity: pl.Series | None = None,
    source: str = "cli",
    notes: str = "",
    tags: list[str] | None = None,
    path: Path | None = None,
) -> str:
    with connect(path) as con:
        run_id = _insert_run(
            con,
            kind="walkforward",
            symbol=symbol,
            strategy=strategy,
            params={"grid": grid, **settings},
            config=config,
            bars=bars,
            metrics=metrics,
            benchmark=benchmark,
            source=source,
            notes=notes,
            tags=tags or [],
            parent_id=None,
        )
        _insert_table(con, run_id, folds)
        if oos.height:
            _insert_series(con, run_id, oos, bench_equity)
        con.execute(
            "INSERT INTO actions VALUES (?, 'walkforward', ?, ?, ?)",
            [
                datetime.now(),
                source,
                run_id,
                _dumps({"symbol": symbol, "strategy": strategy, "folds": folds.height}),
            ],
        )
    return run_id


def update_run(
    run_id: str,
    notes: str | None = None,
    tags: list[str] | None = None,
    starred: bool | None = None,
    source: str = "cli",
    path: Path | None = None,
) -> None:
    changes = {
        k: v for k, v in {"notes": notes, "tags": tags, "starred": starred}.items() if v is not None
    }
    if not changes:
        return
    with connect(path) as con:
        sets = ", ".join(f"{k} = ?" for k in changes)
        con.execute(f"UPDATE runs SET {sets} WHERE id = ?", [*changes.values(), run_id])
        con.execute(
            "INSERT INTO actions VALUES (?, 'annotate', ?, ?, ?)",
            [datetime.now(), source, run_id, _dumps(changes)],
        )


def delete_run(run_id: str, source: str = "cli", path: Path | None = None) -> None:
    with connect(path) as con:
        for table in ("run_series", "run_trades", "run_table"):
            con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM runs WHERE id = ?", [run_id])
        con.execute(
            "INSERT INTO actions VALUES (?, 'delete', ?, ?, '{}')",
            [datetime.now(), source, run_id],
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_runs(
    kind: str | None = None,
    symbol: str | None = None,
    strategy: str | None = None,
    starred_only: bool = False,
    limit: int = 500,
    path: Path | None = None,
) -> pl.DataFrame:
    """Runs newest first, with headline metrics flattened into columns."""
    where, args = ["1=1"], []
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if symbol:
        where.append("symbol = ?")
        args.append(symbol.upper())
    if strategy:
        where.append("strategy = ?")
        args.append(strategy)
    if starred_only:
        where.append("starred")
    with connect(path) as con:
        df = con.sql(
            f"SELECT * FROM runs WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT ?",
            params=[*args, int(limit)],
        ).pl()
    if df.is_empty():
        return df
    metrics = [_loads(m) or {} for m in df["metrics"]]
    bench = [_loads(b) or {} for b in df["benchmark"]]
    return df.with_columns(
        pl.Series("total_return", [m.get("total_return") for m in metrics], pl.Float64),
        pl.Series("cagr", [m.get("cagr") for m in metrics], pl.Float64),
        pl.Series("sharpe", [m.get("sharpe") for m in metrics], pl.Float64),
        pl.Series("max_drawdown", [m.get("max_drawdown") for m in metrics], pl.Float64),
        pl.Series("trades", [m.get("trades") for m in metrics], pl.Float64),
        pl.Series("bench_return", [b.get("total_return") for b in bench], pl.Float64),
    )


def get_run(run_id: str, path: Path | None = None) -> dict | None:
    """Full run: metadata dict plus `series`, `trades` and `table` frames."""
    with connect(path) as con:
        meta = con.sql("SELECT * FROM runs WHERE id = ?", params=[run_id]).pl()
        if meta.is_empty():
            return None
        series = con.sql(
            "SELECT date, close, position, equity, bench_equity FROM run_series "
            "WHERE run_id = ? ORDER BY date",
            params=[run_id],
        ).pl()
        trades = con.sql(
            "SELECT * EXCLUDE (run_id) FROM run_trades WHERE run_id = ? ORDER BY entry_date",
            params=[run_id],
        ).pl()
        rows = con.sql(
            "SELECT data FROM run_table WHERE run_id = ? ORDER BY row_idx", params=[run_id]
        ).fetchall()
    run = meta.row(0, named=True)
    for key in ("params", "config", "metrics", "benchmark"):
        run[key] = _loads(run[key])
    run["series"] = series
    run["trades"] = trades
    run["table"] = pl.DataFrame([_loads(r[0]) for r in rows]) if rows else pl.DataFrame()
    return run


def list_actions(limit: int = 200, path: Path | None = None) -> pl.DataFrame:
    with connect(path) as con:
        return con.sql(
            "SELECT ts, action, source, run_id, details FROM actions ORDER BY ts DESC LIMIT ?",
            params=[int(limit)],
        ).pl()
