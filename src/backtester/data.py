"""Data layer: OHLCV bars stored as Parquet, queried with DuckDB, returned as Polars.

Canonical bar schema: date (Date), open, high, low, close, volume (Float64).
One Parquet file per symbol: <data_dir>/<SYMBOL>.parquet
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DEFAULT_DATA_DIR = Path("data")


def parquet_path(symbol: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / f"{symbol.upper()}.parquet"


def validate_bars(df: pl.DataFrame) -> pl.DataFrame:
    """Enforce schema, sort by date, drop duplicate dates and rows with missing prices."""
    missing = set(BAR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    return (
        df.select(
            pl.col("date").cast(pl.Date),
            *[pl.col(c).cast(pl.Float64) for c in BAR_COLUMNS[1:]],
        )
        .drop_nulls(["open", "high", "low", "close"])
        .unique(subset="date", keep="last")
        .sort("date")
    )


def save_bars(df: pl.DataFrame, symbol: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    path = parquet_path(symbol, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_bars(df).write_parquet(path)
    return path


def load_bars(
    symbol: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Load bars for one symbol through DuckDB (date filtering is pushed down into Parquet)."""
    path = parquet_path(symbol, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"no data for {symbol} at {path}; run `backtester fetch {symbol}`")
    query = "SELECT * FROM read_parquet(?) WHERE 1=1"
    params: list = [str(path)]
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    query += " ORDER BY date"
    with duckdb.connect() as con:
        df = con.sql(query, params=params).pl()
    return validate_bars(df)


def fetch_yahoo(
    symbol: str,
    start: str = "2005-01-01",
    end: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    """Download daily bars from Yahoo Finance (split/dividend adjusted) and cache to Parquet."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install the yahoo extra: uv sync --extra yahoo") from exc

    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    if hasattr(raw.columns, "levels"):  # yfinance returns MultiIndex columns for one ticker too
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index().rename(columns=str.lower)
    df = pl.from_pandas(raw[BAR_COLUMNS])
    return save_bars(df, symbol, data_dir)


def synthetic_bars(
    n: int = 2520,
    seed: int = 7,
    start: date = date(2015, 1, 1),
    drift: float = 0.07,
    vol: float = 0.18,
) -> pl.DataFrame:
    """Geometric-Brownian-motion bars on business days. Useful for tests and offline demos."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    rets = rng.normal((drift - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(
        rng.normal(0, vol * 0.2 * np.sqrt(dt), n)
    )
    spread = np.abs(rng.normal(0, vol * 0.5 * np.sqrt(dt), n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    dates = pl.date_range(start, date(start.year + 40, 1, 1), "1d", eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5).head(n)
    return pl.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )
