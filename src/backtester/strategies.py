"""Strategies turn bars into a target-weight array.

Contract: `fn(bars: pl.DataFrame, **params) -> np.ndarray` of length bars.height, where
value i is the desired portfolio weight (-1..1) decided at the close of bar i using only
data up to and including bar i. The engine executes it at the next open.

Register new strategies with @strategy("name", default=..., ...).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

StrategyFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    fn: StrategyFn
    defaults: dict


REGISTRY: dict[str, StrategySpec] = {}


def strategy(name: str, **defaults):
    def deco(fn: StrategyFn) -> StrategyFn:
        REGISTRY[name] = StrategySpec(name, fn, defaults)
        return fn

    return deco


def get_strategy(name: str) -> StrategySpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}") from None


def generate(name: str, bars: pl.DataFrame, **params) -> np.ndarray:
    spec = get_strategy(name)
    merged = {**spec.defaults, **params}
    return spec.fn(bars, **merged)


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


@strategy("buy_hold")
def buy_hold(bars: pl.DataFrame) -> np.ndarray:
    return np.ones(bars.height)


@strategy("sma_cross", fast=20, slow=100, allow_short=False)
def sma_cross(bars: pl.DataFrame, fast: int, slow: int, allow_short: bool) -> np.ndarray:
    """Long when fast SMA > slow SMA; flat (or short) otherwise."""
    fast, slow = int(fast), int(slow)
    if fast >= slow:
        return np.zeros(bars.height)
    sig = bars.select(
        f=pl.col("close").rolling_mean(fast),
        s=pl.col("close").rolling_mean(slow),
    ).select(
        pl.when(pl.col("s").is_null())
        .then(0.0)
        .when(pl.col("f") > pl.col("s"))
        .then(1.0)
        .otherwise(-1.0 if allow_short else 0.0)
    )
    return sig.to_series().to_numpy()


def rsi(close: pl.Expr, period: int) -> pl.Expr:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower_bound=0).ewm_mean(alpha=1 / period, adjust=False, min_samples=period)
    loss = (-delta).clip(lower_bound=0).ewm_mean(alpha=1 / period, adjust=False, min_samples=period)
    return 100 - 100 / (1 + gain / loss)


@strategy("rsi_reversion", period=14, entry=30.0, exit=55.0, trend=200)
def rsi_reversion(
    bars: pl.DataFrame, period: int, entry: float, exit: float, trend: int
) -> np.ndarray:
    """Buy oversold dips (RSI < entry) while above the trend SMA; sell when RSI > exit.

    Stateful (hold until exit), so the state machine is a small loop over precomputed series.
    """
    cols = bars.select(
        r=rsi(pl.col("close"), int(period)),
        up=(pl.col("close") > pl.col("close").rolling_mean(int(trend))).fill_null(False)
        if trend
        else pl.lit(True),
    )
    r = cols["r"].fill_null(50.0).fill_nan(50.0).to_numpy()
    up = cols["up"].to_numpy()
    out = np.zeros(bars.height)
    held = 0.0
    for i in range(bars.height):
        if held == 0.0 and r[i] < entry and up[i]:
            held = 1.0
        elif held == 1.0 and r[i] > exit:
            held = 0.0
        out[i] = held
    return out


@strategy("donchian", lookback=55, exit_lookback=20)
def donchian(bars: pl.DataFrame, lookback: int, exit_lookback: int) -> np.ndarray:
    """Turtle-style breakout: long on a new N-day high, exit on a new M-day low."""
    cols = bars.select(
        c=pl.col("close"),
        hi=pl.col("high").rolling_max(int(lookback)).shift(1),
        lo=pl.col("low").rolling_min(int(exit_lookback)).shift(1),
    )
    c, hi, lo = (cols[k].to_numpy() for k in ("c", "hi", "lo"))
    out = np.zeros(bars.height)
    held = 0.0
    for i in range(bars.height):
        if held == 0.0 and not np.isnan(hi[i]) and c[i] > hi[i]:
            held = 1.0
        elif held == 1.0 and not np.isnan(lo[i]) and c[i] < lo[i]:
            held = 0.0
        out[i] = held
    return out
