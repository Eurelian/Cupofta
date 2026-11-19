"""Simulation engine.

`simulate` is the hot path. It is deliberately a pure function over flat float64 arrays
with no Polars, no objects and no Python callbacks, so it can be ported 1:1 to Rust
(PyO3/maturin) or JIT-compiled later without touching anything else.

Execution model (no lookahead):
  * `target[i]` is the desired portfolio weight decided at the CLOSE of bar i, using data <= i.
  * It is executed at the OPEN of bar i+1.
  * Costs (commission + slippage) are charged in bps of traded notional.
  * Optional stop-loss: if the bar's low (long) / high (short) breaches entry*(1 -/+ stop),
    the position is closed at the stop price (or the open, if it gapped through). After a
    stop the engine stays flat until the target changes value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# Columns of the trades matrix returned by `simulate`.
TRADE_FIELDS = ("entry_idx", "exit_idx", "direction", "entry_price", "exit_price", "return")


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    stop_loss: float = 0.0  # fraction, e.g. 0.08 = 8% stop; 0 disables

    @property
    def cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


def simulate(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    target: np.ndarray,
    initial_capital: float,
    cost_bps: float,
    stop_loss: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the bar loop.

    Returns:
        equity:   (n,) mark-to-market equity at each close
        position: (n,) portfolio weight held during each bar (after the open's execution)
        trades:   (k, 6) closed round trips, columns = TRADE_FIELDS
    """
    n = close.shape[0]
    equity = np.empty(n)
    position = np.zeros(n)
    trades: list[tuple[float, float, float, float, float, float]] = []

    cost_rate = cost_bps / 1e4
    cash = initial_capital
    units = 0.0
    held = 0.0  # target weight currently held
    blocked = np.nan  # target value suppressed after a stop-out
    entry_idx = -1
    entry_price = 0.0

    def close_trade(i: int, price: float) -> None:
        direction = np.sign(held)
        ret = direction * (price / entry_price - 1.0)
        trades.append((entry_idx, i, direction, entry_price, price, ret))

    for i in range(n):
        # 1) Execute yesterday's decision at today's open.
        if i > 0:
            desired = target[i - 1]
            if np.isnan(desired):
                desired = 0.0
            if desired == blocked:
                desired = 0.0
            else:
                blocked = np.nan
            if desired != held:
                px = open_[i]
                value = cash + units * px
                new_units = desired * value / px
                traded = new_units - units
                cash -= traded * px + abs(traded) * px * cost_rate
                if held != 0.0 and (desired == 0.0 or np.sign(desired) != np.sign(held)):
                    close_trade(i, px)
                if desired != 0.0 and (held == 0.0 or np.sign(desired) != np.sign(held)):
                    entry_idx, entry_price = i, px
                units, held = new_units, desired

        # 2) Intrabar stop-loss.
        if stop_loss > 0.0 and held != 0.0:
            if held > 0.0:
                stop_px = entry_price * (1.0 - stop_loss)
                hit = low[i] <= stop_px
                fill = min(open_[i], stop_px) if hit else 0.0
            else:
                stop_px = entry_price * (1.0 + stop_loss)
                hit = high[i] >= stop_px
                fill = max(open_[i], stop_px) if hit else 0.0
            if hit:
                cash += units * fill - abs(units) * fill * cost_rate
                close_trade(i, fill)
                blocked = held
                units, held = 0.0, 0.0

        position[i] = held
        equity[i] = cash + units * close[i]

    # Mark any open position as a trade closed at the final close (for trade stats only).
    if held != 0.0:
        close_trade(n - 1, close[n - 1])

    trades_arr = np.array(trades, dtype=np.float64).reshape(-1, len(TRADE_FIELDS))
    return equity, position, trades_arr


@dataclass
class BacktestResult:
    bars: pl.DataFrame  # input bars + target, position, equity, returns
    trades: pl.DataFrame
    config: BacktestConfig
    params: dict = field(default_factory=dict)

    @property
    def equity(self) -> pl.Series:
        return self.bars["equity"]


def run_backtest(
    bars: pl.DataFrame,
    target: np.ndarray,
    config: BacktestConfig | None = None,
    params: dict | None = None,
) -> BacktestResult:
    """Wrap `simulate` with Polars in/out."""
    config = config or BacktestConfig()
    target = np.asarray(target, dtype=np.float64)
    if target.shape[0] != bars.height:
        raise ValueError("target length must match number of bars")

    equity, position, trades = simulate(
        bars["open"].to_numpy().astype(np.float64),
        bars["high"].to_numpy().astype(np.float64),
        bars["low"].to_numpy().astype(np.float64),
        bars["close"].to_numpy().astype(np.float64),
        target,
        config.initial_capital,
        config.cost_bps,
        config.stop_loss,
    )

    out = bars.with_columns(
        pl.Series("target", target),
        pl.Series("position", position),
        pl.Series("equity", equity),
    ).with_columns(
        returns=pl.col("equity").pct_change().fill_null(equity[0] / config.initial_capital - 1)
    )

    dates = bars["date"]
    trades_df = pl.DataFrame(
        {name: trades[:, j] for j, name in enumerate(TRADE_FIELDS)},
        schema={name: pl.Float64 for name in TRADE_FIELDS},
    ).with_columns(pl.col("entry_idx", "exit_idx").cast(pl.Int64))
    if trades_df.height:
        trades_df = trades_df.with_columns(
            entry_date=pl.Series([dates[int(k)] for k in trades_df["entry_idx"]]),
            exit_date=pl.Series([dates[int(k)] for k in trades_df["exit_idx"]]),
            bars_held=pl.col("exit_idx") - pl.col("entry_idx"),
        )
    return BacktestResult(bars=out, trades=trades_df, config=config, params=params or {})
