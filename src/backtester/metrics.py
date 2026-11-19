"""Performance metrics computed from a BacktestResult."""

from __future__ import annotations

import math

import numpy as np

from backtester.engine import BacktestResult

PERIODS_PER_YEAR = 252


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def compute_metrics(result: BacktestResult, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    equity = result.bars["equity"].to_numpy()
    rets = result.bars["returns"].to_numpy()
    initial = result.config.initial_capital
    n = len(equity)
    years = n / periods_per_year

    total_return = equity[-1] / initial - 1.0
    cagr = (equity[-1] / initial) ** (1 / years) - 1.0 if years > 0 and equity[-1] > 0 else -1.0
    std = rets.std(ddof=1) if n > 1 else 0.0
    downside = rets[rets < 0]
    dstd = math.sqrt(np.mean(downside**2)) if downside.size else 0.0
    ann = math.sqrt(periods_per_year)
    sharpe = rets.mean() / std * ann if std > 0 else 0.0
    sortino = rets.mean() / dstd * ann if dstd > 0 else 0.0
    mdd = max_drawdown(np.concatenate([[initial], equity]))
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    trades = result.trades
    n_trades = trades.height
    if n_trades:
        tr = trades["return"].to_numpy()
        wins, losses = tr[tr > 0], tr[tr <= 0]
        win_rate = wins.size / n_trades
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else math.inf
        avg_trade = float(tr.mean())
        avg_bars = float(trades["bars_held"].mean())
    else:
        win_rate = profit_factor = avg_trade = avg_bars = 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(std * ann),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(calmar),
        "exposure": float(np.mean(result.bars["position"].to_numpy() != 0)),
        "trades": n_trades,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_trade": avg_trade,
        "avg_bars_held": avg_bars,
    }


_PCT = {"total_return", "cagr", "volatility", "max_drawdown", "exposure", "win_rate", "avg_trade"}


def format_metrics(m: dict) -> str:
    lines = []
    for k, v in m.items():
        if k in _PCT:
            s = f"{v:>10.2%}"
        elif isinstance(v, int):
            s = f"{v:>10d}"
        else:
            s = f"{v:>10.2f}"
        lines.append(f"  {k:<15}{s}")
    return "\n".join(lines)
