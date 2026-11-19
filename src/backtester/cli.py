"""Command-line interface.

  backtester fetch SPY QQQ
  backtester run SPY --strategy sma_cross -p fast=20 -p slow=100
  backtester sweep SPY --strategy sma_cross -g fast=10,20,50 -g slow=100,150,200
  backtester walkforward SPY --strategy sma_cross -g fast=10,20,50 -g slow=100,200
  backtester strategies

Use the symbol SYNTH to run against generated data with no network access.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

from backtester.analysis import sweep, walk_forward
from backtester.data import DEFAULT_DATA_DIR, fetch_yahoo, load_bars, synthetic_bars
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics, format_metrics
from backtester.strategies import REGISTRY, generate


def _coerce(v: str):
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _parse_kv(items: list[str], multi: bool) -> dict:
    out = {}
    for item in items or []:
        key, _, val = item.partition("=")
        if not val:
            raise SystemExit(f"expected key=value, got {item!r}")
        out[key] = [_coerce(x) for x in val.split(",")] if multi else _coerce(val)
    return out


def _bars(args) -> pl.DataFrame:
    if args.symbol.upper() == "SYNTH":
        return synthetic_bars()
    return load_bars(args.symbol, Path(args.data_dir), args.start, args.end)


def _config(args) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_loss=args.stop_loss,
    )


def _header(bars: pl.DataFrame, symbol: str, strategy: str) -> str:
    return f"{symbol}  {strategy}  {bars['date'][0]} → {bars['date'][-1]}  ({bars.height} bars)"


def cmd_fetch(args) -> None:
    for sym in args.symbols:
        path = fetch_yahoo(
            sym,
            start=str(args.start or "2005-01-01"),
            end=str(args.end) if args.end else None,
            data_dir=Path(args.data_dir),
        )
        print(f"saved {sym} → {path}")


def cmd_run(args) -> None:
    bars = _bars(args)
    params = _parse_kv(args.param, multi=False)
    res = run_backtest(bars, generate(args.strategy, bars, **params), _config(args), params)
    bench = run_backtest(bars, generate("buy_hold", bars), _config(args))
    m, b = compute_metrics(res), compute_metrics(bench)
    print(_header(bars, args.symbol, args.strategy), params or "")
    print("\nStrategy\n" + format_metrics(m))
    print("\nBuy & hold\n" + format_metrics(b))
    if args.trades:
        res.trades.select("entry_date", "exit_date", "direction", "return", "bars_held").write_csv(
            args.trades
        )
        print(f"\ntrades → {args.trades}")
    if args.equity:
        res.bars.write_csv(args.equity)
        print(f"equity → {args.equity}")


def cmd_sweep(args) -> None:
    bars = _bars(args)
    grid = _parse_kv(args.grid, multi=True)
    table = sweep(bars, args.strategy, grid, _config(args), sort_by=args.sort)
    print(_header(bars, args.symbol, args.strategy))
    with pl.Config(tbl_rows=args.top, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print(table.head(args.top))
    if args.out:
        table.write_csv(args.out)
        print(f"\nfull table → {args.out}")


def cmd_walkforward(args) -> None:
    bars = _bars(args)
    grid = _parse_kv(args.grid, multi=True)
    folds, oos = walk_forward(
        bars, args.strategy, grid, args.splits, args.train_frac, _config(args), args.objective
    )
    print(_header(bars, args.symbol, args.strategy))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print(folds)
    eq = oos["equity"]
    print(
        f"\nStitched out-of-sample: {oos['date'][0]} → {oos['date'][-1]}, "
        f"return {eq[-1] / args.capital - 1:.2%}"
    )


def cmd_strategies(_args) -> None:
    for name, spec in sorted(REGISTRY.items()):
        print(f"{name:<15} {spec.defaults}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backtester", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, symbol=True):
        if symbol:
            sp.add_argument("symbol", help="ticker with cached data, or SYNTH")
        sp.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
        sp.add_argument("--start", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
        sp.add_argument("--end", type=date.fromisoformat, default=None, help="YYYY-MM-DD")

    def costs(sp):
        sp.add_argument("--strategy", "-s", default="sma_cross", choices=sorted(REGISTRY))
        sp.add_argument("--capital", type=float, default=100_000.0)
        sp.add_argument("--commission-bps", type=float, default=1.0)
        sp.add_argument("--slippage-bps", type=float, default=2.0)
        sp.add_argument("--stop-loss", type=float, default=0.0, help="fraction, e.g. 0.08")

    f = sub.add_parser("fetch", help="download daily bars from Yahoo into Parquet")
    f.add_argument("symbols", nargs="+")
    common(f, symbol=False)
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="single backtest vs buy & hold")
    common(r)
    costs(r)
    r.add_argument("--param", "-p", action="append", help="key=value")
    r.add_argument("--trades", help="write trades CSV")
    r.add_argument("--equity", help="write bar-level equity CSV")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="grid search over parameters")
    common(s)
    costs(s)
    s.add_argument("--grid", "-g", action="append", required=True, help="key=v1,v2,...")
    s.add_argument("--sort", default="sharpe")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--out", help="write full results CSV")
    s.set_defaults(func=cmd_sweep)

    w = sub.add_parser("walkforward", help="anchored walk-forward optimisation")
    common(w)
    costs(w)
    w.add_argument("--grid", "-g", action="append", required=True, help="key=v1,v2,...")
    w.add_argument("--splits", type=int, default=5)
    w.add_argument("--train-frac", type=float, default=0.6)
    w.add_argument("--objective", default="sharpe")
    w.set_defaults(func=cmd_walkforward)

    ls = sub.add_parser("strategies", help="list registered strategies")
    ls.set_defaults(func=cmd_strategies)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
