"""Command-line interface.

  backtester ui                               launch the web UI
  backtester fetch SPY QQQ
  backtester run SPY --strategy sma_cross -p fast=20 -p slow=100 --tag trend
  backtester sweep SPY --strategy sma_cross -g fast=10,20,50 -g slow=100,150,200
  backtester walkforward SPY --strategy sma_cross -g fast=10,20,50 -g slow=100,200
  backtester history [--kind backtest] [--symbol SPY]
  backtester show RUN_ID | note RUN_ID "text" | star RUN_ID | rerun RUN_ID | delete RUN_ID
  backtester actions
  backtester strategies

Every run, sweep and walk-forward is saved to the run memory (data/backtests.duckdb) unless
--no-save is given. Use the symbol SYNTH to run against generated data with no network access.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import polars as pl

from backtester import service, store
from backtester.data import DEFAULT_DATA_DIR, fetch_yahoo
from backtester.engine import BacktestConfig
from backtester.metrics import format_metrics
from backtester.strategies import REGISTRY

TABLE_CFG = dict(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=200)


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


def _config(args) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_loss=args.stop_loss,
    )


def _save_kwargs(args) -> dict:
    return dict(
        save=not args.no_save,
        source="cli",
        notes=args.note or "",
        tags=args.tag or [],
        data_dir=Path(args.data_dir),
    )


def _saved(run_id: str | None) -> None:
    if run_id:
        print(f"\nsaved as run {run_id}  (backtester show {run_id})")


def _header(bars: pl.DataFrame, symbol: str, strategy: str) -> str:
    return f"{symbol}  {strategy}  {bars['date'][0]} → {bars['date'][-1]}  ({bars.height} bars)"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_fetch(args) -> None:
    for sym in args.symbols:
        path = fetch_yahoo(
            sym,
            start=str(args.start or "2005-01-01"),
            end=str(args.end) if args.end else None,
            data_dir=Path(args.data_dir),
        )
        store.log_action("fetch", {"symbol": sym.upper(), "path": str(path)})
        print(f"saved {sym} → {path}")


def cmd_run(args) -> None:
    params = _parse_kv(args.param, multi=False)
    job = service.backtest(
        args.symbol,
        args.strategy,
        params,
        _config(args),
        args.start,
        args.end,
        **_save_kwargs(args),
    )
    res = job.result
    print(_header(res.bars, args.symbol, args.strategy), res.params)
    print("\nStrategy\n" + format_metrics(job.metrics))
    print("\nBuy & hold\n" + format_metrics(job.benchmark))
    if args.trades:
        res.trades.select("entry_date", "exit_date", "direction", "return", "bars_held").write_csv(
            args.trades
        )
        print(f"\ntrades → {args.trades}")
    if args.equity:
        res.bars.write_csv(args.equity)
        print(f"equity → {args.equity}")
    _saved(job.run_id)


def cmd_sweep(args) -> None:
    grid = _parse_kv(args.grid, multi=True)
    job = service.run_sweep(
        args.symbol,
        args.strategy,
        grid,
        _config(args),
        args.start,
        args.end,
        args.sort,
        **_save_kwargs(args),
    )
    print(_header(job.bars, args.symbol, args.strategy))
    with pl.Config(**{**TABLE_CFG, "tbl_rows": args.top}):
        print(job.table.head(args.top))
    if args.out:
        job.table.write_csv(args.out)
        print(f"\nfull table → {args.out}")
    _saved(job.run_id)


def cmd_walkforward(args) -> None:
    grid = _parse_kv(args.grid, multi=True)
    job = service.run_walkforward(
        args.symbol,
        args.strategy,
        grid,
        args.splits,
        args.train_frac,
        args.objective,
        _config(args),
        args.start,
        args.end,
        **_save_kwargs(args),
    )
    with pl.Config(**TABLE_CFG):
        print(job.folds)
    print(f"\nStitched out-of-sample: {job.oos['date'][0]} → {job.oos['date'][-1]}")
    print(format_metrics(job.metrics))
    print(f"\n  buy & hold over the same period: {job.benchmark['total_return']:.2%}")
    _saved(job.run_id)


def cmd_history(args) -> None:
    runs = store.list_runs(args.kind, args.symbol, args.strategy, args.starred, args.limit)
    if runs.is_empty():
        print("no saved runs yet")
        return
    view = runs.select(
        "id",
        pl.col("created_at").dt.strftime("%Y-%m-%d %H:%M").alias("created"),
        "kind",
        "symbol",
        "strategy",
        pl.col("params").str.slice(0, 40),
        "total_return",
        "sharpe",
        "max_drawdown",
        pl.when("starred").then(pl.lit("★")).otherwise(pl.lit("")).alias("★"),
        pl.col("tags").list.join(","),
        pl.col("notes").str.slice(0, 30),
    )
    with pl.Config(**TABLE_CFG):
        print(view)


def cmd_show(args) -> None:
    run = store.get_run(args.run_id)
    if run is None:
        raise SystemExit(f"no run {args.run_id}")
    print(
        f"run {run['id']}  [{run['kind']}]  created {run['created_at']:%Y-%m-%d %H:%M}"
        f"  via {run['source']}"
    )
    print(
        f"{run['symbol']}  {run['strategy']}  {run['start_date']} → {run['end_date']}"
        f"  ({run['n_bars']} bars)"
    )
    print(f"params  {run['params']}")
    print(f"config  {run['config']}")
    if run["tags"]:
        print(f"tags    {', '.join(run['tags'])}")
    if run["notes"]:
        print(f"notes   {run['notes']}")
    if run["parent_id"]:
        print(f"re-run of {run['parent_id']}")
    if run["kind"] == "sweep":
        with pl.Config(**{**TABLE_CFG, "tbl_rows": 10}):
            print("\nTop of grid:\n", run["table"].head(10))
        return
    print("\nMetrics\n" + format_metrics(run["metrics"]))
    if run["benchmark"]:
        print("\nBuy & hold\n" + format_metrics(run["benchmark"]))
    if run["kind"] == "walkforward":
        with pl.Config(**TABLE_CFG):
            print("\nFolds\n", run["table"])


def cmd_note(args) -> None:
    store.update_run(args.run_id, notes=args.text)
    print(f"noted {args.run_id}")


def cmd_tag(args) -> None:
    store.update_run(args.run_id, tags=args.tags)
    print(f"tagged {args.run_id}: {', '.join(args.tags)}")


def cmd_star(args) -> None:
    store.update_run(args.run_id, starred=not args.off)
    print(f"{'un' if args.off else ''}starred {args.run_id}")


def cmd_rerun(args) -> None:
    new_id = service.rerun(args.run_id, source="cli")
    print(f"re-ran {args.run_id} → new run {new_id}")


def cmd_delete(args) -> None:
    store.delete_run(args.run_id)
    print(f"deleted {args.run_id}")


def cmd_actions(args) -> None:
    acts = store.list_actions(args.limit)
    with pl.Config(**{**TABLE_CFG, "fmt_str_lengths": 80}):
        print(acts.with_columns(pl.col("ts").dt.strftime("%Y-%m-%d %H:%M:%S")))


def cmd_strategies(_args) -> None:
    for name, spec in sorted(REGISTRY.items()):
        print(f"{name:<15} {spec.defaults}")


def cmd_ui(args) -> None:
    app = Path(__file__).with_name("ui.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--theme.primaryColor",
        "#2a78d6",
        "--browser.gatherUsageStats",
        "false",
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]
    subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backtester",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, symbol=True):
        if symbol:
            sp.add_argument("symbol", help="ticker with cached data, or SYNTH")
        sp.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
        sp.add_argument("--start", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
        sp.add_argument("--end", type=date.fromisoformat, default=None, help="YYYY-MM-DD")

    def job(sp):
        common(sp)
        sp.add_argument("--strategy", "-s", default="sma_cross", choices=sorted(REGISTRY))
        sp.add_argument("--capital", type=float, default=100_000.0)
        sp.add_argument("--commission-bps", type=float, default=1.0)
        sp.add_argument("--slippage-bps", type=float, default=2.0)
        sp.add_argument("--stop-loss", type=float, default=0.0, help="fraction, e.g. 0.08")
        sp.add_argument("--no-save", action="store_true", help="don't save to run memory")
        sp.add_argument("--note", help="note to attach to the saved run")
        sp.add_argument("--tag", action="append", help="tag the saved run (repeatable)")

    f = sub.add_parser("fetch", help="download daily bars from Yahoo into Parquet")
    f.add_argument("symbols", nargs="+")
    common(f, symbol=False)
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="single backtest vs buy & hold")
    job(r)
    r.add_argument("--param", "-p", action="append", help="key=value")
    r.add_argument("--trades", help="write trades CSV")
    r.add_argument("--equity", help="write bar-level equity CSV")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="grid search over parameters")
    job(s)
    s.add_argument("--grid", "-g", action="append", required=True, help="key=v1,v2,...")
    s.add_argument("--sort", default="sharpe")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--out", help="write full results CSV")
    s.set_defaults(func=cmd_sweep)

    w = sub.add_parser("walkforward", help="anchored walk-forward optimisation")
    job(w)
    w.add_argument("--grid", "-g", action="append", required=True, help="key=v1,v2,...")
    w.add_argument("--splits", type=int, default=5)
    w.add_argument("--train-frac", type=float, default=0.6)
    w.add_argument("--objective", default="sharpe")
    w.set_defaults(func=cmd_walkforward)

    h = sub.add_parser("history", help="list saved runs")
    h.add_argument("--kind", choices=["backtest", "sweep", "walkforward"])
    h.add_argument("--symbol")
    h.add_argument("--strategy")
    h.add_argument("--starred", action="store_true")
    h.add_argument("--limit", type=int, default=50)
    h.set_defaults(func=cmd_history)

    for name, fn, helptext in [
        ("show", cmd_show, "show a saved run"),
        ("rerun", cmd_rerun, "re-run a saved run on current data"),
        ("delete", cmd_delete, "delete a saved run"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("run_id")
        sp.set_defaults(func=fn)

    n = sub.add_parser("note", help="set the note on a saved run")
    n.add_argument("run_id")
    n.add_argument("text")
    n.set_defaults(func=cmd_note)

    t = sub.add_parser("tag", help="set the tags on a saved run")
    t.add_argument("run_id")
    t.add_argument("tags", nargs="+")
    t.set_defaults(func=cmd_tag)

    st = sub.add_parser("star", help="star (or --off to unstar) a saved run")
    st.add_argument("run_id")
    st.add_argument("--off", action="store_true")
    st.set_defaults(func=cmd_star)

    a = sub.add_parser("actions", help="show the activity log")
    a.add_argument("--limit", type=int, default=50)
    a.set_defaults(func=cmd_actions)

    ls = sub.add_parser("strategies", help="list registered strategies")
    ls.set_defaults(func=cmd_strategies)

    u = sub.add_parser("ui", help="launch the web UI")
    u.add_argument("--port", type=int, default=8501)
    u.add_argument("--headless", action="store_true")
    u.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
