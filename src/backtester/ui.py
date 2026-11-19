"""Streamlit UI for the backtester.  Launch with:  backtester ui   (or: streamlit run ui.py)

Pages
  Backtest      run one strategy vs buy & hold        -> saved to run memory
  Sweep         grid-search parameters (+ heatmap)    -> saved
  Walk-forward  out-of-sample validation              -> saved
  History       browse, annotate, star, re-run, delete and compare saved runs
  Activity      everything that was done, newest first
  Data          cached symbols, fetch new ones
"""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import streamlit as st

from backtester import charts, service, store
from backtester.data import cached_symbols, fetch_yahoo
from backtester.engine import BacktestConfig
from backtester.strategies import REGISTRY, get_strategy

SRC = "ui"
PCT_METRICS = [
    "total_return",
    "cagr",
    "volatility",
    "max_drawdown",
    "exposure",
    "win_rate",
    "avg_trade",
]
METRIC_ORDER = [
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "exposure",
    "trades",
    "win_rate",
    "profit_factor",
    "avg_trade",
    "avg_bars_held",
]
LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "volatility": "Volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar",
    "exposure": "Exposure",
    "trades": "Trades",
    "win_rate": "Win rate",
    "profit_factor": "Profit factor",
    "avg_trade": "Avg trade",
    "avg_bars_held": "Avg bars held",
}

st.set_page_config(page_title="Backtester", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def fmt(key: str, v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and v in (float("inf"), float("-inf")):
        return "∞" if v > 0 else "-∞"
    if key in PCT_METRICS:
        return f"{v:.2%}"
    if key == "trades":
        return f"{int(v)}"
    return f"{v:,.2f}"


def coerce(v: str):
    v = v.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def plot(fig) -> None:
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def symbol_options() -> list[str]:
    try:
        cached = cached_symbols()["symbol"].to_list()
    except Exception:  # noqa: BLE001 - unreadable cache shouldn't break the UI
        cached = []
    return [*cached, service.SYNTH]


def inputs_common(prefix: str) -> dict:
    """Symbol / dates / strategy / costs. Returns kwargs for the service functions."""
    c1, c2, c3 = st.columns([1, 1, 2])
    symbol = c1.selectbox(
        "Symbol",
        symbol_options(),
        key=f"{prefix}_sym",
        help="Cached tickers (see Data page) or SYNTH for generated data",
    )
    strategy = c2.selectbox(
        "Strategy",
        sorted(REGISTRY),
        key=f"{prefix}_strat",
        index=sorted(REGISTRY).index("sma_cross"),
    )
    with c3:
        limit = st.toggle("Limit date range", key=f"{prefix}_limit")
        start = end = None
        if limit:
            d1, d2 = st.columns(2)
            start = d1.date_input("Start", date(2010, 1, 1), key=f"{prefix}_start")
            end = d2.date_input("End", date.today(), key=f"{prefix}_end")
    with st.expander("Costs, risk & notes"):
        k1, k2, k3, k4 = st.columns(4)
        config = BacktestConfig(
            initial_capital=k1.number_input(
                "Capital", 1_000.0, 1e9, 100_000.0, 1_000.0, key=f"{prefix}_cap"
            ),
            commission_bps=k2.number_input(
                "Commission (bps)", 0.0, 100.0, 1.0, 0.5, key=f"{prefix}_comm"
            ),
            slippage_bps=k3.number_input(
                "Slippage (bps)", 0.0, 100.0, 2.0, 0.5, key=f"{prefix}_slip"
            ),
            stop_loss=k4.number_input(
                "Stop-loss (0 = off)", 0.0, 0.9, 0.0, 0.01, key=f"{prefix}_stop", format="%.2f"
            ),
        )
        n1, n2 = st.columns([2, 1])
        notes = n1.text_input("Note", key=f"{prefix}_note", placeholder="Why are you running this?")
        tags = n2.text_input("Tags", key=f"{prefix}_tags", placeholder="comma,separated")
    return dict(
        symbol=symbol,
        strategy=strategy,
        start=start,
        end=end,
        config=config,
        notes=notes,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )


def param_inputs(strategy: str, prefix: str) -> dict:
    defaults = get_strategy(strategy).defaults
    if not defaults:
        st.caption("This strategy has no parameters.")
        return {}
    cols = st.columns(min(len(defaults), 4))
    out = {}
    for i, (k, v) in enumerate(defaults.items()):
        col = cols[i % len(cols)]
        key = f"{prefix}_{strategy}_{k}"
        if isinstance(v, bool):
            out[k] = col.checkbox(k, v, key=key)
        elif isinstance(v, int):
            out[k] = int(col.number_input(k, value=v, step=1, key=key))
        else:
            out[k] = float(col.number_input(k, value=float(v), key=key))
    return out


def grid_inputs(strategy: str, prefix: str) -> dict:
    defaults = get_strategy(strategy).defaults
    if not defaults:
        st.caption("This strategy has no parameters to sweep.")
        return {}
    st.caption("Comma-separated values per parameter. Every combination is tested.")
    cols = st.columns(min(len(defaults), 4))
    grid = {}
    for i, (k, v) in enumerate(defaults.items()):
        raw = cols[i % len(cols)].text_input(k, str(v), key=f"{prefix}_{strategy}_{k}_grid")
        grid[k] = [coerce(x) for x in raw.split(",") if x.strip()]
    n = 1
    for vals in grid.values():
        n *= max(len(vals), 1)
    st.caption(f"{n} combination{'s' if n != 1 else ''}")
    return grid


def run_header(run: dict) -> None:
    star = "★ " if run["starred"] else ""
    st.subheader(f"{star}{run['symbol']} · {run['strategy']} · {run['kind']}")
    params = run["params"] or {}
    bits = [
        f"run `{run['id']}`",
        f"{run['created_at']:%Y-%m-%d %H:%M} via {run['source']}",
        f"{run['start_date']} → {run['end_date']} ({run['n_bars']} bars)",
    ]
    if run["parent_id"]:
        bits.append(f"re-run of `{run['parent_id']}`")
    st.caption(" · ".join(bits))
    st.caption(
        "Params: " + ", ".join(f"{k}={v}" for k, v in params.items() if k != "grid")
        if run["kind"] == "backtest"
        else "Grid: " + json.dumps(params.get("grid", {}))
    )
    if run["tags"]:
        st.caption("Tags: " + ", ".join(run["tags"]))
    if run["notes"]:
        st.info(run["notes"])


def metric_tiles(m: dict, bench: dict | None) -> None:
    keys = ["total_return", "cagr", "sharpe", "max_drawdown", "trades", "win_rate"]
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys, strict=True):
        delta = None
        if (
            bench
            and k in ("total_return", "cagr", "sharpe", "max_drawdown")
            and m.get(k) is not None
        ):
            d = m[k] - bench[k]
            delta = f"{d:+.2%} vs B&H" if k in PCT_METRICS else f"{d:+.2f} vs B&H"
        col.metric(LABELS[k], fmt(k, m.get(k)), delta)


def metrics_table(columns: dict[str, dict]) -> pl.DataFrame:
    """Side-by-side metrics: one column per name."""
    rows = {"metric": [LABELS[k] for k in METRIC_ORDER]}
    for name, m in columns.items():
        rows[name] = [fmt(k, (m or {}).get(k)) for k in METRIC_ORDER]
    return pl.DataFrame(rows)


def curves_for(run: dict, label: str = "Strategy") -> dict[str, pl.DataFrame]:
    s = run["series"]
    curves = {label: s.select("date", "equity")}
    if s.height and s["bench_equity"].null_count() < s.height:
        curves["Buy & hold"] = s.select("date", equity=pl.col("bench_equity"))
    return curves


# ---------------------------------------------------------------------------
# Renderers (shared by the run pages and History)
# ---------------------------------------------------------------------------


def render_backtest(run: dict) -> None:
    metric_tiles(run["metrics"], run["benchmark"])
    curves = curves_for(run)
    plot(charts.equity_chart(curves, title="Equity vs buy & hold"))
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.drawdown_chart(curves))
    with c2:
        plot(charts.price_with_position(run["series"]))
    trades = run["trades"]
    st.markdown(f"**Trades** ({trades.height})")
    if trades.height:
        plot(charts.trade_returns_chart(trades))
        st.dataframe(
            trades.with_columns(
                pl.when(pl.col("direction") > 0)
                .then(pl.lit("long"))
                .otherwise(pl.lit("short"))
                .alias("direction")
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "return": st.column_config.NumberColumn("return", format="percent"),
                "entry_date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "exit_date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "entry_price": st.column_config.NumberColumn(format="%.2f"),
                "exit_price": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    with st.expander("All metrics"):
        st.dataframe(
            metrics_table({"Strategy": run["metrics"], "Buy & hold": run["benchmark"]}),
            hide_index=True,
            width="stretch",
        )


def render_sweep(run: dict, key: str) -> None:
    table = run["table"]
    if table.is_empty():
        st.warning("This sweep has no results.")
        return
    grid = (run["params"] or {}).get("grid", {})
    varied = [k for k, v in grid.items() if len(v) > 1]
    numeric_metrics = [c for c in METRIC_ORDER if c in table.columns]

    best = table.row(0, named=True)
    st.markdown(
        "**Best combination** (by the sort metric): " + ", ".join(f"`{k}={best[k]}`" for k in grid)
    )
    metric_tiles(best, None)

    if len(varied) == 2:
        metric = st.selectbox(
            "Heatmap metric", numeric_metrics, key=f"{key}_hm", format_func=lambda k: LABELS[k]
        )
        plot(charts.sweep_heatmap(table, varied[1], varied[0], metric))
    elif len(varied) > 2:
        st.caption("Heatmap is shown when exactly two parameters vary.")

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            k: st.column_config.NumberColumn(LABELS[k], format="percent")
            for k in PCT_METRICS
            if k in table.columns
        },
    )
    if st.button("Backtest the best combination", key=f"{key}_best"):
        params = {k: best[k] for k in grid}
        job = service.backtest(
            run["symbol"],
            run["strategy"],
            params,
            BacktestConfig(**run["config"]),
            run["start_date"],
            run["end_date"],
            source=SRC,
            parent_id=run["id"],
            notes=f"best of sweep {run['id']}",
            tags=list(run["tags"] or []),
        )
        st.session_state["history_focus"] = job.run_id
        st.success(f"Saved backtest `{job.run_id}`. It's at the top of History.")


def render_walkforward(run: dict) -> None:
    metric_tiles(run["metrics"], run["benchmark"])
    st.caption(
        "Metrics are for the stitched out-of-sample periods only. Each fold uses the "
        "parameters that scored best on all data before it."
    )
    plot(
        charts.equity_chart(
            curves_for(run, "Walk-forward OOS"), title="Out-of-sample equity vs buy & hold"
        )
    )
    folds = run["table"]
    if folds.height:
        plot(charts.folds_chart(folds))
        st.dataframe(folds, width="stretch", hide_index=True)


def render_run(run: dict, key: str) -> None:
    run_header(run)
    if run["kind"] == "backtest":
        render_backtest(run)
    elif run["kind"] == "sweep":
        render_sweep(run, key)
    else:
        render_walkforward(run)


def run_job(fn, **kwargs) -> str | None:
    try:
        with st.spinner("Running…"):
            return fn(source=SRC, **kwargs).run_id
    except FileNotFoundError as e:
        st.error(f"{e}. Fetch the symbol on the Data page first.")
    except Exception as e:  # noqa: BLE001
        st.exception(e)
    return None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_backtest() -> None:
    st.title("Backtest")
    common = inputs_common("bt")
    st.markdown("**Parameters**")
    params = param_inputs(common["strategy"], "bt")
    if st.button("Run & save", type="primary", key="bt_go"):
        run_id = run_job(service.backtest, params=params, **common)
        if run_id:
            st.session_state["bt_last"] = run_id
    if rid := st.session_state.get("bt_last"):
        if run := store.get_run(rid):
            st.divider()
            render_run(run, "bt_view")


def page_sweep() -> None:
    st.title("Parameter sweep")
    common = inputs_common("sw")
    grid = grid_inputs(common["strategy"], "sw")
    sort_by = st.selectbox(
        "Rank by",
        ["sharpe", "sortino", "calmar", "cagr", "total_return"],
        key="sw_sort",
        format_func=lambda k: LABELS[k],
    )
    if st.button("Run sweep & save", type="primary", key="sw_go", disabled=not grid):
        run_id = run_job(service.run_sweep, grid=grid, sort_by=sort_by, **common)
        if run_id:
            st.session_state["sw_last"] = run_id
    if rid := st.session_state.get("sw_last"):
        if run := store.get_run(rid):
            st.divider()
            render_run(run, "sw_view")


def page_walkforward() -> None:
    st.title("Walk-forward")
    common = inputs_common("wf")
    grid = grid_inputs(common["strategy"], "wf")
    c1, c2, c3 = st.columns(3)
    n_splits = c1.number_input("Out-of-sample folds", 2, 20, 5, key="wf_splits")
    train_frac = c2.slider("Initial in-sample share", 0.2, 0.9, 0.6, 0.05, key="wf_train")
    objective = c3.selectbox(
        "Optimise for",
        ["sharpe", "sortino", "calmar", "cagr"],
        key="wf_obj",
        format_func=lambda k: LABELS[k],
    )
    if st.button("Run walk-forward & save", type="primary", key="wf_go", disabled=not grid):
        run_id = run_job(
            service.run_walkforward,
            grid=grid,
            n_splits=int(n_splits),
            train_frac=train_frac,
            objective=objective,
            **common,
        )
        if run_id:
            st.session_state["wf_last"] = run_id
    if rid := st.session_state.get("wf_last"):
        if run := store.get_run(rid):
            st.divider()
            render_run(run, "wf_view")


def annotate_panel(run: dict) -> None:
    with st.form(f"annotate_{run['id']}"):
        notes = st.text_area("Notes", run["notes"] or "", height=100)
        c1, c2 = st.columns([3, 1])
        tags = c1.text_input("Tags", ", ".join(run["tags"] or []))
        starred = c2.checkbox("★ Starred", bool(run["starred"]))
        if st.form_submit_button("Save notes"):
            store.update_run(
                run["id"],
                notes=notes,
                tags=[t.strip() for t in tags.split(",") if t.strip()],
                starred=starred,
                source=SRC,
            )
            st.rerun()
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Re-run on current data", key=f"rerun_{run['id']}"):
        try:
            with st.spinner("Re-running…"):
                new_id = service.rerun(run["id"], source=SRC)
        except Exception as e:  # noqa: BLE001
            st.exception(e)
        else:
            st.session_state["history_focus"] = new_id
            st.rerun()
    with c2.popover("Delete"):
        st.write("Delete this run permanently?")
        if st.button("Yes, delete", type="primary", key=f"del_{run['id']}"):
            store.delete_run(run["id"], source=SRC)
            st.rerun()


def render_compare(runs: list[dict]) -> None:
    runs = runs[: charts.MAX_SERIES]
    st.subheader(f"Comparing {len(runs)} runs")
    names = {r["id"]: f"{r['symbol']} {r['strategy']} ({r['id'][:6]})" for r in runs}
    st.dataframe(
        metrics_table({names[r["id"]]: r["metrics"] for r in runs if r["kind"] != "sweep"}),
        hide_index=True,
        width="stretch",
    )
    curves = {
        names[r["id"]]: r["series"].select("date", "equity") for r in runs if r["series"].height
    }
    if curves:
        plot(charts.equity_chart(curves, indexed=True, title="Equity, indexed to 100"))
        plot(charts.drawdown_chart(curves))
    if any(r["kind"] == "sweep" for r in runs):
        st.caption("Sweeps have no single equity curve and are left out of the charts.")


def page_history() -> None:
    st.title("History")
    st.caption("Every run is saved automatically, from here and from the CLI.")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 2, 1])
    kind = c1.selectbox("Kind", ["all", "backtest", "sweep", "walkforward"], key="h_kind")
    sym = c2.selectbox("Symbol", ["all", *symbol_options()], key="h_sym")
    strat = c3.selectbox("Strategy", ["all", *sorted(REGISTRY)], key="h_strat")
    search = c4.text_input("Search notes & tags", key="h_q")
    starred = c5.toggle("Starred only", key="h_star")

    runs = store.list_runs(
        None if kind == "all" else kind,
        None if sym == "all" else sym,
        None if strat == "all" else strat,
        starred,
    )
    if runs.is_empty():
        st.info("No saved runs match. Run a backtest to get started.")
        return
    if search:
        q = search.lower()
        runs = runs.filter(
            pl.col("notes").str.to_lowercase().str.contains(q, literal=True)
            | pl.col("tags").list.join(",").str.to_lowercase().str.contains(q, literal=True)
        )

    view = runs.select(
        pl.when("starred").then(pl.lit("★")).otherwise(pl.lit("")).alias("★"),
        "id",
        "created_at",
        "kind",
        "symbol",
        "strategy",
        pl.col("params").alias("params"),
        "total_return",
        "bench_return",
        "sharpe",
        "max_drawdown",
        "trades",
        pl.col("tags").list.join(", ").alias("tags"),
        "notes",
    )
    st.caption("Select one row to open it, or several to compare.")
    event = st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="h_table",
        column_config={
            "created_at": st.column_config.DatetimeColumn("created", format="YYYY-MM-DD HH:mm"),
            "total_return": st.column_config.NumberColumn("return", format="percent"),
            "bench_return": st.column_config.NumberColumn("B&H return", format="percent"),
            "sharpe": st.column_config.NumberColumn(format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn("max DD", format="percent"),
            "trades": st.column_config.NumberColumn(format="%d"),
            "params": st.column_config.TextColumn(width="medium"),
        },
    )
    selected = [view["id"][i] for i in event.selection.rows]
    if not selected and (focus := st.session_state.get("history_focus")):
        selected = [focus] if focus in view["id"].to_list() else []

    st.divider()
    if len(selected) == 1:
        run = store.get_run(selected[0])
        if run is None:
            return
        tab_res, tab_notes = st.tabs(["Results", "Notes, re-run & delete"])
        with tab_res:
            render_run(run, f"h_{run['id']}")
        with tab_notes:
            annotate_panel(run)
    elif len(selected) > 1:
        render_compare([r for rid in selected if (r := store.get_run(rid))])


def page_activity() -> None:
    st.title("Activity")
    acts = store.list_actions(1000)
    if acts.is_empty():
        st.info("Nothing has happened yet.")
        return
    kinds = sorted(acts["action"].unique().to_list())
    chosen = st.multiselect("Actions", kinds, default=kinds, key="act_kinds")
    acts = acts.filter(pl.col("action").is_in(chosen))
    st.dataframe(
        acts,
        width="stretch",
        hide_index=True,
        column_config={
            "ts": st.column_config.DatetimeColumn("when", format="YYYY-MM-DD HH:mm:ss"),
            "details": st.column_config.TextColumn(width="large"),
        },
    )


def page_data() -> None:
    st.title("Data")
    cached = cached_symbols()
    if cached.is_empty():
        st.info("No cached symbols yet. Fetch some below, or use SYNTH.")
    else:
        st.dataframe(cached, width="stretch", hide_index=True)
    with st.form("fetch"):
        c1, c2 = st.columns([3, 1])
        syms = c1.text_input("Tickers to fetch from Yahoo", placeholder="SPY, QQQ, AAPL")
        start = c2.date_input("From", date(2005, 1, 1))
        if st.form_submit_button("Fetch", type="primary"):
            for s in [x.strip().upper() for x in syms.split(",") if x.strip()]:
                try:
                    with st.spinner(f"Fetching {s}…"):
                        path = fetch_yahoo(s, start=str(start))
                    store.log_action("fetch", {"symbol": s, "path": str(path)}, source=SRC)
                    st.success(f"{s} saved to {path}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"{s}: {e}")


nav = st.navigation(
    {
        "Run": [
            st.Page(page_backtest, title="Backtest", icon="📈", default=True),
            st.Page(page_sweep, title="Sweep", icon="🧮"),
            st.Page(page_walkforward, title="Walk-forward", icon="🚶"),
        ],
        "Memory": [
            st.Page(page_history, title="History", icon="🗂️"),
            st.Page(page_activity, title="Activity", icon="📜"),
        ],
        "Setup": [st.Page(page_data, title="Data", icon="💾")],
    }
)
nav.run()
