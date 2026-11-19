# Cupofta: backtest strategy analyzer

Backtest trading strategies on daily OHLCV bars, sweep their parameters, and validate them out of sample with walk-forward analysis.

**Stack:** Python 3.11+, Parquet storage, DuckDB queries, Polars dataframes, and a numpy simulation kernel. The UI is built with Streamlit and Plotly. Managed with `uv`, linted with `ruff`, tested with `pytest`.

## UI

```bash
uv run backtester ui          # opens http://localhost:8501
```

| Page | What it does |
|---|---|
| **Backtest** | Run one strategy against buy & hold: metric tiles, equity, drawdown, price with exposure, trades |
| **Sweep** | Grid-search the parameters, with a heatmap when two of them vary, and backtest the best combination in one click |
| **Walk-forward** | Out-of-sample validation: fold returns, plus the stitched OOS equity vs buy & hold |
| **History** | Every saved run. Filter and search notes and tags. Select one row to open it (notes, tags, star, re-run, delete) or several to compare them |
| **Activity** | Log of everything done from the UI or CLI |
| **Data** | Cached symbols and Yahoo fetch |

## Memory

Every backtest, sweep and walk-forward is **saved automatically**, from both the UI and the CLI (pass `--no-save` to skip). Each saved run holds its parameters, settings, metrics, buy-and-hold benchmark, bar-level equity and trades, sweep or fold tables, notes, tags and a star. Re-runs keep a link to the run they came from. Every action (fetch, run, annotate, re-run, delete) is written to an activity log.

Everything is stored in one DuckDB file, `data/backtests.duckdb`. Set `BACKTESTER_DB=/path/to/file.duckdb` to use a different one. It's plain SQL, so you can query it directly:

```bash
duckdb data/backtests.duckdb "SELECT id, symbol, strategy, metrics->>'sharpe' FROM runs ORDER BY created_at DESC"
```

CLI equivalents:

```bash
uv run backtester history [--kind backtest] [--symbol SPY] [--starred]
uv run backtester show RUN_ID
uv run backtester note RUN_ID "worked in 2020 but not after"
uv run backtester tag RUN_ID trend candidate
uv run backtester star RUN_ID            # --off to unstar
uv run backtester rerun RUN_ID           # same settings, current data, linked to the original
uv run backtester delete RUN_ID
uv run backtester actions                # activity log
```

## Quick start

```bash
uv sync --extra yahoo                       # install (yahoo extra = data download)
uv run backtester fetch SPY QQQ             # cache daily bars to data/<SYMBOL>.parquet
uv run backtester run SPY -s sma_cross -p fast=50 -p slow=200
uv run backtester sweep SPY -s sma_cross -g fast=10,20,50 -g slow=100,150,200 --out sweep.csv
uv run backtester walkforward SPY -s donchian -g lookback=20,55,100 -g exit_lookback=10,20
uv run backtester strategies                # list strategies and default params
```

Use the symbol `SYNTH` to run against generated price data, with no download needed.

Common flags: `--start/--end YYYY-MM-DD`, `--commission-bps`, `--slippage-bps`, `--stop-loss 0.08`, `--capital`.

## Layout

| Module | Role |
|---|---|
| `data.py` | Parquet cache, DuckDB loader (date filters pushed down), Yahoo fetch, synthetic bars |
| `strategies.py` | Strategy registry. A strategy is `fn(bars, **params) -> target weights` |
| `engine.py` | `simulate()` (the hot loop) plus the `run_backtest()` Polars wrapper |
| `metrics.py` | CAGR, Sharpe, Sortino, max drawdown, Calmar, exposure, trade stats |
| `analysis.py` | Parameter grid sweep and anchored walk-forward optimisation |
| `store.py` | Run memory and activity log (DuckDB) |
| `service.py` | Run, save and re-run jobs, shared by the CLI and the UI |
| `charts.py` | Plotly figures |
| `ui.py` | Streamlit app |
| `cli.py` | `backtester` command |

## Execution model

- `target[i]` is decided at the **close** of bar *i* and filled at the **open** of bar *i+1*, so a strategy can't trade on information it wouldn't have had yet.
- Commission and slippage are charged in basis points of the traded notional.
- Optional stop-loss: if a bar trades through the stop, the position closes at the stop price, or at the open if price gapped past it. The engine then stays flat until the strategy's signal changes.
- Tests check that every registered strategy is **causal**: cutting off future bars must not change past signals.

## Adding a strategy

```python
from backtester.strategies import strategy


@strategy("my_idea", lookback=20)
def my_idea(
    bars, lookback
): ...  # return np.ndarray of target weights in [-1, 1], len == bars.height
```

## Path to Rust

`engine.simulate` is a pure function: float64 arrays in, arrays out, with no Python objects or callbacks. If profiling shows it is the bottleneck (tick data, huge grids), port that one function to Rust with PyO3/maturin and keep everything else in Python.
