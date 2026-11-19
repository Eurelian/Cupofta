import numpy as np
import polars as pl
import pytest

from backtester.analysis import sweep, walk_forward
from backtester.data import load_bars, save_bars, synthetic_bars
from backtester.engine import BacktestConfig, run_backtest
from backtester.metrics import compute_metrics, max_drawdown
from backtester.strategies import REGISTRY, generate

FREE = BacktestConfig(commission_bps=0, slippage_bps=0)


def flat_bars(prices, opens=None):
    prices = np.asarray(prices, dtype=float)
    opens = prices if opens is None else np.asarray(opens, dtype=float)
    n = len(prices)
    return pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, n), "1d", eager=True),
            "open": opens,
            "high": np.maximum(opens, prices),
            "low": np.minimum(opens, prices),
            "close": prices,
            "volume": np.ones(n),
        }
    )


def test_buy_hold_matches_price_return_from_first_execution():
    bars = synthetic_bars(500)
    res = run_backtest(bars, np.ones(bars.height), FREE)
    # Signal at close of bar 0 executes at open of bar 1.
    expected = bars["close"][-1] / bars["open"][1]
    assert res.equity[-1] / FREE.initial_capital == pytest.approx(expected)


def test_no_lookahead_signal_executes_next_open():
    # Price jumps at bar 2. A signal raised at close of bar 2 must not capture that jump.
    bars = flat_bars([100, 100, 200, 200, 200])
    target = np.array([0, 0, 1, 1, 1], dtype=float)
    res = run_backtest(bars, target, FREE)
    assert res.equity[-1] == pytest.approx(FREE.initial_capital)
    assert res.bars["position"].to_list() == [0, 0, 0, 1, 1]


def test_costs_are_charged_on_both_sides():
    bars = flat_bars([100] * 5)
    cfg = BacktestConfig(commission_bps=5, slippage_bps=5)
    target = np.array([1, 1, 0, 0, 0], dtype=float)
    res = run_backtest(bars, target, cfg)
    # 10 bps of notional on entry and on exit
    assert res.equity[-1] == pytest.approx(cfg.initial_capital - 100 - 100)
    assert res.trades.height == 1


def test_short_profits_when_price_falls():
    bars = flat_bars([100, 100, 90, 80])
    res = run_backtest(bars, np.full(4, -1.0), FREE)
    assert res.equity[-1] == pytest.approx(FREE.initial_capital * 1.2)
    assert res.trades["direction"][0] == -1


def test_stop_loss_exits_and_blocks_reentry():
    # Bar 3 opens at 100 and trades down to 85 intrabar -> stop fills at 90.
    bars = flat_bars([100.0, 100, 100, 85, 90, 95], opens=[100.0, 100, 100, 100, 90, 95])
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0, stop_loss=0.1)
    res = run_backtest(bars, np.ones(6), cfg)
    assert res.trades.height == 1
    assert res.trades["exit_price"][0] == pytest.approx(90.0)  # stop at entry*0.9
    assert res.bars["position"].to_list()[-2:] == [0, 0]  # stays flat while target unchanged
    assert res.equity[-1] == pytest.approx(cfg.initial_capital * 0.9)


def test_stop_loss_gap_fills_at_open():
    bars = flat_bars([100.0, 100, 100, 80], opens=[100.0, 100, 100, 80])
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0, stop_loss=0.1)
    res = run_backtest(bars, np.ones(4), cfg)
    assert res.trades["exit_price"][0] == pytest.approx(80.0)


def test_flip_long_to_short_records_trade():
    bars = flat_bars([100, 110, 100, 90])
    res = run_backtest(bars, np.array([1, -1, -1, -1.0]), FREE)
    assert res.trades.height == 2
    assert res.trades["direction"].to_list() == [1, -1]


def test_metrics_shape_and_drawdown():
    assert max_drawdown(np.array([100, 120, 90, 130])) == pytest.approx(-0.25)
    res = run_backtest(synthetic_bars(), generate("sma_cross", synthetic_bars()))
    m = compute_metrics(res)
    assert -1 <= m["max_drawdown"] <= 0
    assert 0 <= m["exposure"] <= 1
    assert m["trades"] == res.trades.height


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_strategies_are_causal(name):
    """Truncating future data must not change past signals."""
    bars = synthetic_bars(800)
    full = generate(name, bars)
    part = generate(name, bars.head(500))
    assert len(full) == bars.height
    np.testing.assert_array_equal(full[:500], part)


def test_parquet_roundtrip_with_date_filter(tmp_path):
    bars = synthetic_bars(300)
    save_bars(bars, "test", tmp_path)
    loaded = load_bars("TEST", tmp_path, start=bars["date"][100])
    assert loaded.height == 200
    assert loaded["close"].to_list() == pytest.approx(bars["close"][100:].to_list())


def test_sweep_and_walk_forward_run():
    bars = synthetic_bars(1500)
    table = sweep(bars, "sma_cross", {"fast": [10, 20], "slow": [50, 100]})
    assert table.height == 4
    folds, oos = walk_forward(bars, "sma_cross", {"fast": [10, 20], "slow": [50, 100]}, n_splits=3)
    assert folds.height == 3
    assert oos["date"].is_sorted() and oos["date"].n_unique() == oos.height
    assert oos["date"][0] == bars["date"][int(1500 * 0.6)]
