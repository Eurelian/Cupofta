import pytest

from backtester import charts, service, store


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTESTER_DB", str(tmp_path / "runs.duckdb"))


def test_backtest_is_saved_and_reloaded():
    job = service.backtest("SYNTH", "sma_cross", {"fast": 10}, tags=["t1"], notes="hello")
    run = store.get_run(job.run_id)
    assert run["kind"] == "backtest"
    assert run["params"]["fast"] == 10 and run["params"]["slow"] == 100
    assert run["tags"] == ["t1"] and run["notes"] == "hello"
    assert run["series"].height == job.result.bars.height
    assert run["series"]["bench_equity"].null_count() == 0
    assert run["trades"].height == job.result.trades.height
    assert run["metrics"]["sharpe"] == pytest.approx(job.metrics["sharpe"])


def test_no_save_writes_nothing():
    service.backtest("SYNTH", "buy_hold", save=False)
    assert store.list_runs().is_empty()


def test_annotate_rerun_delete_and_activity_log():
    rid = service.backtest("SYNTH", "donchian").run_id
    store.update_run(rid, notes="keep", tags=["a", "b"], starred=True)
    assert store.list_runs(starred_only=True)["id"].to_list() == [rid]

    new = service.rerun(rid)
    child = store.get_run(new)
    assert child["parent_id"] == rid
    assert child["metrics"]["total_return"] == pytest.approx(
        store.get_run(rid)["metrics"]["total_return"]
    )

    store.delete_run(rid)
    assert store.get_run(rid) is None
    actions = store.list_actions()["action"].to_list()
    assert actions[:4] == ["delete", "rerun", "backtest", "annotate"]


def test_sweep_and_walkforward_saved():
    grid = {"fast": [10, 20], "slow": [50, 100]}
    sw = service.run_sweep("SYNTH", "sma_cross", grid)
    run = store.get_run(sw.run_id)
    assert run["table"].height == 4 and run["params"]["grid"] == grid
    fig = charts.sweep_heatmap(run["table"], "slow", "fast", "sharpe")
    assert list(fig.data[0].x) == ["50", "100"]  # numeric, not lexical, order

    wf = service.run_walkforward("SYNTH", "sma_cross", grid, n_splits=3)
    run = store.get_run(wf.run_id)
    assert run["table"].height == 3
    assert run["series"].height == wf.oos.height
    assert run["metrics"]["win_rate"] is None
    assert service.rerun(wf.run_id)
