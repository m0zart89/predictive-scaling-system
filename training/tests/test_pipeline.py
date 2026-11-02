"""
End-to-end against a real (SQLite-backed) MLflow tracking server and registry.
"""

import json

import pytest
from mlflow.tracking import MlflowClient

from conftest import daily_series
from forecast_training.__main__ import main
from forecast_training.registry import TrainConfig, load_champion, run_pipeline

FAST = ["rf_level", "hgb_delta"]


def cfg(**kw):
    base = dict(horizon=3, candidates=FAST, min_fresh_windows=24)
    base.update(kw)
    return TrainConfig(**base)


def test_first_run_promotes_and_registers_full_contract(mlflow_tmp, series):
    ts, values = series
    out = run_pipeline(values, cfg(), "test", timestamps=ts)
    assert out.decision == "promoted" and out.version == "1"
    assert out.winner == "hgb_delta"            # rf_level fails level tracking

    client = MlflowClient()
    mv = client.get_model_version_by_alias("cpu-forecasting", "champion")
    assert str(mv.version) == "1"
    assert mv.tags["candidate"] == "hgb_delta"
    assert int(mv.tags["data_end_ts"]) == int(ts[-1])     # refit on all data

    bundle, version, _ = load_champion(client, "cpu-forecasting", "champion")
    assert bundle.meta.horizon == 3 and bundle.meta.target == "delta"
    assert bundle.predict(values[-10:]).shape == (1, 3)

    run = client.get_run(out.run_id)
    assert run.data.tags["decision"] == "promoted"
    children = client.search_runs(run.info.experiment_id,
                                  f"tags.mlflow.parentRunId = '{out.run_id}'")
    assert {c.info.run_name for c in children} == set(FAST)


def test_same_data_again_keeps_champion(mlflow_tmp, series):
    ts, values = series
    run_pipeline(values, cfg(), "test", timestamps=ts)
    out = run_pipeline(values, cfg(), "test", timestamps=ts)
    assert out.decision == "kept_champion"
    assert "newer than champion" in out.reason
    assert out.version is None


def test_new_data_compares_on_unseen_windows_only(mlflow_tmp):
    ts, values = daily_series(days=12)
    first = int(len(values) * 0.8)
    run_pipeline(values[:first], cfg(), "test", timestamps=ts[:first])

    out = run_pipeline(values, cfg(min_improvement=-1.0), "test", timestamps=ts)
    # min_improvement=-1 promotes whenever a fair comparison happens
    assert out.decision == "promoted", out.reason
    assert out.metrics["fresh_windows"] == pytest.approx(len(values) - first - 3 + 1)
    assert "champion_rmse" in out.metrics


def test_unsafe_champion_is_replaced(mlflow_tmp, series):
    ts, values = series
    first = run_pipeline(values, cfg(candidates=["rf_level"], min_tracking=-10), "test",
                         timestamps=ts)
    assert first.decision == "promoted"            # forced in with tracking check off

    out = run_pipeline(values, cfg(), "test", timestamps=ts)
    assert out.decision == "promoted"
    assert "fails level tracking" in out.reason


def test_no_eligible_candidate_is_rejected(mlflow_tmp, series):
    ts, values = series
    out = run_pipeline(values, cfg(min_tracking=5), "test", timestamps=ts)
    assert out.decision == "rejected"
    assert MlflowClient().search_registered_models() == []


def test_dry_run_registers_nothing(mlflow_tmp, series):
    ts, values = series
    out = run_pipeline(values, cfg(dry_run=True), "test", timestamps=ts)
    assert out.decision == "dry_run"
    assert MlflowClient().search_registered_models() == []


def test_cli_with_export_file(mlflow_tmp, tmp_path, capsys):
    ts, values = daily_series(days=6)
    export = {"resultType": "matrix", "result": [{"metric": {}, "values": [
        [int(t), str(v)] for t, v in zip(ts, values)]}]}
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export))

    rc = main(["--export", str(path), "--horizon", "2", "--candidates", "hgb_delta",
               "--exclude", f"{ts[100]}/{ts[110]}", "--summary", str(tmp_path / "s.json")])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["decision"] == "promoted"
    assert json.loads((tmp_path / "s.json").read_text()) == printed


def test_cli_errors_exit_1(mlflow_tmp, tmp_path):
    assert main([]) == 1                                   # no source
    assert main(["--prometheus", "http://x"]) == 1         # no query
    assert main(["--export", str(tmp_path / "missing.json")]) == 1


def test_unknown_candidate_is_an_error(mlflow_tmp, series):
    ts, values = series
    with pytest.raises(ValueError, match="unknown"):
        run_pipeline(values, cfg(candidates=["xgb"]), "test", timestamps=ts)


def test_baselines_are_logged_and_the_chosen_one_gates_promotion(mlflow_tmp, series):
    """Both baselines land in the summary; --baseline says which the winner must
    beat. An impossible bar (persistence scored with an absurd weight is still
    the same windows) is simulated by asking for the seasonal baseline on a
    series where it is unbeatable: a noiseless repeat of one day."""
    ts, values = series
    out = run_pipeline(values, cfg(dry_run=True), "test", timestamps=ts)
    m = out.metrics
    assert {"winner_cost", "winner_coverage", "naive_cost", "seasonal_naive_cost",
            "seasonal_naive_rmse"} <= set(m)
    assert out.decision == "dry_run" and "cost" in out.reason

    client = MlflowClient()
    run = client.get_run(out.run_id)
    assert run.data.params["under_weight"] == "3.0" and run.data.params["baseline"] == "persistence"
    assert "seasonal_naive_cost" in run.data.metrics and "naive_coverage" in run.data.metrics

    import numpy as np
    day = values[:288]
    exact = np.tile(day, 10)                      # every day identical: yesterday is perfect
    out = run_pipeline(exact, cfg(baseline="seasonal", dry_run=True), "test", timestamps=ts[:2880])
    assert out.decision == "rejected" and "seasonal persistence baseline" in out.reason
    assert out.metrics["seasonal_naive_cost"] == pytest.approx(0.0)
    out = run_pipeline(exact, cfg(baseline="persistence", dry_run=True), "test", timestamps=ts[:2880])
    assert out.decision == "dry_run"              # the easy baseline is still beaten
    with pytest.raises(ValueError, match="baseline"):
        cfg(baseline="none")
    with pytest.raises(ValueError, match="under_weight"):
        cfg(under_weight=0)


def test_registered_version_carries_cost_and_weight_tags(mlflow_tmp, series):
    ts, values = series
    out = run_pipeline(values, cfg(under_weight=2.0), "test", timestamps=ts)
    tags = MlflowClient().get_model_version("cpu-forecasting", out.version).tags
    assert tags["under_weight"] == "2" and float(tags["holdout_cost"]) > 0
    assert 0.0 <= float(tags["holdout_coverage"]) <= 1.0
    assert MlflowClient().get_run(out.run_id).data.params["under_weight"] == "2.0"


def test_cli_accepts_under_weight_and_baseline(mlflow_tmp, tmp_path, capsys):
    from conftest import daily_series
    ts, values = daily_series(days=6)
    export = tmp_path / "e.json"
    export.write_text(json.dumps({"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": {}, "values": [[int(t), str(v)] for t, v in zip(ts, values)]}]}}))
    assert main(["--export", str(export), "--candidates", "hgb_delta", "--horizon", "2",
                 "--under-weight", "1", "--baseline", "best", "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["decision"] == "dry_run" and "winner_cost" in printed["metrics"]
    import mlflow
    run = mlflow.search_runs(experiment_names=["cpu-forecasting"], output_format="list",
                             filter_string="tags.decision = 'dry_run'")[0]
    assert run.data.params["under_weight"] == "1.0" and run.data.params["baseline"] == "best"
