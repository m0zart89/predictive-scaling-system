"""Multivariate channels: contract, fitting, registry round trip, data alignment, CLI."""

import json

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from conftest import STEP, daily_series
from forecast_training.__main__ import main
from forecast_training.data import DataError, Series, align_series, load_export, regularize
from forecast_training.features import ForecastMeta, build_features_mv, make_windows_mv, target_channel
from forecast_training.model import CANDIDATES, fit_bundle, level_tracking, train_and_select
from forecast_training.registry import TrainConfig, load_champion, run_pipeline

BY_NAME = {c.name: c for c in CANDIDATES}


def with_leading_channel(days=10, lead=3, seed=0):
    """CPU as in conftest, plus a request-rate channel that is CPU shifted `lead`
    steps earlier (it rises before CPU does) with its own noise and scale."""
    ts, cpu = daily_series(days=days, seed=seed)
    rng = np.random.default_rng(seed + 1)
    rps = np.roll(cpu, -lead) * 400 + rng.normal(0, 5, len(cpu))
    rps[-lead:] = rps[-lead - 1]
    return ts, np.stack([cpu, rps])


def test_meta_and_windows_carry_channels():
    m = ForecastMeta(time_steps=4, horizon=2, features="anchored", target="delta", channels=["cpu", "rps"])
    assert m.channels == ("cpu", "rps") and m.n_channels == 2 and m.multivariate
    assert ForecastMeta.from_dict(json.loads(m.to_json())) == m
    assert ForecastMeta().n_channels == 1 and not ForecastMeta().multivariate
    with pytest.raises(ValueError, match="unique"):
        ForecastMeta(channels=("a", "a"))

    values = np.stack([np.arange(10.0), np.arange(10.0) * 10])
    X, Y, first = make_windows_mv(values, time_steps=3, horizon=2)
    assert X.shape == (6, 2, 3) and Y.shape == (6, 2) and list(first[:2]) == [3, 4]
    assert list(X[0, 1]) == [0, 10, 20] and list(Y[0]) == [3, 4]          # Y from channel 0
    f = build_features_mv(X[:1], "anchored", time_steps=3, horizon=2)
    np.testing.assert_allclose(f[0], [0 - 2, 1 - 2, 2, 0 - 20, 10 - 20, 20])   # target block, then rps block
    assert build_features_mv(X[:1, 0, :], "anchored").shape == (1, 3)
    assert target_channel(X).shape == (6, 3) and target_channel(X[:, 0, :]).shape == (6, 3)


def test_multivariate_bundle_fits_scales_per_channel_and_predicts_the_target():
    ts, values = with_leading_channel()
    b = fit_bundle(values[:, :2000], BY_NAME["hgb_delta"], 10, 3, STEP, ts[:2000], channels=("cpu", "rps"))
    assert b.meta.channels == ("cpu", "rps") and len(b.scalers) == 2 and b.scaler is b.scalers[0]
    assert b.scalers[1].data_max_[0] > 100                       # the rps scaler saw rps units
    out = b.predict(values[:, 2000:2010][None, :, :], ts[2009:2010])
    assert out.shape == (1, 3) and np.all(np.isfinite(out)) and 0 < out[0, 0] < 2
    with pytest.raises(ValueError, match="channels"):
        b.predict(values[0, 2000:2010])                          # a single series is not enough
    with pytest.raises(ValueError, match="names"):
        fit_bundle(values[:, :2000], BY_NAME["hgb_delta"], 10, 3, STEP, channels=("cpu",))
    split = 2000
    assert level_tracking(b, values, split, ts) > 0.9            # still follows the demand level


def test_a_leading_channel_lowers_the_cost_on_the_ramp():
    ts, values = with_leading_channel(lead=3)
    uni, _ = train_and_select(values[0], 10, 3, STEP, 0.2, 0.5, [BY_NAME["hgb_delta"]], ts)
    multi, _ = train_and_select(values, 10, 3, STEP, 0.2, 0.5, [BY_NAME["hgb_delta"]], ts,
                                channels=("cpu", "rps"))
    assert multi[0].score.cost < uni[0].score.cost * 0.9         # the rate leads CPU by three steps
    assert multi[0].bundle.meta.channels == ("cpu", "rps")


def test_pipeline_registers_channel_scalers_and_the_champion_loads_them(mlflow_tmp):
    ts, values = with_leading_channel(days=8)
    out = run_pipeline(values, TrainConfig(horizon=2, candidates=["hgb_delta"], channels=("cpu", "rps")),
                       "test", timestamps=ts)
    assert out.promoted
    client = MlflowClient()
    mv = client.get_model_version_by_alias("cpu-forecasting", "champion")
    assert mv.tags["channels"] == "cpu,rps"
    assert client.get_run(out.run_id).data.params["channels"] == "cpu,rps"
    bundle, _, _ = load_champion(client, "cpu-forecasting", "champion")
    assert bundle.meta.channels == ("cpu", "rps") and len(bundle.scalers) == 2
    window = values[:, -10:][None, :, :]
    np.testing.assert_allclose(bundle.predict(window), bundle.predict(window))
    with pytest.raises(ValueError, match="names"):
        run_pipeline(values, TrainConfig(horizon=2, candidates=["hgb_delta"]), "test", timestamps=ts)
    # the next run compares on fresh windows with the same channels
    again = run_pipeline(values, TrainConfig(horizon=2, candidates=["hgb_delta"], channels=("cpu", "rps")),
                         "test", timestamps=ts)
    assert again.decision == "kept_champion"
    # a changed contract (back to one series) replaces the champion outright
    back = run_pipeline(values[0], TrainConfig(horizon=2, candidates=["hgb_delta"]), "test", timestamps=ts)
    assert back.decision == "promoted" and "takes channels ['cpu', 'rps']" in back.reason


def test_align_series_intersects_the_grids():
    a = Series(np.arange(1000, 1600, 100), np.arange(6.0), 100, filled=1)
    b = Series(np.arange(1200, 1900, 100), np.arange(7.0) * 10, 100, filled=0)
    m = align_series({"cpu": a, "rps": b})
    assert list(m.timestamps) == [1200, 1300, 1400, 1500] and m.channels == ("cpu", "rps")
    np.testing.assert_allclose(m.values, [[2, 3, 4, 5], [0, 10, 20, 30]])
    assert m.filled == {"cpu": 1, "rps": 0} and len(m) == 4
    with pytest.raises(DataError, match="overlap"):
        align_series({"cpu": a, "rps": Series(np.arange(5000, 5300, 100), np.zeros(3), 100)})
    with pytest.raises(DataError, match="steps"):
        align_series({"cpu": a, "rps": Series(np.arange(1000, 1600, 100), np.zeros(6), 50)})


def test_cli_takes_channel_exports(mlflow_tmp, tmp_path, capsys):
    ts, values = with_leading_channel(days=6)
    for i, name in enumerate(("cpu", "rps")):
        (tmp_path / f"{name}.json").write_text(json.dumps({"status": "success", "data": {
            "resultType": "matrix", "result": [{"metric": {}, "values": [[int(t), str(v)] for t, v in zip(ts, values[i])]}]}}))
    assert load_export(str(tmp_path / "rps.json"))[0][0] == int(ts[0])
    assert main(["--export", str(tmp_path / "cpu.json"), "--channel", f"rps={tmp_path / 'rps.json'}",
                 "--candidates", "hgb_delta", "--horizon", "2", "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["decision"] == "dry_run"
    import mlflow
    run = mlflow.search_runs(experiment_names=["cpu-forecasting"], output_format="list",
                             filter_string="tags.decision = 'dry_run'")[0]
    assert run.data.params["channels"] == "cpu,rps"
    assert main(["--export", str(tmp_path / "cpu.json"), "--channel", "nonsense", "--dry-run"]) == 1
    assert regularize([(0, 1.0), (300, 2.0)], 300).filled == 0
