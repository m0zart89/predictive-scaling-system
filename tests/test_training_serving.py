"""
Contract test between training/ and serving/: a model registered by the
retraining pipeline must load in the serving app and produce exactly the
forecast the pipeline computed; a model registered the old way (notebook:
stage "Production", no alias, no forecast_meta.json) must keep working.
"""

import math
import os
import pickle
import sys
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import app as serving_app  # noqa: E402  (serving/ is on sys.path via conftest)

from forecast_training.registry import TrainConfig, load_champion, run_pipeline  # noqa: E402

STEP = 300


def demand(days=8, seed=1):
    rng = np.random.default_rng(seed)
    n = days * 86400 // STEP
    ts = 1_700_000_000 + np.arange(n) * STEP
    hour = (ts % 86400) / 3600
    return ts, np.clip(0.2 + 0.8 * np.exp(-((hour - 13) ** 2) / 8) + rng.normal(0, 0.05, n),
                       0.01, None)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("cpu-forecasting", artifact_location=str(tmp_path / "artifacts"))
    monkeypatch.setattr(serving_app, "MLFLOW_TRACKING_URI", uri)
    yield uri
    mlflow.set_tracking_uri(None)


@pytest.fixture
def serving(monkeypatch):
    """Serving app with clean model state; restored afterwards."""
    monkeypatch.setattr(serving_app, "_bundle", None)
    monkeypatch.setattr(serving_app, "MODEL_ALIAS", "champion")
    monkeypatch.setattr(serving_app, "MODEL_STAGE", "Production")
    serving_app.TRACKER.reset()
    return TestClient(serving_app.app)


def test_pipeline_model_serves_identical_multistep_forecast(registry, serving):
    ts, values = demand()
    out = run_pipeline(values, TrainConfig(horizon=4, candidates=["hgb_delta"]), "test",
                       timestamps=ts)
    assert out.promoted

    serving_app.load_model()
    info = serving.get("/model-info").json()
    assert info["horizon"] == 4 and info["target"] == "delta" and info["features"] == "anchored"
    assert info["model_stage"] == "@champion"

    window = [float(v) for v in values[-10:]]
    body = serving.post("/predict", json={"values": window}).json()
    assert len(body["predictions"]) == 4
    assert body["prediction"] == body["predictions"][0]
    assert body["step_seconds"] == STEP

    from mlflow.tracking import MlflowClient
    bundle, _, _ = load_champion(MlflowClient(), "cpu-forecasting", "champion")
    np.testing.assert_allclose(body["predictions"], bundle.predict(window)[0], rtol=1e-9)

    # out-of-range load: the delta model follows it instead of capping at the training max
    high = [v + 3.0 for v in window]
    assert serving.post("/predict", json={"values": high}).json()["prediction"] > 2.5


def test_window_length_comes_from_the_model(registry, serving):
    """At least input_length values; a longer history is fine and its tail is
    used, so a scaler fetching a season of history serves any model."""
    ts, values = demand()
    run_pipeline(values, TrainConfig(time_steps=6, horizon=2, candidates=["hgb_delta"]), "test",
                 timestamps=ts)
    serving_app.load_model()
    info = serving.get("/model-info").json()
    assert info["input_length"] == 6 and info["season_steps"] == 0
    assert serving.post("/predict", json={"values": [0.1] * 6}).status_code == 200
    r = serving.post("/predict", json={"values": [0.1] * 5})
    assert r.status_code == 422 and "Expected at least 6" in r.json()["detail"]
    short = serving.post("/predict", json={"values": [0.3] * 6}).json()["predictions"]
    long = serving.post("/predict", json={"values": [None, 9.0, None] + [0.3] * 6}).json()["predictions"]
    assert long == short                                  # only the tail counts
    r = serving.post("/predict", json={"values": [0.3] * 5 + [None, 0.3]})
    assert r.status_code == 422 and "must all be present" in r.json()["detail"]


def test_seasonal_model_takes_a_season_of_history_and_fills_gaps(registry, serving):
    """A model with anchored_season features needs time_steps + season_steps
    values; serving reports that in /model-info, forecasts exactly what the
    pipeline would, interpolates nulls in the history part and still answers
    (from the window alone) when there is no history at all."""
    from prometheus_client import REGISTRY

    ts, values = demand(days=12)
    out = run_pipeline(values, TrainConfig(horizon=3, candidates=["hgb_delta_season"]), "test",
                       timestamps=ts)
    assert out.promoted
    serving_app.load_model()
    info = serving.get("/model-info").json()
    assert info["features"] == "anchored_season"
    assert info["season_steps"] == 288 and info["input_length"] == 298

    from mlflow.tracking import MlflowClient
    bundle, _, _ = load_champion(MlflowClient(), "cpu-forecasting", "champion")
    history = [float(v) for v in values[-298:]]
    body = serving.post("/predict", json={"values": history}).json()
    np.testing.assert_allclose(body["predictions"], bundle.predict(history)[0], rtol=1e-9)

    r = serving.post("/predict", json={"values": history[-10:]})     # window only: too short
    assert r.status_code == 422 and "288 of history" in r.json()["detail"]

    filled_before = REGISTRY.get_sample_value("forecast_input_filled_total")
    gappy = list(history)
    gappy[100:103] = [None, None, None]
    body = serving.post("/predict", json={"values": gappy}).json()
    interpolated = list(history)
    interpolated[100:103] = list(np.interp([100, 101, 102], [99, 103], [history[99], history[103]]))
    np.testing.assert_allclose(body["predictions"], bundle.predict(interpolated)[0], rtol=1e-9)
    assert REGISTRY.get_sample_value("forecast_input_filled_total") == filled_before + 3

    # a new cluster: nothing but the window is known yet
    cold = [None] * 288 + history[-10:]
    body = serving.post("/predict", json={"values": cold})
    assert body.status_code == 200 and all(math.isfinite(p) for p in body.json()["predictions"])


def test_legacy_notebook_model_still_loads_via_stage(registry, serving):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler

    _, values = demand(days=3)
    scaler = MinMaxScaler().fit(values.reshape(-1, 1))
    scaled = scaler.transform(values.reshape(-1, 1)).ravel()
    X = np.array([scaled[i:i + 10] for i in range(len(scaled) - 10)])
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, scaled[10:])

    mlflow.set_experiment("cpu-forecasting")
    with mlflow.start_run() as run:                 # exactly what the notebook logs
        mlflow.sklearn.log_model(rf, "model")
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(f.name, "scaler")
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "cpu-forecasting")
    MlflowClient().transition_model_version_stage("cpu-forecasting", mv.version, "Production")

    serving_app.load_model()                        # no @champion → falls back to stage
    info = serving.get("/model-info").json()
    assert info["model_stage"] == "Production"
    assert info["horizon"] == 1 and info["target"] == "level"

    window = [float(v) for v in values[-10:]]
    body = serving.post("/predict", json={"values": window}).json()
    expected = scaler.inverse_transform(
        rf.predict(scaler.transform(np.array(window).reshape(-1, 1)).reshape(1, -1)).reshape(-1, 1))
    assert body["predictions"] == [pytest.approx(float(expected[0, 0]))]


def test_alias_wins_over_stage(registry, serving):
    ts, values = demand()
    run_pipeline(values, TrainConfig(horizon=2, candidates=["hgb_delta"]), "test", timestamps=ts)
    from mlflow.tracking import MlflowClient
    MlflowClient().transition_model_version_stage("cpu-forecasting", "1", "Production")
    serving_app.load_model()
    assert serving.get("/model-info").json()["model_stage"] == "@champion"


def test_no_alias_and_no_stage_fails_cleanly(registry, serving, monkeypatch):
    monkeypatch.setattr(serving_app, "MODEL_STAGE", "")
    with pytest.raises(Exception):
        serving_app.load_model()
    assert serving.get("/ready").status_code == 503


def test_failed_reload_keeps_the_loaded_model(registry, serving, monkeypatch):
    from prometheus_client import REGISTRY

    ts, values = demand()
    run_pipeline(values, TrainConfig(horizon=2, candidates=["hgb_delta"]), "test", timestamps=ts)
    serving_app.load_model()
    serving_app.TRACKER.record("x", [0.5] * 10, [0.5, 0.5], 300, timestamp=1_700_000_000)

    monkeypatch.setattr(serving_app, "MODEL_NAME", "does-not-exist")
    client = TestClient(serving_app.app, raise_server_exceptions=False)
    assert client.post("/reload").status_code == 500
    assert client.get("/ready").status_code == 200
    assert REGISTRY.get_sample_value("forecast_model_loaded") == 1
    assert REGISTRY.get_sample_value("forecast_pending_forecasts") == 1   # untouched
    assert client.post("/predict", json={"values": [0.1] * 10}).status_code == 200


def test_predict_scores_earlier_forecasts_against_later_windows(registry, serving):
    ts, values = demand()
    run_pipeline(values, TrainConfig(horizon=3, candidates=["hgb_delta"]), "test", timestamps=ts)
    serving_app.load_model()

    t0 = int(ts[-2])
    window = [float(v) for v in values[-11:-1]]
    first = serving.post("/predict", json={"values": window, "series": "demo/cpu-burner",
                                           "timestamp": t0}).json()
    later = [float(v) for v in values[-10:]]            # one step on: holds the actual
    serving.post("/predict", json={"values": later, "series": "demo/cpu-burner",
                                   "timestamp": t0 + STEP})

    expected = abs(later[-1] - first["prediction"])
    info = serving.get("/model-info").json()
    assert info["rolling_mae"] == {"1": pytest.approx(expected, abs=1e-6)}
    assert info["model_version"] == "1"

    metrics = serving.get("/metrics").text
    assert 'forecast_rolling_mae{horizon_step="1"}' in metrics
    assert 'forecast_abs_error_count{horizon_step="1"} 1.0' in metrics
    assert 'forecast_model_info{' in metrics and 'model_version="1"' in metrics


def test_bad_timestamp_is_rejected(registry, serving):
    r = serving.post("/predict", json={"values": [0.1] * 10, "timestamp": -5})
    assert r.status_code == 422
    assert serving.post("/predict", json={"values": [0.1] * 10, "step_seconds": 0}).status_code == 422


def test_compressed_time_windows_are_scored_at_their_own_spacing(registry, serving):
    """A caller feeding 30-second samples to a 5-minute model (the demo's
    stepSeconds=30) says so, and the forecast is scored one 30 s step later."""
    ts, values = demand()
    run_pipeline(values, TrainConfig(horizon=2, candidates=["hgb_delta"]), "test", timestamps=ts)
    serving_app.load_model()
    t0 = 1_700_000_000
    first = serving.post("/predict", json={"values": [0.5] * 10, "series": "x", "timestamp": t0,
                                           "step_seconds": 30}).json()
    serving.post("/predict", json={"values": [0.5] * 9 + [0.7], "series": "x", "timestamp": t0 + 30,
                                   "step_seconds": 30})
    info = serving.get("/model-info").json()
    assert info["rolling_mae"] == {"1": pytest.approx(abs(0.7 - first["prediction"]), abs=1e-6)}


def test_time_of_day_model_serves_the_pipeline_forecast_for_the_given_timestamp(registry, serving):
    ts, values = demand()
    out = run_pipeline(values, TrainConfig(horizon=3, candidates=["hgb_delta_tod"]), "test",
                       timestamps=ts)
    assert out.promoted
    serving_app.load_model()
    info = serving.get("/model-info").json()
    assert info["features"] == "anchored_tod" and info["period_seconds"] == 86400

    from mlflow.tracking import MlflowClient
    bundle, _, _ = load_champion(MlflowClient(), "cpu-forecasting", "champion")
    window = [float(v) for v in values[-10:]]
    for t in (int(ts[-1]), int(ts[-1]) + 6 * 3600):
        body = serving.post("/predict", json={"values": window, "timestamp": t}).json()
        np.testing.assert_allclose(body["predictions"], bundle.predict(window, [t])[0], rtol=1e-9)
    morning = serving.post("/predict", json={"values": window, "timestamp": int(ts[-1])}).json()
    evening = serving.post("/predict", json={"values": window, "timestamp": int(ts[-1]) + 6 * 3600}).json()
    assert morning["predictions"] != evening["predictions"]   # the hour matters now
    assert serving.post("/predict", json={"values": window}).status_code == 200   # defaults to now


def test_reload_requires_the_token_when_one_is_configured(registry, serving, monkeypatch):
    ts, values = demand()
    run_pipeline(values, TrainConfig(horizon=2, candidates=["hgb_delta"]), "test", timestamps=ts)
    monkeypatch.setattr(serving_app, "RELOAD_TOKEN", "s3cret")
    client = TestClient(serving_app.app, raise_server_exceptions=False)
    assert client.post("/reload").status_code == 401
    assert client.post("/reload", headers={"X-Reload-Token": "wrong"}).status_code == 401
    assert client.get("/ready").status_code == 503          # nothing was loaded by the refusals
    assert client.post("/reload", headers={"X-Reload-Token": "s3cret"}).status_code == 200
    assert client.get("/ready").status_code == 200

    # the training side sends the header the service expects
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        seen["token"] = req.get_header("X-reload-token")
        return Resp()

    from forecast_training import registry as reg
    monkeypatch.setattr(reg.urllib.request, "urlopen", fake_urlopen)
    assert reg.reload_serving("http://forecast-serving:8080", token="s3cret") is True
    assert seen["token"] == "s3cret"


def test_serving_exports_the_models_holdout_mae_for_the_drift_alert(registry, serving):
    from prometheus_client import REGISTRY

    ts, values = demand()
    out = run_pipeline(values, TrainConfig(horizon=3, candidates=["hgb_delta"]), "test", timestamps=ts)
    from mlflow.tracking import MlflowClient
    tags = MlflowClient().get_model_version("cpu-forecasting", out.version).tags
    assert {"holdout_mae_step_1", "holdout_mae_step_2", "holdout_mae_step_3"} <= set(tags)

    serving_app.load_model()
    for step in (1, 2, 3):
        got = REGISTRY.get_sample_value("forecast_model_holdout_mae", {"horizon_step": str(step)})
        assert got == pytest.approx(float(tags[f"holdout_mae_step_{step}"]))
    assert REGISTRY.get_sample_value("forecast_model_holdout_mae", {"horizon_step": "4"}) is None
