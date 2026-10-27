"""
Contract test between training/ and serving/: a model registered by the
retraining pipeline must load in the serving app and produce exactly the
forecast the pipeline computed; a model registered the old way (notebook:
stage "Production", no alias, no forecast_meta.json) must keep working.
"""

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
    for name in ("model", "scaler", "_meta", "forecast_meta"):
        monkeypatch.setattr(serving_app, name, getattr(serving_app, name))
    monkeypatch.setattr(serving_app, "MODEL_ALIAS", "champion")
    monkeypatch.setattr(serving_app, "MODEL_STAGE", "Production")
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

    rec = serving.post("/recommend-scale", json={"values": window,
                                                 "scale_up_threshold": 10}).json()
    assert rec["predicted_peak"] == pytest.approx(max(body["predictions"]), abs=1e-4)
    assert rec["predicted_utilization"] == pytest.approx(body["prediction"], abs=1e-4)


def test_wrong_window_length_uses_model_time_steps(registry, serving):
    ts, values = demand()
    run_pipeline(values, TrainConfig(time_steps=6, horizon=2, candidates=["hgb_delta"]), "test",
                 timestamps=ts)
    serving_app.load_model()
    assert serving.post("/predict", json={"values": [0.1] * 6}).status_code == 200
    r = serving.post("/predict", json={"values": [0.1] * 10})
    assert r.status_code == 422 and "Expected exactly 6" in r.json()["detail"]


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
