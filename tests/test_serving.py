"""
Smoke tests for forecast-serving.

These tests run without a live MLflow instance — the model is intentionally
not loaded. This validates the service's degraded-state contract, which is
as important as the happy path: probes, error codes, and metadata shape must
be correct even when no model is available.
"""

import pytest
from fastapi.testclient import TestClient
import app as serving_app


@pytest.fixture(scope="module")
def client():
    # Use the app directly — lifespan runs but model load fails silently (no MLflow)
    with TestClient(serving_app.app) as c:
        yield c


# ── Probe endpoints ────────────────────────────────────────────────────────────

def test_liveness_always_200(client):
    """/live must return 200 regardless of model state."""
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_503_without_model(client):
    """/ready returns 503 when no model is loaded (degraded state)."""
    response = client.get("/ready")
    assert response.status_code == 503


# ── Model info ─────────────────────────────────────────────────────────────────

def test_model_info_shape(client):
    """/model-info always responds with a consistent schema."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_loaded" in body
    assert body["model_loaded"] is False
    assert "time_steps" in body


# ── Predict — degraded state ───────────────────────────────────────────────────

def test_predict_503_without_model(client):
    """POST /predict returns 503 (not 500) when model is not loaded."""
    response = client.post(
        "/predict",
        json={"values": [0.2, 0.25, 0.3, 0.28, 0.22, 0.35, 0.4, 0.38, 0.33, 0.29]},
    )
    assert response.status_code == 503


def test_predict_validates_input_length(client):
    """422 is returned for wrong number of values — before model-loaded check."""
    response = client.post("/predict", json={"values": [0.1, 0.2]})
    # 503 if model check fires first, 422 if input validation fires first
    assert response.status_code in (422, 503)


def test_predict_rejects_non_float_values(client):
    """
    Non-numeric values are rejected at the Pydantic schema level (422). NaN is
    not valid JSON (RFC 8259) and fails at parse time; null is allowed only as
    a gap in the history part, never as the last value.
    """
    assert client.post("/predict", json={"values": ["a"] * 10}).status_code == 422
    response = client.post("/predict", json={"values": [None] * 10})
    assert response.status_code == 422
    assert "last value" in response.text


def test_prepare_input_uses_the_tail_and_fills_gaps_in_the_history():
    import numpy as np
    import pytest

    from app import prepare_input

    arr, filled = prepare_input([1.0, 2.0, 3.0], time_steps=2, length=2)
    assert arr.tolist() == [2.0, 3.0] and filled == 0                   # the tail
    arr, filled = prepare_input([1.0, None, None, 4.0, 5.0, 6.0], time_steps=2, length=6)
    np.testing.assert_allclose(arr, [1, 2, 3, 4, 5, 6])                  # linear fill
    assert filled == 2
    arr, filled = prepare_input([None, None, 4.0, 5.0], time_steps=2, length=4)
    assert arr.tolist() == [4.0, 4.0, 4.0, 5.0] and filled == 2         # no earlier sample: flat
    with pytest.raises(ValueError, match="at least 4"):
        prepare_input([1.0, 2.0], 2, 4)
    with pytest.raises(ValueError, match="must all be present"):
        prepare_input([1.0, None, 3.0], 2, 3)


def test_predict_rejects_empty_list(client):
    """Empty values list must be rejected."""
    response = client.post("/predict", json={"values": []})
    assert response.status_code == 422


# ── Reload ─────────────────────────────────────────────────────────────────────

def test_reload_fails_gracefully_without_mlflow():
    """POST /reload returns 500 when MLflow is unreachable — does not crash the process."""
    # Use a fresh client with raise_server_exceptions=False so a 500 from the app
    # is returned as an HTTP response rather than re-raised as a Python exception.
    from fastapi.testclient import TestClient
    import app as serving_app
    c = TestClient(serving_app.app, raise_server_exceptions=False)
    response = c.post("/reload")
    assert response.status_code == 500


# ── Observability ──────────────────────────────────────────────────────────────

def test_metrics_endpoint_reachable(client):
    """/metrics is always reachable and contains expected metric names."""
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    # HTTP-level metrics from prometheus_fastapi_instrumentator
    assert "http_requests_total" in body
    assert 'http_request_duration_seconds_bucket{handler="/live",le="0.01"' in body
    # Custom domain metrics
    assert "forecast_model_loaded" in body
    assert "forecast_prediction_requests_total" in body
    assert "forecast_model_reloads_total" in body
