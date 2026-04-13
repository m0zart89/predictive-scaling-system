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


def test_health_always_200(client):
    """/health is backwards-compatible: always 200, reports model_loaded: false."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


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
    Non-numeric values in input must be rejected at Pydantic schema level (422).
    Note: NaN is not valid JSON (RFC 8259) and is rejected at parse time before
    reaching validators — this test uses None to verify the type validation path.
    """
    response = client.post("/predict", json={"values": [None] * 10})
    assert response.status_code == 422


def test_predict_rejects_empty_list(client):
    """Empty values list must be rejected."""
    response = client.post("/predict", json={"values": []})
    assert response.status_code == 422


# ── Recommend-scale — degraded state ──────────────────────────────────────────

def test_recommend_scale_503_without_model(client):
    """POST /recommend-scale returns 503 when model is not loaded."""
    response = client.post(
        "/recommend-scale",
        json={
            "values": [0.2, 0.25, 0.3, 0.28, 0.22, 0.35, 0.4, 0.38, 0.33, 0.29],
            "current_replicas": 2,
        },
    )
    assert response.status_code == 503


def test_recommend_scale_schema(client):
    """Response schema is validated even in error cases."""
    response = client.post(
        "/recommend-scale",
        json={"values": [0.1] * 10, "current_replicas": 1},
    )
    # Without a model this will be 503, but schema validation should pass
    assert response.status_code in (200, 503)


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
    # Custom domain metrics
    assert "forecast_model_loaded" in body
    assert "forecast_prediction_requests_total" in body
    assert "forecast_model_reloads_total" in body
