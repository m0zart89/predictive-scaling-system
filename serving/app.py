import math
import os
import pickle
import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Configuration ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME",          "cpu-forecasting")
MODEL_STAGE         = os.getenv("MODEL_STAGE",         "Production")
TIME_STEPS          = int(os.getenv("TIME_STEPS",      "10"))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("forecast-serving")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Custom Prometheus metrics ──────────────────────────────────────────────────
# prometheus_fastapi_instrumentator covers HTTP-level metrics (rate, latency, status).
# These counters track domain-level events that HTTP metrics don't capture.
PREDICTION_REQUESTS = Counter(
    "forecast_prediction_requests_total",
    "Total prediction requests that reached inference",
)
PREDICTION_FAILURES = Counter(
    "forecast_prediction_failures_total",
    "Prediction failures: validation errors + inference errors",
    ["reason"],  # label: "validation" | "inference"
)
MODEL_RELOADS = Counter(
    "forecast_model_reloads_total",
    "Model reload attempts (successful or failed)",
    ["result"],  # label: "success" | "failure"
)
MODEL_LOADED = Gauge(
    "forecast_model_loaded",
    "1 if a model is loaded and ready to serve predictions, 0 otherwise",
)

# ── Model state ────────────────────────────────────────────────────────────────
model    = None
scaler   = None
_meta: dict = {}   # operational metadata populated on each successful load


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model() -> None:
    global model, scaler, _meta

    log.info(
        "Loading model name=%s stage=%s from MLflow at %s",
        MODEL_NAME, MODEL_STAGE, MLFLOW_TRACKING_URI,
    )

    client    = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    loaded_model = mlflow.sklearn.load_model(model_uri)

    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(f"No model version found in stage '{MODEL_STAGE}'")

    v        = versions[0]
    run_id   = v.run_id
    version  = v.version

    scaler_dir  = client.download_artifacts(run_id, "scaler", dst_path=tempfile.mkdtemp())
    scaler_path = os.path.join(scaler_dir, os.listdir(scaler_dir)[0])
    with open(scaler_path, "rb") as f:
        loaded_scaler = pickle.load(f)

    # Commit atomically — avoid partial state if the scaler load fails
    model  = loaded_model
    scaler = loaded_scaler
    _meta  = {
        "model_name":    MODEL_NAME,
        "model_stage":   MODEL_STAGE,
        "model_version": version,
        "run_id":        run_id,
        "loaded_at":     datetime.now(timezone.utc).isoformat(),
    }

    MODEL_LOADED.set(1)
    MODEL_RELOADS.labels(result="success").inc()
    log.info(
        "Model loaded: name=%s stage=%s version=%s run_id=%s",
        MODEL_NAME, MODEL_STAGE, version, run_id,
    )


# ── Startup model load ─────────────────────────────────────────────────────────
def _load_model_safe() -> None:
    """
    Background model load. Called from lifespan so it does not block container startup.
    The /ready probe returns 503 until this completes successfully, which is the
    correct Kubernetes mechanism for gating traffic — not a blocking startup hook.
    """
    try:
        load_model()
    except Exception as exc:
        MODEL_LOADED.set(0)
        log.warning(
            "Model not available at startup (%s). "
            "Service running in degraded state. POST /reload to recover.",
            exc,
        )


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    import asyncio

    log.info(
        "Workload Forecast Service starting — MODEL=%s STAGE=%s STEPS=%d MLFLOW=%s",
        MODEL_NAME, MODEL_STAGE, TIME_STEPS, MLFLOW_TRACKING_URI,
    )
    # Run model load in a thread pool so the container starts immediately.
    # /ready returns 503 until load_model() completes — Kubernetes readiness probe
    # is the correct gating mechanism, not a blocking startup hook.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_model_safe)
    log.info("Model loading initiated in background. Traffic held until /ready returns 200.")

    yield

    log.info("Workload Forecast Service shutting down")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Workload Forecast Service",
    description=(
        "Inference service for infrastructure workload forecasting. "
        "Intended as a decision-support component for autoscalers and capacity tools. "
        "See /recommend-scale for the scaling integration prototype."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)


# ── Request / response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Last TIME_STEPS cpu_usage values at 5-minute intervals, raw (unscaled).
    # The service applies the stored MinMaxScaler internally.
    values: List[float]

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("values must not be empty")
        for i, x in enumerate(v):
            if not math.isfinite(x):
                raise ValueError(f"values[{i}] is not finite: {x!r}")
        return v


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction:    float
    model:         str
    stage:         str
    model_version: Optional[str] = None


class RecommendScaleRequest(BaseModel):
    # Same input window as /predict
    values:               List[float]
    current_replicas:     int   = 1
    # Thresholds are caller-controlled — different workloads have different headroom needs
    scale_up_threshold:   float = 0.75
    scale_down_threshold: float = 0.30

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("values must not be empty")
        for i, x in enumerate(v):
            if not math.isfinite(x):
                raise ValueError(f"values[{i}] is not finite: {x!r}")
        return v


class RecommendScaleResponse(BaseModel):
    # "scale_up" | "scale_down" | "hold"
    recommendation:        str
    predicted_utilization: float
    current_replicas:      int
    reason:                str
    model:                 str
    stage:                 str


# ── Operations endpoints ───────────────────────────────────────────────────────

@app.get("/live", tags=["operations"], summary="Liveness probe")
def liveness():
    """
    Returns 200 if the process is running.
    Kubernetes restarts the pod if this endpoint fails.
    Does NOT check whether a model is loaded.
    """
    return {"status": "alive"}


@app.get("/ready", tags=["operations"], summary="Readiness probe")
def readiness():
    """
    Returns 200 only if a model is loaded and inference is possible.
    Kubernetes holds traffic back from this pod until this returns 200.
    Distinction from /live: a pod can be alive but not ready (model loading in progress).
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready to serve predictions.",
        )
    return {"status": "ready"}


@app.get("/model-info", tags=["operations"], summary="Loaded model metadata")
def model_info():
    """
    Operational metadata about the currently loaded model.
    Intended for operators, dashboards, and debugging — not for inference callers.
    Returns model version and run_id so the serving state is auditable without
    querying MLflow directly.
    """
    return {
        "model_loaded": model is not None,
        "time_steps":   TIME_STEPS,
        **_meta,
    }


@app.get("/health", tags=["operations"], summary="Legacy health check")
def health():
    """
    Backwards-compatible health check.
    New deployments should use /live (liveness) and /ready (readiness) instead.
    """
    return {"status": "ok", "model_loaded": model is not None}


# ── Inference endpoints ────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    """
    Predict CPU utilization for the next 5-minute interval.

    Input: the last TIME_STEPS observations (raw, unscaled).
    Output: predicted utilization value + model provenance.

    The model_version field in the response allows callers to detect a model
    swap that occurred between requests (e.g. after POST /reload).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /reload to recover.")

    if len(req.values) != TIME_STEPS:
        PREDICTION_FAILURES.labels(reason="validation").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly {TIME_STEPS} values, got {len(req.values)}.",
        )

    PREDICTION_REQUESTS.inc()

    try:
        arr        = np.array(req.values).reshape(-1, 1)
        scaled     = scaler.transform(arr).flatten().reshape(1, -1)
        pred_raw   = model.predict(scaled)
        prediction = float(scaler.inverse_transform(pred_raw.reshape(-1, 1)).flatten()[0])
    except Exception as exc:
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error. Check service logs.")

    if not math.isfinite(prediction):
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.error("Model returned non-finite prediction %r for input %s", prediction, req.values)
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction.")

    return PredictResponse(
        prediction=prediction,
        model=MODEL_NAME,
        stage=MODEL_STAGE,
        model_version=_meta.get("model_version"),
    )


@app.post(
    "/recommend-scale",
    response_model=RecommendScaleResponse,
    tags=["control-plane"],
    summary="Predictive scaling recommendation (prototype)",
)
def recommend_scale(req: RecommendScaleRequest):
    """
    Decision-support endpoint for predictive autoscaling.

    Runs the same inference as /predict and applies a configurable threshold policy
    to produce a scaling recommendation. This endpoint does NOT interact with any
    Kubernetes API — it returns a recommendation that an external controller
    (KEDA scaler, custom operator) is expected to act on.

    Thresholds are caller-supplied so different workloads can apply their own
    headroom policies without changing the service configuration.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(req.values) != TIME_STEPS:
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly {TIME_STEPS} values, got {len(req.values)}.",
        )

    try:
        arr        = np.array(req.values).reshape(-1, 1)
        scaled     = scaler.transform(arr).flatten().reshape(1, -1)
        pred_raw   = model.predict(scaled)
        predicted  = float(scaler.inverse_transform(pred_raw.reshape(-1, 1)).flatten()[0])
    except Exception as exc:
        log.exception("Inference failed in recommend-scale: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error.")

    up   = req.scale_up_threshold
    down = req.scale_down_threshold

    if predicted > up:
        recommendation = "scale_up"
        reason = (
            f"Predicted utilization {predicted:.3f} exceeds scale-up threshold {up}. "
            f"Recommend increasing capacity before load arrives."
        )
    elif predicted < down:
        recommendation = "scale_down"
        reason = (
            f"Predicted utilization {predicted:.3f} is below scale-down threshold {down}. "
            f"Capacity can be safely reduced."
        )
    else:
        recommendation = "hold"
        reason = (
            f"Predicted utilization {predicted:.3f} is within normal range "
            f"[{down}, {up}]. No scaling action required."
        )

    log.info(
        "recommend-scale: predicted=%.3f recommendation=%s current_replicas=%d",
        predicted, recommendation, req.current_replicas,
    )

    return RecommendScaleResponse(
        recommendation=recommendation,
        predicted_utilization=round(predicted, 4),
        current_replicas=req.current_replicas,
        reason=reason,
        model=MODEL_NAME,
        stage=MODEL_STAGE,
    )


@app.post("/reload", tags=["operations"], summary="Hot-reload model from MLflow")
def reload():
    """
    Re-fetch the current Production model from MLflow without restarting the pod.
    Use this after promoting a new model version to the Production stage.
    Returns the new model metadata on success.
    """
    log.info("Model reload triggered via POST /reload")
    try:
        load_model()
        return {"status": "reloaded", "model": _meta}
    except Exception as exc:
        MODEL_LOADED.set(0)
        MODEL_RELOADS.labels(result="failure").inc()
        log.error("Model reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")
