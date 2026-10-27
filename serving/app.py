import json
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
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from quality import ForecastTracker

# ── Configuration ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME",          "cpu-forecasting")
# Registry alias set by the retraining pipeline (training/). If the alias does
# not exist yet, the service falls back to the legacy stage the notebook uses.
MODEL_ALIAS         = os.getenv("MODEL_ALIAS",         "champion")
MODEL_STAGE         = os.getenv("MODEL_STAGE",         "Production")
# Only used for models without a forecast_meta.json (registered by the notebook).
TIME_STEPS          = int(os.getenv("TIME_STEPS",      "10"))
META_ARTIFACT       = "forecast_meta.json"
# Forecast quality tracking (see quality.py): rolling-MAE window per horizon
# step, and memory bounds for the forecasts waiting to be scored.
FORECAST_MAE_WINDOW = int(os.getenv("FORECAST_MAE_WINDOW", "100"))
FORECAST_MAX_SERIES = int(os.getenv("FORECAST_MAX_SERIES", "64"))
FORECAST_MAX_PENDING = int(os.getenv("FORECAST_MAX_PENDING", "512"))

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
MODEL_INFO = Gauge(
    "forecast_model_info",
    "Loaded model, as labels (always 1); lets dashboards show which version served",
    ["model_name", "model_version", "model_stage", "run_id", "horizon"],
)
# Scores past forecasts against the actual values that later windows carry:
# forecast_abs_error / forecast_rolling_mae, both by horizon step.
TRACKER = ForecastTracker(max_series=FORECAST_MAX_SERIES, max_pending=FORECAST_MAX_PENDING,
                          mae_window=FORECAST_MAE_WINDOW)

# ── Model state ────────────────────────────────────────────────────────────────
model    = None
scaler   = None
_meta: dict = {}   # operational metadata populated on each successful load

# How to turn a window into model features and outputs back into values.
# Must match training/forecast_training/features.py (tests/test_training_serving.py
# checks that both sides agree). Models without a forecast_meta.json artifact
# were registered by the notebook: TIME_STEPS scaled values in, one value out.
LEGACY_FORECAST_META = {"time_steps": TIME_STEPS, "horizon": 1, "step_seconds": 300,
                        "features": "window", "target": "level"}
forecast_meta: dict = dict(LEGACY_FORECAST_META)


def _features(scaled: np.ndarray, kind: str) -> np.ndarray:
    """scaled: (1, time_steps). 'anchored' = window relative to its last value, plus that value."""
    if kind == "window":
        return scaled
    if kind == "anchored":
        last = scaled[:, -1:]
        return np.hstack([scaled[:, :-1] - last, last])
    raise ValueError(f"unknown feature kind {kind!r}")


def forecast(values: List[float]) -> List[float]:
    """Raw window → forecast for each step of the model's horizon (raw units)."""
    meta   = forecast_meta
    arr    = np.array(values, dtype=float).reshape(-1, 1)
    scaled = scaler.transform(arr).reshape(1, -1)
    out    = np.asarray(model.predict(_features(scaled, meta["features"])), dtype=float)
    out    = out.reshape(1, -1)
    if meta["target"] == "delta":
        out = out + scaled[:, -1:]
    return [float(v) for v in scaler.inverse_transform(out.reshape(-1, 1)).ravel()]


def _resolve_version(client: MlflowClient):
    """Find the model version to serve: the alias if set, else the legacy stage."""
    if MODEL_ALIAS:
        try:
            v = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            return v, f"models:/{MODEL_NAME}@{MODEL_ALIAS}", f"@{MODEL_ALIAS}"
        except MlflowException as exc:
            if not MODEL_STAGE:
                raise
            log.info("Alias @%s not found (%s); falling back to stage %s",
                     MODEL_ALIAS, exc.error_code, MODEL_STAGE)
    if not MODEL_STAGE:
        raise RuntimeError("Neither MODEL_ALIAS nor MODEL_STAGE is set")
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(f"No model version found in stage '{MODEL_STAGE}'")
    return versions[0], f"models:/{MODEL_NAME}/{MODEL_STAGE}", MODEL_STAGE


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model() -> None:
    global model, scaler, _meta, forecast_meta

    log.info(
        "Loading model name=%s alias=%s (fallback stage=%s) from MLflow at %s",
        MODEL_NAME, MODEL_ALIAS or "-", MODEL_STAGE or "-", MLFLOW_TRACKING_URI,
    )

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    v, model_uri, selector = _resolve_version(client)
    # The REST store returns versions as strings, the SQL store as ints.
    run_id, version = v.run_id, str(v.version)

    loaded_model = mlflow.sklearn.load_model(model_uri)

    tmp = tempfile.mkdtemp()
    scaler_dir  = client.download_artifacts(run_id, "scaler", dst_path=tmp)
    scaler_path = os.path.join(scaler_dir, sorted(os.listdir(scaler_dir))[0])
    with open(scaler_path, "rb") as f:
        loaded_scaler = pickle.load(f)

    try:
        meta_path = client.download_artifacts(run_id, META_ARTIFACT, dst_path=tmp)
        with open(meta_path) as f:
            loaded_meta = {**LEGACY_FORECAST_META, **json.load(f)}
    except (MlflowException, OSError):
        loaded_meta = dict(LEGACY_FORECAST_META)
    if loaded_meta["features"] not in ("window", "anchored") or \
            loaded_meta["target"] not in ("level", "delta"):
        raise RuntimeError(f"Unsupported {META_ARTIFACT}: {loaded_meta}")

    # Commit atomically — avoid partial state if any artifact fails to load
    model, scaler, forecast_meta = loaded_model, loaded_scaler, loaded_meta
    _meta  = {
        "model_name":    MODEL_NAME,
        "model_stage":   selector,
        "model_version": version,
        "run_id":        run_id,
        "loaded_at":     datetime.now(timezone.utc).isoformat(),
    }

    MODEL_LOADED.set(1)
    MODEL_INFO.clear()   # one series only: drop the previous version's labels
    MODEL_INFO.labels(model_name=MODEL_NAME, model_version=version, model_stage=selector,
                      run_id=run_id, horizon=str(loaded_meta["horizon"])).set(1)
    MODEL_RELOADS.labels(result="success").inc()
    log.info(
        "Model loaded: name=%s %s version=%s run_id=%s time_steps=%s horizon=%s target=%s",
        MODEL_NAME, selector, version, run_id, loaded_meta["time_steps"],
        loaded_meta["horizon"], loaded_meta["target"],
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
        "Workload Forecast Service starting — MODEL=%s ALIAS=%s STAGE=%s MLFLOW=%s",
        MODEL_NAME, MODEL_ALIAS or "-", MODEL_STAGE or "-", MLFLOW_TRACKING_URI,
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
    # Last time_steps cpu_usage values at step_seconds intervals (see /model-info),
    # raw (unscaled).
    # The service applies the stored MinMaxScaler internally.
    values: List[float]
    # Optional, for forecast quality tracking: which series this window belongs
    # to (callers forecasting several workloads should pass a distinct key per
    # workload) and the unix time of the last value (default: arrival time).
    series: str = ""
    timestamp: Optional[float] = None

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("values must not be empty")
        for i, x in enumerate(v):
            if not math.isfinite(x):
                raise ValueError(f"values[{i}] is not finite: {x!r}")
        return v

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_finite(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (not math.isfinite(v) or v <= 0):
            raise ValueError("timestamp must be a positive unix time in seconds")
        return v


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction:    float          # next interval (first step of `predictions`)
    predictions:   List[float]    # one value per step of the model's horizon
    step_seconds:  int            # spacing of `predictions`
    model:         str
    stage:         str            # registry selector the model was loaded from
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
    predicted_utilization: float   # next interval
    predicted_peak:        float   # max over the forecast horizon (drives the decision)
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
        "time_steps":   forecast_meta["time_steps"],
        "horizon":      forecast_meta["horizon"],
        "step_seconds": forecast_meta["step_seconds"],
        "features":     forecast_meta["features"],
        "target":       forecast_meta["target"],
        "rolling_mae":  {str(k): round(v, 6) for k, v in sorted(TRACKER.snapshot().items())},
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
    Predict CPU utilization for the next interval(s).

    Input: the last `time_steps` observations (raw, unscaled; see /model-info).
    Output: `prediction` for the next interval, `predictions` for every step of
    the model's horizon (one value for models registered by the notebook), and
    model provenance.

    The model_version field in the response allows callers to detect a model
    swap that occurred between requests (e.g. after POST /reload).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /reload to recover.")

    time_steps = forecast_meta["time_steps"]
    if len(req.values) != time_steps:
        PREDICTION_FAILURES.labels(reason="validation").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly {time_steps} values, got {len(req.values)}.",
        )

    PREDICTION_REQUESTS.inc()

    try:
        predictions = forecast(req.values)
    except Exception as exc:
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error. Check service logs.")

    if not all(math.isfinite(p) for p in predictions):
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.error("Model returned non-finite prediction %r for input %s", predictions, req.values)
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction.")

    try:
        TRACKER.record(req.series, req.values, predictions, forecast_meta["step_seconds"],
                       req.timestamp)
    except Exception:   # quality tracking must never fail a prediction
        log.exception("forecast quality tracking failed")

    return PredictResponse(
        prediction=predictions[0],
        predictions=predictions,
        step_seconds=forecast_meta["step_seconds"],
        model=MODEL_NAME,
        stage=_meta.get("model_stage", MODEL_STAGE),
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
    to the *peak* of the forecast horizon: scale up if load is expected to exceed
    the threshold at any point in the horizon, scale down only if it stays below
    the low threshold for the whole horizon. For single-step models this is the
    same as the next-interval forecast. This endpoint does NOT interact with any
    Kubernetes API — it returns a recommendation that an external controller
    (KEDA scaler, custom operator) is expected to act on.

    Thresholds are caller-supplied so different workloads can apply their own
    headroom policies without changing the service configuration.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    time_steps = forecast_meta["time_steps"]
    if len(req.values) != time_steps:
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly {time_steps} values, got {len(req.values)}.",
        )

    try:
        predictions = forecast(req.values)
    except Exception as exc:
        log.exception("Inference failed in recommend-scale: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error.")
    if not all(math.isfinite(p) for p in predictions):
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction.")

    predicted = max(predictions)

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
        predicted_utilization=round(predictions[0], 4),
        predicted_peak=round(predicted, 4),
        current_replicas=req.current_replicas,
        reason=reason,
        model=MODEL_NAME,
        stage=_meta.get("model_stage", MODEL_STAGE),
    )


@app.post("/reload", tags=["operations"], summary="Hot-reload model from MLflow")
def reload():
    """
    Re-fetch the model from MLflow (MODEL_ALIAS, falling back to MODEL_STAGE)
    without restarting the pod. The retraining job calls this after moving the alias.
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
