import json
import math
import os
import pickle
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics
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
# While no model is loaded, retry MLflow this often (0 disables). A fresh
# cluster has no champion until the first training run; this picks it up
# without anyone having to call /reload.
MODEL_RETRY_SECONDS = float(os.getenv("MODEL_RETRY_SECONDS", "30"))

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
# How to turn a window into model features and outputs back into values.
# Must match training/forecast_training/features.py (tests/test_training_serving.py
# checks that both sides agree). Models without a forecast_meta.json artifact
# were registered by the notebook: TIME_STEPS scaled values in, one value out.
LEGACY_FORECAST_META = {"time_steps": TIME_STEPS, "horizon": 1, "step_seconds": 300,
                        "features": "window", "target": "level"}
FEATURE_KINDS = ("window", "anchored")
TARGET_KINDS = ("level", "delta")


@dataclass(frozen=True)
class Bundle:
    """Everything a forecast needs, swapped atomically on (re)load so a request
    never pairs a new model with the old scaler or metadata."""
    model: object
    scaler: object
    meta: Dict[str, object]       # forecast_meta.json merged over the legacy defaults
    info: Dict[str, str]          # operational metadata (version, run_id, loaded_at)

    def forecast(self, values: List[float]) -> List[float]:
        """Raw window → forecast for each step of the model's horizon (raw units)."""
        arr    = np.array(values, dtype=float).reshape(-1, 1)
        scaled = self.scaler.transform(arr).reshape(1, -1)
        out    = np.asarray(self.model.predict(_features(scaled, self.meta["features"])), dtype=float)
        out    = out.reshape(1, -1)
        if self.meta["target"] == "delta":
            out = out + scaled[:, -1:]
        return [float(v) for v in self.scaler.inverse_transform(out.reshape(-1, 1)).ravel()]


_bundle: Optional[Bundle] = None


def _features(scaled: np.ndarray, kind: str) -> np.ndarray:
    """scaled: (1, time_steps). 'anchored' = window relative to its last value, plus that value."""
    if kind == "window":
        return scaled
    if kind == "anchored":
        last = scaled[:, -1:]
        return np.hstack([scaled[:, :-1] - last, last])
    raise ValueError(f"unknown feature kind {kind!r}")


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
    global _bundle

    log.info(
        "Loading model name=%s alias=%s (fallback stage=%s) from MLflow at %s",
        MODEL_NAME, MODEL_ALIAS or "-", MODEL_STAGE or "-", MLFLOW_TRACKING_URI,
    )

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    v, model_uri, selector = _resolve_version(client)
    # The REST store returns versions as strings, the SQL store as ints.
    run_id, version = v.run_id, str(v.version)

    loaded_model = mlflow.sklearn.load_model(model_uri)

    with tempfile.TemporaryDirectory() as tmp:
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
    if loaded_meta["features"] not in FEATURE_KINDS or loaded_meta["target"] not in TARGET_KINDS:
        raise RuntimeError(f"Unsupported {META_ARTIFACT}: {loaded_meta}")

    info = {
        "model_name":    MODEL_NAME,
        "model_stage":   selector,
        "model_version": version,
        "run_id":        run_id,
        "loaded_at":     datetime.now(timezone.utc).isoformat(),
    }
    previous = _bundle
    # One assignment: requests in flight keep the bundle they started with.
    _bundle = Bundle(loaded_model, loaded_scaler, loaded_meta, info)
    if previous is not None and previous.info["run_id"] != run_id:
        # Forecasts the old model made are not the new model's errors.
        TRACKER.reset(pending_only=True)

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
_stop_loading = threading.Event()


def _load_model_until_available() -> None:
    """
    Background model load, retried every MODEL_RETRY_SECONDS until a model is
    loaded (or /reload gets there first). Runs in a thread started from the
    lifespan so container startup is never blocked; /ready returns 503 until
    a model is in memory, which is how Kubernetes gates traffic.
    """
    attempt = 0
    while not _stop_loading.is_set() and _bundle is None:
        attempt += 1
        try:
            load_model()
            return
        except Exception as exc:
            MODEL_LOADED.set(0)
            if attempt == 1:
                log.warning("Model not available at startup (%s). Service running in degraded "
                            "state; retrying every %gs, or POST /reload.", exc, MODEL_RETRY_SECONDS)
            elif attempt % 10 == 0:
                log.warning("Still no model after %d attempts: %s", attempt, exc)
        if MODEL_RETRY_SECONDS <= 0:
            return
        _stop_loading.wait(MODEL_RETRY_SECONDS)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    log.info(
        "Workload Forecast Service starting — MODEL=%s ALIAS=%s STAGE=%s MLFLOW=%s",
        MODEL_NAME, MODEL_ALIAS or "-", MODEL_STAGE or "-", MLFLOW_TRACKING_URI,
    )
    # Load (and keep retrying) in a daemon thread so the container starts
    # immediately; /ready holds traffic until a model is in memory.
    _stop_loading.clear()
    threading.Thread(target=_load_model_until_available, name="model-loader", daemon=True).start()
    log.info("Model loading initiated in background. Traffic held until /ready returns 200.")

    yield

    _stop_loading.set()
    log.info("Workload Forecast Service shutting down")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Workload Forecast Service",
    description=(
        "Inference service for infrastructure workload forecasting: a window of recent "
        "observations in, a forecast horizon out. The KEDA external scaler in scaler/ "
        "turns it into replica counts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
# Inference is sub-millisecond; the default (0.1, 0.5, 1) buckets could not
# resolve the p99 < 200 ms alert.
Instrumentator().add(metrics.default(
    latency_lowr_buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2.5),
)).instrument(app).expose(app)


# ── Request / response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # Last time_steps cpu_usage values at step_seconds intervals (see /model-info),
    # raw (unscaled).
    # The service applies the stored MinMaxScaler internally.
    values: List[float]
    # Optional, for forecast quality tracking: which series this window belongs
    # to (callers forecasting several workloads should pass a distinct key per
    # workload), the unix time of the last value (default: arrival time) and
    # the spacing of the values (default: the model's step_seconds; a caller
    # compressing time, e.g. the demo with stepSeconds=30, must say so).
    series: str = ""
    timestamp: Optional[float] = None
    step_seconds: Optional[int] = None

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

    @field_validator("step_seconds")
    @classmethod
    def step_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("step_seconds must be > 0")
        return v


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction:    float          # next interval (first step of `predictions`)
    predictions:   List[float]    # one value per step of the model's horizon
    step_seconds:  int            # spacing of `predictions`
    model:         str
    stage:         str            # registry selector the model was loaded from
    model_version: Optional[str] = None


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
    if _bundle is None:
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
    b = _bundle
    meta = b.meta if b is not None else LEGACY_FORECAST_META
    return {
        "model_loaded": b is not None,
        "time_steps":   meta["time_steps"],
        "horizon":      meta["horizon"],
        "step_seconds": meta["step_seconds"],
        "features":     meta["features"],
        "target":       meta["target"],
        "rolling_mae":  {str(k): round(v, 6) for k, v in sorted(TRACKER.snapshot().items())},
        **(b.info if b is not None else {}),
    }


# ── Inference endpoint ─────────────────────────────────────────────────────────

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
    b = _bundle
    if b is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /reload to recover.")

    time_steps = b.meta["time_steps"]
    if len(req.values) != time_steps:
        PREDICTION_FAILURES.labels(reason="validation").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly {time_steps} values, got {len(req.values)}.",
        )

    PREDICTION_REQUESTS.inc()

    try:
        predictions = b.forecast(req.values)
    except Exception as exc:
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error. Check service logs.")

    if not all(math.isfinite(p) for p in predictions):
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.error("Model returned non-finite prediction %r for input %s", predictions, req.values)
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction.")

    try:
        TRACKER.record(req.series, req.values, predictions,
                       req.step_seconds or b.meta["step_seconds"], req.timestamp)
    except Exception:   # quality tracking must never fail a prediction
        log.exception("forecast quality tracking failed")

    return PredictResponse(
        prediction=predictions[0],
        predictions=predictions,
        step_seconds=b.meta["step_seconds"],
        model=MODEL_NAME,
        stage=b.info["model_stage"],
        model_version=b.info["model_version"],
    )


@app.post("/reload", tags=["operations"], summary="Hot-reload model from MLflow")
def reload():
    """
    Re-fetch the model from MLflow (MODEL_ALIAS, falling back to MODEL_STAGE)
    without restarting the pod. The retraining job calls this after moving the alias.
    Returns the new model metadata on success. On failure the previously loaded
    model, if any, keeps serving.
    """
    log.info("Model reload triggered via POST /reload")
    try:
        load_model()
        return {"status": "reloaded", "model": _bundle.info}
    except Exception as exc:
        MODEL_LOADED.set(1 if _bundle is not None else 0)
        MODEL_RELOADS.labels(result="failure").inc()
        log.error("Model reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")
