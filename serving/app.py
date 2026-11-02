"""
Workload forecast service: a window of recent observations in, a forecast
horizon out. The KEDA external scaler in scaler/ turns it into replicas.

Lifecycle
---------
The model comes from the MLflow registry (`models:/<name>@champion`, falling
back to the legacy `Production` stage for notebook-registered versions) and
is pulled, never pushed. There is no reload endpoint: with several replicas
behind one Service an HTTP reload reaches one pod and the others keep the
old version (split brain, replica flapping in the cluster). Instead a
lifecycle worker in every replica polls the registry alias every
MODEL_REFRESH_SECONDS and hot-swaps the model when the alias moved. The
swap happens under an asyncio.Lock so two loads never interleave, and the
served state is one immutable `Bundle` behind one reference, so a request
either sees the previous bundle or the new one, never a half-built mix.

Startup fetches the champion once, synchronously, before the port opens
(bounded by STARTUP_LOAD_TIMEOUT). If the registry is unreachable the pod
starts anyway, alive but not ready (`/readyz` → 503), serving the
last-known-good copy from MODEL_CACHE_DIR when there is one, and keeps
retrying every MODEL_RETRY_SECONDS; no CrashLoopBackOff for a registry
outage.

Inference
---------
`POST /predict` is asynchronous; the CPU-bound scikit-learn work (scaling,
features, predict, inverse scaling) runs in a bounded thread pool of
INFERENCE_THREADS workers, so KEDA polling a dozen ScaledObjects at once
never blocks the event loop that answers probes and metrics. Feature and
target transforms are shared with the training pipeline
(forecast_training.features, copied into the image), so both sides apply
the same code by construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pickle
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Union

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from pydantic import BaseModel, ConfigDict, Field, field_validator

from forecast_training.features import (FEATURE_KINDS, TARGET_KINDS, build_features_mv, from_target,
                                        target_channel)
from quality import ForecastTracker

# ── Configuration ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "cpu-forecasting")
# Registry alias moved by the retraining pipeline (training/). If the alias does
# not exist yet, the service falls back to the legacy stage the notebook uses.
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
# Only used for models without a forecast_meta.json (registered by the notebook).
TIME_STEPS = int(os.getenv("TIME_STEPS", "10"))
META_ARTIFACT = "forecast_meta.json"
CHANNELS_ARTIFACT = "channels.pkl"
# Forecast quality tracking (see quality.py): rolling-MAE window per horizon
# step, and memory bounds for the forecasts waiting to be scored. Exact with
# one replica; with several, set FORECAST_MAX_SERIES=0 and read the scaler's
# own scoring (predictive_scaler_forecast_abs_error) instead.
FORECAST_MAE_WINDOW = int(os.getenv("FORECAST_MAE_WINDOW", "100"))
FORECAST_MAX_SERIES = int(os.getenv("FORECAST_MAX_SERIES", "64"))
FORECAST_MAX_PENDING = int(os.getenv("FORECAST_MAX_PENDING", "512"))
# Startup: how long the synchronous first fetch may take before the pod starts
# without a model and the worker takes over.
STARTUP_LOAD_TIMEOUT = float(os.getenv("STARTUP_LOAD_TIMEOUT", "20"))
# While no model is loaded, the worker retries the registry this often (0
# disables). A fresh cluster has no champion until the first training run.
MODEL_RETRY_SECONDS = float(os.getenv("MODEL_RETRY_SECONDS", "30"))
# Once a model is loaded, the worker checks the alias this often and swaps the
# model when it moved (0 disables).
MODEL_REFRESH_SECONDS = float(os.getenv("MODEL_REFRESH_SECONDS", "60"))
# Last-known-good copy of the loaded model (model/, scalers, meta and tags).
# A pod that restarts while MLflow is down serves from it, with
# model_stage="cache", until the registry answers again. Empty: no cache.
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "")
# Threads for the CPU-bound inference; the event loop never runs it.
INFERENCE_THREADS = int(os.getenv("INFERENCE_THREADS", "4"))
# Input length policy. The model needs `input_length` values per channel
# (time_steps, plus one season for seasonal models). By default a longer
# history is accepted and its tail used, because the chart's scaler sends
# the same history length to every model; set "1" to reject anything but
# exactly input_length values.
STRICT_INPUT_LENGTH = os.getenv("STRICT_INPUT_LENGTH", "0") == "1"
# With the API behind mTLS (uvicorn --ssl-cert-reqs 2) Prometheus cannot scrape
# /metrics without a client certificate; set this to serve the same registry
# over plain HTTP on a second port (0 = off, /metrics on the API port only).
METRICS_PORT = int(os.getenv("METRICS_PORT", "0"))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("forecast-serving")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
# prometheus_fastapi_instrumentator covers HTTP-level metrics (rate, latency,
# status). These track the domain events HTTP metrics cannot see.
PREDICTION_REQUESTS = Counter(
    "forecast_prediction_requests_total",
    "Total prediction requests that reached inference",
)
PREDICTION_FAILURES = Counter(
    "forecast_prediction_failures_total",
    "Prediction failures: validation errors + inference errors",
    ["reason"],  # validation | inference
)
MODEL_RELOADS = Counter(
    "forecast_model_reloads_total",
    "Model loads from the registry by the lifecycle worker or at startup (success | failure)",
    ["result"],
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
MODEL_HOLDOUT_MAE = Gauge(
    "forecast_model_holdout_mae",
    "Holdout MAE the loaded model achieved in training, by horizon step (from the registry "
    "version's tags); the drift alert compares the live MAE with it",
    ["horizon_step"],
)
INPUT_FILLED = Counter(
    "forecast_input_filled_total",
    "Values filled in by interpolation in the seasonal (older) part of /predict inputs: gaps "
    "in the caller's history, or less than a season of it",
)
# Scores past forecasts against the actual values that later windows carry:
# forecast_abs_error / forecast_rolling_mae, both by horizon step.
TRACKER = ForecastTracker(max_series=FORECAST_MAX_SERIES, max_pending=FORECAST_MAX_PENDING,
                          mae_window=FORECAST_MAE_WINDOW)

# ── Model contract ─────────────────────────────────────────────────────────────
# Models without a forecast_meta.json artifact were registered by the notebook:
# TIME_STEPS scaled values in, one value out.
LEGACY_FORECAST_META: dict[str, Any] = {
    "time_steps": TIME_STEPS, "horizon": 1, "step_seconds": 300, "features": "window",
    "target": "level", "period_seconds": 86400, "season_steps": 0, "channels": [],
}


def input_length(meta: dict[str, Any]) -> int:
    """Samples a caller must send per channel: the window, preceded by one season for seasonal models."""
    return int(meta["time_steps"]) + int(meta.get("season_steps", 0))


def channel_names(meta: dict[str, Any]) -> list[str]:
    """Names of the model's input channels, the target first; [] for a one-series model."""
    return [str(c) for c in (meta.get("channels") or [])]


def prepare_input(values: list[Optional[float]], time_steps: int, length: int) -> tuple[np.ndarray, int]:
    """
    One channel's history → the model's input of exactly `length` values.
    The final `time_steps` (the window) must all be present. Gaps (null) in
    the older, seasonal part are filled by linear interpolation between the
    neighbouring samples: a missing scrape yesterday must not cost a forecast
    today, and a caller with less than a season of history (a new cluster)
    gets a forecast from the window alone, since a flat fill makes the
    season-ago change zero. Returns (input array, filled count); raises
    ValueError with a message fit for a 422.
    """
    if len(values) < length:
        extra = f" ({time_steps} window + {length - time_steps} of history)" if length > time_steps else ""
        raise ValueError(f"Expected at least {length} values{extra}, got {len(values)}.")
    if STRICT_INPUT_LENGTH and len(values) != length:
        raise ValueError(f"Expected exactly {length} values, got {len(values)}.")
    arr = np.array([math.nan if v is None else v for v in values[-length:]], dtype=float)
    if not np.all(np.isfinite(arr[-time_steps:])):
        raise ValueError(f"The last {time_steps} values (the window) must all be present.")
    missing = ~np.isfinite(arr)
    if missing.any():
        idx = np.arange(len(arr))
        arr[missing] = np.interp(idx[missing], idx[~missing], arr[~missing])
    return arr, int(missing.sum())


def prepare_channels(values: Union[list[Optional[float]], dict[str, list[Optional[float]]]],
                     meta: dict[str, Any]) -> tuple[np.ndarray, int, list[float]]:
    """
    A request's `values` (one list, or {channel: list}) → (input, filled,
    window): the model input as (channels, input_length) for a multivariate
    model or (input_length,) for a one-series one, the number of interpolated
    samples, and the target channel's window (the last time_steps values).
    Raises ValueError with a message fit for a 422.
    """
    time_steps, length = int(meta["time_steps"]), input_length(meta)
    names = channel_names(meta)
    if len(names) > 1:
        if not isinstance(values, dict):
            raise ValueError(f"This model takes {len(names)} channels {names}: send values as "
                             "{channel: [...]}.")
        missing = [n for n in names if n not in values]
        if missing:
            raise ValueError(f"Missing channel(s) {missing}; the model takes {names}.")
        rows, filled = [], 0
        for n in names:
            arr, f = prepare_input(values[n], time_steps, length)
            rows.append(arr)
            filled += f
        return np.stack(rows), filled, [float(v) for v in rows[0][-time_steps:]]
    if isinstance(values, dict):
        if len(values) != 1 or (names and names[0] not in values):
            raise ValueError(f"This model takes one series{(' named ' + repr(names[0])) if names else ''}; "
                             f"got channels {sorted(values)}.")
        values = next(iter(values.values()))
    arr, filled = prepare_input(values, time_steps, length)
    return arr, filled, [float(v) for v in arr[-time_steps:]]


@dataclass(frozen=True)
class Bundle:
    """Everything a forecast needs. Immutable and swapped as one reference, so
    a request never pairs a new model with the old scalers or metadata."""
    model: Any
    scalers: tuple                 # one MinMaxScaler per channel, the target first
    meta: dict[str, Any]           # forecast_meta.json merged over the legacy defaults
    info: dict[str, str]           # operational metadata (version, run_id, stage, loaded_at)
    tags: dict[str, str]           # the registry version's tags

    @property
    def scaler(self):
        """The target channel's scaler."""
        return self.scalers[0]

    def forecast(self, values: np.ndarray, timestamp: Optional[float] = None) -> list[float]:
        """
        CPU-bound: raw input (`input_length` values, or (channels, input_length)
        for a multivariate model; see prepare_channels) → forecast for each step
        of the model's horizon (raw units). `timestamp`: unix time of the last
        value, used by time-of-day features (defaults to now, right for a caller
        sending live windows). Called from the inference thread pool, never from
        the event loop.
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            scaled = self.scaler.transform(arr.reshape(-1, 1)).reshape(1, -1)
        else:
            scaled = np.stack([sc.transform(row.reshape(-1, 1)).ravel()
                               for sc, row in zip(self.scalers, arr)])[None, :, :]
        last_ts = np.array([timestamp if timestamp is not None else time.time()])
        out = self.model.predict(build_features_mv(scaled, self.meta["features"], last_ts,
                                                   int(self.meta.get("period_seconds", 86400)),
                                                   int(self.meta["time_steps"]),
                                                   int(self.meta["horizon"])))
        out = from_target(target_channel(scaled), out, self.meta["target"])
        return [float(v) for v in self.scaler.inverse_transform(out.reshape(-1, 1)).ravel()]


# ── Registry access (blocking; always called from a thread) ────────────────────

class RegistryError(RuntimeError):
    """The registry could not be reached or holds no usable version."""


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def resolve_version(client: MlflowClient):
    """The model version to serve: the alias if set, else the legacy stage.
    Returns (version object, model uri, selector label)."""
    try:
        if MODEL_ALIAS:
            try:
                v = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
                return v, f"models:/{MODEL_NAME}@{MODEL_ALIAS}", f"@{MODEL_ALIAS}"
            except MlflowException as exc:
                if not MODEL_STAGE:
                    raise RegistryError(f"alias @{MODEL_ALIAS} of {MODEL_NAME}: {exc}") from exc
                log.info("Alias @%s not found (%s); falling back to stage %s",
                         MODEL_ALIAS, exc.error_code, MODEL_STAGE)
        if not MODEL_STAGE:
            raise RegistryError("neither MODEL_ALIAS nor MODEL_STAGE is set")
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            raise RegistryError(f"no version of {MODEL_NAME} in stage {MODEL_STAGE!r}")
        return versions[0], f"models:/{MODEL_NAME}/{MODEL_STAGE}", MODEL_STAGE
    except RegistryError:
        raise
    except (MlflowException, OSError, ConnectionError, TimeoutError) as exc:   # network, auth, 5xx
        raise RegistryError(f"registry at {MLFLOW_TRACKING_URI} unavailable: {exc}") from exc


def fetch_bundle() -> Bundle:
    """
    Download the champion from the registry: the sklearn model, the scaler
    artifacts (`scaler/scaler.pkl` for the target, `scaler/channels.pkl` for
    every channel of a multivariate model) and `forecast_meta.json`. Blocking:
    network and unpickling. Raises RegistryError when the registry cannot be
    reached, RuntimeError when the artifacts are unusable.
    """
    client = _client()
    v, model_uri, selector = resolve_version(client)
    run_id, version = v.run_id, str(v.version)   # the REST store returns strings, the SQL store ints
    try:
        model = mlflow.sklearn.load_model(model_uri)
        with tempfile.TemporaryDirectory() as tmp:
            scaler_dir = client.download_artifacts(run_id, "scaler", dst_path=tmp)
            names = sorted(os.listdir(scaler_dir))
            with open(os.path.join(scaler_dir, "scaler.pkl" if "scaler.pkl" in names else names[0]), "rb") as f:
                scalers = (pickle.load(f),)
            if CHANNELS_ARTIFACT in names:
                with open(os.path.join(scaler_dir, CHANNELS_ARTIFACT), "rb") as f:
                    scalers = tuple(pickle.load(f))
            try:
                meta_path = client.download_artifacts(run_id, META_ARTIFACT, dst_path=tmp)
                with open(meta_path) as f:
                    meta = {**LEGACY_FORECAST_META, **json.load(f)}
            except (MlflowException, OSError):
                meta = dict(LEGACY_FORECAST_META)          # registered by the notebook
    except (MlflowException, OSError, ConnectionError, TimeoutError) as exc:
        raise RegistryError(f"artifacts of {MODEL_NAME} v{version} unavailable: {exc}") from exc
    tags = dict(v.tags or {})
    info = {
        "model_name": MODEL_NAME,
        "model_stage": selector,
        "model_version": version,
        "run_id": run_id,
        "candidate": tags.get("candidate", ""),   # which candidate the pipeline promoted
    }
    return _validated(Bundle(model, scalers, meta, info, tags))


def _validated(bundle: Bundle) -> Bundle:
    meta = bundle.meta
    if meta["features"] not in FEATURE_KINDS or meta["target"] not in TARGET_KINDS:
        raise RuntimeError(f"unsupported {META_ARTIFACT}: {meta}")
    if len(bundle.scalers) != max(1, len(channel_names(meta))):
        raise RuntimeError(f"{len(bundle.scalers)} scaler(s) for channels {channel_names(meta)}")
    return bundle


# ── Last-known-good cache ──────────────────────────────────────────────────────

def write_cache(bundle: Bundle) -> None:
    """Write the bundle to MODEL_CACHE_DIR atomically (a new directory renamed
    over the old one) so a crash mid-write leaves the previous cache intact."""
    final = os.path.join(MODEL_CACHE_DIR, "champion")
    tmp = os.path.join(MODEL_CACHE_DIR, ".champion.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    mlflow.sklearn.save_model(bundle.model, os.path.join(tmp, "model"))
    with open(os.path.join(tmp, "scalers.pkl"), "wb") as f:
        pickle.dump(list(bundle.scalers), f)
    with open(os.path.join(tmp, "cache.json"), "w") as f:
        json.dump({"meta": bundle.meta, "info": bundle.info, "tags": bundle.tags}, f)
    old = final + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(final):
        os.rename(final, old)
    os.rename(tmp, final)
    shutil.rmtree(old, ignore_errors=True)
    log.info("model cache written: %s (v%s)", final, bundle.info["model_version"])


def read_cache() -> Bundle:
    """The last-known-good bundle from MODEL_CACHE_DIR, marked model_stage="cache"."""
    if not MODEL_CACHE_DIR:
        raise RuntimeError("MODEL_CACHE_DIR is not set")
    final = os.path.join(MODEL_CACHE_DIR, "champion")
    with open(os.path.join(final, "cache.json")) as f:
        saved = json.load(f)
    model = mlflow.sklearn.load_model(os.path.join(final, "model"))
    with open(os.path.join(final, "scalers.pkl"), "rb") as f:
        scalers = tuple(pickle.load(f))
    info = {**saved["info"], "model_stage": "cache"}
    return _validated(Bundle(model, scalers, {**LEGACY_FORECAST_META, **saved["meta"]}, info,
                             saved.get("tags", {})))


# ── Model store: the one reference requests read, the lock writers take ───────

class ModelStore:
    """
    Holds the served Bundle. Readers take no lock: they read one reference,
    and a Bundle is immutable, so they see the previous or the new model in
    full. Writers serialise on `lock` (asyncio.Lock for the worker and
    startup; a threading.Lock under it for the synchronous entry points the
    tests and tooling use), so two loads never interleave and metrics are
    updated once per swap.
    """

    def __init__(self) -> None:
        self._bundle: Optional[Bundle] = None
        self._swap = threading.Lock()
        self._lock: Optional[asyncio.Lock] = None   # created in the running loop

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def get(self) -> Optional[Bundle]:
        return self._bundle

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    # synchronous entry points (blocking; the worker calls them in a thread)

    def install(self, bundle: Bundle) -> None:
        """Hot-swap: one reference assignment under the swap lock, then metrics."""
        info = {**bundle.info, "loaded_at": datetime.now(timezone.utc).isoformat()}
        bundle = Bundle(bundle.model, bundle.scalers, bundle.meta, info, bundle.tags)
        with self._swap:
            previous = self._bundle
            self._bundle = bundle
            if previous is not None and previous.info["run_id"] != bundle.info["run_id"]:
                TRACKER.reset(pending_only=True)   # the old model's forecasts are not the new one's errors
            MODEL_LOADED.set(1)
            MODEL_INFO.clear()                     # one series only: drop the previous version's labels
            MODEL_INFO.labels(model_name=MODEL_NAME, model_version=info["model_version"],
                              model_stage=info["model_stage"], run_id=info["run_id"],
                              horizon=str(bundle.meta["horizon"])).set(1)
            MODEL_HOLDOUT_MAE.clear()
            for key, raw in bundle.tags.items():
                if key.startswith("holdout_mae_step_"):
                    try:
                        MODEL_HOLDOUT_MAE.labels(horizon_step=key.rsplit("_", 1)[1]).set(float(raw))
                    except ValueError:
                        log.warning("ignoring tag %s=%r on v%s", key, raw, info["model_version"])
        log.info("Model loaded: name=%s %s version=%s run_id=%s candidate=%s time_steps=%s horizon=%s "
                 "channels=%s", MODEL_NAME, info["model_stage"], info["model_version"], info["run_id"],
                 info.get("candidate") or "-", bundle.meta["time_steps"], bundle.meta["horizon"],
                 channel_names(bundle.meta) or "-")

    def load_from_registry(self) -> Bundle:
        """Fetch the champion and swap it in. Raises on failure; the current
        model, if any, keeps serving."""
        try:
            bundle = fetch_bundle()
        except Exception:
            MODEL_RELOADS.labels(result="failure").inc()
            if self._bundle is None:
                MODEL_LOADED.set(0)
            raise
        self.install(bundle)
        MODEL_RELOADS.labels(result="success").inc()
        if MODEL_CACHE_DIR:
            try:
                write_cache(self._bundle)        # a convenience, never a reason to fail a load
            except Exception as exc:
                log.warning("could not write the model cache at %s: %s", MODEL_CACHE_DIR, exc)
        return bundle

    def load_from_cache(self) -> Bundle:
        bundle = read_cache()
        self.install(bundle)
        log.warning("serving model v%s from the cache at %s; the registry is unavailable",
                    bundle.info["model_version"], MODEL_CACHE_DIR)
        return bundle

    def refresh(self) -> bool:
        """One registry check: swap when the alias points elsewhere than what is
        served, or when the served model came from the cache. True on a swap.
        A registry outage logs a warning and keeps the current model."""
        current = self._bundle
        if current is None:
            return False
        try:
            v, _, _ = resolve_version(_client())
        except RegistryError as exc:
            log.warning("model refresh check failed (%s); keeping v%s", exc, current.info["model_version"])
            return False
        if str(v.version) == current.info["model_version"] and current.info["model_stage"] != "cache":
            return False
        log.info("registry alias moved to v%s (serving v%s %s); loading", v.version,
                 current.info["model_version"], current.info["model_stage"])
        try:
            self.load_from_registry()
        except Exception as exc:
            log.warning("model load failed (%s); keeping v%s", exc, current.info["model_version"])
            return False
        return True

    # asynchronous entry points (the worker and startup)

    async def load_from_registry_async(self) -> Bundle:
        async with self.lock:
            return await asyncio.to_thread(self.load_from_registry)

    async def load_from_cache_async(self) -> Bundle:
        async with self.lock:
            return await asyncio.to_thread(self.load_from_cache)

    async def refresh_async(self) -> bool:
        async with self.lock:
            return await asyncio.to_thread(self.refresh)


STORE = ModelStore()


# Synchronous names the tests and tooling use.
def load_model() -> None:
    STORE.load_from_registry()


def load_cached_model() -> None:
    STORE.load_from_cache()


def refresh_model() -> bool:
    return STORE.refresh()


# ── Lifecycle worker ───────────────────────────────────────────────────────────

async def lifecycle_worker() -> None:
    """
    Runs in every replica for the life of the process. Without a model: retry
    the registry every MODEL_RETRY_SECONDS. With one: check the alias every
    MODEL_REFRESH_SECONDS and swap on a move. Every registry call is wrapped;
    an outage is logged and never unloads a working model or ends the loop.
    """
    attempt = 0
    while True:
        if not STORE.loaded:
            interval = MODEL_RETRY_SECONDS
            attempt += 1
            try:
                await STORE.load_from_registry_async()
                attempt = 0
            except Exception as exc:
                if attempt == 1 or attempt % 10 == 0:
                    log.warning("still no model after %d attempt(s): %s; retrying every %gs",
                                attempt, exc, MODEL_RETRY_SECONDS)
        else:
            interval = MODEL_REFRESH_SECONDS
            try:
                await STORE.refresh_async()
            except Exception as exc:                    # refresh() handles its own errors; belt and braces
                log.warning("model refresh failed: %s", exc)
        if interval <= 0:
            log.info("lifecycle worker stopped: interval is 0")
            return
        await asyncio.sleep(interval)


async def startup_load() -> None:
    """
    The synchronous first fetch, bounded by STARTUP_LOAD_TIMEOUT. On failure
    the pod starts alive but not ready, from the cache when one exists.
    """
    try:
        await asyncio.wait_for(STORE.load_from_registry_async(), timeout=STARTUP_LOAD_TIMEOUT)
        return
    except asyncio.TimeoutError:
        log.critical("registry did not answer within %gs at startup; starting without a model",
                     STARTUP_LOAD_TIMEOUT)
    except Exception as exc:
        log.critical("could not load the model at startup (%s); starting without a model", exc)
    if MODEL_CACHE_DIR:
        try:
            await STORE.load_from_cache_async()
        except Exception as exc:
            log.info("no usable model cache at %s (%s)", MODEL_CACHE_DIR, exc)


_INFERENCE_POOL: Optional[ThreadPoolExecutor] = None


def inference_pool() -> ThreadPoolExecutor:
    """The bounded pool the CPU-bound inference runs in. Owned by the lifespan
    (shut down with the app) and recreated on demand, so a process that serves
    several app lifetimes (tests, embedding) never hits a closed executor."""
    global _INFERENCE_POOL
    pool = _INFERENCE_POOL
    if pool is None or pool._shutdown:
        pool = _INFERENCE_POOL = ThreadPoolExecutor(max_workers=INFERENCE_THREADS,
                                                    thread_name_prefix="inference")
    return pool


@asynccontextmanager
async def lifespan(application: FastAPI):
    log.info("Workload Forecast Service starting — MODEL=%s ALIAS=%s STAGE=%s MLFLOW=%s "
             "refresh=%gs retry=%gs inference_threads=%d",
             MODEL_NAME, MODEL_ALIAS or "-", MODEL_STAGE or "-", MLFLOW_TRACKING_URI,
             MODEL_REFRESH_SECONDS, MODEL_RETRY_SECONDS, INFERENCE_THREADS)
    if METRICS_PORT:
        from prometheus_client import start_http_server
        start_http_server(METRICS_PORT)
        log.info("Plain-HTTP /metrics on :%d (the API port may require client certificates)", METRICS_PORT)
    inference_pool()
    await startup_load()                      # before the port opens
    worker = asyncio.create_task(lifecycle_worker(), name="model-lifecycle")
    if STORE.loaded:
        log.info("Ready: serving v%s", STORE.get().info["model_version"])
    else:
        log.warning("Started without a model: /readyz is 503 until the worker loads one")
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except (asyncio.CancelledError, Exception):
            pass
        inference_pool().shutdown(wait=False, cancel_futures=True)
        log.info("Workload Forecast Service shutting down")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Workload Forecast Service",
    description=(
        "Inference service for infrastructure workload forecasting: a window of recent "
        "observations in, a forecast horizon out. The KEDA external scaler in scaler/ "
        "turns it into replica counts."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
# Inference is sub-millisecond; the default (0.1, 0.5, 1) buckets could not
# resolve the p99 < 200 ms alert.
Instrumentator().add(metrics.default(
    latency_lowr_buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2.5),
)).instrument(app).expose(app)


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """One request: the recent history of the series to forecast."""
    model_config = ConfigDict(extra="forbid")

    # The last `input_length` values of the series at `step_seconds` intervals
    # (both in /model-info), raw (unscaled); the service applies the stored
    # scaler. The final `time_steps` values must be present; older ones may be
    # null where the caller has a gap and are interpolated. A longer history is
    # accepted and its tail used unless STRICT_INPUT_LENGTH is set. A
    # multivariate model takes one such list per channel: {"cpu": [...],
    # "rps": [...]} (names in /model-info).
    values: Union[list[Optional[float]], dict[str, list[Optional[float]]]]
    # Optional, for forecast quality tracking: which series this window belongs
    # to (callers forecasting several workloads pass a distinct key per
    # workload), the unix time of the last value (default: arrival time) and the
    # spacing of the values (default: the model's step_seconds; a caller
    # compressing time, e.g. the demo with stepSeconds=30, must say so).
    series: str = Field(default="", max_length=253)
    timestamp: Optional[float] = None
    step_seconds: Optional[int] = None

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, v):
        rows = v.items() if isinstance(v, dict) else [("", v)]
        if not rows:
            raise ValueError("values must not be empty")
        for name, row in rows:
            where = f"values[{name!r}]" if name else "values"
            if not row:
                raise ValueError(f"{where} must not be empty")
            if row[-1] is None:
                raise ValueError(f"the last value of {where} must be present")
            for i, x in enumerate(row):
                if x is not None and not math.isfinite(x):
                    raise ValueError(f"{where}[{i}] is not finite: {x!r}")
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

    prediction: float                 # next interval (first step of `predictions`)
    predictions: list[float]          # one value per step of the model's horizon
    step_seconds: int                 # spacing of `predictions`
    model: str
    stage: str                        # registry selector the model was loaded from (or "cache")
    model_version: Optional[str] = None


class ReadyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: Optional[str] = None
    model_stage: Optional[str] = None


# ── Probes and operations ──────────────────────────────────────────────────────

@app.get("/healthz", tags=["operations"], summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """200 while the process and its event loop answer. Says nothing about the
    model: a pod waiting for the registry is alive and must not be restarted."""
    return {"status": "alive"}


@app.get("/readyz", tags=["operations"], summary="Readiness probe", response_model=ReadyResponse)
async def readyz() -> ReadyResponse:
    """200 with the served version once a model is loaded; 503 while the store
    is empty (startup without a registry, a fresh cluster with no champion
    yet), which is how Kubernetes keeps traffic away from this pod."""
    b = STORE.get()
    if b is None:
        raise HTTPException(status_code=503, detail="Model not loaded; the lifecycle worker is retrying.")
    return ReadyResponse(status="ready", model_version=b.info["model_version"],
                         model_stage=b.info["model_stage"])


# Previous probe paths, kept so charts and scripts of either generation work.
app.add_api_route("/live", healthz, methods=["GET"], include_in_schema=False)
app.add_api_route("/ready", readyz, methods=["GET"], include_in_schema=False)


@app.get("/model-info", tags=["operations"], summary="Loaded model metadata")
async def model_info() -> dict[str, Any]:
    """Operational metadata about the served model: version, run id, the input
    contract (input_length, channels, step) and the live rolling MAE."""
    b = STORE.get()
    meta = b.meta if b is not None else LEGACY_FORECAST_META
    return {
        "model_loaded": b is not None,
        "time_steps": meta["time_steps"],
        # values a /predict call must carry per channel: time_steps, plus a season for seasonal models
        "input_length": input_length(meta),
        "season_steps": meta.get("season_steps", 0),
        "channels": channel_names(meta),       # [] for a one-series model; else the names, target first
        "horizon": meta["horizon"],
        "step_seconds": meta["step_seconds"],
        "features": meta["features"],
        "target": meta["target"],
        "period_seconds": meta.get("period_seconds", 86400),
        "refresh_seconds": MODEL_REFRESH_SECONDS,
        "rolling_mae": {str(k): round(v, 6) for k, v in sorted(TRACKER.snapshot().items())},
        **(b.info if b is not None else {}),
    }


# ── Inference ──────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest) -> PredictResponse:
    """
    Forecast the series for the next interval(s).

    Input: the last `input_length` observations (raw, unscaled; see
    /model-info), one list per channel for a multivariate model, gaps in the
    older part as null. Output: `prediction` for the next interval,
    `predictions` for every step of the horizon, and model provenance
    (`model_version` lets a caller see a swap between requests).

    The bundle is read once at the top: a swap during the request does not
    mix two models' parts. Scaling, features and the scikit-learn predict run
    in the inference thread pool; the event loop only awaits them.
    """
    b = STORE.get()
    if b is None:
        raise HTTPException(status_code=503, detail="Model not loaded; the lifecycle worker is retrying.")

    try:
        model_input, filled, window = prepare_channels(req.values, b.meta)
    except ValueError as exc:
        PREDICTION_FAILURES.labels(reason="validation").inc()
        raise HTTPException(status_code=422, detail=str(exc))
    if filled:
        INPUT_FILLED.inc(filled)

    PREDICTION_REQUESTS.inc()
    try:
        loop = asyncio.get_running_loop()
        predictions = await loop.run_in_executor(inference_pool(), b.forecast, model_input, req.timestamp)
    except Exception as exc:
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error. Check service logs.")

    if not all(math.isfinite(p) for p in predictions):
        PREDICTION_FAILURES.labels(reason="inference").inc()
        log.error("Model v%s returned a non-finite prediction %r for window %s",
                  b.info["model_version"], predictions, window)
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction.")

    try:
        # the window only: filled-in history is not an actual value to score against
        TRACKER.record(req.series, window, predictions,
                       req.step_seconds or int(b.meta["step_seconds"]), req.timestamp)
    except Exception:   # quality tracking must never fail a prediction
        log.exception("forecast quality tracking failed")

    return PredictResponse(
        prediction=predictions[0],
        predictions=predictions,
        step_seconds=int(b.meta["step_seconds"]),
        model=MODEL_NAME,
        stage=b.info["model_stage"],
        model_version=b.info["model_version"],
    )
