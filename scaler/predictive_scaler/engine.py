"""
Scaling decision logic — independent of gRPC so it can be tested directly.

One decision per (ScaledObject, config):

    Prometheus window ──► normalise ──► POST /predict ──► denormalise
                                                             │
    observed (last sample) ──────────────────────────────────┤
                                                             ▼
                                  mode=max:      max(predicted, observed)
                                  mode=forecast: predicted
                                                             │
                                                             ▼
                                         reported demand (KEDA divides it by
                                         targetValue to get desired replicas)

Failure policy
--------------
* Forecast service down / not ready / bad answer  → report observed demand.
  The workload keeps scaling reactively, like a plain HPA, and
  predictive_scaler_forecast_failures_total records it.
* Not enough history for a full model window      → report observed demand.
* Prometheus down or query returns nothing        → raise. The gRPC layer turns
  this into an error so KEDA applies the ScaledObject's `fallback` replicas;
  inventing a number here would be worse than admitting we have no data.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .clients import ForecastClient, ForecastError, PrometheusClient, PrometheusError
from .config import ScalerConfig
from .metrics import ScalerMetrics

log = logging.getLogger("predictive-scaler.engine")


class NoDataError(PrometheusError):
    """The query succeeded but returned no samples."""


@dataclass(frozen=True)
class Decision:
    value: float                 # demand reported to KEDA
    observed: float              # last observed demand
    predicted: Optional[float]   # forecast demand, None when not available
    source: str                  # "forecast" | "observed"
    reason: str                  # human-readable explanation, logged

    def desired_replicas(self, target_value: float) -> int:
        return max(0, math.ceil(self.value / target_value))


class PredictiveEngine:
    def __init__(
        self,
        metrics: ScalerMetrics,
        prometheus: Optional[PrometheusClient] = None,
        forecast: Optional[ForecastClient] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.metrics = metrics
        self.prometheus = prometheus or PrometheusClient()
        self.forecast = forecast or ForecastClient()
        self._clock = clock
        self._cache: Dict[Tuple[str, str, str], Tuple[float, Decision]] = {}
        self._lock = threading.Lock()

    # KEDA calls GetMetrics on every HPA sync and IsActive on every polling
    # interval, often for the same ScaledObject within seconds. Caching keeps
    # Prometheus and the forecast service from being hit on every call.
    def decide(self, namespace: str, name: str, cfg: ScalerConfig) -> Decision:
        key = (namespace, name, cfg.cache_key)
        now = self._clock()
        if cfg.cache_seconds > 0:
            with self._lock:
                cached = self._cache.get(key)
                if cached and cached[0] > now:
                    return cached[1]

        decision = self._compute(namespace, name, cfg)

        if cfg.cache_seconds > 0:
            with self._lock:
                self._cache[key] = (now + cfg.cache_seconds, decision)
        return decision

    def _compute(self, namespace: str, name: str, cfg: ScalerConfig) -> Decision:
        labels = {"namespace": namespace, "scaledobject": name}

        try:
            timestamps, values = self.prometheus.fetch_series(
                cfg.prometheus_address, cfg.query, cfg.time_steps,
                cfg.step_seconds, cfg.timeout_seconds,
            )
        except PrometheusError:
            self.metrics.prometheus_failures.labels(**labels).inc()
            raise
        if not values:
            self.metrics.prometheus_failures.labels(**labels).inc()
            raise NoDataError(f"query returned no samples for {namespace}/{name}")

        observed = max(0.0, values[-1])
        predicted: Optional[float] = None

        if len(values) < cfg.time_steps:
            source = "observed"
            reason = (f"only {len(values)}/{cfg.time_steps} samples of history; "
                      "scaling on observed demand until the window fills")
        else:
            model_input = [v / cfg.capacity for v in values]
            started = time.perf_counter()
            try:
                # series + the last sample's time let forecast-serving score this
                # forecast against the actual value once a later window carries it
                horizon = self.forecast.predict(cfg.forecast_url, model_input, cfg.timeout_seconds,
                                                series=f"{namespace}/{name}", timestamp=timestamps[-1],
                                                step_seconds=cfg.step_seconds)
            except ForecastError as exc:
                self.metrics.forecast_failures.labels(**labels, reason=exc.reason).inc()
                source = "observed"
                reason = f"forecast unavailable ({exc}); falling back to observed demand"
            else:
                # next: the coming interval. max: the peak over the model's whole
                # horizon, which buys lead time equal to the horizon.
                model_out = max(horizon) if cfg.forecast_aggregation == "max" else horizon[0]
                predicted = max(0.0, model_out * cfg.capacity)
                source = "forecast"
                reason = (f"forecast {predicted:.4g} ({cfg.forecast_aggregation} of "
                          f"{len(horizon)} step(s)), observed {observed:.4g}, mode={cfg.mode}")
            finally:
                self.metrics.forecast_latency.observe(time.perf_counter() - started)

        if predicted is None:
            value = observed
        elif cfg.mode == "forecast":
            value = predicted
        else:
            # Never scale below what the workload needs right now: the forecast
            # can add capacity ahead of load but cannot remove capacity that is
            # currently in use.
            value = max(predicted, observed)

        decision = Decision(value=value, observed=observed, predicted=predicted,
                            source=source, reason=reason)

        self.metrics.observed_demand.labels(**labels).set(observed)
        if predicted is not None:
            self.metrics.predicted_demand.labels(**labels).set(predicted)
        else:
            # Drop the series instead of leaving a stale forecast on dashboards.
            try:
                self.metrics.predicted_demand.remove(namespace, name)
            except KeyError:
                pass
        self.metrics.reported_demand.labels(**labels).set(value)
        self.metrics.desired_replicas.labels(**labels).set(decision.desired_replicas(cfg.target_value))
        self.metrics.decisions.labels(**labels, source=source).inc()

        log.info("%s/%s: reported=%.4g source=%s (%s)", namespace, name, value, source, reason)
        return decision
