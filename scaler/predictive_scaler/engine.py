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

from .breaker import BreakerConfig, CircuitBreaker
from .clients import ForecastClient, ForecastError, PrometheusClient, PrometheusError
from .config import ScalerConfig
from .metrics import ScalerMetrics
from .quality import ForecastLedger

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
        # past forecasts, scored against the history each decision fetches
        self.ledger = ForecastLedger()
        # one breaker per ScaledObject; its thresholds come from the trigger metadata
        self._breakers: Dict[Tuple[str, str], CircuitBreaker] = {}

    def _channels(self, cfg: ScalerConfig, grid_ts, length: int) -> Dict[str, list]:
        """
        Each extra channel's history, aligned to the target's grid: fetched up to
        the target's newest instant, shifted onto its timestamps, holes as None.
        A channel whose newest sample lags the target's (a slower scrape) is
        carried forward over the trailing gap: a slightly stale rate is a fine
        input, a missing one would cost the forecast.
        """
        out: Dict[str, list] = {}
        newest = grid_ts[-1]
        for name, query in cfg.channels:
            ts, vals = self.prometheus.fetch_history(cfg.prometheus_address, query, cfg.history_steps,
                                                     cfg.step_seconds, cfg.timeout_seconds, now=newest)
            by_ts = {t: v for t, v in zip(ts, vals)}
            row = [by_ts.get(t) for t in grid_ts[-length:]]
            last = next((v for v in reversed(row) if v is not None), None)
            if last is not None:
                for i in range(len(row) - 1, -1, -1):
                    if row[i] is not None:
                        break
                    row[i] = last
            out[name] = row
        return out

    def _breaker(self, key: str, cfg: ScalerConfig) -> CircuitBreaker:
        bk = (key, cfg.cache_key)
        if bk not in self._breakers:
            self._breakers = {k: v for k, v in self._breakers.items() if k[0] != key}
            self._breakers[bk] = CircuitBreaker(BreakerConfig(
                enabled=cfg.breaker, residual_z=cfg.breaker_residual_z,
                residual_z_hard=2 * cfg.breaker_residual_z,
                level_shift_high=cfg.breaker_level_shift, level_shift_low=1 / cfg.breaker_level_shift,
                jump_factor=cfg.breaker_jump_factor, min_open_seconds=cfg.breaker_min_open_seconds,
                max_open_seconds=max(cfg.breaker_min_open_seconds, 3600.0),
                calm_polls=cfg.breaker_calm_polls), clock=time.time)
        return self._breakers[bk]

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
            # history_steps grid instants (None where a sample is missing): the
            # window the model's features are built on, preceded by the older
            # history a seasonal model reads
            grid_ts, history = self.prometheus.fetch_history(
                cfg.prometheus_address, cfg.query, cfg.history_steps,
                cfg.step_seconds, cfg.timeout_seconds,
            )
        except PrometheusError:
            self.metrics.prometheus_failures.labels(**labels).inc()
            raise
        if not history:
            self.metrics.prometheus_failures.labels(**labels).inc()
            raise NoDataError(f"query returned no samples for {namespace}/{name}")

        # The window: the trailing run of consecutive samples (the grid ends at
        # the newest sample, so it is never empty). A hole shortens it.
        n = 0
        while n < len(history) and history[-1 - n] is not None:
            n += 1
        timestamps, values = grid_ts[len(history) - n:], history[len(history) - n:]

        observed = max(0.0, values[-1])
        predicted: Optional[float] = None
        key = f"{namespace}/{name}"

        # Score the forecasts made earlier for instants this history now holds.
        residuals = []
        for r in self.ledger.score(key, grid_ts, history, cfg.capacity):
            self.metrics.forecast_abs_error.labels(**labels, step=str(r.step)).observe(abs(r.error))
            if r.step == 1:
                self.metrics.forecast_residual.labels(**labels).set(r.error)
                residuals.append(r.error)

        # Does the present resemble the history the model learned? (model units)
        verdict = self._breaker(key, cfg).observe(
            key, observed / cfg.capacity,
            [None if v is None else v / cfg.capacity for v in history],
            cfg.history_steps - cfg.time_steps, residuals)
        self.metrics.breaker_open.labels(**labels).set(0 if verdict.use_forecast else 1)
        if verdict.tripped:
            self.metrics.breaker_trips.labels(**labels, reason=verdict.tripped).inc()

        if len(values) < cfg.time_steps:
            source = "observed"
            reason = (f"only {len(values)}/{cfg.time_steps} samples of history; "
                      "scaling on observed demand until the window fills")
        else:
            if cfg.history_steps > cfg.time_steps:
                # the whole history, holes as null: serving interpolates the old part
                model_input = [None if v is None else v / cfg.capacity for v in history]
            else:
                model_input = [v / cfg.capacity for v in values[-cfg.time_steps:]]
            if cfg.channels:
                # multivariate: the extra series on the target's grid, keyed by channel name
                try:
                    model_input = {cfg.target_channel: model_input,
                                   **self._channels(cfg, grid_ts, len(model_input))}
                except PrometheusError:
                    self.metrics.prometheus_failures.labels(**labels).inc()
                    raise
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
                self.ledger.record(key, timestamps[-1], cfg.step_seconds, horizon)
                for k, v in enumerate(horizon, start=1):
                    self.metrics.forecast_demand.labels(**labels, step=str(k)).set(max(0.0, v * cfg.capacity))
                reason = (f"forecast {predicted:.4g} ({cfg.forecast_aggregation} of "
                          f"{len(horizon)} step(s)), observed {observed:.4g}, mode={cfg.mode}")
                cap = observed * cfg.forecast_surge_factor
                if cfg.mode == "max" and cfg.forecast_surge_factor > 0 and observed > 0 and predicted > cap:
                    reason += f"; capped at {cfg.forecast_surge_factor:g}x observed ({cap:.4g})"
                    predicted = cap
            finally:
                self.metrics.forecast_latency.observe(time.perf_counter() - started)

        if predicted is not None and not verdict.use_forecast:
            # The breaker is open: the forecast is recorded (metrics, scoring)
            # but the decision is reactive until the series looks normal again.
            self.metrics.breaker_bypass.labels(**labels).inc()
            source = "observed"
            reason = f"{verdict.reason}; forecast {predicted:.4g} bypassed, observed {observed:.4g}"
            value = observed
        elif predicted is None:
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
            for k in range(1, 65):
                try:
                    self.metrics.forecast_demand.remove(namespace, name, str(k))
                except KeyError:
                    break
        self.metrics.reported_demand.labels(**labels).set(value)
        self.metrics.desired_replicas.labels(**labels).set(decision.desired_replicas(cfg.target_value))
        self.metrics.decisions.labels(**labels, source=source).inc()

        log.info("%s/%s: reported=%.4g source=%s (%s)", namespace, name, value, source, reason)
        return decision
