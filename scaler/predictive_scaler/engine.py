"""
Scaling decision logic — independent of gRPC so it can be tested directly.

One decision per (ScaledObject, config):

    Prometheus window ──► normalise ──► POST /predict ──► denormalise
                                                             │
    observed (last sample) ──────────────────────────────────┤
                                                             ▼
                          conditioning (smoothing.py): service breaker,
                          surge cap, hysteresis gate, EWMA
                                                             │
                                                             ▼
                                  mode=max:      max(smoothed, observed)
                                  mode=forecast: smoothed
                                                             │
                                                             ▼
                                         reported demand (KEDA divides it by
                                         targetValue to get desired replicas)

Failure policy
--------------
* Forecast service down / not ready / bad answer  → report observed demand.
  The workload keeps scaling reactively, like a plain HPA, and
  predictive_scaler_forecast_failures_total records it. After
  forecastBreakerFailures such failures in a row the service breaker opens
  and the service is not called for forecastBreakerOpenSeconds
  (predictive_scaler_forecast_service_breaker_open); one probe then decides.
* Forecast rise shorter than spikeMinSteps consecutive horizon steps → the
  hysteresis gate suppresses it (predictive_scaler_spikes_suppressed_total)
  and the decision rests on observed demand.
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
from .capacity import CapacityRef, CapacityResolver
from .clients import ForecastClient, ForecastError, PrometheusClient, PrometheusError
from .config import ScalerConfig
from .metrics import ScalerMetrics
from .quality import ForecastLedger
from .smoothing import EwmaSmoother, ServiceBreaker, ServiceBreakerConfig, spike_gate

log = logging.getLogger("predictive-scaler.engine")


class NoDataError(PrometheusError):
    """The query succeeded but returned no samples."""


@dataclass(frozen=True)
class Decision:
    value: float                 # demand reported to KEDA
    observed: float              # last observed demand
    predicted: Optional[float]   # forecast demand (after the surge cap), None when not available
    source: str                  # "forecast" | "observed"
    reason: str                  # human-readable explanation, logged
    smoothed: Optional[float] = None   # the EWMA of the forecast the decision used, if it did

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
        # metric conditioning, keyed by namespace/name (smoothing.py)
        self.smoother = EwmaSmoother(clock=clock)
        # the reference capacity in cores: capacityQuery at runtime, else the static key
        self.capacity = CapacityResolver(self.prometheus, clock=clock)
        self._service_breakers: Dict[Tuple[str, str], ServiceBreaker] = {}

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
                jump_factor=cfg.breaker_jump_factor, jump_min_samples=cfg.breaker_jump_min_samples,
                min_open_seconds=cfg.breaker_min_open_seconds,
                max_open_seconds=max(cfg.breaker_min_open_seconds, 3600.0),
                calm_polls=cfg.breaker_calm_polls), clock=time.time)
        return self._breakers[bk]

    def _service_breaker(self, key: str, cfg: ScalerConfig) -> ServiceBreaker:
        bk = (key, cfg.cache_key)
        with self._lock:
            if bk not in self._service_breakers:
                self._service_breakers = {k: v for k, v in self._service_breakers.items() if k[0] != key}
                self._service_breakers[bk] = ServiceBreaker(
                    ServiceBreakerConfig(failures=cfg.forecast_breaker_failures,
                                         open_seconds=cfg.forecast_breaker_open_seconds),
                    clock=self._clock)
            return self._service_breakers[bk]

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
        # Cores the model's fractions refer to: everything below in model
        # units is demand / ref.value, and a forecast is coefficient x ref.value.
        ref = self.capacity.resolve(key, cfg)
        self.metrics.capacity_reference.labels(**labels, source=ref.source).set(ref.value)
        other = "static" if ref.source == "query" else "query"
        try:
            self.metrics.capacity_reference.remove(namespace, name, other)
        except KeyError:
            pass

        # Score the forecasts made earlier for instants this history now holds
        # (each against the reference it was made with).
        residuals = []
        for r in self.ledger.score(key, grid_ts, history, ref.value):
            self.metrics.forecast_abs_error.labels(**labels, step=str(r.step)).observe(abs(r.error))
            if r.step == 1:
                self.metrics.forecast_residual.labels(**labels).set(r.error)
                residuals.append(r.error)

        # Does the present resemble the history the model learned? (model units)
        verdict = self._breaker(key, cfg).observe(
            key, observed / ref.value,
            [None if v is None else v / ref.value for v in history],
            cfg.history_steps - cfg.time_steps, residuals)
        self.metrics.breaker_open.labels(**labels).set(0 if verdict.use_forecast else 1)
        if verdict.tripped:
            self.metrics.breaker_trips.labels(**labels, reason=verdict.tripped).inc()

        smoothed: Optional[float] = None
        horizon_demand: list = []
        gate_reason = ""
        service = self._service_breaker(key, cfg)
        if len(values) < cfg.time_steps:
            source = "observed"
            reason = (f"only {len(values)}/{cfg.time_steps} samples of history; "
                      "scaling on observed demand until the window fills")
        else:
            admission = service.admit(key)
            if not admission.allowed:
                # The service failed too often: reactive mode without paying
                # the call (or its timeout) at all until the hold elapses.
                self.metrics.forecast_skipped.labels(**labels).inc()
                source = "observed"
                reason = f"{admission.reason}; reactive mode on observed demand"
            else:
                source, reason, predicted, horizon_demand = self._forecast(
                    namespace, name, cfg, labels, key, grid_ts, history, values, timestamps, observed, service,
                    ref)
        self.metrics.forecast_service_breaker_open.labels(**labels).set(0 if service.state(key) == "closed" else 1)

        if predicted is not None:
            # Hysteresis gate on the horizon vector (demand units): a rise that
            # does not last spikeMinSteps consecutive steps is noise.
            gate = spike_gate(horizon_demand, observed, cfg.target_value, cfg.spike_min_steps)
            if gate.suppressed:
                self.metrics.spikes_suppressed.labels(**labels).inc()
                gate_reason = gate.reason
                log.info("event=spike_suppressed namespace=%s scaledobject=%s observed=%.4g forecast=%.4g "
                         "rise_steps=%d longest_run=%d min_steps=%d horizon=%s",
                         namespace, name, observed, predicted, gate.rise_steps, gate.longest_run,
                         cfg.spike_min_steps, [round(h, 4) for h in horizon_demand])

        if predicted is not None and not verdict.use_forecast:
            # The anomaly breaker is open: the forecast is recorded (metrics,
            # scoring) but the decision is reactive until the series looks normal again.
            self.metrics.breaker_bypass.labels(**labels).inc()
            source = "observed"
            reason = f"{verdict.reason}; forecast {predicted:.4g} bypassed, observed {observed:.4g}"
            value = observed
        elif predicted is not None and gate_reason:
            source = "observed"
            reason = f"{gate_reason}; forecast {predicted:.4g} suppressed, observed {observed:.4g}"
            value = observed
        elif predicted is None:
            value = observed
        else:
            # EWMA on the forecast that will be used: S_t = a*Y_t + (1-a)*S_{t-1}.
            sm = self.smoother.update(key, predicted, cfg.smoothing_alpha, cfg.smoothing_max_age_seconds)
            smoothed = max(0.0, sm.value)
            self.metrics.smoothed_demand.labels(**labels).set(smoothed)
            if not sm.seeded and abs(sm.damped) > 1e-9:
                reason += f"; smoothed to {smoothed:.4g} (alpha={cfg.smoothing_alpha:g}, previous {sm.previous:.4g})"
                log.info("event=forecast_smoothed namespace=%s scaledobject=%s raw=%.4g smoothed=%.4g "
                         "previous=%.4g alpha=%g", namespace, name, predicted, smoothed, sm.previous,
                         cfg.smoothing_alpha)
            if cfg.mode == "forecast":
                value = smoothed
            else:
                # Never scale below what the workload needs right now: the forecast
                # can add capacity ahead of load but cannot remove capacity that is
                # currently in use.
                value = max(smoothed, observed)

        decision = Decision(value=value, observed=observed, predicted=predicted,
                            source=source, reason=reason, smoothed=smoothed)

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
        if smoothed is None:
            try:
                self.metrics.smoothed_demand.remove(namespace, name)
            except KeyError:
                pass
        self.metrics.reported_demand.labels(**labels).set(value)
        self.metrics.desired_replicas.labels(**labels).set(decision.desired_replicas(cfg.target_value))
        self.metrics.decisions.labels(**labels, source=source).inc()

        log.info("event=decision namespace=%s scaledobject=%s source=%s reported=%.4g observed=%.4g "
                 "forecast=%s smoothed=%s replicas=%d service_breaker=%s anomaly_breaker=%s reason=%r",
                 namespace, name, source, value, observed,
                 f"{predicted:.4g}" if predicted is not None else "-",
                 f"{smoothed:.4g}" if smoothed is not None else "-",
                 decision.desired_replicas(cfg.target_value), service.state(key), verdict.state, reason)
        return decision

    def _forecast(self, namespace: str, name: str, cfg: ScalerConfig, labels: dict, key: str,
                  grid_ts, history, values, timestamps, observed: float, service: ServiceBreaker,
                  ref: CapacityRef):
        """
        The forecast-service call with its fault boundary: build the input
        (demand / ref.value), POST /predict with forecast_timeout_seconds,
        read the answer's units, multiply the coefficients by ref.value,
        record the horizon; every failure (timeout, 5xx, network, unusable
        body) counts against the service breaker and yields observed demand.
        Returns (source, reason, predicted, horizon in demand units).
        """
        capacity = ref.value
        if cfg.history_steps > cfg.time_steps:
            # the whole history, holes as null: serving interpolates the old part
            model_input = [None if v is None else v / capacity for v in history]
        else:
            model_input = [v / capacity for v in values[-cfg.time_steps:]]
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
            horizon = self.forecast.predict(cfg.forecast_url, model_input, cfg.forecast_timeout_seconds,
                                            series=key, timestamp=timestamps[-1],
                                            step_seconds=cfg.step_seconds)
        except ForecastError as exc:
            self.metrics.forecast_failures.labels(**labels, reason=exc.reason).inc()
            opened = service.failure(key)
            if opened:
                self.metrics.forecast_service_breaker_trips.labels(**labels).inc()
                log.warning("event=forecast_service_breaker_open namespace=%s scaledobject=%s reason=%s "
                            "failures=%d hold=%gs error=%r", namespace, name, exc.reason,
                            cfg.forecast_breaker_failures, cfg.forecast_breaker_open_seconds, str(exc))
            else:
                log.warning("event=forecast_failed namespace=%s scaledobject=%s reason=%s error=%r "
                            "breaker=%s", namespace, name, exc.reason, str(exc), service.state(key))
            return "observed", f"forecast unavailable ({exc}); falling back to observed demand", None, []
        except Exception as exc:                 # a client bug is a service failure, not a crash
            self.metrics.forecast_failures.labels(**labels, reason="bad_response").inc()
            if service.failure(key):
                self.metrics.forecast_service_breaker_trips.labels(**labels).inc()
            log.exception("event=forecast_failed namespace=%s scaledobject=%s reason=unexpected error=%r",
                          namespace, name, str(exc))
            return "observed", f"forecast call failed ({exc!r}); falling back to observed demand", None, []
        finally:
            self.metrics.forecast_latency.observe(time.perf_counter() - started)

        if service.success(key):
            log.info("event=forecast_service_breaker_closed namespace=%s scaledobject=%s", namespace, name)
        # What the numbers are (docs/relative-delta-alignment.md): fractions of
        # the model's capacity_reference, or the training series' own units
        # for models before format 5. Either way the scaler multiplies by its
        # reference; with fractions it can also tell whether the two agree.
        units = getattr(horizon, "units", "scaled")
        trained_ref = getattr(horizon, "capacity_reference", None)
        if units == "capacity_fraction" and trained_ref:
            drift = capacity / trained_ref
            self.metrics.capacity_drift.labels(**labels).set(drift)
            factor = cfg.capacity_drift_factor
            if factor and (drift > factor or drift < 1 / factor):
                log.warning("event=capacity_drift namespace=%s scaledobject=%s runtime=%g trained=%g "
                            "factor=%.2f source=%s", namespace, name, capacity, trained_ref, drift, ref.source)
        else:
            try:
                self.metrics.capacity_drift.remove(namespace, name)
            except KeyError:
                pass
        # next: the coming interval. max: the peak over the model's whole
        # horizon, which buys lead time equal to the horizon.
        model_out = max(horizon) if cfg.forecast_aggregation == "max" else horizon[0]
        predicted = max(0.0, model_out * capacity)
        horizon_demand = [max(0.0, v * capacity) for v in horizon]
        log.info("event=forecast_scaled namespace=%s scaledobject=%s coefficient=%.4f units=%s capacity=%g "
                 "source=%s demand=%.4g", namespace, name, model_out, units, capacity, ref.source, predicted)
        self.ledger.record(key, timestamps[-1], cfg.step_seconds, horizon, capacity=capacity)
        for k, v in enumerate(horizon_demand, start=1):
            self.metrics.forecast_demand.labels(**labels, step=str(k)).set(v)
        reason = (f"forecast {predicted:.4g} ({cfg.forecast_aggregation} of "
                  f"{len(horizon)} step(s)), observed {observed:.4g}, mode={cfg.mode}")
        cap = observed * cfg.forecast_surge_factor
        if cfg.mode == "max" and cfg.forecast_surge_factor > 0 and observed > 0 and predicted > cap:
            reason += f"; capped at {cfg.forecast_surge_factor:g}x observed ({cap:.4g})"
            predicted = cap
            horizon_demand = [min(v, cap) for v in horizon_demand]
        return "forecast", reason, predicted, horizon_demand
