"""
Prometheus metrics exposed by the scaler itself.

These are what makes the predictive behaviour observable: for every
ScaledObject you can plot observed demand, predicted demand and the value
actually handed to KEDA on one graph, and see how often the scaler had to
fall back to reactive mode.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, REGISTRY

_LABELS = ("namespace", "scaledobject")


class ScalerMetrics:
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.observed_demand = Gauge(
            "predictive_scaler_observed_demand",
            "Most recent observed demand returned by the ScaledObject query",
            _LABELS, registry=registry,
        )
        self.predicted_demand = Gauge(
            "predictive_scaler_predicted_demand",
            "Forecast demand for the next interval (absent while in fallback)",
            _LABELS, registry=registry,
        )
        self.reported_demand = Gauge(
            "predictive_scaler_reported_demand",
            "Demand value reported to KEDA (what the HPA scales on)",
            _LABELS, registry=registry,
        )
        self.desired_replicas = Gauge(
            "predictive_scaler_desired_replicas",
            "ceil(reported_demand / targetValue): the replica count KEDA will aim for "
            "before min/max bounds are applied",
            _LABELS, registry=registry,
        )
        self.decisions = Counter(
            "predictive_scaler_decisions_total",
            "Scaling decisions by source: forecast, or observed (reactive fallback)",
            _LABELS + ("source",), registry=registry,
        )
        self.forecast_failures = Counter(
            "predictive_scaler_forecast_failures_total",
            "Forecast service calls that failed; the scaler fell back to observed demand",
            _LABELS + ("reason",), registry=registry,
        )
        self.prometheus_failures = Counter(
            "predictive_scaler_prometheus_failures_total",
            "Prometheus queries that failed; the gRPC call returned an error to KEDA",
            _LABELS, registry=registry,
        )
        self.forecast_demand = Gauge(
            "predictive_scaler_forecast_demand",
            "Forecast demand for each step of the model's horizon, in the query's units "
            "(absent while in fallback)",
            _LABELS + ("step",), registry=registry,
        )
        # Scored by the scaler itself from the history it fetches (see quality.py):
        # exact, per replica, in model units so it compares with forecast_model_holdout_mae.
        self.forecast_abs_error = Histogram(
            "predictive_scaler_forecast_abs_error",
            "Absolute error of past forecasts once the actual value was read from Prometheus, "
            "by horizon step, in model units (demand / capacity)",
            _LABELS + ("step",),
            buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
            registry=registry,
        )
        self.forecast_residual = Gauge(
            "predictive_scaler_forecast_residual",
            "Latest (actual - forecast) for the first horizon step, in model units; "
            "what the circuit breaker watches",
            _LABELS, registry=registry,
        )
        self.breaker_open = Gauge(
            "predictive_scaler_breaker_open",
            "1 while the circuit breaker keeps the forecast out of the decision (reactive "
            "scaling on observed demand); retraining excludes these intervals",
            _LABELS, registry=registry,
        )
        self.breaker_trips = Counter(
            "predictive_scaler_breaker_trips_total",
            "Circuit breaker trips by signal: residual (forecast wrong now), level (far from "
            "one season ago), jump (a step the history never contained)",
            _LABELS + ("reason",), registry=registry,
        )
        self.breaker_bypass = Counter(
            "predictive_scaler_breaker_bypass_total",
            "Decisions that had a forecast and reported observed demand instead because the "
            "breaker was open",
            _LABELS, registry=registry,
        )
        # Metric conditioning (smoothing.py): what KEDA got vs what the model said.
        self.smoothed_demand = Gauge(
            "predictive_scaler_smoothed_demand",
            "Forecast demand after the EWMA (smoothingAlpha), the value that meets observed "
            "demand in the decision (absent while in fallback)",
            _LABELS, registry=registry,
        )
        self.spikes_suppressed = Counter(
            "predictive_scaler_spikes_suppressed_total",
            "Forecasts the hysteresis gate classified as noise (a rise shorter than spikeMinSteps "
            "consecutive horizon steps); the decision used observed demand",
            _LABELS, registry=registry,
        )
        self.forecast_service_breaker_open = Gauge(
            "predictive_scaler_forecast_service_breaker_open",
            "1 while the forecast-service breaker is open or half-open: the service failed "
            "forecastBreakerFailures times in a row and the scaler is reactive",
            _LABELS, registry=registry,
        )
        self.forecast_service_breaker_trips = Counter(
            "predictive_scaler_forecast_service_breaker_trips_total",
            "Times the forecast-service breaker opened",
            _LABELS, registry=registry,
        )
        self.forecast_skipped = Counter(
            "predictive_scaler_forecast_skipped_total",
            "Decisions that did not call the forecast service because its breaker was open",
            _LABELS, registry=registry,
        )
        self.forecast_latency = Histogram(
            "predictive_scaler_forecast_latency_seconds",
            "Latency of calls to the forecast service",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=registry,
        )
        self.grpc_requests = Counter(
            "predictive_scaler_grpc_requests_total",
            "gRPC calls received from KEDA",
            ("method", "code"), registry=registry,
        )
