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
