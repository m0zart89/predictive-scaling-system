"""
Per-ScaledObject configuration.

KEDA passes the `metadata` block of the ScaledObject trigger to every gRPC call
as `scalerMetadata` (a string→string map). This module turns that map into a
validated, typed config so the rest of the scaler never touches raw strings.

Units
-----
`query` returns workload *demand* in whatever unit fits the workload (CPU cores,
requests/s, queue depth). Everything the scaler reports to KEDA is in that same
unit; `targetValue` says how much demand one replica should carry.

The forecasting model only knows the units of its training data (the January
2022 CPU series, roughly 0–2). `capacity` converts between the two:
model input = demand / capacity, forecast demand = model output × capacity.
Pick it so the workload's expected range lands inside the training range — a
tree-based model cannot predict beyond the largest value it was trained on.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping


class ConfigError(ValueError):
    """Raised when ScaledObject metadata is missing or invalid."""


MODES = ("max", "forecast")
AGGREGATIONS = ("next", "max")


@dataclass(frozen=True)
class ScalerConfig:
    prometheus_address: str
    query: str
    forecast_url: str
    target_value: float
    capacity: float = 1.0
    time_steps: int = 10
    history_steps: int = 10    # samples fetched and sent; >= time_steps (a season more for
                               # models with seasonal features; the chart computes it)
    step_seconds: int = 300
    mode: str = "max"
    forecast_aggregation: str = "next"
    activation_value: float = 0.0
    cache_seconds: float = 15.0
    timeout_seconds: float = 3.0
    metric_name: str = "predicted-demand"

    @property
    def cache_key(self) -> str:
        """Short hash of the settings. The engine combines it with the
        ScaledObject namespace/name, so editing the trigger invalidates the cache."""
        raw = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _required(meta: Mapping[str, str], key: str) -> str:
    value = (meta.get(key) or "").strip()
    if not value:
        raise ConfigError(f"metadata.{key} is required")
    return value


def _positive_float(meta: Mapping[str, str], key: str, default: float | None = None,
                    allow_zero: bool = False) -> float:
    raw = meta.get(key)
    if raw is None or str(raw).strip() == "":
        if default is None:
            raise ConfigError(f"metadata.{key} is required")
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"metadata.{key} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"metadata.{key} must be {bound}, got {raw!r}")
    return value


def _positive_int(meta: Mapping[str, str], key: str, default: int) -> int:
    raw = meta.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"metadata.{key} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"metadata.{key} must be > 0, got {raw!r}")
    return value


def parse_metadata(meta: Mapping[str, str]) -> ScalerConfig:
    """Validate ScaledObject trigger metadata. Raises ConfigError on bad input."""
    mode = (meta.get("mode") or "max").strip().lower()
    if mode not in MODES:
        raise ConfigError(f"metadata.mode must be one of {MODES}, got {mode!r}")

    agg = (meta.get("forecastAggregation") or "next").strip().lower()
    if agg not in AGGREGATIONS:
        raise ConfigError(f"metadata.forecastAggregation must be one of {AGGREGATIONS}, got {agg!r}")

    time_steps = _positive_int(meta, "timeSteps", default=10)
    history_steps = _positive_int(meta, "historySteps", default=time_steps)
    if history_steps < time_steps:
        raise ConfigError(f"metadata.historySteps must be >= timeSteps ({time_steps}), "
                          f"got {history_steps}")

    return ScalerConfig(
        prometheus_address=_required(meta, "prometheusServerAddress").rstrip("/"),
        query=_required(meta, "query"),
        forecast_url=_required(meta, "forecastServiceURL").rstrip("/"),
        target_value=_positive_float(meta, "targetValue"),
        capacity=_positive_float(meta, "capacity", default=1.0),
        time_steps=time_steps,
        history_steps=history_steps,
        step_seconds=_positive_int(meta, "stepSeconds", default=300),
        mode=mode,
        forecast_aggregation=agg,
        activation_value=_positive_float(meta, "activationValue", default=0.0, allow_zero=True),
        cache_seconds=_positive_float(meta, "cacheSeconds", default=15.0, allow_zero=True),
        timeout_seconds=_positive_float(meta, "timeoutSeconds", default=3.0),
        metric_name=(meta.get("metricName") or "predicted-demand").strip(),
    )
