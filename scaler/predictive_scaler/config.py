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
from typing import Mapping, Tuple


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
    # mode=max: the forecast may exceed observed demand by at most this factor
    # (0 = no cap). A model asking for 10x current demand is wrong far more
    # often than demand is about to grow 10x.
    forecast_surge_factor: float = 3.0
    # Multivariate models: extra input series by channel name → PromQL, sent
    # next to the demand series (target_channel) as {name: [...]}. Must match the
    # channels the model was trained on (/model-info). Empty: one series.
    channels: Tuple[Tuple[str, str], ...] = ()
    target_channel: str = "cpu"
    # Circuit breaker (see breaker.py): bypass the forecast while the series
    # does not resemble the history. Thresholds are in the breaker's units.
    breaker: bool = True
    breaker_residual_z: float = 4.0
    breaker_level_shift: float = 2.5
    breaker_jump_factor: float = 2.0
    breaker_jump_min_samples: int = 48
    breaker_min_open_seconds: float = 600.0
    breaker_calm_polls: int = 6
    # Metric conditioning (see smoothing.py). EWMA on the forecast demand:
    # S_t = alpha*Y_t + (1-alpha)*S_{t-1}; 1 = raw forecast. A state older
    # than smoothing_max_age_seconds is seeded again (0 = never expires).
    smoothing_alpha: float = 0.3
    smoothing_max_age_seconds: float = 600.0
    # Hysteresis gate: a forecast needing more replicas than observed demand
    # must do so for this many consecutive horizon steps, else it is noise
    # and the decision rests on observed demand (0/1 = off).
    spike_min_steps: int = 3
    # Forecast-service call: its own timeout (Prometheus keeps
    # timeout_seconds) and the failure budget that opens the service
    # breaker, after which the scaler is reactive for the hold time.
    forecast_timeout_seconds: float = 0.2
    forecast_breaker_failures: int = 3
    forecast_breaker_open_seconds: float = 30.0
    # Reference capacity at runtime (capacity.py): a PromQL instant query for
    # the workload's envelope in cores (max limit per pod x maxReplicas),
    # cached capacity_refresh_seconds; "" = the static `capacity` above. A
    # resolved value more than capacity_drift_factor away from the model's
    # own capacity_reference is flagged (0 = no check).
    capacity_query: str = ""
    capacity_refresh_seconds: float = 300.0
    capacity_drift_factor: float = 2.0

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


def _positive_int(meta: Mapping[str, str], key: str, default: int, allow_zero: bool = False) -> int:
    raw = meta.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"metadata.{key} must be an integer, got {raw!r}") from exc
    if value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"metadata.{key} must be {bound}, got {raw!r}")
    return value


def _channels(raw) -> Tuple[Tuple[str, str], ...]:
    """metadata.channels: a JSON object {name: PromQL}, in order."""
    if raw is None or str(raw).strip() == "":
        return ()
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"metadata.channels must be a JSON object of name -> PromQL: {exc}") from exc
    if not isinstance(obj, dict) or not obj:
        raise ConfigError("metadata.channels must be a non-empty JSON object of name -> PromQL")
    out = []
    for name, query in obj.items():
        if not isinstance(query, str) or not query.strip() or not str(name).strip():
            raise ConfigError(f"metadata.channels: {name!r} needs a non-empty PromQL string")
        out.append((str(name).strip(), query.strip()))
    return tuple(out)


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
    channels = _channels(meta.get("channels"))
    target_channel = (meta.get("targetChannel") or "cpu").strip()
    if any(name == target_channel for name, _ in channels):
        raise ConfigError(f"metadata.channels must not contain the target channel {target_channel!r}")
    breaker = (meta.get("breaker") or "on").strip().lower()
    if breaker not in ("on", "off", "true", "false"):
        raise ConfigError(f"metadata.breaker must be on|off, got {breaker!r}")
    level_shift = _positive_float(meta, "breakerLevelShift", default=2.5)
    if level_shift <= 1:
        raise ConfigError(f"metadata.breakerLevelShift must be > 1, got {level_shift!r}")
    alpha = _positive_float(meta, "smoothingAlpha", default=0.3)
    if alpha > 1:
        raise ConfigError(f"metadata.smoothingAlpha must be in (0, 1], got {alpha!r}")
    drift = _positive_float(meta, "capacityDriftFactor", default=2.0, allow_zero=True)
    if 0 < drift < 1:
        raise ConfigError(f"metadata.capacityDriftFactor must be >= 1 (or 0 for no check), got {drift!r}")

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
        channels=channels,
        target_channel=target_channel,
        forecast_surge_factor=_positive_float(meta, "forecastSurgeFactor", default=3.0,
                                              allow_zero=True),
        breaker=breaker in ("on", "true"),
        breaker_residual_z=_positive_float(meta, "breakerResidualZ", default=4.0),
        breaker_level_shift=level_shift,
        breaker_jump_factor=_positive_float(meta, "breakerJumpFactor", default=2.0),
        breaker_jump_min_samples=_positive_int(meta, "breakerJumpMinSamples", default=48),
        breaker_min_open_seconds=_positive_float(meta, "breakerMinOpenSeconds", default=600.0),
        breaker_calm_polls=_positive_int(meta, "breakerCalmPolls", default=6),
        smoothing_alpha=alpha,
        smoothing_max_age_seconds=_positive_float(meta, "smoothingMaxAgeSeconds", default=600.0,
                                                  allow_zero=True),
        spike_min_steps=_positive_int(meta, "spikeMinSteps", default=3, allow_zero=True),
        forecast_timeout_seconds=_positive_float(meta, "forecastTimeoutSeconds", default=0.2),
        forecast_breaker_failures=_positive_int(meta, "forecastBreakerFailures", default=3, allow_zero=True),
        forecast_breaker_open_seconds=_positive_float(meta, "forecastBreakerOpenSeconds", default=30.0),
        capacity_query=(meta.get("capacityQuery") or "").strip(),
        capacity_refresh_seconds=_positive_float(meta, "capacityRefreshSeconds", default=300.0,
                                                 allow_zero=True),
        capacity_drift_factor=drift,
    )
