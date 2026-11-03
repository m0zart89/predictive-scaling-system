"""
The reference capacity of a ScaledObject, in cores.

Models trained with units=capacity_fraction answer in fractions of the
capacity their series was divided by; the scaler divides its history by a
capacity on the way in and multiplies the forecast by it on the way out. The
value used to be only the static `capacity` trigger key. `capacityQuery`
lets it be read from Prometheus at runtime (the workload's envelope:
`max(limit per pod) * maxReplicas`, see docs/relative-delta-alignment.md),
cached for capacityRefreshSeconds, so a limit change in a rollout reaches
the scaler without editing every ScaledObject.

Never raises: a Prometheus error, an empty answer or a non-positive value
falls back to the static `capacity` and is logged once per key, so a bad
capacity query degrades to the previous behaviour, not to a scaling outage.
The reference must be the fixed envelope, not the live replica total: a
reference the scaler itself moves would feed back into the series the model
forecasts from.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .clients import PrometheusClient, PrometheusError
from .config import ScalerConfig

log = logging.getLogger("predictive-scaler.capacity")


@dataclass(frozen=True)
class CapacityRef:
    value: float          # cores
    source: str           # "query" | "static"
    resolved_at: float    # monotonic time the value was read (0 for static)


class CapacityResolver:
    def __init__(self, prometheus: PrometheusClient, clock: Callable[[], float] = time.monotonic):
        self._prom = prometheus
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: Dict[str, CapacityRef] = {}
        self._warned: set = set()

    def resolve(self, key: str, cfg: ScalerConfig) -> CapacityRef:
        """The reference for `key` (namespace/name) under `cfg`: the query's
        value when capacityQuery is set and answers, else the static capacity."""
        if not cfg.capacity_query:
            return CapacityRef(cfg.capacity, "static", 0.0)
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and now - hit.resolved_at < cfg.capacity_refresh_seconds:
                return hit
        try:
            value: Optional[float] = self._prom.query_scalar(cfg.prometheus_address, cfg.capacity_query,
                                                             cfg.timeout_seconds)
        except PrometheusError as exc:
            return self._fallback(key, cfg, f"capacityQuery failed: {exc}")
        if value is None or not math.isfinite(value) or value <= 0:
            return self._fallback(key, cfg, f"capacityQuery returned {value!r}")
        ref = CapacityRef(float(value), "query", now)
        with self._lock:
            previous = self._cache.get(key)
            self._cache[key] = ref
            recovered = key in self._warned
            self._warned.discard(key)
        if recovered:
            log.info("event=capacity_resolved scaledobject=%s capacity=%g source=query", key, ref.value)
        elif previous is None or abs(previous.value - ref.value) > 1e-9:
            log.info("event=capacity_changed scaledobject=%s capacity=%g previous=%s source=query", key,
                     ref.value, f"{previous.value:g}" if previous else "-")
        return ref

    def _fallback(self, key: str, cfg: ScalerConfig, why: str) -> CapacityRef:
        with self._lock:
            first = key not in self._warned
            self._warned.add(key)
            # keep serving the last good value from the query while it fails
            hit = self._cache.get(key)
        if first:
            log.warning("event=capacity_fallback scaledobject=%s static_capacity=%g last_query=%s reason=%r",
                        key, cfg.capacity, f"{hit.value:g}" if hit else "-", why)
        if hit is not None:
            return CapacityRef(hit.value, "query", hit.resolved_at)
        return CapacityRef(cfg.capacity, "static", 0.0)

    def forget(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)
            self._warned.discard(key)
