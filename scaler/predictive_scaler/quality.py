"""
Forecast quality measured where the data is: in the scaler.

Every decision fetches a history that contains the actual values for the
instants earlier forecasts aimed at, so the scaler can score its own past
forecasts without any help from forecast-serving and without shared state
between replicas: a replica scores exactly the forecasts it made, from the
history it reads, whichever replica KEDA polled in between. The result is
`predictive_scaler_forecast_abs_error{step}` (a histogram in model units,
comparable with the model's own `forecast_model_holdout_mae`) and the
residual stream the circuit breaker reads.

Memory is bounded: at most `max_series` ScaledObjects (least recently used
evicted) × `max_pending` forecasts each, and a forecast is dropped once the
history has moved past its last step.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple


@dataclass
class _Pending:
    made_at: float            # unix time of the window's last sample
    step_seconds: int
    values: List[float]       # forecast per horizon step, model units


@dataclass(frozen=True)
class Resolved:
    step: int                 # 1-based horizon step
    forecast: float           # model units
    actual: float             # model units
    at: float                 # unix time of the actual sample

    @property
    def error(self) -> float:
        return self.actual - self.forecast


class ForecastLedger:
    def __init__(self, max_series: int = 256, max_pending: int = 64, tolerance: float = 0.25):
        self.max_series = max_series
        self.max_pending = max_pending
        self.tolerance = tolerance
        self._lock = threading.Lock()
        self._pending: "OrderedDict[str, Deque[_Pending]]" = OrderedDict()

    def record(self, key: str, made_at: float, step_seconds: int, values: Sequence[float]) -> None:
        """Remember a forecast made for the instants made_at + k*step, k = 1..len(values)."""
        if not values or step_seconds <= 0 or self.max_series <= 0:
            return
        with self._lock:
            queue = self._pending.get(key)
            if queue is None:
                if len(self._pending) >= self.max_series:
                    self._pending.popitem(last=False)
                queue = self._pending[key] = deque(maxlen=self.max_pending)
            else:
                self._pending.move_to_end(key)
            # one forecast per instant: a cached decision or a retry must not count twice
            if queue and abs(queue[-1].made_at - made_at) <= self.tolerance * step_seconds:
                return
            queue.append(_Pending(float(made_at), int(step_seconds), [float(v) for v in values]))

    def score(self, key: str, timestamps: Sequence[float], history: Sequence[Optional[float]],
              capacity: float = 1.0) -> List[Resolved]:
        """
        Resolve every pending forecast of `key` whose target instants are in
        `history` (query units; divided by `capacity` into model units).
        Returns the resolved (step, forecast, actual) triples, oldest first;
        a forecast leaves the queue once its last step is older than the
        newest sample.
        """
        if not timestamps or self.max_series <= 0:
            return []
        with self._lock:
            queue = self._pending.get(key)
            if not queue:
                return []
            newest = float(timestamps[-1])
            by_ts = {float(t): v for t, v in zip(timestamps, history) if v is not None}
            resolved: List[Resolved] = []
            keep: Deque[_Pending] = deque(maxlen=self.max_pending)
            for p in queue:
                tol = self.tolerance * p.step_seconds
                for k, forecast in enumerate(p.values, start=1):
                    target = p.made_at + k * p.step_seconds
                    actual = _lookup(by_ts, target, tol)
                    if actual is not None:
                        resolved.append(Resolved(k, forecast, actual / capacity, target))
                last_target = p.made_at + len(p.values) * p.step_seconds
                if last_target > newest + tol:
                    keep.append(p)          # later steps still to come
            # a resolved step must not be scored again on the next poll: drop
            # the forecast's steps that are now in the past
            self._pending[key] = deque(
                (_Pending(p.made_at, p.step_seconds, p.values) for p in keep), maxlen=self.max_pending)
            for p in self._pending[key]:
                done = int((newest + self.tolerance * p.step_seconds - p.made_at) // p.step_seconds)
                if done > 0:
                    p.values = [float("nan")] * min(done, len(p.values)) + p.values[min(done, len(p.values)):]
            return [r for r in resolved if r.forecast == r.forecast]   # drop NaN placeholders

    def pending(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._pending.values())

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()


def _lookup(by_ts: Dict[float, float], target: float, tol: float) -> Optional[float]:
    v = by_ts.get(target)
    if v is not None:
        return v
    for t, v in by_ts.items():            # grids are aligned; this is the rare off-by-skew case
        if abs(t - target) <= tol:
            return v
    return None


def rolling_mae(resolved: Sequence[Resolved]) -> Dict[int, Tuple[float, int]]:
    """{step: (mean abs error, n)} over a batch of resolved forecasts."""
    acc: Dict[int, List[float]] = {}
    for r in resolved:
        acc.setdefault(r.step, []).append(abs(r.error))
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}
