"""
Forecast quality tracking: compare past forecasts with what actually happened.

The service is stateless per request, but every /predict call carries the
truth about earlier calls: a window of the last `time_steps` observations.
When a window arrives whose samples cover the target time of a forecast made
earlier, the forecast is scored against that sample and two metrics are
updated, both labelled by horizon step:

* `forecast_abs_error`   histogram of |actual - forecast|
* `forecast_rolling_mae` mean absolute error over the last N resolved forecasts

Everything is in memory and bounded: at most `max_series` series, at most
`max_pending` unresolved forecasts per series, at most `mae_window` errors
per horizon step. Nothing is persisted; a restart starts the window empty.

Matching
--------
A forecast made at time t0 targets t0 + k*step for step k = 1..horizon. A
later window whose last sample is at t1 holds samples at t1 - j*step. Sample j
is the actual for step k when the two instants coincide within
`tolerance * step` (the scaler's windows are aligned to its own clock, not to
a fixed grid). Each (forecast, step) pair is scored at most once.

Callers that forecast several series through one service should pass a
`series` key with each request; without it, windows from different workloads
would be matched against each other's forecasts.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence

from prometheus_client import REGISTRY, CollectorRegistry, Gauge, Histogram

# Errors are in the model's units (the scaler normalises demand by `capacity`,
# so for the January data they are roughly cores on a 0-2 scale).
ERROR_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0)


@dataclass
class _Pending:
    made_at: float               # unix time of the window's last sample
    step_seconds: int
    predictions: List[float]     # one per horizon step, raw units
    resolved: set = field(default_factory=set)   # horizon steps already scored

    @property
    def horizon(self) -> int:
        return len(self.predictions)

    @property
    def last_target(self) -> float:
        return self.made_at + self.horizon * self.step_seconds


class ForecastTracker:
    def __init__(self, registry: CollectorRegistry = REGISTRY, *, max_series: int = 64,
                 max_pending: int = 512, mae_window: int = 100, tolerance: float = 0.25,
                 clock: Callable[[], float] = time.time):
        if not 0 < tolerance < 0.5:
            raise ValueError("tolerance must be in (0, 0.5): half a step would be ambiguous")
        self.max_series = max_series
        self.max_pending = max_pending
        self.mae_window = mae_window
        self.tolerance = tolerance
        self._clock = clock
        self._lock = threading.Lock()
        # series key → pending forecasts, oldest first; OrderedDict gives LRU eviction
        self._pending: "OrderedDict[str, Deque[_Pending]]" = OrderedDict()
        self._errors: Dict[int, Deque[float]] = {}   # horizon step → recent abs errors

        self.abs_error = Histogram(
            "forecast_abs_error",
            "Absolute error of past forecasts once the actual value arrived, by horizon step",
            ["horizon_step"], buckets=ERROR_BUCKETS, registry=registry,
        )
        self.rolling_mae = Gauge(
            "forecast_rolling_mae",
            f"Mean absolute error over the last {mae_window} resolved forecasts, by horizon step",
            ["horizon_step"], registry=registry,
        )
        self.pending = Gauge(
            "forecast_pending_forecasts",
            "Forecasts held in memory waiting for their actual values",
            registry=registry,
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def record(self, series: str, values: Sequence[float], predictions: Sequence[float],
               step_seconds: int, timestamp: Optional[float] = None) -> int:
        """
        Score earlier forecasts of `series` against `values`, then remember
        `predictions` for scoring later. Returns how many (forecast, step)
        pairs were resolved by this window.
        """
        if step_seconds <= 0 or not predictions:
            return 0
        now = float(timestamp) if timestamp is not None else self._clock()
        with self._lock:
            queue = self._pending.get(series)
            if queue is None:
                if len(self._pending) >= self.max_series:
                    self._pending.popitem(last=False)   # evict the least recently used series
                queue = self._pending[series] = deque(maxlen=self.max_pending)
            else:
                self._pending.move_to_end(series)

            resolved = self._resolve(queue, list(values), step_seconds, now)
            self._expire(queue, len(values), step_seconds, now)

            # One forecast per instant: a second call for the same window
            # (another caller, a retry) must not count its error twice.
            if not queue or abs(queue[-1].made_at - now) > self.tolerance * step_seconds:
                queue.append(_Pending(now, step_seconds, [float(p) for p in predictions]))

            self.pending.set(sum(len(q) for q in self._pending.values()))
        return resolved

    def snapshot(self) -> Dict[int, float]:
        """Current rolling MAE per horizon step (for tests and /model-info)."""
        with self._lock:
            return {k: sum(v) / len(v) for k, v in self._errors.items() if v}

    # ── internals ──────────────────────────────────────────────────────────────

    def _resolve(self, queue: Deque[_Pending], values: List[float], step_seconds: int,
                 now: float) -> int:
        n = len(values)
        count = 0
        for p in queue:
            for k in range(1, p.horizon + 1):
                if k in p.resolved:
                    continue
                target = p.made_at + k * p.step_seconds
                # position of `target` in the new window, counted from its end
                back = (now - target) / step_seconds
                j = round(back)
                if j < 0 or j >= n or abs(back - j) > self.tolerance:
                    continue
                actual = values[n - 1 - j]
                err = abs(actual - p.predictions[k - 1])
                if not math.isfinite(err):
                    continue
                p.resolved.add(k)
                self._observe(k, err)
                count += 1
        return count

    def _expire(self, queue: Deque[_Pending], window_len: int, step_seconds: int,
                now: float) -> None:
        # The oldest sample any future window can carry is now - (n-1)*step;
        # a forecast whose last target is older than that can never be scored.
        earliest_visible = now - (window_len - 1 + self.tolerance) * step_seconds
        while queue and (len(queue[0].resolved) == queue[0].horizon
                         or queue[0].last_target < earliest_visible):
            queue.popleft()

    def _observe(self, step: int, err: float) -> None:
        self.abs_error.labels(horizon_step=str(step)).observe(err)
        window = self._errors.get(step)
        if window is None:
            window = self._errors[step] = deque(maxlen=self.mae_window)
        window.append(err)
        self.rolling_mae.labels(horizon_step=str(step)).set(sum(window) / len(window))
