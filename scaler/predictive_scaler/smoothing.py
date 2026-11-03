"""
Metric conditioning between the forecast and the number handed to KEDA.

Three guards sit between `POST /predict` and the value `GetMetrics` returns,
each configured per ScaledObject (config.py) and keyed by `namespace/name`
so workloads never share state:

EWMA smoother (`EwmaSmoother`)
    S_t = α · Y_t + (1 − α) · S_{t−1}, applied to the forecast demand before
    it meets `observed`. α = 1 is the raw forecast; α = 0.3 lets a sustained
    rise through in a few polls (90 % after seven) while a one-poll excursion
    moves the reported value by 30 % of its height. The first sample seeds
    the state; a state older than `max_age` (the forecast was unavailable,
    the breaker bypassed it) is dropped and seeded again, so stale history
    never damps a fresh forecast. With `mode=max` the smoothed forecast is
    still floored at observed demand: smoothing delays scale-up on a false
    peak, never scale-up on real load.

Hysteresis gate (`spike_gate`)
    The HPA's scale-down stabilisation (300 s by default) turns every
    scale-up into at least five minutes of paid replicas. A forecast that
    asks for more replicas than the present load needs is therefore checked
    along the horizon: if the steps that need more replicas than observed
    do not form a run of at least `min_steps` consecutive steps, the rise is
    classified as noise, the forecast is suppressed and the decision rests
    on observed demand. A horizon shorter than `min_steps` cannot be judged
    and passes through.

Forecast-service breaker (`ServiceBreaker`)
    A failure budget for the inference call: after `failures` consecutive
    errors (5xx, timeout, connection refused, unusable body) the breaker
    opens for `open_seconds` and the engine does not call the service at
    all, reporting observed demand (plain reactive scaling). When the hold
    elapses one call probes the service (half-open); success closes the
    breaker, a failure reopens it. KEDA keeps scaling whatever the forecast
    service does.

All three are in-process and thread-safe: the gRPC server answers KEDA from
a thread pool and two calls for the same ScaledObject can overlap.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

# ── EWMA ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Smoothed:
    value: float                  # S_t
    raw: float                    # Y_t
    previous: Optional[float]     # S_{t-1}, None when the state was (re)seeded
    seeded: bool                  # True: first sample, or the old state had expired

    @property
    def damped(self) -> float:
        """Y_t - S_t: how much of the raw forecast the smoothing withheld (0 when seeded or alpha = 1)."""
        return self.raw - self.value


@dataclass
class _EwmaState:
    value: float
    updated_at: float


class EwmaSmoother:
    """One EWMA per key (ScaledObject); state in memory, per scaler replica."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._states: Dict[str, _EwmaState] = {}

    def update(self, key: str, raw: float, alpha: float, max_age: float) -> Smoothed:
        """
        Feed the forecast for this poll and return S_t. `alpha` in (0, 1];
        1 returns `raw` unchanged (the state is still kept so a later alpha
        has history). `max_age` seconds without an update expires the state.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not math.isfinite(raw):
            raise ValueError(f"cannot smooth a non-finite forecast ({raw})")
        now = self._clock()
        with self._lock:
            st = self._states.get(key)
            if st is None or (max_age > 0 and now - st.updated_at > max_age):
                self._states[key] = _EwmaState(raw, now)
                return Smoothed(raw, raw, None, True)
            value = alpha * raw + (1.0 - alpha) * st.value
            previous = st.value
            st.value, st.updated_at = value, now
            return Smoothed(value, raw, previous, False)

    def peek(self, key: str) -> Optional[float]:
        with self._lock:
            st = self._states.get(key)
            return st.value if st else None

    def forget(self, key: str) -> None:
        with self._lock:
            self._states.pop(key, None)


# ── Hysteresis gate ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateVerdict:
    suppressed: bool
    active: bool                  # False when the horizon is too short to judge
    rise_steps: int               # horizon steps needing more replicas than observed
    longest_run: int              # longest run of consecutive such steps
    reason: str                   # "" when the forecast passes


def _replicas(demand: float, target_value: float) -> int:
    return max(0, math.ceil(demand / target_value)) if target_value > 0 else 0


def spike_gate(horizon: Sequence[float], observed: float, target_value: float,
               min_steps: int) -> GateVerdict:
    """
    Judge the multi-step forecast `horizon` (demand units) against `observed`
    demand. A step "rises" when it needs more replicas than observed does
    (replicas = ceil(demand / target_value), what KEDA will compute). The
    forecast is suppressed when some step rises but no `min_steps`
    consecutive steps do. `min_steps` <= 1 disables the gate.
    """
    n = len(horizon)
    if min_steps <= 1:
        return GateVerdict(False, False, 0, 0, "")
    if n < min_steps:
        return GateVerdict(False, False, 0, 0,
                           f"gate inactive: horizon of {n} step(s) is shorter than spikeMinSteps={min_steps}")
    base = _replicas(observed, target_value)
    rises = [_replicas(h, target_value) > base for h in horizon]
    rise_steps = sum(rises)
    longest = run = 0
    for r in rises:
        run = run + 1 if r else 0
        longest = max(longest, run)
    if rise_steps and longest < min_steps:
        return GateVerdict(True, True, rise_steps, longest,
                           f"spike gate: {rise_steps} of {n} horizon step(s) need more than the "
                           f"{base} replica(s) observed demand needs, longest run {longest} < "
                           f"spikeMinSteps={min_steps}; treated as noise")
    return GateVerdict(False, True, rise_steps, longest, "")


# ── Forecast-service breaker ───────────────────────────────────────────────────

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass(frozen=True)
class ServiceBreakerConfig:
    failures: int = 3             # consecutive failures that open the breaker (0 = never)
    open_seconds: float = 30.0    # hold before one probing call is let through


@dataclass
class _ServiceState:
    state: str = CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    probing: bool = False


@dataclass(frozen=True)
class Admission:
    allowed: bool
    state: str
    reason: str                   # "" when the call may go ahead


class ServiceBreaker:
    """Failure budget for the forecast-service call, one per key."""

    def __init__(self, cfg: ServiceBreakerConfig = ServiceBreakerConfig(),
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._states: Dict[str, _ServiceState] = {}

    def admit(self, key: str) -> Admission:
        """May the engine call the forecast service for `key` now?"""
        if self.cfg.failures <= 0:
            return Admission(True, CLOSED, "")
        now = self._clock()
        with self._lock:
            st = self._states.setdefault(key, _ServiceState())
            if st.state == CLOSED:
                return Admission(True, CLOSED, "")
            if st.state == OPEN:
                elapsed = now - st.opened_at
                if elapsed < self.cfg.open_seconds:
                    return Admission(False, OPEN,
                                     f"forecast service breaker open ({self.cfg.open_seconds - elapsed:.0f}s "
                                     f"left after {st.consecutive_failures} consecutive failures)")
                st.state, st.probing = HALF_OPEN, True
                return Admission(True, HALF_OPEN, "")
            # HALF_OPEN: one probe at a time; concurrent callers stay reactive
            if st.probing:
                return Admission(False, HALF_OPEN, "forecast service breaker half-open, a probe is in flight")
            st.probing = True
            return Admission(True, HALF_OPEN, "")

    def success(self, key: str) -> bool:
        """Record a good answer. Returns True when this closed an open breaker."""
        with self._lock:
            st = self._states.get(key)
            if st is None:
                return False
            was_open = st.state != CLOSED
            st.state, st.consecutive_failures, st.probing = CLOSED, 0, False
            return was_open

    def failure(self, key: str) -> bool:
        """Record a failed call. Returns True when this opened the breaker."""
        if self.cfg.failures <= 0:
            return False
        now = self._clock()
        with self._lock:
            st = self._states.setdefault(key, _ServiceState())
            st.consecutive_failures += 1
            st.probing = False
            if st.state == HALF_OPEN or (st.state == CLOSED and st.consecutive_failures >= self.cfg.failures):
                st.state, st.opened_at = OPEN, now
                return True
            if st.state == OPEN:
                st.opened_at = now
            return False

    def state(self, key: str) -> str:
        with self._lock:
            st = self._states.get(key)
            return st.state if st else CLOSED

    def forget(self, key: str) -> None:
        with self._lock:
            self._states.pop(key, None)
