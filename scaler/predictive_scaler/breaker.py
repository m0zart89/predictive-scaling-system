"""
Circuit breaker: when the present stops resembling the history the model
learned, take the forecast out of the loop until it does.

With mode=max the scaler never scales below current demand, so a spike is
handled reactively whatever the model says. The damage a flash sale or a
DDoS does comes afterwards: the model's input now contains the spike, and
for the next horizon to a day the forecast is confidently wrong — a delta
model extrapolates the climb, a quantile model forecasts a distribution that
just moved, a seasonal model expects the spike again tomorrow. The breaker's
job is to notice that moment from signals the engine already has, report
observed demand only while it lasts (KEDA then behaves like a plain
reactive scaler; a `cpu` trigger on the ScaledObject is the always-on
floor), record the interval so retraining can exclude it, and close again
only after the series has been calm for a while.

Signals, all in model units (demand / capacity):

* residual shock   — z-score of (actual − forecast made for now) against a
                     robust spread (MAD) of recent residuals; the model is
                     wrong *now* by more than noise
* level shift      — observed / value one season ago, far from 1 for
                     several polls; the seasonal lags are misleading
* jump             — one-step change larger than anything the fetched
                     history contains; a step the model has never seen

State machine: CLOSED → (trip) → OPEN → (min open elapsed) → HALF_OPEN →
(calm polls) → CLOSED; a trip in HALF_OPEN reopens with a doubled hold time.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass(frozen=True)
class BreakerConfig:
    enabled: bool = True
    residual_window: int = 48          # residuals kept for the spread estimate
    residual_min: int = 12             # residuals needed before the z-score counts
    residual_z: float = 4.0            # |z| on two consecutive polls trips
    residual_z_hard: float = 8.0       # |z| once trips
    level_shift_high: float = 2.5      # observed / season-ago value
    level_shift_low: float = 0.4
    level_shift_polls: int = 3
    jump_factor: float = 2.0           # vs the largest one-step rise in the history
    min_open_seconds: float = 600.0
    max_open_seconds: float = 3600.0
    calm_polls: int = 6
    calm_z: float = 2.0


@dataclass
class _State:
    state: str = CLOSED
    opened_at: float = 0.0
    open_for: float = 600.0
    residuals: Deque[float] = field(default_factory=lambda: deque(maxlen=48))
    hot_residual: int = 0
    hot_level: int = 0
    calm: int = 0
    last_observed: Optional[float] = None
    last_z: Optional[float] = None
    trips: int = 0


@dataclass(frozen=True)
class Verdict:
    use_forecast: bool
    state: str
    reason: str                 # "" when the forecast may be used
    tripped: Optional[str]      # signal that tripped on this poll: residual | level | jump


class CircuitBreaker:
    """One breaker per ScaledObject key; all state in memory, per scaler replica."""

    def __init__(self, cfg: BreakerConfig = BreakerConfig(), clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self._clock = clock
        self._states: Dict[str, _State] = {}

    def observe(self, key: str, observed: float, history: Sequence[Optional[float]],
                season_steps: int, residuals: Sequence[float] = ()) -> Verdict:
        """
        Feed one poll: the observed value, the fetched history (oldest first,
        None for holes, in the same units as `observed`), the season length in
        steps (0 when unknown) and the step-1 residuals (actual − forecast)
        resolved on this poll. Returns whether the forecast may be used.
        """
        cfg = self.cfg
        if not cfg.enabled:
            return Verdict(True, CLOSED, "", None)
        st = self._states.get(key)
        if st is None:
            st = self._states[key] = _State(open_for=cfg.min_open_seconds,
                                            residuals=deque(maxlen=cfg.residual_window))
        tripped: List[Tuple[str, str]] = []

        # 1. residual shock
        z = self._residual_z(st, residuals)
        st.last_z = z
        if z is not None:
            st.hot_residual = st.hot_residual + 1 if abs(z) > cfg.residual_z else 0
            if abs(z) > cfg.residual_z_hard or st.hot_residual >= 2:
                tripped.append(("residual", f"residual z={z:.1f}"))

        # 2. level vs one season ago
        ref = _season_ago(history, season_steps)
        if ref is not None and ref > 0 and observed > 0:
            ratio = observed / ref
            shifted = ratio > cfg.level_shift_high or ratio < cfg.level_shift_low
            st.hot_level = st.hot_level + 1 if shifted else 0
            if st.hot_level >= cfg.level_shift_polls:
                tripped.append(("level", f"level x{ratio:.2f} vs one season ago"))
        else:
            st.hot_level = 0

        # 3. a jump the history never contained
        if st.last_observed is not None:
            rise = observed - st.last_observed
            biggest = _max_rise(history)
            if biggest > 0 and rise > cfg.jump_factor * biggest:
                tripped.append(("jump", f"jump +{rise:.3g} > {cfg.jump_factor:g}x the largest "
                                        f"rise in the history ({biggest:.3g})"))
        st.last_observed = observed

        now = self._clock()
        if tripped:
            signal, why = tripped[0]
            self._trip(st, now)
            return Verdict(False, st.state, "breaker: " + "; ".join(w for _, w in tripped), signal)

        if st.state == OPEN:
            if now - st.opened_at < st.open_for:
                left = int(st.open_for - (now - st.opened_at))
                return Verdict(False, OPEN, f"breaker: open, {left}s before half-open", None)
            st.state, st.calm = HALF_OPEN, 0

        if st.state == HALF_OPEN:
            calm = (z is None or abs(z) < cfg.calm_z) and st.hot_level == 0 and st.hot_residual == 0
            st.calm = st.calm + 1 if calm else 0
            if st.calm >= cfg.calm_polls:
                st.state, st.open_for = CLOSED, cfg.min_open_seconds
                return Verdict(True, CLOSED, "", None)
            return Verdict(False, HALF_OPEN,
                           f"breaker: half-open, {st.calm}/{cfg.calm_polls} calm polls", None)

        return Verdict(True, CLOSED, "", None)

    def state(self, key: str) -> str:
        st = self._states.get(key)
        return st.state if st else CLOSED

    def is_open(self, key: str) -> bool:
        return self.state(key) != CLOSED

    def forget(self, key: str) -> None:
        self._states.pop(key, None)

    # ── internals ──────────────────────────────────────────────────────────────
    def _trip(self, st: _State, now: float) -> None:
        if st.state == HALF_OPEN:                 # re-trip while probing: hold longer
            st.open_for = min(st.open_for * 2, self.cfg.max_open_seconds)
        st.state, st.opened_at, st.calm, st.trips = OPEN, now, 0, st.trips + 1
        st.hot_residual = st.hot_level = 0

    def _residual_z(self, st: _State, residuals: Sequence[float]) -> Optional[float]:
        if not residuals:
            return None
        latest = float(residuals[-1])
        # the spread is estimated from residuals seen *before* this poll, so a
        # shock cannot widen its own yardstick
        baseline = list(st.residuals)
        for r in residuals:
            st.residuals.append(float(r))
        if len(baseline) < self.cfg.residual_min:
            return None
        med = _median(baseline)
        mad = _median([abs(x - med) for x in baseline]) * 1.4826
        if mad < 1e-9:
            return 0.0 if abs(latest - med) < 1e-9 else math.copysign(99.0, latest - med)
        return (latest - med) / mad


def _season_ago(history: Sequence[Optional[float]], season_steps: int) -> Optional[float]:
    if season_steps <= 0 or len(history) <= season_steps:
        return None
    return history[-1 - season_steps]


def _max_rise(history: Sequence[Optional[float]]) -> float:
    prev, best = None, 0.0
    for v in history[:-1]:                        # the current sample is the one under test
        if v is not None and prev is not None:
            best = max(best, v - prev)
        prev = v
    return best


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
