"""
Forecast quality tracking: compare past forecasts with what actually happened.

The service is stateless per request, but every /predict call carries the
truth about earlier calls: a window of the last `time_steps` observations.
When a window arrives whose samples cover the target time of a forecast made
earlier, the forecast is scored against that sample and two metrics are
updated, both labelled by horizon step:

* `forecast_abs_error`   histogram of |actual - forecast|
* `forecast_rolling_mae` mean absolute error over the last N resolved forecasts

Two backends implement the same contract (`QualityBackend`):

`ForecastTracker` (memory)
    Everything in the pod's memory, bounded: at most `max_series` series,
    `max_pending` unresolved forecasts per series, `mae_window` errors per
    horizon step. Exact with one replica; with several, each replica only
    sees the windows routed to it, so forecasts made on one pod are never
    scored on another. Nothing is persisted; a restart starts empty.

`RedisMetricsEngine` (redis)
    The state lives in Redis, shared by every replica and surviving pod
    restarts; the pod holds nothing per series. Every forecast is written
    with a TTL (default 24 h) so forecasts for series that stop calling
    (a deleted ScaledObject, a renamed workload) clean themselves up, and the
    error windows are bounded Redis lists trimmed on the server. Redis being
    down degrades the service to "no quality figures", never to an error on
    /predict. Schema and round trips are documented on the class.

Matching (both backends)
--------------------------
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

import json
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Protocol, Sequence

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

log = logging.getLogger("forecast-serving.quality")

# Errors are in the model's units (the scaler normalises demand by `capacity`,
# so for the January data they are roughly cores on a 0-2 scale).
ERROR_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0)

DEFAULT_FORECAST_TTL = 24 * 3600        # a forecast nobody scored in a day never will be
DEFAULT_MAE_TTL = 7 * 24 * 3600         # error windows of a retired model version fade out
DEFAULT_ROLLING_WINDOW = 100


class QualityBackend(Protocol):
    """What serving/app.py needs from a quality tracker, whichever backend."""

    def record(self, series: str, values: Sequence[float], predictions: Sequence[float],
               step_seconds: int, timestamp: Optional[float] = None,
               model_version: Optional[str] = None) -> int: ...

    def reset(self, pending_only: bool = False, model_version: Optional[str] = None) -> None: ...

    def snapshot(self, model_version: Optional[str] = None, horizon: int = 0) -> Dict[int, float]: ...


def _metrics(registry: CollectorRegistry, mae_window: int) -> tuple[Histogram, Gauge]:
    abs_error = Histogram(
        "forecast_abs_error",
        "Absolute error of past forecasts once the actual value arrived, by horizon step",
        ["horizon_step"], buckets=ERROR_BUCKETS, registry=registry,
    )
    rolling_mae = Gauge(
        "forecast_rolling_mae",
        f"Mean absolute error over the last {mae_window} resolved forecasts, by horizon step",
        ["horizon_step"], registry=registry,
    )
    return abs_error, rolling_mae


# ══════════════════════════════════════════════════════════════════════════════
# In-memory backend
# ══════════════════════════════════════════════════════════════════════════════

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
    """Per-pod, bounded, exact with one replica. See the module docstring."""

    def __init__(self, registry: CollectorRegistry = REGISTRY, *, max_series: int = 64,
                 max_pending: int = 512, mae_window: int = DEFAULT_ROLLING_WINDOW,
                 tolerance: float = 0.25, clock: Callable[[], float] = time.time):
        if not 0 < tolerance < 0.5:
            raise ValueError("tolerance must be in (0, 0.5): half a step would be ambiguous")
        self.max_series = max_series
        self.max_pending = max_pending
        self.mae_window = mae_window
        self.tolerance = tolerance
        self._clock = clock
        self._lock = threading.Lock()
        # The model version whose forecasts are tracked (set by reset on a swap, or
        # by the first versioned record). A record from another version is a
        # request that was already in flight when the model was swapped.
        self._version: Optional[str] = None
        # series key → pending forecasts, oldest first; OrderedDict gives LRU eviction
        self._pending: "OrderedDict[str, Deque[_Pending]]" = OrderedDict()
        self._errors: Dict[int, Deque[float]] = {}   # horizon step → recent abs errors

        self.abs_error, self.rolling_mae = _metrics(registry, mae_window)
        self.pending = Gauge(
            "forecast_pending_forecasts",
            "Forecasts held in memory waiting for their actual values",
            registry=registry,
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def record(self, series: str, values: Sequence[float], predictions: Sequence[float],
               step_seconds: int, timestamp: Optional[float] = None,
               model_version: Optional[str] = None) -> int:
        """
        Score earlier forecasts of `series` against `values`, then remember
        `predictions` for scoring later. Returns how many (forecast, step)
        pairs were resolved by this window.

        `model_version` guards the swap: the record of a request that began
        on the previous model and reaches this tracker after
        `reset(model_version=new)` is dropped (returns 0). Its forecasts
        belong to a model that no longer serves, and queued here they would
        be scored as the new model's errors. Without a version (older
        callers, tests) nothing is dropped.
        """
        if step_seconds <= 0 or not predictions or self.max_series <= 0:
            return 0      # max_series=0: tracking off (the scaler scores forecasts itself)
        now = float(timestamp) if timestamp is not None else self._clock()
        with self._lock:
            if model_version is not None:
                if self._version is None:
                    self._version = str(model_version)
                elif str(model_version) != self._version:
                    log.debug("quality: dropped a record of v%s, v%s is serving", model_version, self._version)
                    return 0
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

    def reset(self, pending_only: bool = False, model_version: Optional[str] = None) -> None:
        """Forget pending forecasts (after a model swap: the old model's forecasts
        are not the new model's errors) and, unless pending_only, the errors too.
        `model_version` names the model that serves from now on: records of any
        other version that are still in flight are dropped (see record). A full
        reset without a version forgets the version too."""
        with self._lock:
            self._pending.clear()
            self.pending.set(0)
            if not pending_only:
                self._errors.clear()
                self.rolling_mae.clear()
            if model_version is not None:
                self._version = str(model_version)
            elif not pending_only:
                self._version = None

    def snapshot(self, model_version: Optional[str] = None, horizon: int = 0) -> Dict[int, float]:
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


# ══════════════════════════════════════════════════════════════════════════════
# Redis backend
# ══════════════════════════════════════════════════════════════════════════════

# Claim one (forecast, step) pair and push its error into the bounded window,
# atomically on the server. Two replicas scoring the same pair from two
# windows race on SADD; exactly one wins and pushes.
#
#   KEYS[1]  forecast:{series}:{timestamp}:scored    (SET of claimed steps)
#   KEYS[2]  metrics:mae:{model_version}:{horizon_idx}  (LIST of recent errors)
#   ARGV     step, forecast TTL, error, rolling_window_size, MAE TTL
_CLAIM_AND_PUSH = """
if redis.call('SADD', KEYS[1], ARGV[1]) == 0 then
  return 0
end
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('LPUSH', KEYS[2], ARGV[3])
redis.call('LTRIM', KEYS[2], 0, tonumber(ARGV[4]) - 1)
redis.call('EXPIRE', KEYS[2], ARGV[5])
return 1
"""


@dataclass(frozen=True)
class _Claim:
    """A (forecast, step) pair this window can score: computed in Python from
    one forecast payload and one sample, pushed by the script above."""
    series: str
    forecast_key: str
    model_version: str
    step: int
    error: float


class RedisMetricsEngine:
    """
    Forecast quality state in Redis, shared by all serving replicas.

    Design
    ------
    * The pod keeps no per-series state: a replica can be restarted, scaled
      or drained at any time, and a window arriving at replica B scores the
      forecast replica A made. Memory on the pod is O(1) per request.
    * Every key carries a TTL: a forecast is kept `forecast_ttl_seconds`
      (24 h); a window that never comes (the workload was deleted, the
      ScaledObject renamed, the scaler moved to another series key) costs
      nothing after that. Error windows are capped to `rolling_window_size`
      entries by LTRIM on the server and expire `mae_ttl_seconds` after the
      last push, so a retired model version fades out by itself.
    * Two round trips per /predict (`record`): one pipeline reads the
      candidate forecasts and their claimed steps, one pipeline runs the
      claim-and-push script per unclaimed pair, re-reads the windows it
      touched for the gauge, and stores the new forecast (SET NX EX: the
      first forecast for an instant wins, a retry seconds later is not a
      second forecast). No array is ever pulled to compute an error: each
      error is one subtraction on one sample.
    * Strict error boundary: every Redis call is wrapped; `redis.RedisError`,
      `ConnectionError` and `OSError` are logged (once per outage, then at
      debug) and counted, the call returns as if nothing matched, and the
      backend is skipped for `cooldown_seconds` so a dead Redis does not
      cost a socket timeout on every request. /predict never fails or waits
      beyond the configured socket timeouts because of this class.
    * The client and its `ConnectionPool` are thread-safe; the engine's own
      bookkeeping is guarded by one lock, so the async handler can call it
      from any worker thread.

    Redis data schema
    -----------------
    ``forecast:{series_id}:{timestamp}``   STRING, TTL 24 h
        JSON ``{"predictions": [..], "model_version": "..", "made_at": t0,
        "step_seconds": s, "targets": [t0+s, .., t0+H*s]}``. ``timestamp`` is
        ``made_at`` quantised to ``tolerance * step`` seconds (75 s for a
        5-minute step), which is the matching tolerance: a window probes the
        quantum its sample points at and its two neighbours, so matching is
        the same ±¼-step rule as the in-memory tracker, with exact keys.
    ``forecast:{series_id}:{timestamp}:scored``   SET of horizon steps, TTL 24 h
        The steps of that forecast already scored (claimed by SADD).
    ``metrics:mae:{model_version}:{horizon_idx}``   LIST, newest first, length ≤ N
        The last N absolute errors of that model version at that horizon
        step (``horizon_idx`` is 1-based, as the ``horizon_step`` label).
        ``get_rolling_mae`` averages it.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", *,
                 registry: CollectorRegistry = REGISTRY,
                 rolling_window_size: int = DEFAULT_ROLLING_WINDOW,
                 forecast_ttl_seconds: int = DEFAULT_FORECAST_TTL,
                 mae_ttl_seconds: int = DEFAULT_MAE_TTL,
                 tolerance: float = 0.25,
                 max_connections: int = 16,
                 socket_timeout: float = 0.25,
                 socket_connect_timeout: float = 0.25,
                 cooldown_seconds: float = 5.0,
                 key_prefix: str = "",
                 clock: Callable[[], float] = time.time,
                 client: Any = None):
        """
        `client` lets tests pass a fake; otherwise a `redis.Redis` is built on
        a `ConnectionPool` from `url` with `max_connections` and the socket
        timeouts (seconds). Keep the timeouts short: they bound how long a
        /predict call can wait on a slow Redis.
        """
        if not 0 < tolerance < 0.5:
            raise ValueError("tolerance must be in (0, 0.5): half a step would be ambiguous")
        if rolling_window_size <= 0:
            raise ValueError("rolling_window_size must be > 0")
        self.rolling_window_size = int(rolling_window_size)
        self.forecast_ttl = int(forecast_ttl_seconds)
        self.mae_ttl = int(mae_ttl_seconds)
        self.tolerance = tolerance
        self.cooldown = float(cooldown_seconds)
        self.prefix = key_prefix
        self._clock = clock
        self._lock = threading.Lock()
        self._down_until = 0.0
        self._outage_logged = False
        self._seen_horizon = 1
        self._current_version: Optional[str] = None

        import redis as _redis       # imported here so the memory backend needs no redis-py
        self._errors_cls = (_redis.RedisError, ConnectionError, OSError)
        if client is None:
            pool = _redis.ConnectionPool.from_url(
                url, max_connections=max_connections, socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout, decode_responses=True,
                health_check_interval=30,
            )
            client = _redis.Redis(connection_pool=pool)
        self._r = client
        self._claim = client.register_script(_CLAIM_AND_PUSH)

        self.abs_error, self.rolling_mae = _metrics(registry, rolling_window_size)
        self.backend_up = Gauge(
            "forecast_quality_backend_up",
            "1 while the Redis quality backend answered its last call, 0 during an outage",
            registry=registry,
        )
        self.backend_errors = Counter(
            "forecast_quality_backend_errors_total",
            "Redis calls of the quality backend that failed (the request went on without scoring)",
            registry=registry,
        )
        self.backend_up.set(1)

    # ── keys ───────────────────────────────────────────────────────────────────

    def _quantum(self, step_seconds: int) -> float:
        return self.tolerance * step_seconds

    def _bucket(self, ts: float, step_seconds: int) -> int:
        return int(round(round(ts / self._quantum(step_seconds)) * self._quantum(step_seconds)))

    def forecast_key(self, series: str, made_at: float, step_seconds: int) -> str:
        return f"{self.prefix}forecast:{series}:{self._bucket(made_at, step_seconds)}"

    @staticmethod
    def scored_key(forecast_key: str) -> str:
        return f"{forecast_key}:scored"

    def mae_key(self, model_version: str, horizon_idx: int) -> str:
        return f"{self.prefix}metrics:mae:{model_version}:{int(horizon_idx)}"

    # ── error boundary ─────────────────────────────────────────────────────────

    def _available(self) -> bool:
        with self._lock:
            return self._clock() >= self._down_until

    def _failed(self, what: str, exc: BaseException) -> None:
        self.backend_errors.inc()
        self.backend_up.set(0)
        with self._lock:
            self._down_until = self._clock() + self.cooldown
            first = not self._outage_logged
            self._outage_logged = True
        if first:
            log.warning("quality backend unavailable during %s (%s: %s); scoring paused for %.0fs "
                        "per failure, predictions unaffected", what, type(exc).__name__, exc, self.cooldown)
        else:
            log.debug("quality backend still unavailable during %s: %s", what, exc)

    def _recovered(self) -> None:
        self.backend_up.set(1)
        with self._lock:
            if self._outage_logged:
                self._outage_logged = False
                log.info("quality backend answered again; scoring resumed")

    # ── public API (QualityBackend) ────────────────────────────────────────────

    def record(self, series: str, values: Sequence[float], predictions: Sequence[float],
               step_seconds: int, timestamp: Optional[float] = None,
               model_version: Optional[str] = None) -> int:
        """
        Score earlier forecasts of `series` against this window, then store
        `predictions` for later scoring; two round trips. Returns the number
        of (forecast, step) pairs this call scored (0 during an outage).
        """
        if step_seconds <= 0 or not predictions or not series:
            return 0
        now = float(timestamp) if timestamp is not None else self._clock()
        version = str(model_version) if model_version else "unknown"
        with self._lock:
            self._seen_horizon = max(self._seen_horizon, len(predictions))
            # The serving version is set by reset() on a swap. A record of the
            # previous version still in flight is stored and scored under its
            # own version's key (nothing of it reaches the new model's window
            # or gauges); only a pod that never saw a swap adopts the version
            # of its first record.
            if self._current_version is None:
                self._current_version = version
        if not self._available():
            return 0
        try:
            claims = self._find_claims(series, list(values), step_seconds, now)
            count = self._execute(claims, store=(series, now, list(predictions), version, step_seconds))
        except self._errors_cls as exc:
            self._failed("record", exc)
            return 0
        except Exception:                      # a bug here must not reach /predict
            log.exception("quality backend: unexpected error in record")
            return 0
        self._recovered()
        return count

    def store_forecast(self, series: str, made_at: float, predictions: Sequence[float],
                       model_version: str, step_seconds: int, pipe: Any = None) -> bool:
        """
        Write one forecast under ``forecast:{series}:{timestamp}`` with the
        TTL, as a single atomic set-with-expire (SET … EX ttl NX). NX keeps
        the first forecast made for an instant: the scaler polls more often
        than its windows advance a quantum, and a retry must not be counted
        as a second forecast. With `pipe` the command is queued on that
        pipeline (the caller executes it); otherwise it runs now. Returns
        False when the payload could not be serialised or Redis failed.
        """
        payload = self._serialize(made_at, predictions, model_version, step_seconds)
        if payload is None:
            return False
        key = self.forecast_key(series, made_at, step_seconds)
        target = pipe if pipe is not None else self._r
        if pipe is not None:
            target.set(key, payload, ex=self.forecast_ttl, nx=True)
            return True
        if not self._available():
            return False
        try:
            self._r.set(key, payload, ex=self.forecast_ttl, nx=True)
        except self._errors_cls as exc:
            self._failed("store_forecast", exc)
            return False
        self._recovered()
        return True

    def resolve_and_score(self, series: str, values: Sequence[float], step_seconds: int,
                          timestamp: Optional[float] = None) -> int:
        """
        Score the forecasts whose targets this window carries, without storing
        a new forecast. Round trip 1 reads the candidate forecasts and their
        claimed steps; errors are computed sample by sample; round trip 2 runs
        the claim-and-push script per pair and refreshes the gauges. Returns
        how many pairs were scored by this call.
        """
        if step_seconds <= 0 or not series or not values:
            return 0
        now = float(timestamp) if timestamp is not None else self._clock()
        if not self._available():
            return 0
        try:
            claims = self._find_claims(series, list(values), step_seconds, now)
            count = self._execute(claims, store=None)
        except self._errors_cls as exc:
            self._failed("resolve_and_score", exc)
            return 0
        self._recovered()
        return count

    def get_rolling_mae(self, model_version: str, horizon_idx: int) -> Optional[float]:
        """
        Mean of the error window ``metrics:mae:{model_version}:{horizon_idx}``.
        None when the window is empty, Redis is unavailable, or no entry
        parses as a finite number; entries that do not parse are skipped.
        """
        if not self._available():
            return None
        try:
            raw = self._r.lrange(self.mae_key(str(model_version), horizon_idx), 0, -1)
        except self._errors_cls as exc:
            self._failed("get_rolling_mae", exc)
            return None
        self._recovered()
        return _mean(raw)

    def snapshot(self, model_version: Optional[str] = None, horizon: int = 0) -> Dict[int, float]:
        """Rolling MAE per horizon step of `model_version` (default: the version
        of the last recorded forecast) for steps 1..horizon (default: the
        longest horizon seen), one pipelined round trip."""
        with self._lock:
            version = str(model_version) if model_version else self._current_version
            steps = int(horizon) if horizon > 0 else self._seen_horizon
        if version is None or steps <= 0 or not self._available():
            return {}
        try:
            pipe = self._r.pipeline(transaction=False)
            for k in range(1, steps + 1):
                pipe.lrange(self.mae_key(version, k), 0, -1)
            windows = pipe.execute()
        except self._errors_cls as exc:
            self._failed("snapshot", exc)
            return {}
        self._recovered()
        out: Dict[int, float] = {}
        for k, raw in enumerate(windows, start=1):
            mean = _mean(raw)
            if mean is not None:
                out[k] = mean
        return out

    def reset(self, pending_only: bool = False, model_version: Optional[str] = None) -> None:
        """
        On a model swap the pod's gauges stop describing the new version until
        it has errors of its own; the shared state is left alone (another
        replica may still serve the old version, and its forecasts score into
        their own ``metrics:mae:{old_version}:*`` windows, never the new one's).
        `model_version` names the version that serves from now on: gauges and
        `snapshot` follow it, and a record of the old version still in flight
        cannot move them back. Without it the version is forgotten and the next
        record sets it again.
        """
        self.rolling_mae.clear()
        with self._lock:
            # without a version the pod forgets it and adopts that of the next record
            self._current_version = str(model_version) if model_version is not None else None

    def ping(self) -> bool:
        """True when Redis answers; used by the startup log and tests."""
        try:
            return bool(self._r.ping())
        except self._errors_cls as exc:
            self._failed("ping", exc)
            return False

    # ── internals ──────────────────────────────────────────────────────────────

    def _serialize(self, made_at: float, predictions: Sequence[float], model_version: str,
                   step_seconds: int) -> Optional[str]:
        try:
            preds = [float(p) for p in predictions]
        except (TypeError, ValueError):
            log.warning("quality backend: predictions are not numeric (%r); forecast not stored",
                        predictions)
            return None
        if not preds or not all(math.isfinite(p) for p in preds) or not math.isfinite(made_at):
            log.warning("quality backend: non-finite forecast or timestamp; forecast not stored")
            return None
        doc = {
            "predictions": preds,
            "model_version": str(model_version),
            "made_at": float(made_at),
            "step_seconds": int(step_seconds),
            "targets": [float(made_at) + k * int(step_seconds) for k in range(1, len(preds) + 1)],
        }
        try:
            return json.dumps(doc, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            log.warning("quality backend: could not serialise forecast (%s)", exc)
            return None

    def _find_claims(self, series: str, values: List[float], step_seconds: int,
                     now: float) -> List[_Claim]:
        """
        Round trip 1. A forecast made m steps before this window's last sample
        has its step-k target at position j = m - k from the window's end; so
        the forecasts that can match were made 1 .. (n - 1 + H) steps ago. Each
        expected instant is probed at its quantum and the two neighbours.
        """
        n = len(values)
        if n == 0:
            return []
        with self._lock:
            horizon = self._seen_horizon
        q = self._quantum(step_seconds)
        keys: List[str] = []
        seen: set = set()
        for m in range(1, n + horizon):
            t0 = now - m * step_seconds
            centre = round(t0 / q)
            for b in (centre - 1, centre, centre + 1):
                key = f"{self.prefix}forecast:{series}:{int(round(b * q))}"
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        pipe = self._r.pipeline(transaction=False)
        for key in keys:
            pipe.get(key)
            pipe.smembers(self.scored_key(key))
        replies = pipe.execute()

        claims: List[_Claim] = []
        for i, key in enumerate(keys):
            raw, scored = replies[2 * i], replies[2 * i + 1]
            if not raw:
                continue
            doc = self._parse(key, raw)
            if doc is None:
                continue
            made_at, step, preds, version = doc
            done = {int(s) for s in (scored or ()) if str(s).lstrip("-").isdigit()}
            for k in range(1, len(preds) + 1):
                if k in done:
                    continue
                target = made_at + k * step
                back = (now - target) / step_seconds
                j = round(back)
                if j < 0 or j >= n or abs(back - j) > self.tolerance:
                    continue
                actual = values[n - 1 - j]
                try:
                    err = abs(float(actual) - preds[k - 1])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(err):
                    continue
                claims.append(_Claim(series, key, version, k, err))
        return claims

    def _execute(self, claims: List[_Claim], store: Optional[tuple]) -> int:
        """
        Round trip 2: claim-and-push every pair, re-read the touched windows
        of the current version for the gauges, store the new forecast. The
        pipeline runs the commands in order, so the LRANGE sees the pushes.
        """
        pipe = self._r.pipeline(transaction=False)
        with self._lock:
            current = self._current_version
        touched: List[tuple] = []
        for c in claims:
            self._claim(keys=[self.scored_key(c.forecast_key), self.mae_key(c.model_version, c.step)],
                        args=[c.step, self.forecast_ttl, repr(c.error), self.rolling_window_size,
                              self.mae_ttl], client=pipe)
        for c in claims:
            if c.model_version == current and (c.model_version, c.step) not in touched:
                touched.append((c.model_version, c.step))
        for version, step in touched:
            pipe.lrange(self.mae_key(version, step), 0, -1)
        if store is not None:
            series, made_at, predictions, version, step_seconds = store
            self.store_forecast(series, made_at, predictions, version, step_seconds, pipe=pipe)
        if not claims and store is None:
            return 0
        replies = pipe.execute()

        count = 0
        for c, won in zip(claims, replies[:len(claims)]):
            if won:
                self.abs_error.labels(horizon_step=str(c.step)).observe(c.error)
                count += 1
        for (version, step), raw in zip(touched, replies[len(claims):len(claims) + len(touched)]):
            mean = _mean(raw)
            if mean is not None:
                self.rolling_mae.labels(horizon_step=str(step)).set(mean)
        return count

    @staticmethod
    def _parse(key: str, raw: Any) -> Optional[tuple]:
        """Decode a stored forecast defensively; a corrupt entry is skipped, not fatal."""
        try:
            doc = json.loads(raw)
            preds = [float(p) for p in doc["predictions"]]
            made_at = float(doc["made_at"])
            step = int(doc["step_seconds"])
            version = str(doc.get("model_version") or "unknown")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            log.warning("quality backend: unreadable forecast at %s; skipped", key)
            return None
        if step <= 0 or not preds or not math.isfinite(made_at):
            return None
        return made_at, step, preds, version


def _mean(raw: Optional[Iterable[Any]]) -> Optional[float]:
    """Mean of the entries that parse as finite floats; None when there are none."""
    total, n = 0.0, 0
    for item in raw or ():
        try:
            v = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            total += v
            n += 1
    return total / n if n else None
