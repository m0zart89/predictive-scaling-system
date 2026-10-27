"""
Load a CPU demand series and put it on a regular grid.

Two sources:

* a Prometheus `query_range` export (the JSON shape of notebooks/data-jan-2022.json)
* a live Prometheus server (the CronJob path)

Both return a `Series`: timestamps on an exact `step_seconds` grid with no
missing points. Missing points, and any ranges the caller marks as bad (an
outage that recorded near-zero usage, say), are filled with the mean value for
the same time of day — the same treatment the notebook applies by hand to the
17 January outage.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Prometheus caps a single query_range response at 11,000 points per series.
_MAX_POINTS_PER_QUERY = 10_000


class DataError(RuntimeError):
    """The source returned no usable data."""


@dataclass
class Series:
    timestamps: np.ndarray  # int64 unix seconds, strictly increasing, regular
    values: np.ndarray      # float64, finite
    step_seconds: int
    filled: int = 0         # how many points were imputed

    def __len__(self) -> int:
        return len(self.values)


# ── sources ────────────────────────────────────────────────────────────────────

def _parse_matrix(body: dict) -> List[Tuple[int, float]]:
    if body.get("status", "success") != "success":
        raise DataError(f"Prometheus error: {body.get('error', 'unknown')}")
    data = body.get("data", body)  # accept both the API envelope and bare `data`
    result = data.get("result", [])
    if len(result) != 1:
        raise DataError(
            f"expected exactly one series, got {len(result)}; aggregate the query (e.g. sum(...))"
        )
    points = []
    for ts, raw in result[0].get("values", []):
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            points.append((int(float(ts)), v))
    return points


def load_export(path: str) -> List[Tuple[int, float]]:
    """Read a saved query_range response (as in notebooks/data-jan-2022.json)."""
    with open(path) as f:
        return _parse_matrix(json.load(f))


def query_prometheus(address: str, query: str, days: float, step_seconds: int,
                     timeout: float = 30.0, now: float | None = None) -> List[Tuple[int, float]]:
    """Fetch the last `days` of `query` from Prometheus, in chunks."""
    end = time.time() if now is None else now
    start = end - days * 86400
    chunk = _MAX_POINTS_PER_QUERY * step_seconds
    points: List[Tuple[int, float]] = []
    t = start
    while t < end:
        t_end = min(end, t + chunk)
        params = urllib.parse.urlencode({
            "query": query, "start": f"{t:.0f}", "end": f"{t_end:.0f}", "step": str(step_seconds),
        })
        url = f"{address.rstrip('/')}/api/v1/query_range?{params}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise DataError(f"Prometheus returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise DataError(f"Prometheus unreachable at {address}: {exc}") from exc
        try:
            points.extend(_parse_matrix(body))
        except DataError as exc:
            if "got 0" not in str(exc):   # an empty chunk is fine, a malformed one is not
                raise
        t = t_end + step_seconds
    return points


# ── regularisation ─────────────────────────────────────────────────────────────

def regularize(points: Sequence[Tuple[int, float]], step_seconds: int,
               exclude: Iterable[Tuple[int, int]] = ()) -> Series:
    """
    Snap points onto a regular grid and fill holes.

    Points are bucketed to the nearest grid slot (duplicates averaged). Empty
    slots and slots inside any `exclude` [start, end] range are filled with the
    mean of the same time of day across the rest of the series; if a time of
    day has no data at all, linear interpolation is used instead.
    """
    if not points:
        raise DataError("no data points")

    ts = np.array([p[0] for p in points], dtype=np.int64)
    vs = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(ts)
    ts, vs = ts[order], vs[order]

    t0 = int(round(ts[0] / step_seconds) * step_seconds)
    slots = np.rint((ts - t0) / step_seconds).astype(np.int64)
    n = int(slots[-1]) + 1
    sums = np.bincount(slots, weights=vs, minlength=n)
    counts = np.bincount(slots, minlength=n)
    grid_t = t0 + np.arange(n, dtype=np.int64) * step_seconds

    values = np.full(n, np.nan)
    have = counts > 0
    values[have] = sums[have] / counts[have]

    for start, end in exclude:
        values[(grid_t >= start) & (grid_t <= end)] = np.nan

    missing = np.isnan(values)
    filled = int(missing.sum())
    if filled == n:
        raise DataError("every point is missing or excluded")

    if filled:
        slots_per_day = max(1, 86400 // step_seconds)
        tod = ((grid_t % 86400) // step_seconds) % slots_per_day
        tod_sum = np.bincount(tod[~missing], weights=values[~missing], minlength=slots_per_day)
        tod_cnt = np.bincount(tod[~missing], minlength=slots_per_day)
        profile = np.divide(tod_sum, tod_cnt, out=np.full(slots_per_day, np.nan), where=tod_cnt > 0)
        values[missing] = profile[tod[missing]]

        still = np.isnan(values)
        if still.any():
            idx = np.arange(n)
            values[still] = np.interp(idx[still], idx[~still], values[~still])

    return Series(timestamps=grid_t, values=values, step_seconds=step_seconds, filled=filled)


def parse_exclude(spec: str) -> Tuple[int, int]:
    """'2022-01-17T00:00:00Z/2022-01-17T08:00:00Z' or 'START_UNIX/END_UNIX' → (start, end)."""
    from datetime import datetime, timezone

    try:
        a, b = spec.split("/", 1)
    except ValueError as exc:
        raise ValueError(f"exclude range must be START/END, got {spec!r}") from exc

    def to_ts(s: str) -> int:
        s = s.strip()
        if s.lstrip("-").isdigit():
            return int(s)
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return int(d.timestamp())

    start, end = to_ts(a), to_ts(b)
    if end < start:
        raise ValueError(f"exclude range ends before it starts: {spec!r}")
    return start, end
