"""
Windowing and the model contract shared with serving.

A registered model is a sklearn estimator plus two artifacts:

* `scaler/`               MinMaxScaler fitted on the training values
* `forecast_meta.json`    how to turn a window into features and outputs back
                          into values (see ForecastMeta)

serving/app.py implements the same `features`/`target` transforms for a single
window; tests/test_training_serving.py checks that both sides agree.

Why time-of-day features ("anchored_tod")
-----------------------------------------
A window of the last ten samples says where the load is and which way it is
moving, not what time it is. On a daily pattern the morning ramp starts at
the same hour every day; the window only sees it once it has begun. Encoding
the last sample's position in the seasonal cycle (sin/cos of the phase, so
23:55 and 00:05 are neighbours) lets the model learn "at 06:00 the next half
hour goes up" and move before the window shows it. `period_seconds` is the
cycle length: 86400 for real days, shorter when time is compressed, as in
the benchmark.

Why seasonal lags ("anchored_season")
-------------------------------------
In the autoscaling setting the window is short next to the cycle: ten
5-minute samples against a 24-hour day. The phase feature tells the model
*when* it is; it still has to remember from training what usually happens at
that hour. Giving it the values from one season ago — the level then, and the
change that followed over the horizon it is about to forecast — lets it read
"yesterday at this time the ramp started" straight off the input, including
weekends and irregular days that a phase alone averages away. The input grows
to `time_steps + season_steps` samples (`season_steps = period / step`); the
scaler fetches that much history and serving fills gaps in the older part.

Why a "delta" target
--------------------
A tree ensemble predicts by averaging training targets, so it can never forecast
above the highest level it saw in training, and on a ramp it pulls back towards
typical values. Predicting the *change* from the last observed value, with
features expressed relative to that value, lets the same model follow a level
it has never seen: forecast = last value + predicted change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

FORMAT_VERSION = 3
FEATURE_KINDS = ("window", "anchored", "anchored_tod", "anchored_season")
TARGET_KINDS = ("level", "delta")
# Feature kinds that need the timestamp of each window's last sample.
TIMED_FEATURE_KINDS = ("anchored_tod",)
# Feature kinds whose input reaches back one season before the window.
SEASONAL_FEATURE_KINDS = ("anchored_season",)


@dataclass(frozen=True)
class ForecastMeta:
    time_steps: int = 10
    horizon: int = 1
    step_seconds: int = 300
    features: str = "window"   # window: scaled window as-is; anchored: relative to last value;
                               # anchored_tod: anchored + sin/cos of the seasonal phase;
                               # anchored_season: anchored + the values one season ago
    target: str = "level"      # level: model outputs values; delta: outputs change from last value
    period_seconds: int = 86400   # seasonal cycle for anchored_tod (a real day; 2400 in the benchmark)
    season_steps: int = 0      # anchored_season: samples per cycle (period / step); the input is
                               # time_steps + season_steps samples long. 0 for the other kinds.
    format_version: int = FORMAT_VERSION

    def __post_init__(self):
        if self.features not in FEATURE_KINDS:
            raise ValueError(f"features must be one of {FEATURE_KINDS}")
        if self.target not in TARGET_KINDS:
            raise ValueError(f"target must be one of {TARGET_KINDS}")
        if self.time_steps < 2 or self.horizon < 1:
            raise ValueError("time_steps must be >= 2 and horizon >= 1")
        if self.period_seconds <= 0:
            raise ValueError("period_seconds must be > 0")
        if self.features in SEASONAL_FEATURE_KINDS:
            if self.season_steps < self.horizon:
                raise ValueError("season_steps must be >= horizon for seasonal features")
        elif self.season_steps != 0:
            raise ValueError(f"season_steps is only for {SEASONAL_FEATURE_KINDS}")

    @property
    def needs_time(self) -> bool:
        return self.features in TIMED_FEATURE_KINDS

    @property
    def input_length(self) -> int:
        """Samples a caller must send: the window, preceded by one season for seasonal kinds."""
        return self.time_steps + self.season_steps

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "ForecastMeta":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# Models registered before this pipeline existed (the notebook's Random Forest):
# 10 scaled values in, one scaled value out, no meta artifact.
LEGACY_META = ForecastMeta(time_steps=10, horizon=1, step_seconds=300,
                           features="window", target="level")


def make_windows(values: np.ndarray, time_steps: int, horizon: int, season_steps: int = 0):
    """
    Slide over `values` (1-D). Returns X (n, time_steps + season_steps) of inputs,
    Y (n, horizon) of the values that follow, and `first_target` (n,) — the index
    in `values` of each row's first target, used for time-based splitting. With
    `season_steps` each input starts one season before the window, so the row
    also holds the values from a cycle ago (see build_features).
    """
    values = np.asarray(values, dtype=np.float64)
    length = time_steps + season_steps
    n = len(values) - length - horizon + 1
    if n <= 0:
        raise ValueError(
            f"need at least {length + horizon} points for time_steps={time_steps}, "
            f"horizon={horizon}, season_steps={season_steps}; got {len(values)}"
        )
    idx = np.arange(n)[:, None]
    X = values[idx + np.arange(length)]
    Y = values[idx + length + np.arange(horizon)]
    return X, Y, np.arange(n) + length


def time_features(last_ts, period_seconds: int = 86400) -> np.ndarray:
    """(n, 2) sin/cos of where each window's last sample sits in the seasonal cycle."""
    ts = np.asarray(last_ts, dtype=np.float64).reshape(-1)
    phase = 2 * np.pi * (ts % period_seconds) / period_seconds
    return np.column_stack([np.sin(phase), np.cos(phase)])


def seasonal_lags(X: np.ndarray, time_steps: int, horizon: int):
    """
    Split a seasonal input row into (window, lag_now, lag_next): the last
    `time_steps` samples, the sample one season before the window's last one,
    and the `horizon` samples that followed it — what happened a cycle ago over
    the stretch being forecast. Column arithmetic: the row holds
    `time_steps + season_steps` samples ending now, so one season before "now"
    is column `time_steps - 1` and the season-ago targets are the columns after it.
    """
    X = np.atleast_2d(X)
    season = X.shape[1] - time_steps
    if season < horizon:
        raise ValueError(f"input of {X.shape[1]} samples holds no season of {horizon}+ steps "
                         f"before a window of {time_steps}")
    window = X[:, season:]
    lag_now = X[:, time_steps - 1:time_steps]
    lag_next = X[:, time_steps:time_steps + horizon]
    return window, lag_now, lag_next


def build_features(X: np.ndarray, kind: str, last_ts=None, period_seconds: int = 86400,
                   time_steps: int | None = None, horizon: int | None = None) -> np.ndarray:
    """`last_ts`: unix time of each row's last sample; required for timed kinds.
    `time_steps`/`horizon`: required for seasonal kinds, whose rows are longer than the window."""
    X = np.atleast_2d(X)
    if kind == "window":
        return X
    if kind == "anchored_season":
        if time_steps is None or horizon is None:
            raise ValueError("anchored_season features need time_steps and horizon")
        window, lag_now, lag_next = seasonal_lags(X, time_steps, horizon)
        last = window[:, -1:]
        # the window as for "anchored", then: how far today's level is from a
        # season ago, and the change that followed then over the coming horizon
        return np.hstack([window[:, :-1] - last, last, lag_now - last, lag_next - lag_now])
    if kind in ("anchored", "anchored_tod"):
        last = X[:, -1:]
        # shape of the window relative to now, plus the current level
        feats = np.hstack([X[:, :-1] - last, last])
        if kind == "anchored_tod":
            if last_ts is None:
                raise ValueError("anchored_tod features need the timestamp of the last sample")
            feats = np.hstack([feats, time_features(last_ts, period_seconds)])
        return feats
    raise ValueError(kind)


def to_target(X: np.ndarray, Y: np.ndarray, kind: str) -> np.ndarray:
    return Y - np.atleast_2d(X)[:, -1:] if kind == "delta" else Y


def from_target(X: np.ndarray, out: np.ndarray, kind: str) -> np.ndarray:
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(-1, 1) if np.atleast_2d(X).shape[0] > 1 else out.reshape(1, -1)
    return out + np.atleast_2d(X)[:, -1:] if kind == "delta" else out
