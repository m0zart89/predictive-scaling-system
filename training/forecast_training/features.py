"""
Windowing and the model contract shared with serving.

A registered model is a sklearn estimator plus two artifacts:

* `scaler/`               MinMaxScaler fitted on the training values
* `forecast_meta.json`    how to turn a window into features and outputs back
                          into values (see ForecastMeta)

serving/app.py implements the same `features`/`target` transforms for a single
window; tests/test_training_serving.py checks that both sides agree.

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

FORMAT_VERSION = 1
FEATURE_KINDS = ("window", "anchored")
TARGET_KINDS = ("level", "delta")


@dataclass(frozen=True)
class ForecastMeta:
    time_steps: int = 10
    horizon: int = 1
    step_seconds: int = 300
    features: str = "window"   # window: scaled window as-is; anchored: relative to last value
    target: str = "level"      # level: model outputs values; delta: outputs change from last value
    format_version: int = FORMAT_VERSION

    def __post_init__(self):
        if self.features not in FEATURE_KINDS:
            raise ValueError(f"features must be one of {FEATURE_KINDS}")
        if self.target not in TARGET_KINDS:
            raise ValueError(f"target must be one of {TARGET_KINDS}")
        if self.time_steps < 2 or self.horizon < 1:
            raise ValueError("time_steps must be >= 2 and horizon >= 1")

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


def make_windows(values: np.ndarray, time_steps: int, horizon: int):
    """
    Slide over `values` (1-D). Returns X (n, time_steps) of inputs, Y (n, horizon)
    of the values that follow, and `first_target` (n,) — the index in `values`
    of each row's first target, used for time-based splitting.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values) - time_steps - horizon + 1
    if n <= 0:
        raise ValueError(
            f"need at least {time_steps + horizon} points for time_steps={time_steps}, "
            f"horizon={horizon}; got {len(values)}"
        )
    idx = np.arange(n)[:, None]
    X = values[idx + np.arange(time_steps)]
    Y = values[idx + time_steps + np.arange(horizon)]
    return X, Y, np.arange(n) + time_steps


def build_features(X: np.ndarray, kind: str) -> np.ndarray:
    X = np.atleast_2d(X)
    if kind == "window":
        return X
    if kind == "anchored":
        last = X[:, -1:]
        # shape of the window relative to now, plus the current level
        return np.hstack([X[:, :-1] - last, last])
    raise ValueError(kind)


def to_target(X: np.ndarray, Y: np.ndarray, kind: str) -> np.ndarray:
    return Y - np.atleast_2d(X)[:, -1:] if kind == "delta" else Y


def from_target(X: np.ndarray, out: np.ndarray, kind: str) -> np.ndarray:
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(-1, 1) if np.atleast_2d(X).shape[0] > 1 else out.reshape(1, -1)
    return out + np.atleast_2d(X)[:, -1:] if kind == "delta" else out
