"""
Candidate models, training and evaluation.

All candidates are trained on the same chronological split and scored on the
same holdout, in original units (the notebook reported RMSE on scaled values;
these numbers are directly comparable to the demand the scaler sees).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

from .features import ForecastMeta, build_features, from_target, make_windows, to_target

log = logging.getLogger("forecast-training")


@dataclass(frozen=True)
class Candidate:
    name: str
    features: str
    target: str
    factory: Callable[[int], BaseEstimator]   # horizon → unfitted estimator
    params: Dict[str, object] = field(default_factory=dict)


def _rf(_h: int) -> BaseEstimator:
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=0)


def _hgb(h: int) -> BaseEstimator:
    base = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
    return base if h == 1 else MultiOutputRegressor(base)


def _ridge(_h: int) -> BaseEstimator:
    return Ridge(alpha=1.0)


# rf_level is what the notebook promoted; the others predict change from the
# last value so they can follow levels outside the training range.
CANDIDATES: List[Candidate] = [
    Candidate("rf_level", "window", "level", _rf, {"model_type": "RandomForestRegressor",
                                                   "n_estimators": 100, "min_samples_leaf": 2}),
    Candidate("rf_delta", "anchored", "delta", _rf, {"model_type": "RandomForestRegressor",
                                                     "n_estimators": 100, "min_samples_leaf": 2}),
    Candidate("hgb_delta", "anchored", "delta", _hgb, {"model_type": "HistGradientBoostingRegressor",
                                                       "max_iter": 200, "learning_rate": 0.05}),
    Candidate("ridge_delta", "anchored", "delta", _ridge, {"model_type": "Ridge", "alpha": 1.0}),
]
CANDIDATE_NAMES = [c.name for c in CANDIDATES]


@dataclass
class ForecastBundle:
    """A fitted model with everything needed to turn raw windows into forecasts."""
    model: BaseEstimator
    scaler: MinMaxScaler
    meta: ForecastMeta

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """windows (n, time_steps) in original units → forecasts (n, horizon)."""
        windows = np.atleast_2d(np.asarray(windows, dtype=np.float64))
        n, t = windows.shape
        scaled = self.scaler.transform(windows.reshape(-1, 1)).reshape(n, t)
        out = self.model.predict(build_features(scaled, self.meta.features))
        out = from_target(scaled, out, self.meta.target)
        h = out.shape[1]
        return self.scaler.inverse_transform(out.reshape(-1, 1)).reshape(n, h)


def fit_bundle(values: np.ndarray, candidate: Candidate, time_steps: int, horizon: int,
               step_seconds: int) -> ForecastBundle:
    scaler = MinMaxScaler().fit(np.asarray(values).reshape(-1, 1))
    scaled = scaler.transform(np.asarray(values).reshape(-1, 1)).ravel()
    X, Y, _ = make_windows(scaled, time_steps, horizon)
    target = to_target(X, Y, candidate.target)
    model = candidate.factory(horizon)
    model.fit(build_features(X, candidate.features), target if horizon > 1 else target.ravel())
    meta = ForecastMeta(time_steps=time_steps, horizon=horizon, step_seconds=step_seconds,
                        features=candidate.features, target=candidate.target)
    return ForecastBundle(model=model, scaler=scaler, meta=meta)


@dataclass
class Score:
    rmse: float                 # mean over horizon steps (selection metric)
    rmse_by_step: List[float]
    mae: float
    n: int

    def as_metrics(self, prefix: str = "") -> Dict[str, float]:
        m = {f"{prefix}rmse": self.rmse, f"{prefix}mae": self.mae}
        for i, v in enumerate(self.rmse_by_step, start=1):
            m[f"{prefix}rmse_step_{i}"] = v
        return m


def score(y_true: np.ndarray, y_pred: np.ndarray, steps: int | None = None) -> Score:
    steps = steps or y_true.shape[1]
    err = y_pred[:, :steps] - y_true[:, :steps]
    by_step = np.sqrt(np.mean(err ** 2, axis=0))
    return Score(rmse=float(np.mean(by_step)), rmse_by_step=[float(x) for x in by_step],
                 mae=float(np.mean(np.abs(err))), n=int(len(err)))


def holdout_windows(values: np.ndarray, split: int, time_steps: int, horizon: int):
    """Windows whose first target is at or after `split` (inputs may look back across it)."""
    X, Y, first_target = make_windows(values, time_steps, horizon)
    keep = first_target >= split
    if not keep.any():
        raise ValueError("holdout is empty; lower --test-fraction or provide more data")
    return X[keep], Y[keep]


def evaluate_bundle(bundle: ForecastBundle, values: np.ndarray, split: int,
                    steps: int | None = None) -> Score:
    X, Y = holdout_windows(values, split, bundle.meta.time_steps, bundle.meta.horizon)
    return score(Y, bundle.predict(X), steps)


def level_tracking(bundle: ForecastBundle, values: np.ndarray, split: int) -> float:
    """
    Behavioural check: how much does the forecast move when load moves *beyond*
    anything seen in training?

    Every holdout window is lifted twice — by m and by 2m, where m is the
    training maximum — so both copies are entirely out of range, and

        tracking = mean(forecast(X + 2m) - forecast(X + m)) / m

    ~1.0: the forecast moves one-for-one with the level. ~0: the model is
    pinned to its training range and would forecast *less* load than is
    arriving — for an autoscaler, scaling down into a spike. Measuring between
    two out-of-range levels (rather than against the in-range baseline) keeps
    a saturated model from getting credit for the distance up to its ceiling.
    """
    X, _ = holdout_windows(values, split, bundle.meta.time_steps, bundle.meta.horizon)
    m = float(np.max(values[:split]))
    if m <= 0:
        return 1.0
    return float(np.mean(bundle.predict(X + 2 * m) - bundle.predict(X + m)) / m)


def naive_score(values: np.ndarray, split: int, time_steps: int, horizon: int) -> Score:
    """Persistence baseline: every future step equals the last observed value."""
    X, Y = holdout_windows(values, split, time_steps, horizon)
    return score(Y, np.repeat(X[:, -1:], horizon, axis=1))


@dataclass
class CandidateResult:
    candidate: Candidate
    bundle: ForecastBundle
    score: Score
    tracking: float
    eligible: bool


def train_and_select(values: np.ndarray, time_steps: int, horizon: int, step_seconds: int,
                     test_fraction: float, min_tracking: float = 0.5,
                     candidates: List[Candidate] | None = None):
    """
    Fit every candidate on the training part and score it on the holdout.

    A candidate is eligible only if its level tracking is at least
    `min_tracking`; the best eligible candidate by holdout RMSE wins. Returns
    (results ranked eligible-first then by RMSE, split index). If nothing is
    eligible, results[0].eligible is False and the caller must not promote.
    """
    candidates = candidates or CANDIDATES
    split = int(len(values) * (1 - test_fraction))
    if split < time_steps + horizon:
        raise ValueError("training part is too short for the requested window")

    results: List[CandidateResult] = []
    for cand in candidates:
        bundle = fit_bundle(values[:split], cand, time_steps, horizon, step_seconds)
        s = evaluate_bundle(bundle, values, split)
        tr = level_tracking(bundle, values, split)
        ok = tr >= min_tracking
        log.info("candidate %-12s holdout rmse=%.4f mae=%.4f tracking=%.2f%s",
                 cand.name, s.rmse, s.mae, tr, "" if ok else "  (ineligible: does not follow level)")
        results.append(CandidateResult(cand, bundle, s, tr, ok))
    results.sort(key=lambda r: (not r.eligible, r.score.rmse))
    return results, split
