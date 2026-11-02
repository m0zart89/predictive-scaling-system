"""
Candidate models, training and evaluation.

All candidates are trained on the same chronological split and scored on the
same holdout, in original units (the notebook reported RMSE on scaled values;
these numbers are directly comparable to the demand the scaler sees).

Selection metric
----------------
For an autoscaler the two kinds of error are not worth the same: a forecast
below the demand that arrives means capacity that is not there (latency,
errors), a forecast above it means replicas that idle. Candidates are ranked
by an asymmetric cost,

    cost = mean( under_weight * shortfall + excess )

where shortfall = max(actual − forecast, 0) and excess = max(forecast −
actual, 0), both per forecast step. With `under_weight = 1` this is the MAE;
the default 3 says one unit of missing capacity costs three idle ones. The
forecast that minimises this expected cost is the w/(1+w) quantile of the
demand distribution (0.75 for w = 3), which is what the `_q75` candidates
estimate directly instead of a mean. `coverage` — the share of forecast steps
at or above the actual — is reported alongside so the trade-off is visible.
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

from .features import (SEASONAL_FEATURE_KINDS, TIMED_FEATURE_KINDS, ForecastMeta, build_features,
                       from_target, make_windows, seasonal_lags, to_target)

log = logging.getLogger("forecast-training")

DEFAULT_UNDER_WEIGHT = 3.0


@dataclass(frozen=True)
class Candidate:
    name: str
    features: str
    target: str
    factory: Callable[[int], BaseEstimator]   # horizon → unfitted estimator
    params: Dict[str, object] = field(default_factory=dict)

    @property
    def needs_time(self) -> bool:
        return self.features in TIMED_FEATURE_KINDS

    @property
    def seasonal(self) -> bool:
        return self.features in SEASONAL_FEATURE_KINDS


def season_steps_for(candidate: Candidate, period_seconds: int, step_seconds: int) -> int:
    """Samples per seasonal cycle for a seasonal candidate, 0 for the others."""
    if not candidate.seasonal:
        return 0
    return max(1, int(round(period_seconds / step_seconds)))


def _rf(_h: int) -> BaseEstimator:
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=0)


def _hgb(h: int) -> BaseEstimator:
    base = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
    return base if h == 1 else MultiOutputRegressor(base)


def _hgb_q75(h: int) -> BaseEstimator:
    # pinball loss: estimates the 75th percentile of the change, not its mean
    base = HistGradientBoostingRegressor(loss="quantile", quantile=0.75, max_iter=200,
                                         learning_rate=0.05, random_state=0)
    return base if h == 1 else MultiOutputRegressor(base)


def _ridge(_h: int) -> BaseEstimator:
    return Ridge(alpha=1.0)


_RF = {"model_type": "RandomForestRegressor", "n_estimators": 100, "min_samples_leaf": 2}
_HGB = {"model_type": "HistGradientBoostingRegressor", "max_iter": 200, "learning_rate": 0.05}
_HGB_Q75 = {**_HGB, "loss": "quantile", "quantile": 0.75}

# rf_level is what the notebook promoted; the others predict change from the
# last value so they can follow levels outside the training range.
CANDIDATES: List[Candidate] = [
    Candidate("rf_level", "window", "level", _rf, _RF),
    Candidate("rf_delta", "anchored", "delta", _rf, _RF),
    Candidate("hgb_delta", "anchored", "delta", _hgb, _HGB),
    Candidate("ridge_delta", "anchored", "delta", _ridge, {"model_type": "Ridge", "alpha": 1.0}),
    # + sin/cos of the seasonal phase: can move before a daily ramp shows in the window
    Candidate("hgb_delta_tod", "anchored_tod", "delta", _hgb, _HGB),
    Candidate("rf_delta_tod", "anchored_tod", "delta", _rf, _RF),
    # upper quantile of the change: a forecast with the headroom the cost asks for
    Candidate("hgb_q75_delta", "anchored", "delta", _hgb_q75, _HGB_Q75),
    # + the values one season ago: the level then and the change that followed
    Candidate("hgb_delta_season", "anchored_season", "delta", _hgb, _HGB),
    Candidate("rf_delta_season", "anchored_season", "delta", _rf, _RF),
    Candidate("hgb_q75_delta_season", "anchored_season", "delta", _hgb_q75, _HGB_Q75),
]
CANDIDATE_NAMES = [c.name for c in CANDIDATES]


@dataclass
class ForecastBundle:
    """A fitted model with everything needed to turn raw windows into forecasts."""
    model: BaseEstimator
    scaler: MinMaxScaler
    meta: ForecastMeta

    def predict(self, windows: np.ndarray, last_ts=None) -> np.ndarray:
        """windows (n, input_length) in original units → forecasts (n, horizon).
        `last_ts` (n,): unix time of each window's last sample, for timed features."""
        windows = np.atleast_2d(np.asarray(windows, dtype=np.float64))
        n, t = windows.shape
        scaled = self.scaler.transform(windows.reshape(-1, 1)).reshape(n, t)
        out = self.model.predict(build_features(scaled, self.meta.features, last_ts,
                                                self.meta.period_seconds, self.meta.time_steps,
                                                self.meta.horizon))
        out = from_target(scaled, out, self.meta.target)
        h = out.shape[1]
        return self.scaler.inverse_transform(out.reshape(-1, 1)).reshape(n, h)


def _last_ts(timestamps, first_target: np.ndarray):
    """Timestamp of each window's last input sample, or None without timestamps."""
    if timestamps is None:
        return None
    return np.asarray(timestamps, dtype=np.float64)[first_target - 1]


def fit_bundle(values: np.ndarray, candidate: Candidate, time_steps: int, horizon: int,
               step_seconds: int, timestamps=None, period_seconds: int = 86400) -> ForecastBundle:
    if candidate.needs_time and timestamps is None:
        raise ValueError(f"candidate {candidate.name} needs timestamps")
    season = season_steps_for(candidate, period_seconds, step_seconds)
    scaler = MinMaxScaler().fit(np.asarray(values).reshape(-1, 1))
    scaled = scaler.transform(np.asarray(values).reshape(-1, 1)).ravel()
    X, Y, first_target = make_windows(scaled, time_steps, horizon, season)
    target = to_target(X, Y, candidate.target)
    model = candidate.factory(horizon)
    feats = build_features(X, candidate.features, _last_ts(timestamps, first_target), period_seconds,
                           time_steps, horizon)
    model.fit(feats, target if horizon > 1 else target.ravel())
    meta = ForecastMeta(time_steps=time_steps, horizon=horizon, step_seconds=step_seconds,
                        features=candidate.features, target=candidate.target,
                        period_seconds=period_seconds, season_steps=season)
    return ForecastBundle(model=model, scaler=scaler, meta=meta)


@dataclass
class Score:
    rmse: float                 # mean over horizon steps
    rmse_by_step: List[float]
    mae: float
    n: int
    mae_by_step: List[float] = field(default_factory=list)   # serving compares its rolling MAE to this
    cost: float = 0.0           # asymmetric cost (selection metric), see module docstring
    coverage: float = 0.0       # share of forecast steps at or above the actual value
    under_mae: float = 0.0      # mean shortfall (actual above forecast), over all steps
    over_mae: float = 0.0       # mean excess (forecast above actual), over all steps
    under_weight: float = DEFAULT_UNDER_WEIGHT

    def as_metrics(self, prefix: str = "") -> Dict[str, float]:
        m = {f"{prefix}rmse": self.rmse, f"{prefix}mae": self.mae, f"{prefix}cost": self.cost,
             f"{prefix}coverage": self.coverage, f"{prefix}under_mae": self.under_mae,
             f"{prefix}over_mae": self.over_mae}
        for i, v in enumerate(self.rmse_by_step, start=1):
            m[f"{prefix}rmse_step_{i}"] = v
        for i, v in enumerate(self.mae_by_step, start=1):
            m[f"{prefix}mae_step_{i}"] = v
        return m


def score(y_true: np.ndarray, y_pred: np.ndarray, steps: int | None = None,
          under_weight: float = DEFAULT_UNDER_WEIGHT) -> Score:
    steps = steps or y_true.shape[1]
    err = y_pred[:, :steps] - y_true[:, :steps]
    by_step = np.sqrt(np.mean(err ** 2, axis=0))
    mae_by_step = np.mean(np.abs(err), axis=0)
    under = np.clip(-err, 0, None)
    over = np.clip(err, 0, None)
    return Score(rmse=float(np.mean(by_step)), rmse_by_step=[float(x) for x in by_step],
                 mae=float(np.mean(np.abs(err))), n=int(len(err)),
                 mae_by_step=[float(x) for x in mae_by_step],
                 cost=float(np.mean(under_weight * under + over)),
                 coverage=float(np.mean(err >= 0)),
                 under_mae=float(np.mean(under)), over_mae=float(np.mean(over)),
                 under_weight=under_weight)


def holdout_windows(values: np.ndarray, split: int, time_steps: int, horizon: int,
                    timestamps=None, season_steps: int = 0):
    """Windows whose first target is at or after `split` (inputs may look back across it).
    Returns X, Y and the last-sample timestamps (None without timestamps)."""
    X, Y, first_target = make_windows(values, time_steps, horizon, season_steps)
    keep = first_target >= split
    if not keep.any():
        raise ValueError("holdout is empty; lower --test-fraction or provide more data")
    last_ts = _last_ts(timestamps, first_target)
    return X[keep], Y[keep], (last_ts[keep] if last_ts is not None else None)


def evaluate_bundle(bundle: ForecastBundle, values: np.ndarray, split: int,
                    steps: int | None = None, timestamps=None,
                    under_weight: float = DEFAULT_UNDER_WEIGHT) -> Score:
    X, Y, last_ts = holdout_windows(values, split, bundle.meta.time_steps, bundle.meta.horizon,
                                    timestamps, bundle.meta.season_steps)
    return score(Y, bundle.predict(X, last_ts), steps, under_weight)


def level_tracking(bundle: ForecastBundle, values: np.ndarray, split: int,
                   timestamps=None) -> float:
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
    X, _, last_ts = holdout_windows(values, split, bundle.meta.time_steps, bundle.meta.horizon,
                                    timestamps, bundle.meta.season_steps)
    m = float(np.max(values[:split]))
    if m <= 0:
        return 1.0
    return float(np.mean(bundle.predict(X + 2 * m, last_ts) - bundle.predict(X + m, last_ts)) / m)


def naive_score(values: np.ndarray, split: int, time_steps: int, horizon: int,
                under_weight: float = DEFAULT_UNDER_WEIGHT) -> Score:
    """Persistence baseline: every future step equals the last observed value."""
    X, Y, _ = holdout_windows(values, split, time_steps, horizon)
    return score(Y, np.repeat(X[:, -1:], horizon, axis=1), under_weight=under_weight)


def seasonal_naive_score(values: np.ndarray, split: int, time_steps: int, horizon: int,
                         season_steps: int, under_weight: float = DEFAULT_UNDER_WEIGHT) -> Score:
    """
    Seasonal persistence: each future step equals the value one season earlier
    (the same time yesterday). On traffic with a daily pattern this is the
    baseline a model has to beat to be worth having; plain persistence is too
    easy. Raises ValueError when the holdout has no window with a full season
    of history before it.
    """
    if season_steps < horizon:
        raise ValueError("season_steps must be >= horizon")
    X, Y, _ = holdout_windows(values, split, time_steps, horizon, season_steps=season_steps)
    _, _, lag_next = seasonal_lags(X, time_steps, horizon)
    return score(Y, lag_next, under_weight=under_weight)


@dataclass
class CandidateResult:
    candidate: Candidate
    bundle: ForecastBundle
    score: Score
    tracking: float
    eligible: bool


def train_and_select(values: np.ndarray, time_steps: int, horizon: int, step_seconds: int,
                     test_fraction: float, min_tracking: float = 0.5,
                     candidates: List[Candidate] | None = None, timestamps=None,
                     period_seconds: int = 86400, under_weight: float = DEFAULT_UNDER_WEIGHT):
    """
    Fit every candidate on the training part and score it on the holdout.

    A candidate is eligible only if its level tracking is at least
    `min_tracking`; the best eligible candidate by holdout cost wins. Returns
    (results ranked eligible-first then by cost, split index). If nothing is
    eligible, results[0].eligible is False and the caller must not promote.
    Candidates the data cannot support are skipped with a warning: timed ones
    without timestamps, seasonal ones when the training part is shorter than
    a season plus a window.
    """
    candidates = candidates or CANDIDATES
    split = int(len(values) * (1 - test_fraction))
    if split < time_steps + horizon:
        raise ValueError("training part is too short for the requested window")

    results: List[CandidateResult] = []
    for cand in candidates:
        if cand.needs_time and timestamps is None:
            log.warning("candidate %s skipped: no timestamps for time-of-day features", cand.name)
            continue
        season = season_steps_for(cand, period_seconds, step_seconds)
        if season and (season < horizon or split < time_steps + season + horizon):
            log.warning("candidate %s skipped: training part of %d points holds no season of %d "
                        "steps before a window", cand.name, split, season)
            continue
        train_ts = timestamps[:split] if timestamps is not None else None
        bundle = fit_bundle(values[:split], cand, time_steps, horizon, step_seconds, train_ts,
                            period_seconds)
        s = evaluate_bundle(bundle, values, split, timestamps=timestamps, under_weight=under_weight)
        tr = level_tracking(bundle, values, split, timestamps)
        ok = tr >= min_tracking
        log.info("candidate %-20s holdout cost=%.4f coverage=%.2f rmse=%.4f mae=%.4f tracking=%.2f%s",
                 cand.name, s.cost, s.coverage, s.rmse, s.mae, tr,
                 "" if ok else "  (ineligible: does not follow level)")
        results.append(CandidateResult(cand, bundle, s, tr, ok))
    if not results:
        raise ValueError("no candidate could be trained (timed candidates need timestamps, "
                         "seasonal ones a season of history)")
    results.sort(key=lambda r: (not r.eligible, r.score.cost))
    return results, split
