"""
MLflow side of the pipeline: log every candidate, compare the winner with the
current champion, and move the alias only when the winner is better.

Promotion rules (in order):

1. Some candidate must pass the level-tracking check, otherwise nothing changes.
2. The winner must beat the baseline: persistence (last value repeated) by
   default, or with `baseline=seasonal|best` the seasonal persistence (the
   value one season earlier) / the better of the two. Both baselines are
   logged either way.
3. If there is no champion yet, promote.
4. If the champion fails the level-tracking check on today's data, replace it:
   a model that cannot follow load above its training range is unsafe to
   autoscale on, whatever its score.
5. Otherwise promote only if the winner's cost beats the champion's by at least
   `min_improvement` (relative), on data neither model was trained on: windows
   after both the holdout split and the champion's own training data
   (`data_end_ts` version tag). With fewer than `min_fresh_windows` such
   windows the champion is kept — comparing on data it was fitted to would
   always favour it. When the horizons differ, the comparison uses the steps
   both models predict.

"Cost" throughout is the asymmetric cost of forecast_training.model: a unit
of demand the forecast missed costs `under_weight` idle units.

Models use registry *aliases* (`models:/name@champion`) rather than the
deprecated stages API.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .features import LEGACY_META, ForecastMeta
from .model import (CANDIDATES, DEFAULT_UNDER_WEIGHT, Candidate, CandidateResult, ForecastBundle,
                    evaluate_bundle, fit_bundle, level_tracking, naive_score, seasonal_naive_score,
                    train_and_select)

log = logging.getLogger("forecast-training")

META_ARTIFACT = "forecast_meta.json"
BASELINES = ("persistence", "seasonal", "best")


@dataclass
class TrainConfig:
    model_name: str = "cpu-forecasting"
    alias: str = "champion"
    experiment: str = "cpu-forecasting"
    time_steps: int = 10
    horizon: int = 6
    step_seconds: int = 300
    period_seconds: int = 86400      # seasonal cycle for time-of-day features
    test_fraction: float = 0.2
    min_tracking: float = 0.5
    min_improvement: float = 0.0
    min_fresh_windows: int = 48      # 4 hours of 5-minute data
    under_weight: float = DEFAULT_UNDER_WEIGHT   # cost of a unit of shortfall, in idle units
    baseline: str = "persistence"    # the winner must beat: persistence | seasonal | best
    refit: bool = True
    dry_run: bool = False
    candidates: List[str] = field(default_factory=lambda: [c.name for c in CANDIDATES])

    def __post_init__(self):
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}, got {self.baseline!r}")
        if self.under_weight <= 0:
            raise ValueError("under_weight must be > 0")


@dataclass
class Outcome:
    decision: str                       # promoted | kept_champion | rejected
    reason: str
    winner: Optional[str] = None
    run_id: Optional[str] = None
    version: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def promoted(self) -> bool:
        return self.decision == "promoted"

    def to_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason, "winner": self.winner,
                "run_id": self.run_id, "version": self.version, "metrics": self.metrics}


# ── champion ───────────────────────────────────────────────────────────────────

def load_champion(client: MlflowClient, name: str, alias: str) -> Optional[tuple]:
    """Return (bundle, version, tags) for the aliased model, or None if there is none."""
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None

    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    tmp = tempfile.mkdtemp()
    scaler_dir = client.download_artifacts(mv.run_id, "scaler", dst_path=tmp)
    with open(os.path.join(scaler_dir, sorted(os.listdir(scaler_dir))[0]), "rb") as f:
        scaler = pickle.load(f)
    try:
        meta_path = client.download_artifacts(mv.run_id, META_ARTIFACT, dst_path=tmp)
        with open(meta_path) as f:
            meta = ForecastMeta.from_dict(json.load(f))
    except (MlflowException, OSError):
        meta = LEGACY_META   # registered by the notebook, before this contract
    return ForecastBundle(model=model, scaler=scaler, meta=meta), mv.version, dict(mv.tags or {})


# ── pipeline ───────────────────────────────────────────────────────────────────

def _log_bundle(bundle: ForecastBundle) -> None:
    mlflow.sklearn.log_model(bundle.model, "model")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scaler.pkl")
        with open(path, "wb") as f:
            pickle.dump(bundle.scaler, f)
        mlflow.log_artifact(path, "scaler")
    mlflow.log_dict(json.loads(bundle.meta.to_json()), META_ARTIFACT)


def run_pipeline(values: np.ndarray, cfg: TrainConfig, source: str,
                 filled_points: int = 0, timestamps: Optional[np.ndarray] = None) -> Outcome:
    """`timestamps` (unix seconds, aligned with `values`) enable the fresh-data
    comparison against the champion and are recorded on the registered version."""
    client = MlflowClient()
    if timestamps is not None and len(timestamps) != len(values):
        raise ValueError("timestamps and values differ in length")
    by_name = {c.name: c for c in CANDIDATES}
    unknown = set(cfg.candidates) - set(by_name)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    candidates: List[Candidate] = [by_name[n] for n in cfg.candidates]

    mlflow.set_experiment(cfg.experiment)
    run_name = "retrain-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params({
            "source": source, "n_points": len(values), "filled_points": filled_points,
            "time_steps": cfg.time_steps, "horizon": cfg.horizon,
            "step_seconds": cfg.step_seconds, "period_seconds": cfg.period_seconds,
            "test_fraction": cfg.test_fraction,
            "min_tracking": cfg.min_tracking, "min_improvement": cfg.min_improvement,
            "under_weight": cfg.under_weight, "baseline": cfg.baseline,
            "candidates": ",".join(cfg.candidates),
        })

        results, split = train_and_select(values, cfg.time_steps, cfg.horizon, cfg.step_seconds,
                                          cfg.test_fraction, cfg.min_tracking, candidates,
                                          timestamps, cfg.period_seconds, cfg.under_weight)
        for r in results:
            with mlflow.start_run(run_name=r.candidate.name, nested=True):
                mlflow.log_params({**r.candidate.params, "features": r.candidate.features,
                                   "target": r.candidate.target,
                                   "season_steps": r.bundle.meta.season_steps})
                mlflow.log_metrics({**r.score.as_metrics("holdout_"), "level_tracking": r.tracking,
                                    "eligible": float(r.eligible)})

        naive = naive_score(values, split, cfg.time_steps, cfg.horizon, cfg.under_weight)
        mlflow.log_metrics(naive.as_metrics("naive_"))
        best: CandidateResult = results[0]
        metrics = {"winner_rmse": best.score.rmse, "winner_cost": best.score.cost,
                   "winner_coverage": best.score.coverage, "winner_level_tracking": best.tracking,
                   "naive_rmse": naive.rmse, "naive_cost": naive.cost}
        baseline_cost, baseline_name = naive.cost, "persistence"
        season = max(1, int(round(cfg.period_seconds / cfg.step_seconds)))
        try:
            seasonal = seasonal_naive_score(values, split, cfg.time_steps, cfg.horizon, season,
                                            cfg.under_weight)
        except ValueError as exc:      # less than a season of history before the holdout
            log.info("seasonal baseline not available: %s", exc)
        else:
            mlflow.log_metrics(seasonal.as_metrics("seasonal_naive_"))
            metrics.update({"seasonal_naive_rmse": seasonal.rmse,
                            "seasonal_naive_cost": seasonal.cost})
            if cfg.baseline == "seasonal" or (cfg.baseline == "best" and seasonal.cost < naive.cost):
                baseline_cost, baseline_name = seasonal.cost, "seasonal persistence"
        log.info("baselines: persistence cost=%.4f%s; winner %s cost=%.4f", naive.cost,
                 f", seasonal cost={metrics['seasonal_naive_cost']:.4f}"
                 if "seasonal_naive_cost" in metrics else "", best.candidate.name, best.score.cost)
        outcome = _decide(client, cfg, values, timestamps, split, best, baseline_cost,
                          baseline_name, metrics)
        outcome.run_id = parent.info.run_id
        outcome.winner = best.candidate.name if best.eligible else None

        if outcome.promoted and not cfg.dry_run:
            final = (fit_bundle(values, best.candidate, cfg.time_steps, cfg.horizon,
                                cfg.step_seconds, timestamps, cfg.period_seconds)
                     if cfg.refit else best.bundle)
            fitted_until = len(values) if cfg.refit else split
            _log_bundle(final)
            mv = mlflow.register_model(f"runs:/{parent.info.run_id}/model", cfg.model_name)
            client.set_registered_model_alias(cfg.model_name, cfg.alias, mv.version)
            tags = {"candidate": best.candidate.name, "horizon": cfg.horizon,
                    "holdout_rmse": f"{best.score.rmse:.6f}",
                    "holdout_cost": f"{best.score.cost:.6f}",
                    "holdout_coverage": f"{best.score.coverage:.4f}",
                    "under_weight": f"{cfg.under_weight:g}",
                    # per-step holdout MAE: serving exports it so the drift alert
                    # compares the live rolling MAE with what the model achieved
                    **{f"holdout_mae_step_{i}": f"{v:.6f}"
                       for i, v in enumerate(best.score.mae_by_step, start=1)},
                    "level_tracking": f"{best.tracking:.3f}"}
            if timestamps is not None:
                # last point the registered model was fitted on; the next run
                # only compares against it on data after this
                tags["data_end_ts"] = int(timestamps[fitted_until - 1])
            for k, v in tags.items():
                client.set_model_version_tag(cfg.model_name, mv.version, k, str(v))
            outcome.version = str(mv.version)
            log.info("promoted %s v%s → @%s", cfg.model_name, mv.version, cfg.alias)
        elif outcome.promoted:
            outcome.decision, outcome.reason = "dry_run", "would promote: " + outcome.reason

        mlflow.log_metrics(outcome.metrics)
        mlflow.set_tags({"decision": outcome.decision, "decision_reason": outcome.reason,
                         "winner": outcome.winner or ""})
    return outcome


def _decide(client, cfg: TrainConfig, values, timestamps, split, best: CandidateResult,
            baseline_cost: float, baseline_name: str, metrics: dict) -> Outcome:
    if not best.eligible:
        return Outcome("rejected", f"no candidate reached level tracking {cfg.min_tracking}",
                       metrics=metrics)
    if best.score.cost >= baseline_cost:
        return Outcome("rejected", f"winner cost {best.score.cost:.4f} does not beat the "
                                   f"{baseline_name} baseline {baseline_cost:.4f}", metrics=metrics)

    champ = load_champion(client, cfg.model_name, cfg.alias)
    if champ is None:
        return Outcome("promoted", f"no @{cfg.alias} model yet; {best.candidate.name} "
                                   f"cost {best.score.cost:.4f}", metrics=metrics)

    champ_bundle, champ_version, champ_tags = champ
    champ_tracking = level_tracking(champ_bundle, values, split, timestamps)
    metrics["champion_level_tracking"] = champ_tracking
    if champ_tracking < cfg.min_tracking:
        return Outcome("promoted", f"champion v{champ_version} fails level tracking "
                                   f"({champ_tracking:.2f} < {cfg.min_tracking})", metrics=metrics)

    # Compare only on windows neither model was fitted to.
    compare_from, note = split, ""
    data_end = champ_tags.get("data_end_ts")
    if data_end is not None and timestamps is not None:
        compare_from = max(split, int(np.searchsorted(timestamps, int(data_end), side="right")))
    else:
        note = " (champion has no data_end_ts tag; it may have been fitted to this holdout)"
    steps = min(champ_bundle.meta.horizon, best.bundle.meta.horizon)
    t = max(champ_bundle.meta.input_length, best.bundle.meta.input_length)
    fresh = len(values) - max(compare_from, t) - max(champ_bundle.meta.horizon,
                                                     best.bundle.meta.horizon) + 1
    metrics["fresh_windows"] = float(max(fresh, 0))
    if fresh < cfg.min_fresh_windows:
        return Outcome("kept_champion", f"only {max(fresh, 0)} windows of data newer than "
                                        f"champion v{champ_version}; need {cfg.min_fresh_windows} "
                                        f"for a fair comparison", metrics=metrics)

    # The same windows for both: the later of the two inputs' reach sets the start.
    compare_from = max(compare_from, t)
    champ_score = evaluate_bundle(champ_bundle, values, compare_from, steps, timestamps,
                                  cfg.under_weight)
    new_score = evaluate_bundle(best.bundle, values, compare_from, steps, timestamps,
                                cfg.under_weight)
    metrics.update({"champion_cost": champ_score.cost, "champion_rmse": champ_score.rmse,
                    "winner_cost_common_steps": new_score.cost,
                    "winner_rmse_common_steps": new_score.rmse})
    vs = (f"v{champ_version} cost {champ_score.cost:.4f} vs {best.candidate.name} "
          f"{new_score.cost:.4f} over {steps} step(s), {fresh} fresh windows{note}")
    if new_score.cost < champ_score.cost * (1 - cfg.min_improvement):
        return Outcome("promoted", vs, metrics=metrics)
    return Outcome("kept_champion", vs, metrics=metrics)


def reload_serving(url: str, timeout: float = 30.0, token: Optional[str] = None) -> bool:
    """Ask forecast-serving to load the new champion. Failure is logged, not raised.
    `token` goes into X-Reload-Token (forecast-serving's RELOAD_TOKEN)."""
    headers = {"X-Reload-Token": token} if token else {}
    req = urllib.request.Request(f"{url.rstrip('/')}/reload", data=b"", method="POST",
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            log.info("serving reloaded: %s", resp.read().decode()[:200])
            return True
    except (urllib.error.URLError, OSError) as exc:
        log.warning("serving reload failed (%s); it will pick the model up on next restart", exc)
        return False
