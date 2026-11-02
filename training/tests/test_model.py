import numpy as np
import pytest

from forecast_training.features import (LEGACY_META, ForecastMeta, build_features, from_target,
                                        make_windows, to_target)
from forecast_training.model import (CANDIDATES, fit_bundle, level_tracking, naive_score, score,
                                     seasonal_naive_score, train_and_select)

BY_NAME = {c.name: c for c in CANDIDATES}


# ── features / contract ────────────────────────────────────────────────────────

def test_make_windows_shapes_and_alignment():
    X, Y, first = make_windows(np.arange(10.0), time_steps=3, horizon=2)
    assert X.shape == (6, 3) and Y.shape == (6, 2)
    assert list(X[0]) == [0, 1, 2] and list(Y[0]) == [3, 4]
    assert list(X[-1]) == [5, 6, 7] and list(Y[-1]) == [8, 9]
    assert list(first) == [3, 4, 5, 6, 7, 8]


def test_make_windows_too_short():
    with pytest.raises(ValueError, match="at least 5"):
        make_windows(np.arange(4.0), 3, 2)


def test_anchored_features_and_delta_roundtrip():
    X = np.array([[1.0, 2.0, 4.0]])
    assert build_features(X, "anchored").tolist() == [[-3.0, -2.0, 4.0]]
    Y = np.array([[5.0, 7.0]])
    delta = to_target(X, Y, "delta")
    assert delta.tolist() == [[1.0, 3.0]]
    assert from_target(X, delta, "delta").tolist() == Y.tolist()


def test_from_target_handles_1d_model_output():
    X = np.array([[0.0, 1.0], [0.0, 2.0]])
    assert from_target(X, np.array([0.5, 0.5]), "delta").tolist() == [[1.5], [2.5]]
    assert from_target(X[:1], np.array([0.5, 0.25]), "level").tolist() == [[0.5, 0.25]]


def test_meta_validation_and_roundtrip():
    m = ForecastMeta(time_steps=10, horizon=6, features="anchored", target="delta")
    assert ForecastMeta.from_dict({**__import__("json").loads(m.to_json()), "extra": 1}) == m
    with pytest.raises(ValueError):
        ForecastMeta(target="ratio")
    assert LEGACY_META.horizon == 1 and LEGACY_META.target == "level"


# ── models ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(BY_NAME))
@pytest.mark.parametrize("horizon", [1, 4])
def test_every_candidate_fits_and_predicts_the_horizon(series, name, horizon):
    ts, values = series
    b = fit_bundle(values[:1500], BY_NAME[name], 10, horizon, 300, ts[:1500])
    n = b.meta.input_length                       # 10, or 10 + a day (288) for seasonal candidates
    out = b.predict(values[1500:1500 + n], ts[1499 + n:1500 + n])
    assert out.shape == (1, horizon)
    assert np.all(np.isfinite(out))
    assert (n == 298) == BY_NAME[name].seasonal


def test_models_beat_persistence_on_a_daily_pattern(series):
    _, values = series
    results, split = train_and_select(values, 10, 3, 300, 0.2)
    naive = naive_score(values, split, 10, 3)
    assert results[0].score.cost < naive.cost and results[0].score.rmse < naive.rmse


def test_delta_models_follow_levels_above_the_training_range(series):
    _, values = series
    split = int(len(values) * 0.8)
    level = fit_bundle(values[:split], BY_NAME["rf_level"], 10, 3, 300)
    delta = fit_bundle(values[:split], BY_NAME["rf_delta"], 10, 3, 300)
    assert level_tracking(level, values, split) < 0.1      # pinned to the training range
    assert level_tracking(delta, values, split) > 0.9      # follows the new level

    ramp = np.linspace(1.5, 3.0, 10)                       # far above training max (~1.2)
    assert delta.predict(ramp)[0, 0] > 2.0
    assert level.predict(ramp)[0, 0] < 1.5


def test_selection_skips_candidates_that_do_not_track_level(series):
    _, values = series
    results, _ = train_and_select(values, 10, 3, 300, 0.2, min_tracking=0.5)
    assert results[0].eligible
    assert results[0].candidate.target == "delta"
    rf_level = next(r for r in results if r.candidate.name == "rf_level")
    assert not rf_level.eligible
    # eligible results come first, each group sorted by cost
    flags = [r.eligible for r in results]
    assert flags == sorted(flags, reverse=True)
    costs = [r.score.cost for r in results if r.eligible]
    assert costs == sorted(costs)


def test_nothing_eligible_is_reported(series):
    _, values = series
    results, _ = train_and_select(values, 10, 3, 300, 0.2, min_tracking=5.0)
    assert not results[0].eligible


def test_score_limits_steps():
    y = np.array([[1.0, 1.0, 1.0]])
    p = np.array([[1.0, 2.0, 4.0]])
    assert score(y, p).rmse_by_step == [0.0, 1.0, 3.0]
    assert score(y, p, steps=2).rmse == pytest.approx(0.5)


def test_asymmetric_cost_and_coverage():
    y = np.array([[1.0, 1.0, 1.0, 1.0]])
    p = np.array([[1.0, 2.0, 0.5, 0.0]])          # exact, 1 over, 0.5 under, 1 under
    s = score(y, p, under_weight=3.0)
    assert s.coverage == pytest.approx(0.5)
    assert s.over_mae == pytest.approx(0.25) and s.under_mae == pytest.approx(0.375)
    assert s.cost == pytest.approx((1.0 + 3 * 0.5 + 3 * 1.0) / 4)
    assert score(y, p, under_weight=1.0).cost == pytest.approx(s.mae)   # weight 1 is the MAE
    m = s.as_metrics("holdout_")
    assert m["holdout_cost"] == s.cost and m["holdout_coverage"] == 0.5
    # the cost prefers the forecast with headroom; RMSE does not
    low, high = np.array([[0.9, 0.9]]), np.array([[1.1, 1.1]])
    assert score(np.ones((1, 2)), high).cost < score(np.ones((1, 2)), low).cost
    assert score(np.ones((1, 2)), high).rmse == pytest.approx(score(np.ones((1, 2)), low).rmse)


def test_quantile_candidate_forecasts_with_headroom(series):
    """The q75 model estimates the 75th percentile of the change: on the same
    data it covers more of the actual values than the mean model, at a higher
    RMSE, and the asymmetric cost is what tells the two apart."""
    _, values = series
    results, _ = train_and_select(values, 10, 3, 300, 0.2, 0.5,
                                  [BY_NAME["hgb_delta"], BY_NAME["hgb_q75_delta"]])
    by = {r.candidate.name: r for r in results}
    mean, q75 = by["hgb_delta"], by["hgb_q75_delta"]
    assert q75.eligible and q75.bundle.meta.features == "anchored"
    assert q75.score.coverage > mean.score.coverage + 0.1
    assert 0.6 < q75.score.coverage < 0.9                 # near the 0.75 it estimates
    assert q75.score.rmse > mean.score.rmse               # headroom costs accuracy
    assert q75.score.under_mae < mean.score.under_mae     # and buys fewer shortfalls


def test_seasonal_naive_baseline_is_the_same_time_a_day_ago(series):
    _, values = series
    split = int(len(values) * 0.8)
    s = seasonal_naive_score(values, split, 10, 3, 288)
    naive = naive_score(values, split, 10, 3)
    assert s.n == naive.n                                  # scored on the same windows
    # the synthetic bump drifts slowly, so here it is only on a par with persistence;
    # on real daily traffic (docs/traces.md) it is the harder baseline
    assert s.rmse == pytest.approx(naive.rmse, rel=0.1)
    # by hand for one window: the forecast for index i is values[i - 288]
    from forecast_training.features import make_windows
    X, Y, first = make_windows(values, 10, 3, 288)
    keep = first >= split
    expected = np.stack([values[first[keep] + k - 288] for k in range(3)], axis=1)
    assert score(Y[keep], expected).rmse == pytest.approx(s.rmse)
    with pytest.raises(ValueError, match="season_steps"):
        seasonal_naive_score(values, split, 10, 3, 2)


def test_seasonal_candidate_reads_yesterday_off_the_input(series):
    """With the values from a day ago in the input, the 30-minute forecast on a
    daily pattern beats the same model on the window alone, and the model keeps
    following levels outside the training range."""
    ts, values = series
    results, split = train_and_select(values, 10, 6, 300, 0.2, 0.5,
                                      [BY_NAME["hgb_delta"], BY_NAME["hgb_delta_season"]], ts)
    by = {r.candidate.name: r for r in results}
    assert by["hgb_delta_season"].score.cost < by["hgb_delta"].score.cost
    assert by["hgb_delta_season"].score.rmse < by["hgb_delta"].score.rmse
    assert by["hgb_delta_season"].eligible and by["hgb_delta_season"].tracking > 0.9
    meta = by["hgb_delta_season"].bundle.meta
    assert meta.season_steps == 288 and meta.input_length == 298
    # too little history for a season: the candidate is skipped, not crashed
    short, _ = train_and_select(values[:300], 10, 6, 300, 0.2, 0.5,
                                [BY_NAME["hgb_delta"], BY_NAME["hgb_delta_season"]])
    assert [r.candidate.name for r in short] == ["hgb_delta"]
    # a compressed day (the benchmark): the season is period / step
    b = fit_bundle(values[:1000], BY_NAME["rf_delta_season"], 10, 6, 30, period_seconds=2400)
    assert b.meta.season_steps == 80 and b.meta.input_length == 90


def test_time_of_day_candidate_anticipates_the_daily_ramp(series):
    """With the phase of the day as a feature, the 30-minute-ahead forecast on a
    daily pattern beats the same model without it: it knows the ramp is coming."""
    from forecast_training.model import CANDIDATES, train_and_select

    ts, values = series
    by_name = {c.name: c for c in CANDIDATES}
    results, split = train_and_select(values, 10, 6, 300, 0.2, 0.5,
                                      [by_name["hgb_delta"], by_name["hgb_delta_tod"]], ts)
    rmse = {r.candidate.name: r.score.rmse for r in results}
    assert rmse["hgb_delta_tod"] < rmse["hgb_delta"]
    assert all(r.eligible for r in results)            # still follows new levels
    tod = next(r for r in results if r.candidate.name == "hgb_delta_tod")
    assert tod.bundle.meta.features == "anchored_tod" and tod.bundle.meta.period_seconds == 86400
    # timed candidates need timestamps; without them they are skipped, not crashed
    results2, _ = train_and_select(values, 10, 6, 300, 0.2, 0.5,
                                   [by_name["hgb_delta"], by_name["hgb_delta_tod"]])
    assert [r.candidate.name for r in results2] == ["hgb_delta"]
