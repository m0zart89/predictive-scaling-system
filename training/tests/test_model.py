import numpy as np
import pytest

from forecast_training.features import (LEGACY_META, ForecastMeta, build_features, from_target,
                                        make_windows, to_target)
from forecast_training.model import (CANDIDATES, fit_bundle, level_tracking, naive_score, score,
                                     train_and_select)

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
    _, values = series
    b = fit_bundle(values[:1500], BY_NAME[name], 10, horizon, 300)
    out = b.predict(values[1500:1510])
    assert out.shape == (1, horizon)
    assert np.all(np.isfinite(out))


def test_models_beat_persistence_on_a_daily_pattern(series):
    _, values = series
    results, split = train_and_select(values, 10, 3, 300, 0.2)
    naive = naive_score(values, split, 10, 3)
    assert results[0].score.rmse < naive.rmse


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
    # eligible results come first, each group sorted by RMSE
    flags = [r.eligible for r in results]
    assert flags == sorted(flags, reverse=True)


def test_nothing_eligible_is_reported(series):
    _, values = series
    results, _ = train_and_select(values, 10, 3, 300, 0.2, min_tracking=5.0)
    assert not results[0].eligible


def test_score_limits_steps():
    y = np.array([[1.0, 1.0, 1.0]])
    p = np.array([[1.0, 2.0, 4.0]])
    assert score(y, p).rmse_by_step == [0.0, 1.0, 3.0]
    assert score(y, p, steps=2).rmse == pytest.approx(0.5)
