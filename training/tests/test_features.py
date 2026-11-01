"""Feature transforms shared with serving (forecast_training.features)."""

import numpy as np
import pytest

def test_time_features_follow_the_seasonal_phase():
    from forecast_training.features import FEATURE_KINDS, build_features, time_features

    assert "anchored_tod" in FEATURE_KINDS
    tf = time_features([0, 21600, 43200, 86400], 86400)
    np.testing.assert_allclose(tf[0], [0, 1], atol=1e-12)        # midnight
    np.testing.assert_allclose(tf[1], [1, 0], atol=1e-12)        # 06:00
    np.testing.assert_allclose(tf[2], [0, -1], atol=1e-12)       # noon
    np.testing.assert_allclose(tf[3], tf[0], atol=1e-12)         # next midnight
    # a compressed day: the same phases at 1/36 of the time
    np.testing.assert_allclose(time_features([600], 2400)[0], [1, 0], atol=1e-12)

    X = np.array([[0.1, 0.2, 0.4], [0.5, 0.5, 0.5]])
    f = build_features(X, "anchored_tod", last_ts=[0, 21600])
    assert f.shape == (2, 5)
    np.testing.assert_allclose(f[0], [0.1 - 0.4, 0.2 - 0.4, 0.4, 0, 1], atol=1e-12)
    with pytest.raises(ValueError, match="timestamp"):
        build_features(X, "anchored_tod")


def test_seasonal_features_hold_yesterdays_level_and_change():
    from forecast_training.features import (SEASONAL_FEATURE_KINDS, ForecastMeta, build_features,
                                            make_windows, seasonal_lags)

    assert SEASONAL_FEATURE_KINDS == ("anchored_season",)
    # 3-sample window, horizon 2, season of 4 steps: each row is 7 samples
    X, Y, first = make_windows(np.arange(20.0), time_steps=3, horizon=2, season_steps=4)
    assert X.shape == (12, 7) and Y.shape == (12, 2) and list(first[:2]) == [7, 8]
    assert list(X[0]) == [0, 1, 2, 3, 4, 5, 6] and list(Y[0]) == [7, 8]
    window, lag_now, lag_next = seasonal_lags(X[:1], 3, 2)
    assert list(window[0]) == [4, 5, 6]            # the last three samples
    assert lag_now[0, 0] == 2                      # one season (4) before "now" (6)
    assert list(lag_next[0]) == [3, 4]             # what followed then, over the horizon (7, 8 minus 4)
    f = build_features(X[:1], "anchored_season", time_steps=3, horizon=2)
    np.testing.assert_allclose(f[0], [4 - 6, 5 - 6, 6, 2 - 6, 3 - 2, 4 - 2])
    with pytest.raises(ValueError, match="time_steps"):
        build_features(X[:1], "anchored_season")
    with pytest.raises(ValueError, match="season"):
        seasonal_lags(X[:1, :4], 3, 2)             # no season before the window

    m = ForecastMeta(time_steps=3, horizon=2, features="anchored_season", season_steps=4)
    assert m.input_length == 7
    with pytest.raises(ValueError, match="season_steps"):
        ForecastMeta(time_steps=3, horizon=2, features="anchored_season", season_steps=1)
    with pytest.raises(ValueError, match="season_steps"):
        ForecastMeta(features="anchored", season_steps=4)
    assert ForecastMeta.from_dict({"time_steps": 10}).season_steps == 0   # older metas
