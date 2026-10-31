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
