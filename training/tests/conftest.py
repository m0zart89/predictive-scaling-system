import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STEP = 300
DAY = 86400 // STEP


def daily_series(days: float = 10, noise: float = 0.05, seed: int = 0,
                 t0: int = 1_700_000_000) -> tuple:
    """Synthetic CPU demand: business-hours bump on a low base, plus noise."""
    rng = np.random.default_rng(seed)
    n = int(days * DAY)
    ts = t0 + np.arange(n, dtype=np.int64) * STEP
    hour = (ts % 86400) / 3600
    level = 0.2 + 0.8 * np.exp(-((hour - 13) ** 2) / 8)
    values = np.clip(level + rng.normal(0, noise, n), 0.01, None)
    return ts, values


@pytest.fixture
def series():
    return daily_series()


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    """A throwaway MLflow tracking server + registry backed by SQLite."""
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("cpu-forecasting", artifact_location=str(tmp_path / "artifacts"))
    yield uri
    mlflow.set_tracking_uri(None)
