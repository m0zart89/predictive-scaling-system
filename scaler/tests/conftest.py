import os
import sys

import pytest
from prometheus_client import CollectorRegistry

HERE = os.path.dirname(__file__)
# `predictive_scaler` lives in scaler/, the shared fakes in scaler/tests/
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from predictive_scaler.metrics import ScalerMetrics  # noqa: E402


@pytest.fixture
def registry():
    return CollectorRegistry()


@pytest.fixture
def metrics(registry):
    return ScalerMetrics(registry)
