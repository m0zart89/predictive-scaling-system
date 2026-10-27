"""Test doubles and helpers shared by the scaler tests."""

from predictive_scaler.config import parse_metadata

BASE_META = {
    "prometheusServerAddress": "http://prometheus:9090",
    "query": "sum(rate(container_cpu_usage_seconds_total[1m]))",
    "forecastServiceURL": "http://forecast-serving:8080",
    "targetValue": "0.5",
}


class FakePrometheus:
    """Returns a fixed window (or raises) and records every call."""

    def __init__(self, values=None, exc=None):
        self.values = list(values or [])
        self.exc = exc
        self.calls = []

    def fetch_window(self, address, query, points, step_seconds, timeout, now=None):
        self.calls.append(dict(address=address, query=query, points=points,
                               step_seconds=step_seconds, timeout=timeout, now=now))
        if self.exc:
            raise self.exc
        return self.values[-points:]


class FakeForecast:
    """Answers with fn(values) (default: last value) or raises. fn may return a
    float (single-step model) or a list (multi-step horizon)."""

    def __init__(self, fn=None, exc=None):
        self.fn = fn or (lambda values: values[-1])
        self.exc = exc
        self.calls = []

    def predict(self, base_url, values, timeout, series=None, timestamp=None, step_seconds=None):
        self.calls.append(list(values))
        self.kwargs = dict(series=series, timestamp=timestamp, step_seconds=step_seconds)
        if self.exc:
            raise self.exc
        out = self.fn(values)
        return list(out) if isinstance(out, (list, tuple)) else [out]


def make_config(**overrides):
    meta = dict(BASE_META)
    meta.update({k: str(v) for k, v in overrides.items()})
    return parse_metadata(meta)
