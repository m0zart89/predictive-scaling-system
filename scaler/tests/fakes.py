"""Test doubles and helpers shared by the scaler tests."""

from predictive_scaler.config import parse_metadata

BASE_META = {
    "prometheusServerAddress": "http://prometheus:9090",
    "query": "sum(rate(container_cpu_usage_seconds_total[1m]))",
    "forecastServiceURL": "http://forecast-serving:8080",
    "targetValue": "0.5",
}


class FakePrometheus:
    """Returns a fixed history (None = missing sample; or raises) and records
    every call. Like the real client, the grid ends at the newest sample."""

    def __init__(self, values=None, exc=None):
        self.values = list(values or [])
        self.exc = exc
        self.calls = []

    NOW = 1_700_000_000.0   # time of the last sample the fake returns

    def fetch_history(self, address, query, points, step_seconds, timeout, now=None):
        self.calls.append(dict(address=address, query=query, points=points,
                               step_seconds=step_seconds, timeout=timeout))
        if self.exc:
            raise self.exc
        values = self.values[-points:]
        timestamps = [self.NOW - (len(values) - 1 - i) * step_seconds for i in range(len(values))]
        return timestamps, values

    def fetch_series(self, *args, **kwargs):
        timestamps, values = self.fetch_history(*args, **kwargs)
        run = 0
        while run < len(values) and values[-1 - run] is not None:
            run += 1
        return timestamps[len(values) - run:], values[len(values) - run:]

    def fetch_window(self, *args, **kwargs):
        return self.fetch_series(*args, **kwargs)[1]


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
