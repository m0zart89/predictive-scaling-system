"""
Redis-backed forecast quality tracking (serving/quality.py, RedisMetricsEngine):
forecasts stored with a TTL under forecast:{series}:{timestamp}, scored once
across replicas into bounded metrics:mae:{version}:{step} lists, and a Redis
outage that costs nothing but the quality figures.
"""

import json

import fakeredis
import pytest
import redis
from prometheus_client import CollectorRegistry

from quality import RedisMetricsEngine

STEP = 300
N = 10
T0 = 1_700_000_000


def window(series_values, end_index):
    return [float(v) for v in series_values[end_index - N + 1:end_index + 1]]


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def client(server):
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def registry():
    return CollectorRegistry()


def engine(client, registry, **kw):
    kw.setdefault("rolling_window_size", 4)
    return RedisMetricsEngine(registry=registry, client=client, **kw)


def mae(registry, step):
    return registry.get_sample_value("forecast_rolling_mae", {"horizon_step": str(step)})


def hist_count(registry, step):
    return registry.get_sample_value("forecast_abs_error_count", {"horizon_step": str(step)})


def test_forecast_is_stored_with_a_ttl_under_the_documented_key(client, registry):
    eng = engine(client, registry, forecast_ttl_seconds=3600)
    eng.record("demo/api", [1.0] * N, [1.5, 1.7], STEP, timestamp=T0, model_version="7")
    keys = client.keys("forecast:demo/api:*")
    assert len(keys) == 1
    key = keys[0]
    assert key == eng.forecast_key("demo/api", T0, STEP)
    doc = json.loads(client.get(key))
    assert doc["predictions"] == [1.5, 1.7] and doc["model_version"] == "7"
    assert doc["made_at"] == T0 and doc["step_seconds"] == STEP
    assert doc["targets"] == [T0 + STEP, T0 + 2 * STEP]
    assert 0 < client.ttl(key) <= 3600


def test_each_horizon_step_is_scored_when_its_actual_arrives(client, registry):
    eng = engine(client, registry)
    series = list(range(100))                      # value == time index
    # forecast made at index 9 for steps 1..3: predicts 10, 12, 11 (actual 10, 11, 12)
    eng.record("s", window(series, 9), [10.0, 12.0, 11.0], STEP, timestamp=T0, model_version="1")
    assert mae(registry, 1) is None

    assert eng.record("s", window(series, 10), [0.0], STEP, timestamp=T0 + STEP, model_version="1") == 1
    assert mae(registry, 1) == 0.0
    assert mae(registry, 2) is None
    assert client.lrange("metrics:mae:1:1", 0, -1) == ["0.0"]

    # two steps later the window holds the actuals for steps 2 and 3 at once
    # (and for the previous call's own one-step forecast)
    assert eng.record("s", window(series, 12), [0.0], STEP, timestamp=T0 + 3 * STEP, model_version="1") == 3
    assert mae(registry, 2) == 1.0 and mae(registry, 3) == 1.0
    assert hist_count(registry, 2) == 1
    assert registry.get_sample_value("forecast_abs_error_sum", {"horizon_step": "3"}) == 1.0
    assert eng.get_rolling_mae("1", 2) == 1.0
    assert eng.snapshot("1", 3) == {1: pytest.approx((0 + 11) / 2), 2: 1.0, 3: 1.0}


def test_a_forecast_step_is_scored_only_once_even_from_another_replica(server, registry):
    """Two engines on two pods share the server: the second pod's window must
    not score the pair the first already claimed."""
    a = engine(fakeredis.FakeRedis(server=server, decode_responses=True), registry)
    b = engine(fakeredis.FakeRedis(server=server, decode_responses=True), CollectorRegistry())
    series = list(range(100))
    a.record("s", window(series, 9), [12.0], STEP, timestamp=T0, model_version="1")
    # windows at +1, +2 and +3 steps all contain the actual for step 1
    a.record("s", window(series, 10), [0.0], STEP, timestamp=T0 + STEP, model_version="1")
    b.record("s", window(series, 11), [0.0], STEP, timestamp=T0 + 2 * STEP, model_version="1")
    a.record("s", window(series, 12), [0.0], STEP, timestamp=T0 + 3 * STEP, model_version="1")
    # the one forecast (error 2) plus the +1 and +2 calls' own forecasts (11, 12)
    errors = [float(x) for x in a._r.lrange("metrics:mae:1:1", 0, -1)]
    assert sorted(errors) == [2.0, 11.0, 12.0]
    assert a._r.smembers(a.scored_key(a.forecast_key("s", T0, STEP))) == {"1"}


def test_rolling_window_is_trimmed_on_the_server(client, registry):
    eng = engine(client, registry, rolling_window_size=2)
    series = [0.0] * 100
    t = T0
    for err in (4.0, 1.0, 3.0):
        eng.record("s", window(series, 9), [err], STEP, timestamp=t, model_version="1")
        t += STEP
        eng.record("s", window(series, 10), [0.0], STEP, timestamp=t, model_version="1")
        t += STEP
    # errors seen: 4, 0, 1, 0, 3 — the list holds the last two, newest first
    assert client.lrange("metrics:mae:1:1", 0, -1) == ["3.0", "0.0"]
    assert mae(registry, 1) == 1.5
    assert hist_count(registry, 1) == 5           # the histogram keeps everything
    assert client.ttl("metrics:mae:1:1") > 0


def test_windows_off_the_grid_are_not_matched(client, registry):
    eng = engine(client, registry)
    series = list(range(100))
    eng.record("s", window(series, 9), [10.0], STEP, timestamp=T0, model_version="1")
    # half a step later: neither the old nor the new sample is the step-1 actual
    assert eng.record("s", window(series, 10), [0.0], STEP, timestamp=T0 + STEP * 0.5, model_version="1") == 0
    # within tolerance (a quarter step) it is
    assert eng.record("s", window(series, 10), [0.0], STEP, timestamp=T0 + STEP * 1.2, model_version="1") == 1


def test_series_are_kept_apart(client, registry):
    eng = engine(client, registry)
    eng.record("a", [1.0] * N, [1.0], STEP, timestamp=T0, model_version="1")
    eng.record("b", [5.0] * N, [1.0], STEP, timestamp=T0, model_version="1")
    assert eng.record("a", [1.0] * N, [1.0], STEP, timestamp=T0 + STEP, model_version="1") == 1
    assert mae(registry, 1) == 0.0                 # scored against a's window, not b's


def test_a_retry_for_the_same_instant_keeps_the_first_forecast(client, registry):
    eng = engine(client, registry)
    eng.record("s", [1.0] * N, [3.0], STEP, timestamp=T0, model_version="1")
    eng.record("s", [1.0] * N, [9.0], STEP, timestamp=T0 + 1, model_version="1")   # a retry
    assert len(client.keys("forecast:s:*")) == 1
    eng.record("s", [1.0] * N, [0.0], STEP, timestamp=T0 + STEP, model_version="1")
    assert hist_count(registry, 1) == 1 and mae(registry, 1) == 2.0


def test_errors_land_in_the_window_of_the_version_that_forecast(client, registry):
    """After a swap the old version's pending forecasts score into its own
    list; the gauge describes the version being served."""
    eng = engine(client, registry)
    eng.record("s", [0.0] * N, [1.0], STEP, timestamp=T0, model_version="1")
    eng.reset(pending_only=True)                   # swap to version 2
    eng.record("s", [0.0] * N, [0.5], STEP, timestamp=T0 + STEP, model_version="2")
    assert client.lrange("metrics:mae:1:1", 0, -1) == ["1.0"]
    assert client.lrange("metrics:mae:2:1", 0, -1) == []
    assert mae(registry, 1) is None                # v2 has no errors yet
    eng.record("s", [0.0] * N, [0.0], STEP, timestamp=T0 + 2 * STEP, model_version="2")
    assert mae(registry, 1) == 0.5
    assert eng.get_rolling_mae("1", 1) == 1.0 and eng.get_rolling_mae("2", 1) == 0.5
    assert eng.get_rolling_mae("3", 1) is None


def test_corrupt_entries_are_skipped(client, registry):
    eng = engine(client, registry)
    client.set(eng.forecast_key("s", T0, STEP), "{not json")
    client.rpush("metrics:mae:1:1", "abc", "0.5", "nan", "1.5")
    assert eng.record("s", [0.0] * N, [0.0], STEP, timestamp=T0 + STEP, model_version="1") == 0
    assert eng.get_rolling_mae("1", 1) == 1.0
    assert eng.store_forecast("s", float("nan"), [1.0], "1", STEP) is False
    assert eng.store_forecast("s", T0, [float("inf")], "1", STEP) is False


class FlakyRedis:
    """Raises until `fail_calls` have been made, then delegates to the fake."""

    def __init__(self, inner, fail_calls):
        self._inner, self.fail_calls, self.calls = inner, fail_calls, 0

    def register_script(self, script):
        return self._inner.register_script(script)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*a, **kw):
            self.calls += 1
            if self.calls <= self.fail_calls:
                raise redis.ConnectionError("connection refused")
            return attr(*a, **kw)
        return wrapped


def test_redis_outage_degrades_to_no_scoring_and_recovers(client, registry):
    clock = {"t": 1000.0}
    flaky = FlakyRedis(client, fail_calls=1)
    eng = engine(flaky, registry, cooldown_seconds=5.0, clock=lambda: clock["t"])
    # first call: Redis is down → 0, counted, backend_up=0, nothing raised
    assert eng.record("s", [0.0] * N, [1.0], STEP, timestamp=T0, model_version="1") == 0
    assert registry.get_sample_value("forecast_quality_backend_up") == 0
    assert registry.get_sample_value("forecast_quality_backend_errors_total") == 1
    # inside the cooldown no call reaches Redis at all
    calls = flaky.calls
    assert eng.record("s", [0.0] * N, [1.0], STEP, timestamp=T0 + 1, model_version="1") == 0
    assert eng.get_rolling_mae("1", 1) is None and eng.snapshot("1", 1) == {}
    assert flaky.calls == calls
    # after the cooldown the engine tries again and the backend answers
    clock["t"] += 6
    assert eng.record("s", [0.0] * N, [1.0], STEP, timestamp=T0 + 2, model_version="1") == 0
    assert registry.get_sample_value("forecast_quality_backend_up") == 1
    assert len(client.keys("forecast:s:*")) == 1


def test_rejects_ambiguous_tolerance_and_empty_windows(client, registry):
    with pytest.raises(ValueError):
        engine(client, registry, tolerance=0.5)
    with pytest.raises(ValueError):
        engine(client, registry, rolling_window_size=0)
    eng = engine(client, registry)
    assert eng.record("", [0.0] * N, [1.0], STEP, timestamp=T0) == 0
    assert eng.record("s", [0.0] * N, [], STEP, timestamp=T0) == 0
    assert eng.resolve_and_score("s", [], STEP, timestamp=T0) == 0
    assert eng.snapshot() == {}
