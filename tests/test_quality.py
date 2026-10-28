"""
Forecast quality tracking (serving/quality.py): past forecasts are scored
against the actual values that later windows carry, per horizon step, in
bounded memory.
"""

import pytest
from prometheus_client import CollectorRegistry

from quality import ForecastTracker

STEP = 300
N = 10


def window(series_values, end_index):
    """The last N values of `series_values` ending at index `end_index`."""
    return [float(v) for v in series_values[end_index - N + 1:end_index + 1]]


@pytest.fixture
def registry():
    return CollectorRegistry()


@pytest.fixture
def tracker(registry):
    return ForecastTracker(registry, mae_window=4, max_series=2, max_pending=3)


def mae(registry, step):
    return registry.get_sample_value("forecast_rolling_mae", {"horizon_step": str(step)})


def hist_count(registry, step):
    return registry.get_sample_value("forecast_abs_error_count", {"horizon_step": str(step)})


def test_each_horizon_step_is_scored_when_its_actual_arrives(tracker, registry):
    series = list(range(100))                      # value == time index
    t0 = 1_000_000
    # forecast made at index 9 for steps 1..3: predicts 10, 12, 11 (actual 10, 11, 12)
    tracker.record("s", window(series, 9), [10.0, 12.0, 11.0], STEP, timestamp=t0)
    assert mae(registry, 1) is None

    assert tracker.record("s", window(series, 10), [0.0], STEP, timestamp=t0 + STEP) == 1
    assert mae(registry, 1) == 0.0
    assert mae(registry, 2) is None

    # two steps later the window holds the actuals for steps 2 and 3 at once
    # (and for the previous call's own one-step forecast)
    assert tracker.record("s", window(series, 12), [0.0], STEP, timestamp=t0 + 3 * STEP) == 3
    assert mae(registry, 2) == 1.0 and mae(registry, 3) == 1.0
    assert hist_count(registry, 2) == 1
    assert registry.get_sample_value("forecast_abs_error_sum", {"horizon_step": "3"}) == 1.0


def test_a_forecast_step_is_scored_only_once(tracker, registry):
    series = list(range(100))
    t0 = 1_000_000
    tracker.record("s", window(series, 9), [12.0], STEP, timestamp=t0)
    # windows at +1, +2 and +3 steps all contain the actual for step 1
    for k in (1, 2, 3):
        tracker.record("s", window(series, 9 + k), [0.0], STEP, timestamp=t0 + k * STEP)
    # the one forecast (error 2), plus the +1 and +2 calls' own forecasts of 0
    # (errors 11 and 12); the +3 call's forecast is still pending
    assert hist_count(registry, 1) == 3
    assert mae(registry, 1) == pytest.approx((2 + 11 + 12) / 3)


def test_rolling_mae_is_over_the_last_n_errors(registry):
    tracker = ForecastTracker(registry, mae_window=2)
    series = [0.0] * 100
    t = 1_000_000
    for err in (4.0, 1.0, 3.0):          # forecasts off by exactly err, each closed by an
        tracker.record("s", window(series, 9), [err], STEP, timestamp=t)   # exact (0) forecast
        t += STEP
        tracker.record("s", window(series, 10), [0.0], STEP, timestamp=t)
        t += STEP
    # errors seen: 4, 0, 1, 0, 3 — the gauge covers the last two
    assert mae(registry, 1) == 1.5
    assert hist_count(registry, 1) == 5  # the histogram keeps everything


def test_windows_off_the_grid_are_not_matched(tracker, registry):
    series = list(range(100))
    t0 = 1_000_000
    tracker.record("s", window(series, 9), [10.0], STEP, timestamp=t0)
    # half a step later: neither the old nor the new sample is the step-1 actual
    assert tracker.record("s", window(series, 10), [0.0], STEP, timestamp=t0 + STEP * 0.5) == 0
    # within tolerance (25% of a step) it is
    assert tracker.record("s", window(series, 10), [0.0], STEP, timestamp=t0 + STEP * 1.2) == 1


def test_series_are_kept_apart(tracker, registry):
    t0 = 1_000_000
    tracker.record("a", [1.0] * N, [1.0], STEP, timestamp=t0)
    tracker.record("b", [5.0] * N, [1.0], STEP, timestamp=t0)
    assert tracker.record("a", [1.0] * N, [1.0], STEP, timestamp=t0 + STEP) == 1
    assert mae(registry, 1) == 0.0       # scored against a's window, not b's


def test_duplicate_calls_for_the_same_instant_record_one_forecast(tracker, registry):
    t0 = 1_000_000
    tracker.record("s", [1.0] * N, [3.0], STEP, timestamp=t0)
    tracker.record("s", [1.0] * N, [9.0], STEP, timestamp=t0 + 1)   # a retry one second later
    assert registry.get_sample_value("forecast_pending_forecasts") == 1
    tracker.record("s", [1.0] * N, [0.0], STEP, timestamp=t0 + STEP)
    assert hist_count(registry, 1) == 1 and mae(registry, 1) == 2.0


def test_memory_is_bounded(tracker, registry):
    t = 1_000_000
    # 12-step forecasts spaced 10.5 steps apart: never scored (off the grid),
    # never expired (their targets stay ahead of the window)
    for i in range(10):                  # max_pending=3 per series
        tracker.record("s", [0.0] * N, [1.0] * 12, STEP, timestamp=t + i * 10.5 * STEP)
    assert registry.get_sample_value("forecast_pending_forecasts") == 3
    assert hist_count(registry, 1) is None
    for name in ("a", "b"):              # max_series=2: "s" is evicted as least recently used
        tracker.record(name, [0.0] * N, [1.0], STEP, timestamp=t)
    assert set(tracker._pending) == {"a", "b"}


def test_unscorable_forecasts_expire(tracker, registry):
    t0 = 1_000_000
    tracker.record("s", [0.0] * N, [1.0] * 2, STEP, timestamp=t0)
    # much later, the window no longer reaches back to the forecast's targets
    tracker.record("s", [0.0] * N, [1.0], STEP, timestamp=t0 + 100 * STEP)
    assert registry.get_sample_value("forecast_pending_forecasts") == 1
    assert hist_count(registry, 1) is None


def test_arrival_time_is_used_when_no_timestamp_is_given(registry):
    clock = iter([1000.0, 1000.0 + STEP])
    tracker = ForecastTracker(registry, clock=lambda: next(clock))
    tracker.record("s", [0.0] * N, [0.5], STEP)
    assert tracker.record("s", [0.0] * N, [0.5], STEP) == 1
    assert mae(registry, 1) == 0.5


def test_snapshot_reports_current_mae(tracker):
    t0 = 1_000_000
    tracker.record("s", [0.0] * N, [0.25, 0.5], STEP, timestamp=t0)
    tracker.record("s", [0.0] * N, [0.0], STEP, timestamp=t0 + 2 * STEP)
    assert tracker.snapshot() == {1: 0.25, 2: 0.5}
