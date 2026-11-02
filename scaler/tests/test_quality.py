"""ForecastLedger: scoring past forecasts against fetched history."""

from predictive_scaler.quality import ForecastLedger, rolling_mae


def grid(t0, n, step=30, values=None):
    ts = [t0 + i * step for i in range(n)]
    return ts, (values if values is not None else [1.0] * n)


def test_resolves_each_step_once_and_expires_the_forecast():
    led = ForecastLedger()
    led.record("a", 1000, 30, [0.5, 0.6, 0.7])
    ts, hist = grid(1000 - 30 * 9, 12, values=[0.1] * 10 + [0.55, 0.65])   # up to 1060
    out = led.score("a", ts, hist)
    assert [(r.step, r.actual) for r in out] == [(1, 0.55), (2, 0.65)]
    assert led.score("a", ts, hist) == []                                 # not twice
    ts, hist = grid(1000 - 30 * 9, 13, values=[0.1] * 10 + [0.55, 0.65, 0.75])
    out = led.score("a", ts, hist)
    assert [(r.step, r.actual) for r in out] == [(3, 0.75)]
    assert led.pending() == 0                                             # all steps in the past


def test_capacity_puts_actuals_in_model_units_and_holes_wait():
    led = ForecastLedger()
    led.record("a", 1000, 30, [0.5, 0.6])
    ts, hist = grid(1000 - 30 * 9, 12, values=[0.1] * 10 + [None, 1.3])    # step 1 missing
    out = led.score("a", ts, hist, capacity=2.0)
    assert [(r.step, r.forecast, r.actual) for r in out] == [(2, 0.6, 0.65)]
    assert led.pending() == 0            # the forecast is past its last step; the hole is never scored


def test_duplicate_forecast_for_the_same_instant_is_ignored_and_memory_is_bounded():
    led = ForecastLedger(max_series=2, max_pending=3)
    led.record("a", 1000, 30, [0.5])
    led.record("a", 1005, 30, [0.9])     # within a quarter step: the same window
    assert led.pending() == 1
    for i in range(5):
        led.record("a", 2000 + i * 30, 30, [0.5])
    assert led.pending() == 3            # deque maxlen
    led.record("b", 1000, 30, [0.5])
    led.record("c", 1000, 30, [0.5])
    assert set(led._pending) == {"b", "c"}   # least recently used series evicted
    assert ForecastLedger(max_series=0).pending() == 0


def test_rolling_mae_groups_by_step():
    led = ForecastLedger()
    led.record("a", 1000, 30, [1.0, 1.0])
    led.record("a", 1030, 30, [1.0, 1.0])
    ts, hist = grid(1000 - 30 * 9, 13, values=[0.1] * 10 + [1.5, 0.5, 1.25])
    mae = rolling_mae(led.score("a", ts, hist))
    assert mae[1] == (0.5, 2) and mae[2] == (0.375, 2)
