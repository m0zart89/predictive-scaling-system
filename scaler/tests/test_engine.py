import pytest

from fakes import FakeForecast, FakePrometheus, make_config
from predictive_scaler.clients import ForecastError, PrometheusError
from predictive_scaler.engine import NoDataError, PredictiveEngine

NS, NAME = "demo", "cpu-burner"
WINDOW = [0.2] * 9 + [0.4]          # ten samples, last observed = 0.4


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def engine_with(metrics, prom, fc, clock=None):
    return PredictiveEngine(metrics, prometheus=prom, forecast=fc, clock=clock or Clock())


def labels(**extra):
    return {"namespace": NS, "scaledobject": NAME, **extra}


# ── decision policy ────────────────────────────────────────────────────────────

def test_max_mode_scales_ahead_when_forecast_is_higher(metrics, registry):
    eng = engine_with(metrics, FakePrometheus(WINDOW), FakeForecast(lambda v: 0.9))
    d = eng.decide(NS, NAME, make_config())
    assert d.source == "forecast"
    assert d.predicted == pytest.approx(0.9)
    assert d.value == pytest.approx(0.9)
    # targetValue 0.5 → ceil(0.9 / 0.5) = 2 replicas
    assert registry.get_sample_value("predictive_scaler_desired_replicas", labels()) == 2


def test_max_mode_never_reports_below_observed(metrics):
    eng = engine_with(metrics, FakePrometheus(WINDOW), FakeForecast(lambda v: 0.1))
    d = eng.decide(NS, NAME, make_config())
    assert d.predicted == pytest.approx(0.1)
    assert d.value == pytest.approx(0.4)        # observed wins


def test_forecast_mode_trusts_the_forecast(metrics):
    eng = engine_with(metrics, FakePrometheus(WINDOW), FakeForecast(lambda v: 0.1))
    d = eng.decide(NS, NAME, make_config(mode="forecast"))
    assert d.value == pytest.approx(0.1)


def test_capacity_normalises_in_and_out_of_model_space(metrics):
    window = [2.0] * 9 + [3.0]                 # demand in cores
    fc = FakeForecast(lambda v: 0.5)           # model answers in [0, 1]
    eng = engine_with(metrics, FakePrometheus(window), fc)
    d = eng.decide(NS, NAME, make_config(capacity="4", targetValue="1"))
    assert fc.calls[0] == pytest.approx([0.5] * 9 + [0.75])
    assert d.predicted == pytest.approx(2.0)   # 0.5 * 4 cores
    assert d.value == pytest.approx(3.0)       # observed 3 cores is higher


def test_negative_forecast_is_clamped(metrics):
    eng = engine_with(metrics, FakePrometheus(WINDOW), FakeForecast(lambda v: -0.3))
    d = eng.decide(NS, NAME, make_config(mode="forecast"))
    assert d.predicted == 0.0
    assert d.value == 0.0


def test_window_size_and_step_are_passed_to_prometheus(metrics):
    prom = FakePrometheus(WINDOW)
    eng = engine_with(metrics, prom, FakeForecast())
    eng.decide(NS, NAME, make_config(timeSteps="10", stepSeconds="30", timeoutSeconds="2"))
    assert prom.calls[0]["points"] == 10
    assert prom.calls[0]["step_seconds"] == 30
    assert prom.calls[0]["timeout"] == 2


# ── degraded modes ─────────────────────────────────────────────────────────────

def test_short_history_scales_on_observed_without_calling_forecast(metrics, registry):
    fc = FakeForecast()
    eng = engine_with(metrics, FakePrometheus([0.1, 0.3, 0.7]), fc)
    d = eng.decide(NS, NAME, make_config())
    assert d.source == "observed"
    assert d.value == pytest.approx(0.7)
    assert d.predicted is None
    assert fc.calls == []
    assert registry.get_sample_value("predictive_scaler_decisions_total",
                                     labels(source="observed")) == 1


@pytest.mark.parametrize("reason", ["unavailable", "not_ready", "bad_response"])
def test_forecast_failure_falls_back_to_observed(metrics, registry, reason):
    fc = FakeForecast(exc=ForecastError("boom", reason))
    eng = engine_with(metrics, FakePrometheus(WINDOW), fc)
    d = eng.decide(NS, NAME, make_config())
    assert d.source == "observed"
    assert d.value == pytest.approx(0.4)
    assert registry.get_sample_value("predictive_scaler_forecast_failures_total",
                                     labels(reason=reason)) == 1


def test_predicted_gauge_is_removed_during_fallback(metrics, registry):
    clock = Clock()
    fc = FakeForecast(lambda v: 0.9)
    eng = engine_with(metrics, FakePrometheus(WINDOW), fc, clock)
    cfg = make_config(cacheSeconds="0")
    eng.decide(NS, NAME, cfg)
    assert registry.get_sample_value("predictive_scaler_predicted_demand", labels()) == 0.9

    fc.exc = ForecastError("down")
    eng.decide(NS, NAME, cfg)
    assert registry.get_sample_value("predictive_scaler_predicted_demand", labels()) is None


def test_prometheus_failure_propagates_and_is_counted(metrics, registry):
    eng = engine_with(metrics, FakePrometheus(exc=PrometheusError("down")), FakeForecast())
    with pytest.raises(PrometheusError):
        eng.decide(NS, NAME, make_config())
    assert registry.get_sample_value("predictive_scaler_prometheus_failures_total", labels()) == 1


def test_no_samples_is_an_error_not_zero(metrics):
    eng = engine_with(metrics, FakePrometheus([]), FakeForecast())
    with pytest.raises(NoDataError):
        eng.decide(NS, NAME, make_config())


# ── caching ────────────────────────────────────────────────────────────────────

def test_decisions_are_cached_until_ttl_expires(metrics):
    clock = Clock()
    prom, fc = FakePrometheus(WINDOW), FakeForecast()
    eng = engine_with(metrics, prom, fc, clock)
    cfg = make_config(cacheSeconds="15")

    first = eng.decide(NS, NAME, cfg)
    clock.t += 14
    assert eng.decide(NS, NAME, cfg) is first
    assert len(prom.calls) == 1

    clock.t += 2
    eng.decide(NS, NAME, cfg)
    assert len(prom.calls) == 2


def test_cache_is_per_scaledobject_and_per_config(metrics):
    prom = FakePrometheus(WINDOW)
    eng = engine_with(metrics, prom, FakeForecast())
    eng.decide(NS, NAME, make_config())
    eng.decide(NS, "other", make_config())
    eng.decide(NS, NAME, make_config(targetValue="0.7"))
    assert len(prom.calls) == 3


def test_cache_can_be_disabled(metrics):
    prom = FakePrometheus(WINDOW)
    eng = engine_with(metrics, prom, FakeForecast())
    cfg = make_config(cacheSeconds="0")
    eng.decide(NS, NAME, cfg)
    eng.decide(NS, NAME, cfg)
    assert len(prom.calls) == 2


def test_errors_are_not_cached(metrics):
    prom = FakePrometheus(exc=PrometheusError("down"))
    eng = engine_with(metrics, prom, FakeForecast())
    cfg = make_config()
    for _ in range(2):
        with pytest.raises(PrometheusError):
            eng.decide(NS, NAME, cfg)
    assert len(prom.calls) == 2


# ── multi-step horizon ─────────────────────────────────────────────────────────

def test_next_aggregation_uses_first_step(metrics):
    eng = engine_with(metrics, FakePrometheus(WINDOW), FakeForecast(lambda v: [0.5, 0.9, 0.7]))
    d = eng.decide(NS, NAME, make_config(mode="forecast"))
    assert d.predicted == pytest.approx(0.5)


def test_max_aggregation_scales_for_the_peak_of_the_horizon(metrics):
    eng = engine_with(metrics, FakePrometheus(WINDOW), FakeForecast(lambda v: [0.5, 0.9, 0.7]))
    d = eng.decide(NS, NAME, make_config(mode="forecast", forecastAggregation="max"))
    assert d.predicted == pytest.approx(0.9)
    assert "max of 3 step(s)" in d.reason


def test_forecast_call_carries_series_key_and_window_time(metrics):
    fc = FakeForecast(lambda v: 0.9)
    prom = FakePrometheus(WINDOW)
    eng = engine_with(metrics, prom, fc)
    eng.decide(NS, NAME, make_config())
    assert fc.kwargs["series"] == f"{NS}/{NAME}"
    # the last sample's own time, so serving can match the actual later
    assert fc.kwargs["timestamp"] == pytest.approx(FakePrometheus.NOW, abs=1e-6)
    assert fc.kwargs["step_seconds"] == 300


# ── history for seasonal models ────────────────────────────────────────────────

def test_long_history_is_sent_whole_with_holes_as_none(metrics):
    """historySteps > timeSteps: the forecast call carries the whole history,
    scaled to model units, with None where Prometheus had no sample; the
    decision itself still rests on the trailing window."""
    history = [None, 0.6, None, 0.8] + [0.2] * 9 + [0.4]      # 14 samples, 10 contiguous at the end
    fc = FakeForecast(lambda v: 0.9)
    prom = FakePrometheus(history)
    eng = engine_with(metrics, prom, fc)
    d = eng.decide(NS, NAME, make_config(timeSteps="10", historySteps="14", capacity="2"))
    assert prom.calls[0]["points"] == 14
    assert fc.calls[0] == [None, 0.3, None, 0.4] + [0.1] * 9 + [0.2]
    assert d.source == "forecast" and d.observed == pytest.approx(0.4)
    assert fc.kwargs["timestamp"] == pytest.approx(FakePrometheus.NOW)
    # history shorter than asked (a young cluster): the older part is simply absent
    prom = FakePrometheus([0.2] * 9 + [0.4])
    fc = FakeForecast(lambda v: 0.9)
    engine_with(metrics, prom, fc).decide(NS, NAME, make_config(historySteps="298"))
    assert len(fc.calls[0]) == 10 and prom.calls[0]["points"] == 298


def test_a_hole_inside_the_window_still_means_observed_demand(metrics):
    history = [0.2] * 20 + [None] + [0.3] * 5                 # only 5 contiguous samples at the end
    fc = FakeForecast(lambda v: 0.9)
    d = engine_with(metrics, FakePrometheus(history), fc).decide(
        NS, NAME, make_config(timeSteps="10", historySteps="26"))
    assert d.source == "observed" and d.value == pytest.approx(0.3)
    assert "5/10 samples" in d.reason and fc.calls == []


def test_default_history_equals_the_window_and_the_call_is_unchanged(metrics):
    fc = FakeForecast(lambda v: 0.9)
    engine_with(metrics, FakePrometheus([0.5] * 12), fc).decide(NS, NAME, make_config())
    assert fc.calls[0] == [0.5] * 10                          # exactly timeSteps values, no nulls


# ── forecast quality scored by the scaler ──────────────────────────────────────

def test_scaler_scores_its_own_forecasts_from_the_next_history(metrics, registry):
    """A forecast made at t for t+1..t+3 is scored once the history fetched at a
    later poll holds those instants: exact matching, no serving state, and it
    works whichever replica polled in between."""
    fc = FakeForecast(lambda v: [0.5, 0.6, 0.7])              # model units; capacity 2 → 1.0, 1.2, 1.4
    prom = FakePrometheus([0.2] * 10)
    eng = engine_with(metrics, prom, fc)
    cfg = make_config(capacity="2", stepSeconds="30", cacheSeconds="0")
    eng.decide(NS, NAME, cfg)
    assert registry.get_sample_value("predictive_scaler_forecast_demand", labels(step="2")) == pytest.approx(1.2)
    assert eng.ledger.pending() == 1

    # two steps later: the history now holds the actuals for steps 1 and 2
    prom.values = [0.2] * 10 + [1.1, 1.3]                       # query units
    prom.NOW = FakePrometheus.NOW + 60
    eng.decide(NS, NAME, cfg)
    def count(step):
        return registry.get_sample_value("predictive_scaler_forecast_abs_error_count", labels(step=step))

    def total(step):
        return registry.get_sample_value("predictive_scaler_forecast_abs_error_sum", labels(step=step))
    assert count("1") == 1 and total("1") == pytest.approx(abs(1.1 / 2 - 0.5))
    assert count("2") == 1 and total("2") == pytest.approx(abs(1.3 / 2 - 0.6))
    assert count("3") is None                                   # not yet
    assert registry.get_sample_value("predictive_scaler_forecast_residual", labels()) == pytest.approx(1.1 / 2 - 0.5)

    # the same poll again (cache off, same newest sample): nothing is scored twice
    eng.decide(NS, NAME, cfg)
    assert count("1") == 1 and count("2") == 1

    # one more step: step 3 resolves and the forecast leaves the ledger
    prom.values = [0.2] * 10 + [1.1, 1.3, 1.5]
    prom.NOW = FakePrometheus.NOW + 90
    eng.decide(NS, NAME, cfg)
    assert count("3") == 1 and total("3") == pytest.approx(abs(1.5 / 2 - 0.7))
    assert eng.ledger.pending() == 2                             # the two forecasts made at +60 and +90


def test_fallback_drops_the_per_step_forecast_series(metrics, registry):
    prom = FakePrometheus(WINDOW)
    fc = FakeForecast(lambda v: [0.9, 0.8])
    eng = engine_with(metrics, prom, fc)
    cfg = make_config(cacheSeconds="0")
    eng.decide(NS, NAME, cfg)
    assert registry.get_sample_value("predictive_scaler_forecast_demand", labels(step="2")) == pytest.approx(0.8)
    fc.exc = ForecastError("down")
    eng.decide(NS, NAME, cfg)
    assert registry.get_sample_value("predictive_scaler_forecast_demand", labels(step="1")) is None
    assert registry.get_sample_value("predictive_scaler_forecast_demand", labels(step="2")) is None


# ── circuit breaker and surge cap ──────────────────────────────────────────────

def test_breaker_opens_on_a_jump_and_the_decision_turns_reactive(metrics, registry):
    """A one-step jump the history never contained: the forecast is still made
    and recorded, but the reported value is observed demand until the breaker
    closes; metrics say so and retraining can exclude the interval."""
    fc = FakeForecast(lambda v: 2.0)                 # model units; capacity 1
    history = [0.2, 0.25, 0.2, 0.3, 0.25, 0.2, 0.3, 0.25, 0.2, 0.3]
    prom = FakePrometheus(history)
    eng = engine_with(metrics, prom, fc)
    cfg = make_config(cacheSeconds="0", breakerJumpMinSamples="5")
    d = eng.decide(NS, NAME, cfg)
    assert d.source == "forecast" and d.value == pytest.approx(0.9)    # 2.0 capped at 3 x 0.3
    assert registry.get_sample_value("predictive_scaler_breaker_open", labels()) == 0

    prom.values = history + [1.5]                    # +1.2 against a largest rise of 0.1
    prom.NOW += 300
    d = eng.decide(NS, NAME, cfg)
    assert d.source == "observed" and d.value == pytest.approx(1.5)
    assert d.predicted is not None                   # the forecast was made, not used
    assert "breaker" in d.reason and "jump" in d.reason
    assert registry.get_sample_value("predictive_scaler_breaker_open", labels()) == 1
    assert registry.get_sample_value("predictive_scaler_breaker_trips_total", labels(reason="jump")) == 1
    assert registry.get_sample_value("predictive_scaler_breaker_bypass_total", labels()) == 1
    assert registry.get_sample_value("predictive_scaler_decisions_total", labels(source="observed")) == 1

    # still open on the next poll (hold time), reactive again
    prom.values = history + [1.5, 1.5]
    prom.NOW += 300
    d = eng.decide(NS, NAME, cfg)
    assert d.source == "observed" and "open" in d.reason


def test_breaker_off_keeps_the_forecast_and_surge_cap_can_be_disabled(metrics):
    fc = FakeForecast(lambda v: 2.0)
    history = [0.2] * 9 + [0.3]
    prom = FakePrometheus(history + [1.5])
    eng = engine_with(metrics, prom, fc)
    d = eng.decide(NS, NAME, make_config(breaker="off", forecastSurgeFactor="0"))
    assert d.source == "forecast" and d.value == pytest.approx(2.0)   # uncapped, unbroken


def test_surge_cap_limits_how_far_the_forecast_may_exceed_observed(metrics):
    fc = FakeForecast(lambda v: 5.0)
    eng = engine_with(metrics, FakePrometheus(WINDOW), fc)
    d = eng.decide(NS, NAME, make_config(forecastSurgeFactor="2"))
    assert d.predicted == pytest.approx(0.8) and "capped at 2x" in d.reason
    d = eng.decide(NS, NAME, make_config(forecastSurgeFactor="2", mode="forecast"))
    assert d.predicted == pytest.approx(5.0)        # the cap is a mode=max guard only


# ── multivariate channels ──────────────────────────────────────────────────────

def test_extra_channels_are_fetched_on_the_targets_grid_and_sent_by_name(metrics):
    import json

    fc = FakeForecast(lambda v: 0.9)
    rps = [100.0] * 8 + [None, 150.0]                     # a hole, then the newest sample
    prom = FakePrometheus(WINDOW, by_query={"sum(rate(http_requests_total[1m]))": rps})
    eng = engine_with(metrics, prom, fc)
    cfg = make_config(capacity="2", channels=json.dumps({"rps": "sum(rate(http_requests_total[1m]))"}),
                      targetChannel="cpu")
    d = eng.decide(NS, NAME, cfg)
    assert d.source == "forecast"
    sent = fc.calls[0]
    assert set(sent) == {"cpu", "rps"}
    assert sent["cpu"] == [v / 2 for v in WINDOW]           # the target, scaled by capacity
    assert sent["rps"] == rps                               # raw units, hole kept (serving fills it)
    assert prom.calls[1]["query"] == "sum(rate(http_requests_total[1m]))"
    assert prom.calls[1]["now"] == pytest.approx(FakePrometheus.NOW)   # up to the target's newest sample


def test_a_lagging_channel_is_carried_forward_and_a_failing_one_is_an_error(metrics, registry):
    import json

    class Lagging(FakePrometheus):
        def fetch_history(self, address, query, points, step_seconds, timeout, now=None):
            ts, vals = super().fetch_history(address, query, points, step_seconds, timeout, now)
            if query == "rps":                              # newest sample one step old
                return ts[:-1], vals[:-1]
            return ts, vals

    fc = FakeForecast(lambda v: 0.9)
    prom = Lagging(WINDOW, by_query={"rps": [1.0] * 8 + [7.0, 9.0]})   # 9.0 not scraped yet
    eng = engine_with(metrics, prom, fc)
    eng.decide(NS, NAME, make_config(channels=json.dumps({"rps": "rps"})))
    assert fc.calls[0]["rps"] == [1.0] * 8 + [7.0, 7.0]      # on the target grid, last value carried

    prom = FakePrometheus(WINDOW, by_query={"rps": []})
    prom.exc = None
    eng = engine_with(metrics, prom, FakeForecast(lambda v: 0.9))

    def failing(address, query, points, step_seconds, timeout, now=None):
        if query == "rps":
            raise PrometheusError("rps query failed")
        return FakePrometheus.fetch_history(prom, address, query, points, step_seconds, timeout, now)

    prom.fetch_history = failing
    with pytest.raises(PrometheusError):
        eng.decide(NS, NAME, make_config(channels=json.dumps({"rps": "rps"})))
    assert registry.get_sample_value("predictive_scaler_prometheus_failures_total", labels()) == 1
