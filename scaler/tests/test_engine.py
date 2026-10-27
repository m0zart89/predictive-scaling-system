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
