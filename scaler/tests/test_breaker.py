"""Circuit breaker: signals, state machine, hold times."""

import pytest

from predictive_scaler.breaker import CLOSED, HALF_OPEN, OPEN, BreakerConfig, CircuitBreaker


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def breaker(**kw):
    clock = Clock()
    cfg = BreakerConfig(**{"residual_min": 4, "calm_polls": 2, "min_open_seconds": 60,
                           "jump_min_samples": 8, **kw})
    return CircuitBreaker(cfg, clock=clock), clock


FLAT = [1.0] * 20          # a history with no rises at all


def test_closed_by_default_and_disabled_never_trips():
    b, _ = breaker()
    assert b.observe("a", 1.0, FLAT, 0).use_forecast
    assert b.state("a") == CLOSED
    off, _ = breaker(enabled=False)
    v = off.observe("a", 100.0, FLAT, 0, residuals=[50.0])
    assert v.use_forecast and v.tripped is None


def test_jump_larger_than_anything_in_the_history_trips():
    b, clock = breaker()
    history = [1.0, 1.1, 1.0, 1.2, 1.1, 1.0] + [1.0] * 10          # largest rise 0.2
    assert b.observe("a", 1.0, history, 0).use_forecast              # first poll: nothing to compare
    v = b.observe("a", 1.6, history + [1.6], 0)                       # +0.6 > 2 x 0.2
    assert not v.use_forecast and v.tripped == "jump" and v.state == OPEN
    assert "jump" in v.reason
    # stays open for min_open_seconds even if the series is calm again
    clock.t += 30
    v = b.observe("a", 1.0, history + [1.6, 1.0], 0)
    assert not v.use_forecast and v.state == OPEN and "before half-open" in v.reason
    # then half-open: calm polls close it
    clock.t += 31
    v = b.observe("a", 1.0, history, 0)
    assert not v.use_forecast and v.state == HALF_OPEN
    v = b.observe("a", 1.0, history, 0)
    assert v.use_forecast and v.state == CLOSED


def test_level_far_from_one_season_ago_trips_after_consecutive_polls():
    b, _ = breaker(level_shift_polls=3)
    season = 5
    history = [1.0] * 10                                              # the value a season ago is 1.0
    for i in range(2):
        v = b.observe("a", 3.0, history, season)                       # x3 > 2.5, but only twice
        assert v.use_forecast, i
    v = b.observe("a", 3.0, history, season)
    assert not v.use_forecast and v.tripped == "level"
    # a drop far below yesterday counts too
    b2, _ = breaker(level_shift_polls=1)
    assert b2.observe("a", 0.2, history, season).tripped == "level"
    # unknown season or a hole a season ago: no level signal
    assert b2.observe("b", 3.0, history, 0).use_forecast
    holed = [1.0] * 4 + [None] + [1.0] * 5
    assert b2.observe("c", 3.0, holed, 5).use_forecast


def test_residual_shock_needs_a_baseline_then_two_polls_or_one_hard_hit():
    b, _ = breaker(residual_min=6, residual_z=4.0, residual_z_hard=8.0)
    noise = [0.01, -0.02, 0.015, -0.01, 0.005, -0.015]                # MAD ~ 0.013
    for r in noise:
        assert b.observe("a", 1.0, FLAT, 0, residuals=[r]).use_forecast
    v = b.observe("a", 1.0, FLAT, 0, residuals=[0.1])                 # z ~ 5: first hot poll
    assert v.use_forecast
    v = b.observe("a", 1.0, FLAT, 0, residuals=[0.1])                 # second: trips
    assert not v.use_forecast and v.tripped == "residual"
    b2, _ = breaker(residual_min=6)
    for r in noise:
        b2.observe("a", 1.0, FLAT, 0, residuals=[r])
    assert b2.observe("a", 1.0, FLAT, 0, residuals=[0.5]).tripped == "residual"   # z >> 8 at once


def test_retrip_while_half_open_doubles_the_hold_up_to_the_cap():
    b, clock = breaker(min_open_seconds=60, max_open_seconds=100)
    history = [1.0, 1.1] * 5                                          # largest rise 0.1
    b.observe("a", 1.0, history, 0)
    assert b.observe("a", 5.0, history + [5.0], 0).tripped == "jump"
    clock.t += 61
    assert b.observe("a", 1.0, history, 0).state == HALF_OPEN
    assert b.observe("a", 5.0, history + [5.0], 0).state == OPEN      # re-trip
    clock.t += 61                                                     # 60 s is no longer enough
    assert b.observe("a", 1.0, history, 0).state == OPEN
    clock.t += 40                                                     # 100 s (the cap) is
    assert b.observe("a", 1.0, history, 0).state == HALF_OPEN
    b.forget("a")
    assert b.state("a") == CLOSED


def test_config_rejects_bad_breaker_values():
    from fakes import make_config
    from predictive_scaler.config import ConfigError

    cfg = make_config(breaker="off", forecastSurgeFactor="0", breakerLevelShift="3",
                      breakerMinOpenSeconds="120", breakerCalmPolls="4")
    assert cfg.breaker is False and cfg.forecast_surge_factor == 0
    assert cfg.breaker_level_shift == 3 and cfg.breaker_min_open_seconds == 120 and cfg.breaker_calm_polls == 4
    assert make_config().breaker is True and make_config().forecast_surge_factor == 3.0
    with pytest.raises(ConfigError, match="breaker"):
        make_config(breaker="maybe")
    with pytest.raises(ConfigError, match="breakerLevelShift"):
        make_config(breakerLevelShift="1")


def test_jump_needs_enough_history_to_know_what_a_normal_rise_is():
    """A workload minutes old (the e2e, a fresh ScaledObject): every ramp would be
    a "jump" against a history that holds nothing yet. The signal waits for
    jump_min_samples known samples."""
    b, _ = breaker(jump_min_samples=20)
    short = [None] * 10 + [1.0, 1.1, 1.0, 1.05, 1.0]              # 5 known samples
    b.observe("a", 1.0, short, 0)
    assert b.observe("a", 5.0, short + [5.0], 0).use_forecast        # not enough history: no verdict
    long = [1.0, 1.1] * 10                                           # 20 known samples
    b.observe("b", 1.0, long, 0)
    assert b.observe("b", 5.0, long + [5.0], 0).tripped == "jump"
