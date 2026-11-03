"""Metric conditioning primitives (predictive_scaler/smoothing.py)."""

import threading

import pytest

from predictive_scaler.smoothing import (CLOSED, HALF_OPEN, OPEN, EwmaSmoother, ServiceBreaker,
                                         ServiceBreakerConfig, spike_gate)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ── EWMA ───────────────────────────────────────────────────────────────────────

def test_ewma_follows_the_formula():
    sm = EwmaSmoother(clock=Clock())
    assert sm.update("k", 1.0, 0.3, 0).value == 1.0            # seeded with the first sample
    s = sm.update("k", 2.0, 0.3, 0)
    assert s.value == pytest.approx(0.3 * 2.0 + 0.7 * 1.0) and s.previous == 1.0 and not s.seeded
    assert s.damped == pytest.approx(2.0 - 1.3)
    s = sm.update("k", 2.0, 0.3, 0)
    assert s.value == pytest.approx(0.3 * 2.0 + 0.7 * 1.3)


def test_ewma_converges_to_a_sustained_level():
    sm = EwmaSmoother(clock=Clock())
    sm.update("k", 0.0, 0.3, 0)
    values = [sm.update("k", 1.0, 0.3, 0).value for _ in range(7)]
    assert values[-1] > 0.9 and all(b > a for a, b in zip(values, values[1:]))


def test_ewma_state_expires_and_is_per_key():
    clock = Clock()
    sm = EwmaSmoother(clock=clock)
    sm.update("a", 1.0, 0.5, 100)
    assert sm.update("b", 5.0, 0.5, 100).seeded                 # b: its own state
    clock.t = 50
    assert sm.update("a", 3.0, 0.5, 100).value == pytest.approx(2.0)
    clock.t = 200                                               # 150 s since the last update
    s = sm.update("a", 3.0, 0.5, 100)
    assert s.seeded and s.value == 3.0
    sm.forget("a")
    assert sm.peek("a") is None and sm.peek("b") == 5.0


def test_ewma_rejects_bad_input():
    sm = EwmaSmoother()
    with pytest.raises(ValueError):
        sm.update("k", 1.0, 0.0, 0)
    with pytest.raises(ValueError):
        sm.update("k", 1.0, 1.5, 0)
    with pytest.raises(ValueError):
        sm.update("k", float("nan"), 0.5, 0)


def test_ewma_is_thread_safe():
    sm = EwmaSmoother()
    errors = []

    def worker(key):
        try:
            for _ in range(2000):
                sm.update(key, 1.0, 0.5, 0)
        except Exception as exc:                                  # pragma: no cover
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(f"k{i % 4}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and all(sm.peek(f"k{i}") == 1.0 for i in range(4))


# ── hysteresis gate ────────────────────────────────────────────────────────────

def test_gate_suppresses_a_short_rise_and_passes_a_sustained_one():
    # observed 0.4 at target 0.5 → 1 replica; 0.9 → 2
    g = spike_gate([0.4, 0.9, 0.9, 0.4, 0.9, 0.4], 0.4, 0.5, 3)
    assert g.suppressed and g.active and g.rise_steps == 3 and g.longest_run == 2
    g = spike_gate([0.4, 0.9, 0.9, 0.9, 0.4, 0.4], 0.4, 0.5, 3)
    assert not g.suppressed and g.longest_run == 3 and g.reason == ""
    g = spike_gate([0.4, 0.4, 0.4, 0.4], 0.4, 0.5, 3)           # nothing rises: nothing to gate
    assert not g.suppressed and g.rise_steps == 0


def test_gate_is_inactive_when_it_cannot_judge():
    assert not spike_gate([0.9, 0.9], 0.4, 0.5, 3).active        # horizon shorter than min_steps
    assert not spike_gate([0.9, 0.4, 0.4], 0.4, 0.5, 1).active   # disabled
    assert not spike_gate([0.9, 0.4, 0.4], 0.4, 0.5, 0).active
    assert not spike_gate([], 0.4, 0.5, 3).active


# ── forecast-service breaker ───────────────────────────────────────────────────

def test_service_breaker_lifecycle():
    clock = Clock()
    br = ServiceBreaker(ServiceBreakerConfig(failures=3, open_seconds=30), clock=clock)
    assert br.admit("k").allowed and br.state("k") == CLOSED
    assert br.failure("k") is False and br.failure("k") is False
    assert br.failure("k") is True and br.state("k") == OPEN       # third in a row opens
    a = br.admit("k")
    assert not a.allowed and "open" in a.reason
    clock.t = 31
    assert br.admit("k").allowed and br.state("k") == HALF_OPEN     # one probe
    assert not br.admit("k").allowed                                # the rest wait for it
    assert br.success("k") is True and br.state("k") == CLOSED
    assert br.admit("k").allowed


def test_a_failed_probe_reopens_and_successes_reset_the_count():
    clock = Clock()
    br = ServiceBreaker(ServiceBreakerConfig(failures=2, open_seconds=10), clock=clock)
    br.failure("k")
    assert br.success("k") is False                               # a success in CLOSED resets the count
    assert br.failure("k") is False                               # 1 again, not 2
    assert br.failure("k") is True
    clock.t = 11
    assert br.admit("k").allowed                                  # probe
    assert br.failure("k") is True and br.state("k") == OPEN       # failed probe: open again
    clock.t = 15
    assert not br.admit("k").allowed                              # the hold restarted at t=11


def test_service_breaker_can_be_disabled_and_is_per_key():
    br = ServiceBreaker(ServiceBreakerConfig(failures=0))
    for _ in range(10):
        assert br.failure("k") is False
    assert br.admit("k").allowed
    br = ServiceBreaker(ServiceBreakerConfig(failures=1, open_seconds=60))
    br.failure("a")
    assert not br.admit("a").allowed and br.admit("b").allowed
    br.forget("a")
    assert br.admit("a").allowed
