"""bench/: the load profile and the report's metric arithmetic (the real runs only happen in CI)."""

import importlib.util
import json
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "bench", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


load = _load("load")
report = _load("report")


def test_profile_is_quiet_at_night_and_ramps_sharply_in_the_morning():
    assert load.clients_at(3) == 1
    assert load.clients_at(6) == 1
    assert load.clients_at(7) == 4          # halfway up the 06-08 ramp (1 → 6)
    assert load.clients_at(10) == 6
    assert load.clients_at(12.5) == 5       # lunch dip
    assert load.clients_at(15) == 7
    assert load.clients_at(24) == load.clients_at(0)
    assert report.RAMPS == load.RAMPS


def synthetic(label, t0=1_000_000, step=15, day=2400, scale_delay=120):
    """One compressed day: clients follow the profile; replicas follow the
    clients after `scale_delay` seconds; p95 is high while under-provisioned."""
    series = {k: [] for k in ("clients", "sim_hour", "replicas", "p95", "forecast_decisions",
                              "observed_decisions")}
    n = day // step
    for i in range(n + 1):
        t = t0 + i * step
        hour = (i * step) / day * 24
        clients = load.clients_at(hour)
        lagged = load.clients_at(max(0.0, (i * step - scale_delay) / day * 24))
        replicas = max(1, (lagged + 1) // 2)
        p95 = 0.1 if replicas * 2 >= clients else 0.1 + 0.3 * (clients - replicas * 2)
        series["clients"].append([t, clients])
        series["sim_hour"].append([t, hour])
        series["replicas"].append([t, replicas])
        series["p95"].append([t, p95])
        series["forecast_decisions"].append([t, i * 2.0])
        series["observed_decisions"].append([t, 5.0])
    return {"label": label, "start": t0, "end": t0 + day, "step": step, "meta": {},
            "series": series,
            "totals": {"requests": 1000.0, "errors": 10.0, "requests_under_500ms": 900.0,
                       "latency_count": 1000.0, "latency_sum": 150.0}}


def test_ramp_windows_are_found_from_the_simulated_clock():
    run = synthetic("hpa")
    run["series"] = {k: [(t, v) for t, v in s] for k, s in run["series"].items()}
    windows = report.ramp_windows(run["series"]["sim_hour"])
    assert [w[0] for w in windows] == [6.0, 13.0]
    start_6 = windows[0][1]
    assert (start_6 - run["start"]) / 2400 * 24 == pytest.approx(6.0, abs=0.2)


def test_summary_metrics(tmp_path):
    run = synthetic("hpa", scale_delay=120)
    path = tmp_path / "hpa.json"
    path.write_text(json.dumps(run))
    s = report.summarize(report.load(str(path)))
    assert s["slow_share"] == pytest.approx(0.1)
    assert s["error_share"] == pytest.approx(0.01)
    assert s["mean_latency"] == pytest.approx(0.15)
    assert s["replicas_max"] == 4
    # replicas = ceil(clients / 2) lagged by 120 s: the ramp reaches 3 clients 60 s in,
    # so the first scale-up lands 180 s after the ramp starts (±1 sample of 15 s)
    assert s["ramps"][0]["time_to_scale_up"] == pytest.approx(180, abs=15)
    assert s["ramps"][0]["p95_max"] > s["p95_day_mean"]
    assert s["replica_minutes"] == pytest.approx(sum(v for _, v in run["series"]["replicas"]) * 15 / 60)
    assert s["forecast_decisions"] == pytest.approx(2.0 * (2400 // 15))


def test_report_markdown_compares_runs(tmp_path):
    a, b = synthetic("hpa", scale_delay=180), synthetic("predictive", scale_delay=30)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    out = tmp_path / "benchmark.md"
    assert report.main([str(pa), str(pb), "--out", str(out), "--no-png"]) == 0
    text = out.read_text()
    assert "| Metric | A: reactive HPA | B: predictive | B vs A |" in text
    assert "time to first scale-up" in text and "better" in text
    sa, sb = report.summarize(report.load(str(pa))), report.summarize(report.load(str(pb)))
    assert sb["ramps"][0]["time_to_scale_up"] < sa["ramps"][0]["time_to_scale_up"]
    assert sb["p95_ramp_mean"] < sa["p95_ramp_mean"]


def test_delta_wording():
    assert report.delta(100, 50) == " (−50% better)"
    assert report.delta(100, 150) == " (+50% worse)"
    assert report.delta(None, 1) == "" and report.delta(0, 1) == ""
