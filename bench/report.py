#!/usr/bin/env python3
"""
Turn two collected benchmark phases into a Markdown report and PNG charts.

    python bench/report.py bench-results/hpa.json bench-results/predictive.json \
        --out docs/benchmark.md --png-dir docs/benchmark

Metrics per run (all from the client-side bench_* series and the pod count
Prometheus scraped; see bench/collect.py):

* p95 latency during the ramps (mean and worst 1-minute p95 inside the ramp
  windows of the load profile) and over the whole day;
* share of requests slower than 500 ms, and client errors (timeouts);
* replica-minutes: the integral of running cpu-burner pods;
* for each ramp: replicas running when it starts, and the time until the
  run had the capacity the ramp needed (the smaller of the two runs' peak
  replica count inside the ramp, so both can reach it; 0 s = already there).

The ramps are located from bench_sim_hour, not from wall-clock offsets, so a
late-starting load Job does not shift them. The charts put both runs on one
simulated-time axis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple

Series = List[Tuple[float, float]]

RAMPS = [(6.0, 8.5), (13.0, 14.5)]   # must match bench/load.py


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    d["series"] = {k: [(float(t), float(v)) for t, v in s] for k, s in d["series"].items()}
    return d


def value_at(series: Series, t: float) -> Optional[float]:
    """Last sample at or before t."""
    best = None
    for ts, v in series:
        if ts <= t:
            best = v
        else:
            break
    return best


def ramp_windows(sim_hour: Series, ramps=RAMPS) -> List[Tuple[float, float, float]]:
    """(ramp start hour, start ts, end ts) for every ramp occurrence in the run."""
    out = []
    for h0, h1 in ramps:
        inside, start = False, None
        for ts, hour in sim_hour:
            if not inside and h0 <= hour < h1:
                inside, start = True, ts
            elif inside and not (h0 <= hour < h1):
                out.append((h0, start, ts))
                inside = False
        if inside:
            out.append((h0, start, sim_hour[-1][0]))
    return sorted(out, key=lambda w: w[1])


def within(series: Series, start: float, end: float) -> List[float]:
    return [v for ts, v in series if start <= ts <= end]


def mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x == x]   # drop NaN
    return sum(xs) / len(xs) if xs else None


def time_to_capacity(replicas: Series, ramp_start: float, need: float,
                     limit: float = 1800) -> Optional[float]:
    """Seconds from ramp_start until `need` replicas were running (0 if already)."""
    if need is None:
        return None
    at_start = value_at(replicas, ramp_start)
    if at_start is not None and at_start >= need:
        return 0.0
    for ts, v in replicas:
        if ramp_start <= ts <= ramp_start + limit and v >= need:
            return ts - ramp_start
    return None


def ramp_need(runs: Sequence[dict], ramp_index: int, tail: float = 120) -> Optional[float]:
    """Replicas a ramp needed: the smaller of the runs' peaks inside it (plus `tail` s)."""
    peaks = []
    for run in runs:
        s = run["series"]
        windows = ramp_windows(s.get("sim_hour", []))
        if ramp_index >= len(windows):
            return None
        _, a, b = windows[ramp_index]
        inside = within(s.get("replicas", []), a, b + tail)
        if not inside:
            return None
        peaks.append(max(inside))
    return min(peaks) if peaks else None


def replica_minutes(replicas: Series, step: float) -> Optional[float]:
    if not replicas:
        return None
    return sum(v for _, v in replicas) * step / 60


def summarize(run: dict, needs: Sequence[Optional[float]] = ()) -> dict:
    s, totals, step = run["series"], run["totals"], run.get("step", 15)
    windows = ramp_windows(s.get("sim_hour", []))
    ramp_p95 = [v for _, a, b in windows for v in within(s.get("p95", []), a, b)]
    ramps = []
    for i, (h0, a, b) in enumerate(windows):
        p95 = within(s.get("p95", []), a, b)
        need = needs[i] if i < len(needs) else None
        ramps.append({
            "hour": h0, "start": a, "end": b,
            "p95_mean": mean(p95), "p95_max": max(p95) if p95 else None,
            "need": need,
            "time_to_capacity": time_to_capacity(s.get("replicas", []), a, need),
            "replicas_at_start": value_at(s.get("replicas", []), a),
        })
    req = totals.get("requests") or 0
    under = totals.get("requests_under_500ms")
    count = totals.get("latency_count")
    return {
        "label": run["label"],
        "duration_min": (run["end"] - run["start"]) / 60,
        "requests": req,
        "errors": totals.get("errors") or 0,
        "error_share": (totals.get("errors") or 0) / req if req else None,
        "slow_share": (1 - under / count) if (under is not None and count) else None,
        "mean_latency": (totals["latency_sum"] / count) if (totals.get("latency_sum") is not None and count) else None,
        "p95_day_mean": mean([v for _, v in s.get("p95", [])]),
        "p95_ramp_mean": mean(ramp_p95),
        "p95_ramp_max": max(ramp_p95) if ramp_p95 else None,
        "replica_minutes": replica_minutes(s.get("replicas", []), step),
        "replicas_max": max((v for _, v in s.get("replicas", [])), default=None),
        "forecast_decisions": _counter_increase(s.get("forecast_decisions")),
        # a counter series that never appeared means the scaler never fell back
        "observed_decisions": _counter_increase(s.get("observed_decisions"))
        if s.get("observed_decisions") or not s.get("forecast_decisions") else 0.0,
        "rolling_mae_step1": s["rolling_mae_step1"][-1][1] if s.get("rolling_mae_step1") else None,
        "ramps": ramps,
    }


def _counter_increase(series: Optional[Series]) -> Optional[float]:
    if not series:
        return None
    return series[-1][1] - series[0][1]


# ── formatting ─────────────────────────────────────────────────────────────────

def fmt(v, unit="", digits=2) -> str:
    if v is None:
        return "n/a"
    if unit == "%":
        return f"{100 * v:.{digits}f}%"
    if unit == "s":
        return f"{v:.{digits}f} s"
    if unit == "ms":
        return f"{1000 * v:.0f} ms"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:.{digits}f}{unit}"
    return f"{v:.0f}{unit}"


def delta(a, b, lower_is_better=True) -> str:
    if a in (None, 0) or b is None:
        return ""
    change = (b - a) / abs(a)
    sign = "−" if change < 0 else "+"
    good = (change < 0) == lower_is_better
    return f" ({sign}{abs(change) * 100:.0f}%{' better' if good else ' worse'})"


def markdown(a: dict, b: dict, runs: Sequence[dict], png: Optional[str]) -> str:
    meta = runs[-1].get("meta", {})
    lines = [
        "# Benchmark: predictive scaler vs reactive HPA",
        "",
        "_Generated by `bench/report.py` from a run of `.github/workflows/benchmark.yml`"
        + (f" (commit `{meta['commit'][:7]}`" if meta.get("commit") else "")
        + (f", {meta['date']})" if meta.get("date") else ")")
        + "; do not edit by hand. How it is measured: [bench/README.md](../bench/README.md)._",
        "",
        f"Same kind cluster, same cpu-burner workload ({meta.get('cpu_request', '500m')} CPU per "
        f"replica, up to {meta.get('max_replicas', '?')} replicas), same compressed day of load "
        f"({a['duration_min']:.0f} min per run): quiet night, sharp ramp at 06:00, lunch dip, "
        f"second ramp at 13:00, evening decline. Run A scales with a plain CPU HPA at "
        f"{meta.get('hpa_target', '70')}%; run B with the KEDA predictive scaler (`mode=max`, "
        f"`forecastAggregation=max`, target {meta.get('target_value', '0.35')} cores per replica, "
        f"{meta.get('step_seconds', '30')} s step) on a model trained on the warm-up days of the "
        "same pattern.",
        "",
        "| Metric | A: reactive HPA | B: predictive | B vs A |",
        "|---|---|---|---|",
        f"| p95 latency during ramps (mean of 1-min p95) | {fmt(a['p95_ramp_mean'], 'ms')} | "
        f"{fmt(b['p95_ramp_mean'], 'ms')} |{delta(a['p95_ramp_mean'], b['p95_ramp_mean'])} |",
        f"| p95 latency during ramps (worst minute) | {fmt(a['p95_ramp_max'], 'ms')} | "
        f"{fmt(b['p95_ramp_max'], 'ms')} |{delta(a['p95_ramp_max'], b['p95_ramp_max'])} |",
        f"| p95 latency over the day (mean of 1-min p95) | {fmt(a['p95_day_mean'], 'ms')} | "
        f"{fmt(b['p95_day_mean'], 'ms')} |{delta(a['p95_day_mean'], b['p95_day_mean'])} |",
        f"| Requests slower than 500 ms | {fmt(a['slow_share'], '%')} | {fmt(b['slow_share'], '%')} |"
        f"{delta(a['slow_share'], b['slow_share'])} |",
        f"| Client errors (timeouts) | {fmt(round(a['errors']))} of {fmt(round(a['requests']))} | "
        f"{fmt(round(b['errors']))} of {fmt(round(b['requests']))} | |",
        f"| Replica-minutes | {fmt(a['replica_minutes'])} | {fmt(b['replica_minutes'])} |"
        f"{delta(a['replica_minutes'], b['replica_minutes'])} |",
        f"| Peak replicas | {fmt(a['replicas_max'])} | {fmt(b['replicas_max'])} | |",
    ]
    for i, (ra, rb) in enumerate(zip(a["ramps"], b["ramps"])):
        label = f"Ramp {i + 1} ({ra['hour']:02.0f}:00)"
        lines.append(
            f"| {label}: replicas running at its start | {fmt(ra['replicas_at_start'])} | "
            f"{fmt(rb['replicas_at_start'])} | |")
        lines.append(
            f"| {label}: time to {fmt(ra['need'])} replicas | "
            f"{fmt(ra['time_to_capacity'], 's', 0)} | {fmt(rb['time_to_capacity'], 's', 0)} |"
            f"{delta(ra['time_to_capacity'], rb['time_to_capacity'])} |")
        lines.append(
            f"| {label}: p95 latency (mean of 1-min p95) | {fmt(ra['p95_mean'], 'ms')} | "
            f"{fmt(rb['p95_mean'], 'ms')} |{delta(ra['p95_mean'], rb['p95_mean'])} |")
    if b.get("forecast_decisions") is not None:
        lines.append(f"| Scaler decisions from forecast / observed (run B) | — | "
                     f"{fmt(b['forecast_decisions'])} / {fmt(b['observed_decisions'])} | |")
    if b.get("rolling_mae_step1") is not None:
        lines.append(f"| Forecast rolling MAE, step 1, end of run B (model units) | — | "
                     f"{fmt(b['rolling_mae_step1'], '', 3)} | |")
    lines.append("")
    if png:
        lines += [f"![load, replicas and p95 latency for both runs]({png})", ""]
    lines += [
        "## Reading the numbers",
        "",
        "- *Time to N replicas* runs from the moment the simulated clock enters the ramp until N "
        "cpu-burner pods are running, where N is the smaller of the two runs' peaks inside the "
        "ramp (0 s: already there). The HPA reacts to CPU already above target; the predictive "
        "scaler reports `max(forecast, observed)`, so it can only be earlier when the forecast "
        "rises before the observed demand does, or when it never scaled down.",
        "- *Replica-minutes* are the cost side: a scaler that is early but over-provisions pays here.",
        "- The chart's *desired* line is the scaler's `ceil(reported / targetValue)`. The HPA KEDA "
        "manages ignores changes within its 10% tolerance, so desired can sit one above the "
        "running count without anything being unschedulable.",
        "- Both runs share one small node (the CI runner), so absolute latencies are noisy; the "
        "comparison between runs is the point, and it is one sample. The weekly run keeps the "
        "history in the workflow artifacts.",
        "",
    ]
    return "\n".join(lines)


def plot(runs: Sequence[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {"hpa": "#888888", "predictive": "#1f77b4"}
    for run in runs:
        s, t0 = run["series"], run["start"]
        label = run["label"]
        color = colors.get(label)
        rel = lambda series: ([(ts - t0) / 60 for ts, _ in series], [v for _, v in series])  # noqa: E731
        if s.get("clients"):
            axes[0].plot(*rel(s["clients"]), label=f"{label}: clients", color=color, drawstyle="steps-post")
        if s.get("replicas"):
            axes[1].plot(*rel(s["replicas"]), label=f"{label}: replicas", color=color, drawstyle="steps-post")
        if label == "predictive" and s.get("desired_replicas"):
            axes[1].plot(*rel(s["desired_replicas"]), label="predictive: desired", color=color,
                         linestyle=":", drawstyle="steps-post")
        if s.get("p95"):
            axes[2].plot(*rel(s["p95"]), label=f"{label}: p95", color=color)
    for h0, h1 in RAMPS:
        for ax in axes:
            day = runs[-1]["series"].get("sim_hour")
            if day:
                # shade the ramp window of the last run (same profile for all)
                w = ramp_windows(day, [(h0, h1)])
                for _, a, b in w:
                    ax.axvspan((a - runs[-1]["start"]) / 60, (b - runs[-1]["start"]) / 60,
                               color="#ffcc00", alpha=0.15)
    axes[0].set_ylabel("load clients")
    axes[1].set_ylabel("cpu-burner replicas")
    axes[2].set_ylabel("client p95 latency (s)")
    axes[2].set_xlabel("minutes into the compressed day (ramps shaded)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Reactive HPA vs predictive scaler — one compressed day of load")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hpa", help="collected JSON of the reactive HPA run")
    ap.add_argument("predictive", help="collected JSON of the predictive run")
    ap.add_argument("--out", default="docs/benchmark.md")
    ap.add_argument("--png-dir", default="docs/benchmark")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--json", help="also write the computed summary here")
    args = ap.parse_args(argv)

    runs = [load(args.hpa), load(args.predictive)]
    needs = [ramp_need(runs, i) for i in range(len(RAMPS))]
    a, b = summarize(runs[0], needs), summarize(runs[1], needs)
    png_rel = None
    if not args.no_png:
        png_path = os.path.join(args.png_dir, "benchmark.png")
        plot(runs, png_path)
        png_rel = os.path.relpath(png_path, os.path.dirname(os.path.abspath(args.out)))
    runs[-1].setdefault("meta", {}).setdefault("date", dt.date.today().isoformat())
    text = markdown(a, b, runs, png_rel)
    with open(args.out, "w") as f:
        f.write(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"hpa": a, "predictive": b}, f, indent=1)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
