#!/usr/bin/env python3
"""
Export the series a benchmark phase needs from Prometheus into one JSON file.

    python bench/collect.py --label hpa --start 1700000000 --end 1700002400 \
        --out bench-results/hpa.json [--prometheus http://127.0.0.1 --host prometheus.localhost]

`--host` sets the Host header, so the kind Ingress is reachable at 127.0.0.1
without DNS. Standard library only; runs on the CI runner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

STEP = 15

# name → PromQL. Rates use 1m windows: short enough to show a ramp, long enough
# to cover four 15-second scrapes.
RANGE_QUERIES = {
    "clients": "max(bench_clients)",
    "sim_hour": "max(bench_sim_hour)",
    "replicas": 'count(up{app="cpu-burner"} == 1)',
    "cpu_demand": 'sum(rate(process_cpu_seconds_total{app="cpu-burner"}[1m]))',
    "p50": "histogram_quantile(0.50, sum by (le) (rate(bench_latency_seconds_bucket[1m])))",
    "p95": "histogram_quantile(0.95, sum by (le) (rate(bench_latency_seconds_bucket[1m])))",
    "slow_share": '1 - sum(rate(bench_latency_seconds_bucket{le="0.5"}[1m])) '
                  "/ sum(rate(bench_latency_seconds_count[1m]))",
    "request_rate": "sum(rate(bench_requests_total[1m]))",
    "error_rate": 'sum(rate(bench_requests_total{result="error"}[1m]))',
    # predictive run only (absent otherwise)
    "desired_replicas": "max(predictive_scaler_desired_replicas)",
    "observed_demand": "max(predictive_scaler_observed_demand)",
    "predicted_demand": "max(predictive_scaler_predicted_demand)",
    "reported_demand": "max(predictive_scaler_reported_demand)",
    "forecast_decisions": 'sum(predictive_scaler_decisions_total{source="forecast"})',
    "observed_decisions": 'sum(predictive_scaler_decisions_total{source="observed"})',
    "rolling_mae_step1": 'max(forecast_rolling_mae{horizon_step="1"})',
}

# Totals over the whole phase, evaluated at --end with a range of its length.
TOTAL_QUERIES = {
    "requests": "sum(increase(bench_requests_total[{r}s]))",
    "errors": 'sum(increase(bench_requests_total{{result="error"}}[{r}s]))',
    "requests_under_500ms": 'sum(increase(bench_latency_seconds_bucket{{le="0.5"}}[{r}s]))',
    "latency_count": "sum(increase(bench_latency_seconds_count[{r}s]))",
    "latency_sum": "sum(increase(bench_latency_seconds_sum[{r}s]))",
}


def api(base: str, host: str | None, path: str, params: dict, timeout: float = 60) -> dict:
    url = f"{base.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    if host:
        req.add_unredirected_header("Host", host)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = json.load(resp)
    if body.get("status") != "success":
        raise RuntimeError(f"Prometheus error for {params.get('query')!r}: {body.get('error')}")
    return body["data"]


def query_range(base, host, expr, start, end):
    data = api(base, host, "/api/v1/query_range",
               {"query": expr, "start": start, "end": end, "step": STEP})
    result = data.get("result") or []
    if not result:
        return []
    return [[float(ts), float(v)] for ts, v in result[0]["values"] if v not in ("NaN", "+Inf", "-Inf")]


def query_scalar(base, host, expr, at):
    data = api(base, host, "/api/v1/query", {"query": expr, "time": at})
    result = data.get("result") or []
    return float(result[0]["value"][1]) if result else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="phase name, e.g. hpa or predictive")
    ap.add_argument("--start", type=float, required=True, help="unix seconds")
    ap.add_argument("--end", type=float, required=True, help="unix seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prometheus", default="http://127.0.0.1")
    ap.add_argument("--host", default="prometheus.localhost", help="Host header (ingress); '' to skip")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                    help="extra metadata to store (commit, runner, settings)")
    args = ap.parse_args(argv)

    host = args.host or None
    out = {
        "label": args.label, "start": args.start, "end": args.end, "step": STEP,
        "collected_at": time.time(),
        "meta": dict(kv.split("=", 1) for kv in args.meta),
        "series": {}, "totals": {},
    }
    for name, expr in RANGE_QUERIES.items():
        try:
            out["series"][name] = query_range(args.prometheus, host, expr, args.start, args.end)
        except Exception as exc:   # one missing series must not lose the rest
            print(f"warning: {name}: {exc}", file=sys.stderr)
            out["series"][name] = []
    length = int(args.end - args.start)
    for name, expr in TOTAL_QUERIES.items():
        try:
            out["totals"][name] = query_scalar(args.prometheus, host, expr.format(r=length), args.end)
        except Exception as exc:
            print(f"warning: {name}: {exc}", file=sys.stderr)
            out["totals"][name] = None

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    have = [n for n, s in out["series"].items() if s]
    print(f"{args.label}: {len(have)} series ({', '.join(have)}), totals {out['totals']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
