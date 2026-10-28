"""
Load generator for the predictive-vs-reactive benchmark.

Replays a daily traffic pattern compressed in time (24 h → DAY_SECONDS, 40 min
by default): a quiet night, a sharp morning ramp, a lunch dip, a second ramp,
an evening decline. The pattern is expressed as the number of concurrent
clients; each client calls cpu-burner's /work back to back with a short think
time, so offered CPU demand is proportional to the client count.

Runs as a Job in the cluster (bench/load-job.yaml) with the cpu-burner image
(python + prometheus-client) and exposes its own metrics on :8000 so the same
Prometheus that feeds the scaler records what the clients experienced:

  bench_requests_total{result="ok|error"}   client-side outcome (timeouts count as errors)
  bench_latency_seconds                     client-side latency histogram, queueing included
  bench_clients                             clients currently active
  bench_sim_hour                            simulated hour of day (0-24), to locate the ramps

Environment: TARGET (http://cpu-burner:8080), DAY_SECONDS (2400), DAYS (1),
WORK_MS (25), THINK_MS (100), METRICS_PORT (8000).
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request

from prometheus_client import Counter, Gauge, Histogram, start_http_server

TARGET = os.getenv("TARGET", "http://cpu-burner:8080")
DAY_SECONDS = float(os.getenv("DAY_SECONDS", "2400"))
DAYS = float(os.getenv("DAYS", "1"))
WORK_MS = int(os.getenv("WORK_MS", "25"))
THINK_MS = int(os.getenv("THINK_MS", "100"))
TIMEOUT = float(os.getenv("TIMEOUT_SECONDS", "10"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

# (hour, concurrent clients); linear in between. Two sharp ramps (06-08 and
# 13-14) are what the forecast is supposed to anticipate.
PROFILE = [
    (0, 1), (6, 1), (8, 6), (12, 6), (13, 4), (14, 7), (17, 7), (22, 2), (24, 1),
]
# Ramps as (start_hour, end_hour): the report measures latency inside these.
RAMPS = [(6, 8.5), (13, 14.5)]

REQUESTS = Counter("bench_requests_total", "Client requests by outcome", ["result"])
LATENCY = Histogram(
    "bench_latency_seconds", "Client-side request latency (includes queueing)",
    buckets=(0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10),
)
CLIENTS = Gauge("bench_clients", "Clients currently generating load")
SIM_HOUR = Gauge("bench_sim_hour", "Simulated hour of day")


def clients_at(hour: float) -> int:
    """Piecewise-linear client count for a simulated hour of day."""
    hour = hour % 24
    for (h0, c0), (h1, c1) in zip(PROFILE, PROFILE[1:]):
        if h0 <= hour <= h1:
            return round(c0 + (c1 - c0) * (hour - h0) / (h1 - h0))
    return PROFILE[-1][1]


def worker(stop: threading.Event, active: threading.Event) -> None:
    url = f"{TARGET}/work?ms={WORK_MS}"
    while not stop.is_set():
        if not active.wait(timeout=0.5):
            continue
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310
                resp.read()
            result = "ok"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            result = "error"
        LATENCY.observe(time.perf_counter() - started)
        REQUESTS.labels(result=result).inc()
        if THINK_MS:
            time.sleep(THINK_MS / 1000)


def main() -> None:
    start_http_server(METRICS_PORT)
    max_clients = max(c for _, c in PROFILE)
    stop = threading.Event()
    gates = [threading.Event() for _ in range(max_clients)]
    for gate in gates:
        threading.Thread(target=worker, args=(stop, gate), daemon=True).start()

    print(f"bench load: {DAYS} day(s) of {DAY_SECONDS:.0f}s against {TARGET}, "
          f"work={WORK_MS}ms think={THINK_MS}ms, up to {max_clients} clients", flush=True)
    t0 = time.time()
    total = DAYS * DAY_SECONDS
    last = None
    while True:
        elapsed = time.time() - t0
        if elapsed >= total:
            break
        hour = (elapsed % DAY_SECONDS) / DAY_SECONDS * 24
        n = clients_at(hour)
        SIM_HOUR.set(hour)
        if n != last:
            for i, gate in enumerate(gates):
                gate.set() if i < n else gate.clear()
            CLIENTS.set(n)
            print(f"t+{elapsed:6.0f}s  day {int(elapsed // DAY_SECONDS) + 1}  "
                  f"hour {hour:5.2f}  clients {n}", flush=True)
            last = n
        time.sleep(1)

    stop.set()
    for gate in gates:
        gate.set()
    CLIENTS.set(0)
    ok = REQUESTS.labels(result="ok")._value.get()
    err = REQUESTS.labels(result="error")._value.get()
    print(f"done: {ok:.0f} ok, {err:.0f} errors", flush=True)
    time.sleep(20)   # let Prometheus scrape the final counters


if __name__ == "__main__":
    main()
