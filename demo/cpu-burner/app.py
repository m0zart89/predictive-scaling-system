"""
cpu-burner — a demo workload whose CPU use is driven by request load.

GET /work?ms=50   spins the CPU for ~50 ms and returns
GET /healthz      liveness/readiness
GET /metrics      Prometheus metrics (includes process_cpu_seconds_total)

Total demand across all replicas is

    sum(rate(process_cpu_seconds_total{app="cpu-burner"}[1m]))   # in cores

which does not change when the workload scales out — the load is spread over
more pods, not multiplied. That makes it the right signal to forecast.
"""

import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

MAX_MS = int(os.getenv("MAX_WORK_MS", "1000"))

REQUESTS = Counter("cpu_burner_requests_total", "Work requests served")
LATENCY = Histogram(
    "cpu_burner_request_duration_seconds", "End-to-end /work latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)


def burn(ms: int) -> int:
    """Busy-loop for `ms` milliseconds of CPU time; returns iterations."""
    deadline = time.process_time() + ms / 1000
    n = 0
    while time.process_time() < deadline:
        n += 1
    return n


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/work":
            started = time.perf_counter()
            try:
                ms = int(urllib.parse.parse_qs(url.query).get("ms", ["50"])[0])
            except ValueError:
                return self._send(400, b"ms must be an integer\n")
            ms = max(0, min(ms, MAX_MS))
            burn(ms)
            REQUESTS.inc()
            LATENCY.observe(time.perf_counter() - started)
            return self._send(200, f"burned {ms}ms\n".encode())
        if url.path == "/healthz":
            return self._send(200, b"ok\n")
        if url.path == "/metrics":
            return self._send(200, generate_latest(), CONTENT_TYPE_LATEST)
        return self._send(404, b"not found\n")

    def _send(self, code, body, ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep pod logs quiet under load


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"cpu-burner listening on :{port}", flush=True)
    ThreadingHTTPServer(("", port), Handler).serve_forever()
