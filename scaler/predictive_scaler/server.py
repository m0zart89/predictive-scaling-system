"""
KEDA external scaler gRPC server.

KEDA calls this service for every ScaledObject whose trigger is:

    triggers:
      - type: external
        metadata:
          scalerAddress: predictive-scaler.keda:6000
          ...

and divides the value from GetMetrics by the targetSize from GetMetricSpec to
get the desired replica count (AverageValue semantics), the same way the HPA
treats any external metric.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import threading
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from prometheus_client import start_http_server

from . import externalscaler_pb2 as pb
from . import externalscaler_pb2_grpc as pb_grpc
from .clients import PrometheusError
from .config import ConfigError, ScalerConfig, parse_metadata
from .engine import PredictiveEngine
from .metrics import ScalerMetrics

log = logging.getLogger("predictive-scaler")

SERVICE_NAME = "externalscaler.ExternalScaler"


class ExternalScalerServicer(pb_grpc.ExternalScalerServicer):
    def __init__(self, engine: PredictiveEngine, metrics: ScalerMetrics,
                 stream_interval: float = 15.0, stop_event: threading.Event | None = None):
        self.engine = engine
        self.metrics = metrics
        self.stream_interval = stream_interval
        self.stop_event = stop_event or threading.Event()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _config(self, ref: pb.ScaledObjectRef, method: str, context) -> ScalerConfig:
        try:
            return parse_metadata(dict(ref.scalerMetadata))
        except ConfigError as exc:
            self._abort(context, method, grpc.StatusCode.INVALID_ARGUMENT,
                        f"{ref.namespace}/{ref.name}: {exc}")

    def _decide(self, ref: pb.ScaledObjectRef, cfg: ScalerConfig, method: str, context):
        try:
            return self.engine.decide(ref.namespace, ref.name, cfg)
        except PrometheusError as exc:
            # KEDA reacts to an error by applying the ScaledObject `fallback`
            # (if configured) — that is the right owner of "no data" policy.
            self._abort(context, method, grpc.StatusCode.UNAVAILABLE,
                        f"{ref.namespace}/{ref.name}: {exc}")

    def _abort(self, context, method: str, code: grpc.StatusCode, message: str):
        self.metrics.grpc_requests.labels(method=method, code=code.name).inc()
        log.warning("%s failed: %s", method, message)
        context.abort(code, message)  # raises

    def _ok(self, method: str):
        self.metrics.grpc_requests.labels(method=method, code="OK").inc()

    # ── ExternalScaler API ─────────────────────────────────────────────────────
    def IsActive(self, request, context):
        cfg = self._config(request, "IsActive", context)
        decision = self._decide(request, cfg, "IsActive", context)
        self._ok("IsActive")
        return pb.IsActiveResponse(result=decision.value > cfg.activation_value)

    def StreamIsActive(self, request, context):
        # Only used by `external-push` triggers. Pushes the active state on a
        # fixed interval until KEDA disconnects or the server shuts down.
        cfg = self._config(request, "StreamIsActive", context)
        self._ok("StreamIsActive")
        while context.is_active() and not self.stop_event.is_set():
            try:
                decision = self.engine.decide(request.namespace, request.name, cfg)
            except PrometheusError as exc:
                log.warning("StreamIsActive %s/%s: %s", request.namespace, request.name, exc)
            else:
                yield pb.IsActiveResponse(result=decision.value > cfg.activation_value)
            self.stop_event.wait(self.stream_interval)

    def GetMetricSpec(self, request, context):
        cfg = self._config(request, "GetMetricSpec", context)
        self._ok("GetMetricSpec")
        return pb.GetMetricSpecResponse(metricSpecs=[
            pb.MetricSpec(
                metricName=cfg.metric_name,
                # int64 field kept for KEDA versions that ignore the float one
                targetSize=max(1, math.ceil(cfg.target_value)),
                targetSizeFloat=cfg.target_value,
            )
        ])

    def GetMetrics(self, request, context):
        ref = request.scaledObjectRef
        cfg = self._config(ref, "GetMetrics", context)
        decision = self._decide(ref, cfg, "GetMetrics", context)
        self._ok("GetMetrics")
        return pb.GetMetricsResponse(metricValues=[
            pb.MetricValue(
                metricName=request.metricName or cfg.metric_name,
                metricValue=int(math.ceil(decision.value)),
                metricValueFloat=decision.value,
            )
        ])


def build_server(engine: PredictiveEngine, metrics: ScalerMetrics, address: str,
                 max_workers: int = 10, stream_interval: float = 15.0,
                 stop_event: threading.Event | None = None) -> tuple[grpc.Server, int]:
    """Create (but do not start) the gRPC server. Returns (server, bound_port)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_ExternalScalerServicer_to_server(
        ExternalScalerServicer(engine, metrics, stream_interval, stop_event), server)

    # Standard gRPC health service, so Kubernetes can use native `grpc` probes.
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    for svc in ("", SERVICE_NAME):
        health_servicer.set(svc, health_pb2.HealthCheckResponse.SERVING)

    port = server.add_insecure_port(address)
    return server, port


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    grpc_port = int(os.getenv("GRPC_PORT", "6000"))
    metrics_port = int(os.getenv("METRICS_PORT", "9100"))
    stream_interval = float(os.getenv("STREAM_INTERVAL_SECONDS", "15"))

    metrics = ScalerMetrics()
    engine = PredictiveEngine(metrics)
    stop_event = threading.Event()
    server, port = build_server(engine, metrics, f"[::]:{grpc_port}",
                                stream_interval=stream_interval, stop_event=stop_event)

    start_http_server(metrics_port)
    server.start()
    log.info("Predictive scaler listening: gRPC :%d, metrics :%d", port, metrics_port)

    def _shutdown(signum, _frame):
        log.info("Received signal %d, shutting down", signum)
        stop_event.set()
        server.stop(grace=10)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
