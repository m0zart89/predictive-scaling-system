"""
gRPC tests: a real server on a random port, called through the generated
client stub — the same wire path KEDA uses.
"""

import threading

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from fakes import BASE_META, FakeForecast, FakePrometheus
from predictive_scaler import externalscaler_pb2 as pb
from predictive_scaler import externalscaler_pb2_grpc as pb_grpc
from predictive_scaler.clients import PrometheusError
from predictive_scaler.engine import PredictiveEngine
from predictive_scaler.server import SERVICE_NAME, build_server

WINDOW = [0.2] * 9 + [0.4]


@pytest.fixture
def scaler(metrics):
    prom = FakePrometheus(WINDOW)
    fc = FakeForecast(lambda v: 0.9)
    engine = PredictiveEngine(metrics, prometheus=prom, forecast=fc)
    stop = threading.Event()
    server, port = build_server(engine, metrics, "127.0.0.1:0",
                                stream_interval=0.05, stop_event=stop)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield pb_grpc.ExternalScalerStub(channel), channel, prom, fc
    stop.set()
    channel.close()
    server.stop(grace=None)


def ref(**meta_overrides):
    meta = dict(BASE_META, cacheSeconds="0")
    meta.update(meta_overrides)
    return pb.ScaledObjectRef(name="cpu-burner", namespace="demo", scalerMetadata=meta)


def test_get_metric_spec(scaler):
    stub, *_ = scaler
    spec = stub.GetMetricSpec(ref(targetValue="0.5", metricName="cpu-forecast")).metricSpecs[0]
    assert spec.metricName == "cpu-forecast"
    assert spec.targetSizeFloat == pytest.approx(0.5)
    assert spec.targetSize == 1          # int field never 0, KEDA rejects 0 targets


def test_get_metrics_reports_forecast_demand(scaler):
    stub, *_ = scaler
    resp = stub.GetMetrics(pb.GetMetricsRequest(scaledObjectRef=ref(), metricName="s0-predicted-demand"))
    value = resp.metricValues[0]
    assert value.metricName == "s0-predicted-demand"   # echo what KEDA asked for
    assert value.metricValueFloat == pytest.approx(0.9)
    assert value.metricValue == 1                      # ceil for the int field


def test_is_active_respects_activation_value(scaler):
    stub, *_ = scaler
    assert stub.IsActive(ref(activationValue="0.5")).result is True
    assert stub.IsActive(ref(activationValue="0.95")).result is False


def test_invalid_metadata_is_invalid_argument(scaler):
    stub, *_ = scaler
    with pytest.raises(grpc.RpcError) as err:
        stub.GetMetricSpec(ref(targetValue="zero"))
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "targetValue" in err.value.details()


def test_prometheus_failure_is_unavailable(scaler):
    stub, _, prom, _ = scaler
    prom.exc = PrometheusError("connection refused")
    with pytest.raises(grpc.RpcError) as err:
        stub.GetMetrics(pb.GetMetricsRequest(scaledObjectRef=ref()))
    assert err.value.code() == grpc.StatusCode.UNAVAILABLE


def test_request_counter(scaler, registry):
    stub, *_ = scaler
    stub.IsActive(ref())
    with pytest.raises(grpc.RpcError):
        stub.IsActive(ref(mode="bogus"))
    get = registry.get_sample_value
    assert get("predictive_scaler_grpc_requests_total", {"method": "IsActive", "code": "OK"}) == 1
    assert get("predictive_scaler_grpc_requests_total",
               {"method": "IsActive", "code": "INVALID_ARGUMENT"}) == 1


def test_stream_is_active_pushes_updates(scaler):
    stub, *_ = scaler
    stream = stub.StreamIsActive(ref())
    results = [next(stream).result for _ in range(3)]
    stream.cancel()
    assert results == [True, True, True]


def test_health_service(scaler):
    _, channel, *_ = scaler
    health = health_pb2_grpc.HealthStub(channel)
    for svc in ("", SERVICE_NAME):
        status = health.Check(health_pb2.HealthCheckRequest(service=svc)).status
        assert status == health_pb2.HealthCheckResponse.SERVING
