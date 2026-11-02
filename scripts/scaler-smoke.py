"""
Smoke test of the predictive scaler's gRPC API, run inside the cluster with
the scaler image (grpcio + generated stubs) by scripts/up.sh. Exercises
GetMetricSpec and GetMetrics exactly as KEDA does; with TLS_DIR set it dials
with the client certificate KEDA uses (Secret keda-scaler-client).

Environment: NS (namespace), STEP (stepSeconds), TLS_DIR (optional).
"""

import os
import time

import grpc

from predictive_scaler import externalscaler_pb2 as pb
from predictive_scaler import externalscaler_pb2_grpc as pbg

ns = os.environ["NS"]
tls_dir = os.getenv("TLS_DIR", "")
scheme = "https" if tls_dir else "http"
ref = pb.ScaledObjectRef(name="cpu-burner", namespace=ns, scalerMetadata={
    "prometheusServerAddress": f"http://prometheus.{ns}.svc.cluster.local:9090",
    "query": 'sum(rate(process_cpu_seconds_total{app="cpu-burner"}[1m]))',
    "forecastServiceURL": f"{scheme}://forecast-serving.{ns}.svc.cluster.local:8080",
    "targetValue": "0.35", "capacity": "2.5", "stepSeconds": os.environ["STEP"], "cacheSeconds": "0"})
target = f"predictive-scaler.{ns}.svc.cluster.local:6000"
if tls_dir:
    def read(name):
        with open(os.path.join(tls_dir, name), "rb") as f:
            return f.read()
    creds = grpc.ssl_channel_credentials(root_certificates=read("ca.crt"), private_key=read("tls.key"),
                                         certificate_chain=read("tls.crt"))
    channel = grpc.secure_channel(target, creds)
    print(f"dialing {target} with mTLS (client certificate from {tls_dir})")
else:
    channel = grpc.insecure_channel(target)
    print(f"dialing {target} in plaintext")
stub = pbg.ExternalScalerStub(channel)

spec = stub.GetMetricSpec(ref, timeout=10).metricSpecs[0]
print(f"GetMetricSpec: {spec.metricName} targetSizeFloat={spec.targetSizeFloat}")
assert spec.metricName == "predicted-demand" and abs(spec.targetSizeFloat - 0.35) < 1e-9
for attempt in range(24):
    try:
        m = stub.GetMetrics(pb.GetMetricsRequest(scaledObjectRef=ref, metricName=spec.metricName), timeout=10)
        v = m.metricValues[0]
        print(f"GetMetrics: {v.metricName}={v.metricValueFloat:.4f}")
        break
    except grpc.RpcError as exc:
        print(f"GetMetrics: {exc.code().name} ({exc.details()}); waiting for samples")
        time.sleep(10)
else:
    raise SystemExit("scaler never returned a metric")
