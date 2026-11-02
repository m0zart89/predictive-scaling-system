"""mTLS on the gRPC server and the HTTP client, with certificates minted by openssl."""

import shutil
import subprocess
import threading

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

from fakes import FakeForecast, FakePrometheus
from predictive_scaler.clients import ssl_context
from predictive_scaler.engine import PredictiveEngine
from predictive_scaler.server import build_health_server, build_server, server_credentials

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")


def mint(tmp_path, name, cn, ca_key, ca_crt, san=None, client=False):
    """A certificate signed by the test CA, laid out like a cert-manager Secret."""
    d = tmp_path / name
    d.mkdir()
    ext = d / "ext.cnf"
    usage = "clientAuth" if client else "serverAuth"
    ext.write_text(f"extendedKeyUsage={usage}\n" + (f"subjectAltName=DNS:{san}\n" if san else ""))
    subprocess.run(["openssl", "req", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", d / "tls.key",
                    "-out", d / "req.csr", "-subj", f"/CN={cn}"], check=True, capture_output=True)
    subprocess.run(["openssl", "x509", "-req", "-in", d / "req.csr", "-CA", ca_crt, "-CAkey", ca_key,
                    "-CAcreateserial", "-out", d / "tls.crt", "-days", "2", "-extfile", ext],
                   check=True, capture_output=True)
    shutil.copy(ca_crt, d / "ca.crt")
    return d


@pytest.fixture
def pki(tmp_path):
    ca_key, ca_crt = tmp_path / "ca.key", tmp_path / "ca.crt"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", ca_key,
                    "-out", ca_crt, "-days", "2", "-subj", "/CN=test-ca"], check=True, capture_output=True)
    other_key, other_crt = tmp_path / "other.key", tmp_path / "other.crt"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", other_key,
                    "-out", other_crt, "-days", "2", "-subj", "/CN=other-ca"], check=True, capture_output=True)
    return {
        "server": mint(tmp_path, "server", "predictive-scaler", ca_key, ca_crt, san="localhost"),
        "client": mint(tmp_path, "client", "keda-operator", ca_key, ca_crt, client=True),
        "stranger": mint(tmp_path, "stranger", "stranger", other_key, other_crt, client=True),
        "ca": ca_crt,
    }


def test_grpc_server_requires_a_client_certificate_from_the_ca(pki, metrics, monkeypatch):
    monkeypatch.setenv("TLS_DIR", str(pki["server"]))
    engine = PredictiveEngine(metrics, prometheus=FakePrometheus([0.2] * 10), forecast=FakeForecast())
    server, port = build_server(engine, metrics, "127.0.0.1:0", stop_event=threading.Event())
    server.start()
    try:
        ca = pki["ca"].read_bytes()

        def channel(cert_dir):
            if cert_dir is None:
                creds = grpc.ssl_channel_credentials(root_certificates=ca)
            else:
                creds = grpc.ssl_channel_credentials(
                    root_certificates=ca, private_key=(cert_dir / "tls.key").read_bytes(),
                    certificate_chain=(cert_dir / "tls.crt").read_bytes())
            return grpc.secure_channel(f"localhost:{port}", creds)

        with channel(pki["client"]) as ch:                       # KEDA with its certificate
            resp = health_pb2_grpc.HealthStub(ch).Check(health_pb2.HealthCheckRequest(), timeout=5)
            assert resp.status == health_pb2.HealthCheckResponse.SERVING
        with channel(None) as ch, pytest.raises(grpc.RpcError):   # no client certificate
            health_pb2_grpc.HealthStub(ch).Check(health_pb2.HealthCheckRequest(), timeout=5)
        with channel(pki["stranger"]) as ch, pytest.raises(grpc.RpcError):   # wrong CA
            health_pb2_grpc.HealthStub(ch).Check(health_pb2.HealthCheckRequest(), timeout=5)
        with pytest.raises(Exception):                            # plaintext against a TLS server
            with grpc.insecure_channel(f"localhost:{port}") as ch:
                health_pb2_grpc.HealthStub(ch).Check(health_pb2.HealthCheckRequest(), timeout=3)
    finally:
        server.stop(grace=None)


def test_no_tls_dir_means_plaintext_and_client_context_loads_the_certificate(pki):
    assert server_credentials("") is None
    assert ssl_context("") is None
    ctx = ssl_context(str(pki["client"]))
    assert ctx is not None and ctx.verify_mode.name == "CERT_REQUIRED"
    assert any("test-ca" in str(c.get("subject")) for c in ctx.get_ca_certs())
    with pytest.raises(FileNotFoundError):
        server_credentials(str(pki["client"].parent / "missing"))


def test_health_server_is_plaintext_for_the_kubelet():
    server, port = build_health_server("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            resp = health_pb2_grpc.HealthStub(ch).Check(
                health_pb2.HealthCheckRequest(service="externalscaler.ExternalScaler"), timeout=5)
            assert resp.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        server.stop(grace=None)
